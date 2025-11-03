#!/usr/bin/env python3
"""
Qwen 2.5-14B Checkpoint Resume + Multi-GPU 최적화
- checkpoint-1250에서 재시작
- Multi-GPU 훈련 (2× H100)
- 배치 크기 최적화 (속도 1.5~2배 향상)
"""

import os
import sys
import torch
from datetime import datetime
from typing import Optional
from dataclasses import dataclass
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 모니터링 모듈
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'z_util'))
from cpu_mntrg import CPUMonitor
from gpu_mnrtg import GPUMonitor

from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 글로벌 모니터
cpu_monitor = CPUMonitor()
gpu_monitor = GPUMonitor()

from dotenv import load_dotenv
load_dotenv()

os.getenv("TOKENIZER")

def log_system_resources(stage: str):
    """시스템 리소스 상태 로깅 (상세)"""
    logger.info("\n" + "="*80)
    logger.info(f"[{stage}] 시스템 리소스 상세 모니터링")
    logger.info("="*80)
    
    # CPU 상세
    cpu_monitor.log_snapshot(logger, stage)
    
    # GPU 상세
    if gpu_monitor.available:
        gpu_monitor.log_all_gpus(logger, stage)
        
        # 메모리 압박 확인
        for i in range(gpu_monitor.device_count):
            if not gpu_monitor.check_memory_available(i, required_gb=2.0):
                logger.warning(f"[ WARNING ]  GPU{i} 메모리 부족! (2GB 미만 남음)")
    
    # RAM 압박 확인
    if cpu_monitor.check_memory_pressure(threshold=85.0):
        logger.warning(f"[ WARNING ]  RAM 메모리 압박! (85% 이상 사용 중)")
    
    logger.info("="*80 + "\n")


class EmbeddingMonitorCallback(TrainerCallback):
    """임베딩 학습 모니터링 콜백"""
    
    def __init__(self, tokenizer, test_tokens=None):
        self.tokenizer = tokenizer
        self.test_tokens = test_tokens or ['데이터', '분석', '안녕', '감사']
        self.initial_embeddings = {}
        
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """학습 시작 시 초기 임베딩 저장"""
        if model is not None:
            embed_layer = model.get_input_embeddings()
            
            logger.info("="*80)
            logger.info("[ MONITOR ] 초기 임베딩 통계")
            logger.info("="*80)
            
            for token in self.test_tokens:
                try:
                    token_id = self.tokenizer.encode(token, add_special_tokens=False)[0]
                    embedding = embed_layer.weight[token_id].detach().cpu().float().numpy()
                    self.initial_embeddings[token] = embedding
                    
                    logger.info(f"  {token}: 평균={embedding.mean():.6f}, 표준편차={embedding.std():.6f}")
                except Exception as e:
                    logger.warning(f"  {token}: 토큰화 실패 - {e}")
            
            logger.info("="*80)
    
    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        """평가 시 임베딩 변화 확인"""
        if model is not None and state.global_step > 0 and state.global_step % 500 == 0:
            embed_layer = model.get_input_embeddings()
            
            logger.info("="*80)
            logger.info(f"[ MONITOR ] Step {state.global_step} - 임베딩 변화")
            logger.info("="*80)
            
            for token in self.test_tokens:
                if token not in self.initial_embeddings:
                    continue
                    
                try:
                    token_id = self.tokenizer.encode(token, add_special_tokens=False)[0]
                    current_emb = embed_layer.weight[token_id].detach().cpu().float().numpy()
                    initial_emb = self.initial_embeddings[token]
                    
                    # 변화량 계산
                    change = np.linalg.norm(current_emb - initial_emb)
                    similarity = np.dot(current_emb, initial_emb) / (
                        np.linalg.norm(current_emb) * np.linalg.norm(initial_emb) + 1e-8
                    )
                    
                    logger.info(f"  {token}:")
                    logger.info(f"    변화량: {change:.6f}")
                    logger.info(f"    초기 유사도: {similarity:.6f}")
                    logger.info(f"    현재 평균: {current_emb.mean():.6f}")
                except Exception as e:
                    logger.warning(f"  {token}: 계산 실패 - {e}")
            
            # '데이터' vs '분석' 유사도
            try:
                token1_id = self.tokenizer.encode('데이터', add_special_tokens=False)[0]
                token2_id = self.tokenizer.encode('분석', add_special_tokens=False)[0]
                
                emb1 = embed_layer.weight[token1_id].detach().cpu().float().numpy()
                emb2 = embed_layer.weight[token2_id].detach().cpu().float().numpy()
                
                similarity = np.dot(emb1, emb2) / (
                    np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
                )
                
                logger.info(f"\n  '데이터' ↔ '분석' 유사도: {similarity:.6f}")
                
                if metrics is not None:
                    metrics['embedding/data_analysis_similarity'] = similarity
            except:
                pass
                
            logger.info("="*80)

@dataclass
class QwenFineTuningConfig:
    """Qwen 2.5-14B 파인튜닝 설정 (Multi-GPU 최적화)"""
    
    # Checkpoint 재시작
    resume_from_checkpoint: str = "/home/work/tesseract/qwen/qwen-KR-14B-output-unsloth/checkpoint-1250"
    
    # 모델
    base_model: str = "Qwen/Qwen2.5-14B-Instruct"
    max_seq_length: int = 4096
    
    # 토크나이저
    tokenizer_name: str = os.getenv("TOKENIZER")
    hf_token: Optional[str] = None
    
    # 데이터
    dataset_name: str = "MyeongHo0621/korean-quality-cleaned"
    
    # 출력
    output_dir: str = "/home/work/tesseract/qwen/qwen-KR-14B-output-unsloth"
    run_name: str = f"qwen25-14b-KR-resumed-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # LoRA
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.0
    
    # ⚡ 훈련 최적화 (Multi-GPU)
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8      # 4 → 8 (2배)
    gradient_accumulation_steps: int = 2      # 4 → 2 (step 수 절반)
    learning_rate: float = 1.4e-4             # √2배 증가
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # 최적화
    use_gradient_checkpointing: str = "unsloth"
    
    # ⚡ 저장 최적화
    save_steps: int = 500                     # 250 → 500 (I/O 절감)
    save_total_limit: int = 5
    logging_steps: int = 20                   # 10 → 20


class QwenFineTuner:
    """Qwen 2.5 Checkpoint Resume 파인튜너"""
    
    def __init__(self, config: QwenFineTuningConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    def load_model_from_checkpoint(self):
        """Checkpoint에서 모델 로드"""
        logger.info("="*80)
        logger.info("Checkpoint에서 모델 재시작")
        logger.info("="*80)
        logger.info(f"Checkpoint: {self.config.resume_from_checkpoint}")
        logger.info(f"토크나이저: {self.config.tokenizer_name}")
        logger.info("="*80)
        
        # HF 토큰 설정
        hf_token = self.config.hf_token or os.getenv("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
            logger.info("[ COMPLETE ] HuggingFace 로그인 완료")
        
        # Checkpoint에서 로드 (Unsloth 방식)
        import warnings
        warnings.filterwarnings("ignore", message="Some weights.*were not used")
        
        logger.info("[ INFO ] Checkpoint에서 LoRA 어댑터 로딩...")
        
        # 베이스 모델 먼저 로드
        self.model, _ = FastLanguageModel.from_pretrained(
            model_name=self.config.base_model,
            max_seq_length=self.config.max_seq_length,
            dtype=None,
            load_in_4bit=False,
            load_in_8bit=True,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
        )
        
        logger.info("[ COMPLETE ] 베이스 모델 로드 완료")
        log_system_resources("베이스 모델 로드 후")
        
        # 토크나이저 로드 (Checkpoint에서)
        from transformers import AutoTokenizer
        
        # Checkpoint 디렉토리에 토크나이저가 있으면 사용, 없으면 환경변수에서
        checkpoint_tokenizer_path = os.path.join(self.config.resume_from_checkpoint, "tokenizer_config.json")
        if os.path.exists(checkpoint_tokenizer_path):
            logger.info("[ INFO ] Checkpoint에서 토크나이저 로드")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.resume_from_checkpoint,
                trust_remote_code=True
            )
        else:
            logger.info("[ INFO ] 환경변수에서 토크나이저 로드")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer_name,
                token=hf_token,
                trust_remote_code=True
            )
        
        logger.info(f"[ COMPLETE ] 토크나이저 로드 완료")
        logger.info(f"  Vocab size: {len(self.tokenizer):,}")
        
        # PAD 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 임베딩 크기 조정 (필요 시)
        original_vocab_size = self.model.get_input_embeddings().weight.shape[0]
        new_vocab_size = len(self.tokenizer)
        
        if original_vocab_size != new_vocab_size:
            logger.info(f"[ ! ] Vocab 크기 조정: {original_vocab_size:,} → {new_vocab_size:,}")
            self.model.resize_token_embeddings(new_vocab_size)
            logger.info("[ COMPLETE ] 임베딩 크기 조정 완료")
        
        # LoRA 적용
        logger.info("[ INFO ] LoRA 설정 적용 중...")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            modules_to_save=["embed_tokens", "lm_head"],
            bias="none",
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
            random_state=42,
            use_rslora=False,
            loftq_config=None
        )
        
        logger.info("[ COMPLETE ] LoRA 적용 완료")
        log_system_resources("LoRA 적용 후")
        
        # Checkpoint 가중치 로드
        logger.info("[ INFO ] Checkpoint 가중치 로딩...")
        try:
            from peft import PeftModel
            
            # LoRA 어댑터만 로드 (Checkpoint에서)
            adapter_path = self.config.resume_from_checkpoint
            if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
                logger.info(f"  LoRA 어댑터 로드: {adapter_path}")
                self.model = PeftModel.from_pretrained(
                    self.model.base_model,
                    adapter_path,
                    is_trainable=True
                )
                logger.info("[ COMPLETE ] LoRA 가중치 로드 완료")
            else:
                logger.warning("[ WARNING ] Checkpoint에 LoRA 어댑터 없음 - 새로 시작")
        except Exception as e:
            logger.warning(f"[ WARNING ] Checkpoint 로드 실패: {e}")
            logger.info("  Trainer.train()의 resume_from_checkpoint로 재시도")
        
        logger.info("="*80)
    
    def load_data(self, streaming: bool = False):
        """데이터셋 로드"""
        logger.info("="*80)
        logger.info(f"데이터셋 로딩: {self.config.dataset_name}")
        logger.info(f"  모드: {'스트리밍' if streaming else '일반'}")
        logger.info("="*80)
        
        if streaming:
            dataset = load_dataset(
                self.config.dataset_name, 
                split="train",
                streaming=True
            )
            logger.info("[ COMPLETE ] 스트리밍 데이터셋 로드 완료")
        else:
            dataset = load_dataset(self.config.dataset_name, split="train")
            logger.info(f"[ COMPLETE ] 전체 데이터: {len(dataset):,}개")
        
        return dataset
    
    def train(self, dataset):
        """훈련 재시작"""
        logger.info("="*80)
        logger.info(" Qwen 2.5-14B 훈련 재시작 (Checkpoint-1250)")
        logger.info(f" Run: {self.config.run_name}")
        logger.info(" Multi-GPU 최적화 적용")
        logger.info("="*80)
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # GPU 메모리 정리
        log_system_resources("학습 준비")
        if gpu_monitor.available:
            gpu_monitor.clear_cache()
            logger.info("[ COMPLETE ] GPU 캐시 정리 완료")
        
        # 스트리밍 데이터셋 변환 (필요 시)
        from datasets import IterableDataset, Dataset
        is_streaming = isinstance(dataset, IterableDataset)
        
        if is_streaming:
            logger.info("[ INFO ] 스트리밍 데이터셋 변환 중...")
            import itertools
            
            all_examples = []
            for i, example in enumerate(itertools.islice(dataset, 54190)):
                all_examples.append(example)
                if (i + 1) % 10000 == 0:
                    logger.info(f"    진행: {i+1:,}/54,190")
            
            dataset = Dataset.from_list(all_examples)
            del all_examples
            logger.info(f"[ COMPLETE ] {len(dataset):,}개 변환 완료")
        
        # 데이터 분할
        train_size = int(0.95 * len(dataset))
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, len(dataset)))
        
        eval_size = len(dataset) - train_size
        
        logger.info(f"훈련: {train_size:,}개")
        logger.info(f"검증: {eval_size:,}개")
        logger.info(f"Epoch: {self.config.num_train_epochs}")
        logger.info(f"배치(단일 GPU): {self.config.per_device_train_batch_size}")
        logger.info(f"배치(전체): {self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps * torch.cuda.device_count()}")
        logger.info("="*80)
        
        # 데이터 포맷팅
        logger.info("[ INFO ] 데이터 포맷팅 시작...")
        log_system_resources("데이터 포맷팅 시작")
        
        try:
            logger.info(f"  훈련 데이터 포맷팅 중... ({train_size:,}개)")
            train_formatted = []
            for i, example in enumerate(train_dataset):
                try:
                    messages = example.get('messages', [])
                    if messages:
                        text = self.tokenizer.apply_chat_template(
                            messages, 
                            tokenize=False, 
                            add_generation_prompt=False
                        )
                        train_formatted.append({"text": text})
                except Exception as e:
                    logger.debug(f"샘플 {i} 실패: {e}")
                
                if (i + 1) % 10000 == 0:
                    logger.info(f"    진행: {i+1:,}/{train_size:,}")
            
            train_dataset = Dataset.from_list(train_formatted)
            del train_formatted
            logger.info(f"[ COMPLETE ] 훈련 데이터: {len(train_dataset):,}개")
            
            logger.info(f"  검증 데이터 포맷팅 중... ({eval_size:,}개)")
            eval_formatted = []
            for i, example in enumerate(eval_dataset):
                try:
                    messages = example.get('messages', [])
                    if messages:
                        text = self.tokenizer.apply_chat_template(
                            messages, 
                            tokenize=False, 
                            add_generation_prompt=False
                        )
                        eval_formatted.append({"text": text})
                except Exception as e:
                    logger.debug(f"샘플 {i} 실패: {e}")
            
            eval_dataset = Dataset.from_list(eval_formatted)
            del eval_formatted
            logger.info(f"[ COMPLETE ] 검증 데이터: {len(eval_dataset):,}개")
            
            log_system_resources("데이터 포맷팅 완료")
            
        except Exception as e:
            logger.error(f"[ FAIL ] 데이터 포맷팅 실패: {e}")
            import traceback
            logger.error(f"  스택 트레이스:\n{traceback.format_exc()}")
            raise
        
        # 훈련 인자
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            run_name=self.config.run_name,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            
            # 최적화
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            optim="adamw_8bit",
            
            # 로깅/저장
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            eval_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # 기타
            remove_unused_columns=False,
            report_to=[],
            seed=42,
            save_safetensors=True,
            
            # ⚡ Multi-GPU (자동 감지)
            # CUDA_VISIBLE_DEVICES로 제어됨
        )
        
        # Trainer 생성
        logger.info("[ INFO ] SFTTrainer 초기화 중...")
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=training_args,
            callbacks=[EmbeddingMonitorCallback(self.tokenizer)]
        )
        logger.info("[ COMPLETE ] Trainer 초기화 완료")
        
        # 훈련 시작
        logger.info("="*80)
        logger.info(" 훈련 재시작 ")
        logger.info(f" Checkpoint: {self.config.resume_from_checkpoint}")
        logger.info(f" GPU 수: {torch.cuda.device_count()}")
        logger.info(f" 배치 크기: {self.config.per_device_train_batch_size} × {self.config.gradient_accumulation_steps} × {torch.cuda.device_count()} = {self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps * torch.cuda.device_count()}")
        logger.info(" 예상 속도: 1.5~2배 향상")
        logger.info("="*80)
        
        log_system_resources("훈련 시작")
        
        # ⚡ Checkpoint에서 재시작
        trainer.train(resume_from_checkpoint=self.config.resume_from_checkpoint)
        
        # 최종 저장
        final_path = os.path.join(self.config.output_dir, "final")
        trainer.save_model(final_path)
        self.tokenizer.save_pretrained(final_path)
        
        logger.info("="*80)
        logger.info(" [ OK ] 훈련 완료!")
        logger.info("="*80)
        logger.info(f"모델 저장: {final_path}")
        logger.info("="*80)
        
        return final_path


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print(" Qwen 2.5-14B Checkpoint Resume + Multi-GPU 최적화")
    print(" Checkpoint-1250 → Multi-GPU 훈련")
    print("="*80)
    
    # GPU 확인
    gpu_count = torch.cuda.device_count()
    print(f"\n사용 가능한 GPU: {gpu_count}개")
    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    if gpu_count < 2:
        print("\n[ WARNING ] GPU가 1개뿐입니다!")
        print("  Multi-GPU 모드는 2개 이상 필요합니다.")
        print("  Single-GPU 최적화 모드로 전환합니다...")
        print("  (배치 크기는 8로 유지하되, GPU 1개만 사용)")
        input("\nEnter를 눌러 계속하거나 Ctrl+C로 중단하세요...")
    
    # 설정
    config = QwenFineTuningConfig()
    
    print(f"\n{'='*80}")
    print(" 설정 요약")
    print(f"{'='*80}")
    print(f"Checkpoint: {config.resume_from_checkpoint}")
    print(f"토크나이저: {config.tokenizer_name}")
    print(f"데이터셋: {config.dataset_name}")
    print(f"출력: {config.output_dir}")
    print(f"LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"Epoch: {config.num_train_epochs}")
    print(f"배치(단일 GPU): {config.per_device_train_batch_size}")
    print(f"배치(전체): {config.per_device_train_batch_size * config.gradient_accumulation_steps * gpu_count}")
    print(f"학습률: {config.learning_rate}")
    print(f"저장 간격: {config.save_steps} steps")
    print(f"{'='*80}\n")
    
    # 파인튜너
    finetuner = QwenFineTuner(config)
    
    # 실행
    try:
        # 1. Checkpoint에서 모델 로드
        finetuner.load_model_from_checkpoint()
        
        # 2. 데이터 로드
        dataset = finetuner.load_data(streaming=True)
        
        # 3. 훈련 재시작
        model_path = finetuner.train(dataset)
        
        print(f"\n{'='*80}")
        print(" 완료!")
        print(f"{'='*80}")
        print(f"모델: {model_path}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

