#!/usr/bin/env python3
"""
Qwen2.5-14B-Instruct 한국어 멀티턴 대화 파인튜닝
- H100 80GB 최적화
- Flash Attention 2
- 멀티턴 대화 데이터셋
- LoRA + 8bit 양자화
"""

import os
import sys
import torch
from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass
import logging
import json

# 모니터링 모듈
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'util'))
from cpu_mntrg import CPUMonitor
from gpu_mnrtg import GPUMonitor

from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from huggingface_hub import login

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 글로벌 모니터
cpu_monitor = CPUMonitor()
gpu_monitor = GPUMonitor()


def log_system_resources(stage: str):
    """시스템 리소스 상태 로깅"""
    logger.info("\n" + "="*80)
    logger.info(f"[{stage}] 시스템 리소스 모니터링")
    logger.info("="*80)
    
    # CPU 상세
    cpu_monitor.log_snapshot(logger, stage)
    
    # GPU 상세
    if gpu_monitor.available:
        gpu_monitor.log_all_gpus(logger, stage)
        
        # 메모리 압박 확인
        for i in range(gpu_monitor.device_count):
            if not gpu_monitor.check_memory_available(i, required_gb=5.0):
                logger.warning(f"[ WARNING ]  GPU{i} 메모리 부족! (5GB 미만)")
    
    # RAM 압박 확인
    if cpu_monitor.check_memory_pressure(threshold=85.0):
        logger.warning(f"[ WARNING ]  RAM 메모리 압박! (85% 이상)")
    
    logger.info("="*80 + "\n")


class MultiTurnDatasetLoader:
    """한국어 멀티턴 대화 데이터셋 로더"""
    
    def __init__(self, data_dirs: List[str]):
        """
        Args:
            data_dirs: JSONL 데이터 디렉토리 리스트
        """
        self.data_dirs = data_dirs
    
    def load_jsonl_files(self, file_paths: List[str]) -> Dataset:
        """JSONL 파일들을 로드"""
        all_data = []
        
        for file_path in file_paths:
            logger.info(f"  로딩: {os.path.basename(file_path)}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    count = 0
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                
                                # messages 필드가 있는 경우
                                if 'messages' in data and isinstance(data['messages'], list):
                                    # 멀티턴 대화인지 확인 (최소 2개 이상)
                                    if len(data['messages']) >= 2:
                                        all_data.append(data)
                                        count += 1
                                
                                # text 필드만 있는 경우 (이미 ChatML 포맷팅됨)
                                elif 'text' in data and isinstance(data['text'], str):
                                    # assistant가 최소 1번 이상 있으면 유효한 대화
                                    if '<|im_start|>assistant' in data['text']:
                                        all_data.append(data)
                                        count += 1
                            except json.JSONDecodeError as e:
                                logger.debug(f"JSON 파싱 실패: {e}")
                                continue
                    
                    logger.info(f"    추가됨: {count:,}개")
            except Exception as e:
                logger.error(f"  파일 로드 실패 ({file_path}): {e}")
                continue
        
        logger.info(f"[ COMPLETE ] 총 {len(all_data):,}개 데이터 로드")
        return Dataset.from_list(all_data)
    
    def load_from_directory(self, directory: str, patterns: Optional[List[str]] = None) -> Dataset:
        """디렉토리에서 JSONL 파일들을 로드"""
        import glob
        
        if patterns is None:
            patterns = ['*.jsonl']
        
        file_paths = []
        for pattern in patterns:
            file_paths.extend(glob.glob(os.path.join(directory, pattern)))
        
        logger.info(f"발견된 파일: {len(file_paths)}개")
        return self.load_jsonl_files(file_paths)


class TrainingMonitorCallback(TrainerCallback):
    """학습 모니터링 콜백"""
    
    def __init__(self):
        self.start_time = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        """학습 시작"""
        self.start_time = datetime.now()
        logger.info("="*80)
        logger.info("[ START ] 학습 시작")
        logger.info(f"  시작 시간: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*80)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """로그 출력 시"""
        if logs and state.global_step % 50 == 0:
            elapsed = datetime.now() - self.start_time
            logger.info(f"Step {state.global_step}: loss={logs.get('loss', 0):.4f}, "
                       f"lr={logs.get('learning_rate', 0):.2e}, "
                       f"elapsed={str(elapsed).split('.')[0]}")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """평가 시"""
        if metrics:
            logger.info("="*80)
            logger.info(f"[ EVAL ] Step {state.global_step}")
            logger.info(f"  Loss: {metrics.get('eval_loss', 0):.4f}")
            logger.info(f"  Runtime: {metrics.get('eval_runtime', 0):.1f}s")
            logger.info("="*80)


@dataclass
class Qwen14BFineTuningConfig:
    """Qwen2.5-14B 파인튜닝 설정"""
    
    # 모델
    base_model: str = "Qwen/Qwen2.5-14B-Instruct"
    max_seq_length: int = 4096
    
    # 토크나이저 (기본 모델과 동일)
    tokenizer_name: Optional[str] = None  # None이면 base_model 사용
    
    # 데이터
    korean_data_dir: str = "/home/work/tesseract/korean_large_data/cleaned_jsonl"
    data_files: List[str] = None  # None이면 전체 JSONL 파일 사용
    
    # 출력
    output_dir: str = "/home/work/tesseract/qwen/2.5_14B_Inst/output"
    run_name: str = f"qwen25-14b-KR-multiturn-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # LoRA
    lora_r: int = 64  # 14B는 64가 적당
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    
    # 학습 설정 (H100 80GB 최적화 - 속도 우선)
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 12  # H100 80GB 최적화: 64GB 여유
    gradient_accumulation_steps: int = 4  # 효과적 배치: 48
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    
    # 최적화
    use_gradient_checkpointing: str = "unsloth"
    load_in_8bit: bool = True
    load_in_4bit: bool = False
    
    # 저장
    save_steps: int = 500
    save_total_limit: int = 3
    logging_steps: int = 10
    eval_steps: int = 500


class Qwen14BFineTuner:
    """Qwen2.5-14B 멀티턴 대화 파인튜너"""
    
    def __init__(self, config: Qwen14BFineTuningConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """모델 로드"""
        logger.info("="*80)
        logger.info("Qwen2.5-14B-Instruct 모델 로딩")
        logger.info("="*80)
        logger.info(f"모델: {self.config.base_model}")
        logger.info(f"Max Seq Length: {self.config.max_seq_length}")
        logger.info(f"8bit: {self.config.load_in_8bit}, 4bit: {self.config.load_in_4bit}")
        logger.info("="*80)
        
        log_system_resources("모델 로드 전")
        
        # HuggingFace 로그인
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
            logger.info("[ COMPLETE ] HuggingFace 로그인 완료")
        
        # 모델 로드 (Unsloth)
        logger.info("[ INFO ] 모델 다운로드 시작...")
        
        # Flash Attention 2 사용 (H100 최적화)
        attn_impl = "flash_attention_2" if torch.cuda.is_available() else "eager"
        logger.info(f"[ INFO ] Attention 구현: {attn_impl}")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.base_model,
            max_seq_length=self.config.max_seq_length,
            dtype=None,  # Auto (BF16 on H100)
            load_in_4bit=self.config.load_in_4bit,
            load_in_8bit=self.config.load_in_8bit,
            trust_remote_code=True,
            attn_implementation=attn_impl
        )
        
        logger.info("[ COMPLETE ] 베이스 모델 로드 완료")
        log_system_resources("베이스 모델 로드 후")
        
        # 토크나이저 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"[ INFO ] Vocab size: {len(self.tokenizer):,}")
        
        # LoRA 적용
        logger.info("[ INFO ] LoRA 설정 적용 중...")
        logger.info(f"  LoRA r={self.config.lora_r}, alpha={self.config.lora_alpha}")
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
            random_state=42,
            use_rslora=False,
            loftq_config=None
        )
        
        logger.info("[ COMPLETE ] LoRA 적용 완료")
        log_system_resources("LoRA 적용 후")
        
        # Trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"[ INFO ] 학습 가능 파라미터: {trainable_params:,} / {all_params:,} "
                   f"({100 * trainable_params / all_params:.2f}%)")
        
        logger.info("="*80)
    
    def load_data(self) -> Dataset:
        """한국어 멀티턴 데이터셋 로드"""
        logger.info("="*80)
        logger.info("한국어 멀티턴 대화 데이터셋 로딩")
        logger.info("="*80)
        logger.info(f"데이터 디렉토리: {self.config.korean_data_dir}")
        
        loader = MultiTurnDatasetLoader([self.config.korean_data_dir])
        
        if self.config.data_files:
            # 특정 파일들만 로드
            file_paths = [
                os.path.join(self.config.korean_data_dir, f) 
                for f in self.config.data_files
            ]
            dataset = loader.load_jsonl_files(file_paths)
        else:
            # 전체 JSONL 파일 로드
            dataset = loader.load_from_directory(self.config.korean_data_dir)
        
        logger.info(f"[ COMPLETE ] 총 {len(dataset):,}개 데이터 로드")
        logger.info("="*80)
        
        return dataset
    
    def format_dataset(self, dataset: Dataset) -> Dataset:
        """데이터셋 포맷팅 (ChatML)"""
        logger.info("[ INFO ] 데이터셋 포맷팅 시작...")
        
        formatted_data = []
        failed_count = 0
        text_field_count = 0
        messages_field_count = 0
        
        for i, example in enumerate(dataset):
            try:
                # text 필드가 이미 있는 경우 (이미 포맷팅됨)
                if 'text' in example and '<|im_start|>' in example['text']:
                    formatted_data.append({"text": example['text']})
                    text_field_count += 1
                
                # messages 필드가 있는 경우 (포맷팅 필요)
                elif 'messages' in example:
                    messages = example.get('messages', [])
                    
                    if not messages or len(messages) < 1:
                        failed_count += 1
                        continue
                    
                    # ChatML 포맷 적용
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    
                    formatted_data.append({"text": text})
                    messages_field_count += 1
                else:
                    failed_count += 1
                
            except Exception as e:
                logger.debug(f"샘플 {i} 포맷팅 실패: {e}")
                failed_count += 1
                continue
            
            if (i + 1) % 50000 == 0:
                logger.info(f"  진행: {i+1:,}/{len(dataset):,}")
        
        logger.info(f"[ COMPLETE ] 포맷팅 완료:")
        logger.info(f"  text 필드 사용: {text_field_count:,}개")
        logger.info(f"  messages 변환: {messages_field_count:,}개")
        logger.info(f"  실패: {failed_count:,}개")
        logger.info(f"  총: {len(formatted_data):,}개")
        
        return Dataset.from_list(formatted_data)
    
    def train(self, dataset: Dataset):
        """학습 실행"""
        logger.info("="*80)
        logger.info(" Qwen2.5-14B 멀티턴 대화 파인튜닝")
        logger.info(f" Run: {self.config.run_name}")
        logger.info("="*80)
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # GPU 캐시 정리
        log_system_resources("학습 준비")
        if gpu_monitor.available:
            gpu_monitor.clear_cache()
            logger.info("[ COMPLETE ] GPU 캐시 정리 완료")
        
        # 데이터 분할
        train_size = int(0.95 * len(dataset))
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, len(dataset)))
        
        eval_size = len(dataset) - train_size
        
        logger.info(f"훈련: {train_size:,}개")
        logger.info(f"검증: {eval_size:,}개")
        logger.info(f"Epoch: {self.config.num_train_epochs}")
        logger.info(f"배치 크기: {self.config.per_device_train_batch_size}")
        logger.info(f"효과적 배치: {self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps}")
        logger.info("="*80)
        
        # 데이터 포맷팅
        logger.info("[ INFO ] 데이터 포맷팅 중...")
        train_formatted = self.format_dataset(train_dataset)
        eval_formatted = self.format_dataset(eval_dataset)
        
        logger.info(f"[ COMPLETE ] 훈련: {len(train_formatted):,}개, 검증: {len(eval_formatted):,}개")
        
        # 샘플 출력
        logger.info("\n" + "="*80)
        logger.info("[ SAMPLE ] 첫 번째 훈련 데이터:")
        logger.info("="*80)
        sample_text = train_formatted[0]['text']
        logger.info(sample_text[:500] + "..." if len(sample_text) > 500 else sample_text)
        logger.info("="*80 + "\n")
        
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
            max_grad_norm=self.config.max_grad_norm,
            
            # 최적화 (H100)
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            optim="adamw_8bit",
            
            # 로깅/저장
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
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
        )
        
        # Trainer 생성
        logger.info("[ INFO ] SFTTrainer 초기화 중...")
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_formatted,
            eval_dataset=eval_formatted,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=training_args,
            callbacks=[TrainingMonitorCallback()]
        )
        logger.info("[ COMPLETE ] Trainer 초기화 완료")
        
        # 훈련 시작
        logger.info("="*80)
        logger.info(" 학습 시작 ")
        logger.info(f" GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f" Flash Attention: {training_args.bf16}")
        logger.info("="*80)
        
        log_system_resources("훈련 시작")
        
        trainer.train()
        
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
    print(" Qwen2.5-14B-Instruct 한국어 멀티턴 대화 파인튜닝")
    print(" H100 80GB 최적화 | Flash Attention 2")
    print("="*80)
    
    # GPU 확인
    if not torch.cuda.is_available():
        print("\n[ ERROR ] CUDA를 사용할 수 없습니다!")
        sys.exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"\nGPU: {gpu_name}")
    print(f"메모리: {gpu_memory_gb:.1f}GB")
    
    # 설정
    config = Qwen14BFineTuningConfig()
    
    print(f"\n{'='*80}")
    print(" 설정 요약")
    print(f"{'='*80}")
    print(f"모델: {config.base_model}")
    print(f"데이터: {config.korean_data_dir}")
    print(f"출력: {config.output_dir}")
    print(f"LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"Epoch: {config.num_train_epochs}")
    print(f"배치: {config.per_device_train_batch_size} × {config.gradient_accumulation_steps} = {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"학습률: {config.learning_rate}")
    print(f"Max Seq: {config.max_seq_length}")
    print(f"{'='*80}\n")
    
    # 파인튜너
    finetuner = Qwen14BFineTuner(config)
    
    try:
        # 1. 모델 로드
        finetuner.load_model()
        
        # 2. 데이터 로드
        dataset = finetuner.load_data()
        
        # 3. 훈련
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

