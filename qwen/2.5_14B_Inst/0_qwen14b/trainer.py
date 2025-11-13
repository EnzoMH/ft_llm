#!/usr/bin/env python3
"""
Qwen2.5-14B 멀티턴 대화 파인튜너
"""

import os
import gc
import json
import logging
import torch
from typing import Optional

# Unsloth를 먼저 import (최적화 활성화)
from unsloth import FastLanguageModel, is_bfloat16_supported

# 나머지 import
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from huggingface_hub import login, HfApi, snapshot_download

from .config import Qwen14BFineTuningConfig
from .dataset_loader import MultiTurnDatasetLoader
from .callbacks import TrainingMonitorCallback, HubUploadCallback
from .utils import log_system_resources, gpu_monitor


logger = logging.getLogger(__name__)


class Qwen14BFineTuner:
    """Qwen2.5-14B 멀티턴 대화 파인튜너"""
    
    def __init__(self, config: Qwen14BFineTuningConfig):
        """
        Args:
            config: 파인튜닝 설정
        """
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
        
        log_system_resources(logger, "모델 로드 전")
        
        # HuggingFace 로그인
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
            logger.info("[ COMPLETE ] HuggingFace 로그인 완료")
        
        # 모델 로드 (Unsloth)
        logger.info("[ INFO ] 모델 다운로드 시작...")
        
        # Flash Attention 2 시도, 실패 시 Unsloth가 자동으로 fallback
        # 현재 환경에서 Flash Attention 2가 깨져있어서 Unsloth가 자동으로 처리하도록 함
        logger.info("[ INFO ] Attention 구현: Unsloth 자동 선택 (Flash Attention 2 실패 시 자동 fallback)")
        
        try:
            # Flash Attention 2 시도
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.base_model,
                max_seq_length=self.config.max_seq_length,
                dtype=None,  # Auto (BF16 on H100)
                load_in_4bit=self.config.load_in_4bit,
                load_in_8bit=self.config.load_in_8bit,
                trust_remote_code=True,
                attn_implementation="flash_attention_2"  # 시도는 하지만 실패 시 Unsloth가 자동 처리
            )
        except Exception as e:
            logger.warning(f"[ WARNING ] Flash Attention 2 로드 실패: {e}")
            logger.info("[ INFO ] Unsloth가 자동으로 최적의 attention을 선택합니다...")
            # attn_implementation 없이 재시도 (Unsloth가 자동 선택)
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.base_model,
                max_seq_length=self.config.max_seq_length,
                dtype=None,
                load_in_4bit=self.config.load_in_4bit,
                load_in_8bit=self.config.load_in_8bit,
                trust_remote_code=True,
                # attn_implementation 제거 - Unsloth가 자동으로 최적화
            )
        
        logger.info("[ COMPLETE ] 베이스 모델 로드 완료")
        log_system_resources(logger, "베이스 모델 로드 후")
        
        # 토크나이저 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"[ INFO ] Vocab size: {len(self.tokenizer):,}")
        
        # 기존 adapter 확인 (로컬 또는 Hub에서 다운로드한 최종 모델)
        # 주의: checkpoint가 있으면 checkpoint에서 adapter를 로드하므로 여기서는 최종 모델만 확인
        adapter_path = os.path.join(self.config.output_dir, "adapter_model.safetensors")
        adapter_config_path = os.path.join(self.config.output_dir, "adapter_config.json")
        
        # checkpoint가 없는 경우에만 최종 adapter 로드 시도
        checkpoints = []
        if os.path.exists(self.config.output_dir):
            checkpoints = [d for d in os.listdir(self.config.output_dir) 
                          if d.startswith("checkpoint-") and os.path.isdir(os.path.join(self.config.output_dir, d))]
        
        if not checkpoints and os.path.exists(adapter_path) and os.path.exists(adapter_config_path):
            # checkpoint가 없고 최종 adapter만 있는 경우 (Hub에서 다운로드한 경우)
            logger.info("[ INFO ] 기존 adapter 발견! 최종 모델을 로드합니다...")
            logger.info(f"  Adapter 경로: {adapter_path}")
            
            # trainer_state.json 확인
            trainer_state_path = os.path.join(self.config.output_dir, "trainer_state.json")
            if os.path.exists(trainer_state_path):
                with open(trainer_state_path, 'r') as f:
                    state = json.load(f)
                global_step = state.get('global_step', 0)
                logger.info(f"  이전 학습: Step {global_step}")
            
            # 기존 adapter 로드
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model,
                self.config.output_dir,
                is_trainable=True
            )
            logger.info("[ COMPLETE ] 기존 adapter 로드 완료 (학습 재개 가능)")
        else:
            # LoRA 적용 (새로 시작 또는 checkpoint에서 재개)
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
        
        log_system_resources(logger, "LoRA 적용 후")
        
        # Trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"[ INFO ] 학습 가능 파라미터: {trainable_params:,} / {all_params:,} "
                   f"({100 * trainable_params / all_params:.2f}%)")
        
        logger.info("="*80)
    
    def load_data(self) -> Dataset:
        """한국어 멀티턴 데이터셋 로드
        
        Returns:
            Dataset: 로드된 데이터셋
        """
        logger.info("="*80)
        logger.info("한국어 멀티턴 대화 데이터셋 로딩")
        logger.info("="*80)
        logger.info(f"데이터 디렉토리: {self.config.korean_data_dir}")
        
        log_system_resources(logger, "데이터 로드 전")
        
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
        
        # max_samples 제한 적용 (RAM 압박 방지)
        if self.config.max_samples and len(dataset) > self.config.max_samples:
            logger.info(f"[ INFO ] 데이터 샘플링: {len(dataset):,} → {self.config.max_samples:,}")
            # 랜덤 샘플링 대신 앞에서부터 선택 (재현성)
            dataset = dataset.select(range(self.config.max_samples))
            logger.info(f"[ COMPLETE ] 샘플링 완료: {len(dataset):,}개")
        
        log_system_resources(logger, "데이터 로드 후")
        logger.info("="*80)
        
        return dataset
    
    def format_dataset(self, dataset: Dataset) -> Dataset:
        """데이터셋 포맷팅 (ChatML) - 메모리 효율적
        
        Args:
            dataset: 원본 데이터셋
            
        Returns:
            Dataset: 포맷팅된 데이터셋
        """
        logger.info(f"[ INFO ] 데이터셋 포맷팅 시작... (입력: {len(dataset):,}개)")
        
        if len(dataset) == 0:
            logger.error("[ ERROR ] 입력 데이터셋이 비어있습니다!")
            raise ValueError("입력 데이터셋이 0개입니다")
        
        # 첫 3개 샘플 키 확인
        logger.info("[ DEBUG ] 첫 3개 샘플 키 확인:")
        for i in range(min(3, len(dataset))):
            logger.info(f"  샘플 {i}: {list(dataset[i].keys())}")
        
        formatted_data = []
        failed_count = 0
        text_field_count = 0
        messages_field_count = 0
        batch = []
        batch_size = 5000  # 5천개씩 배치 처리
        
        for i, example in enumerate(dataset):
            try:
                # text 필드가 이미 있는 경우 (이미 포맷팅됨)
                if 'text' in example and example['text'] is not None and '<|im_start|>' in example['text']:
                    batch.append({"text": example['text']})
                    text_field_count += 1
                
                # messages 필드가 있는 경우 (포맷팅 필요)
                elif 'messages' in example and example.get('messages'):
                    messages = example.get('messages', [])
                    
                    if not messages or len(messages) < 1:
                        failed_count += 1
                        if i < 5:  # 처음 5개만 상세 로그
                            logger.error(f"샘플 {i}: messages 필드 비어있음")
                        continue
                    
                    # ChatML 포맷 적용
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    
                    batch.append({"text": text})
                    messages_field_count += 1
                else:
                    failed_count += 1
                    if i < 5:  # 처음 5개만 상세 로그
                        logger.error(f"샘플 {i}: text/messages 필드 없음. 키={list(example.keys())}")
                
                # 배치가 차면 추가하고 초기화 (메모리 관리)
                if len(batch) >= batch_size:
                    formatted_data.extend(batch)
                    batch = []
                
            except Exception as e:
                if i < 5:  # 처음 5개만 상세 로그
                    logger.error(f"샘플 {i} 포맷팅 실패: {e}, 키={list(example.keys())}")
                failed_count += 1
                continue
            
            if (i + 1) % 50000 == 0:
                logger.info(f"  진행: {i+1:,}/{len(dataset):,}")
        
        # 남은 배치 처리
        if batch:
            formatted_data.extend(batch)
        
        logger.info(f"[ COMPLETE ] 포맷팅 완료:")
        logger.info(f"  text 필드 사용: {text_field_count:,}개")
        logger.info(f"  messages 변환: {messages_field_count:,}개")
        logger.info(f"  실패: {failed_count:,}개")
        logger.info(f"  총: {len(formatted_data):,}개")
        
        if len(formatted_data) == 0:
            raise ValueError(f"포맷팅된 데이터가 0개입니다! 원본 데이터: {len(dataset):,}개")
        
        return Dataset.from_list(formatted_data)
    
    def train(self, dataset: Dataset) -> str:
        """학습 실행
        
        Args:
            dataset: 학습할 데이터셋
            
        Returns:
            str: 최종 모델 저장 경로
        """
        logger.info("="*80)
        logger.info(" Qwen2.5-14B 멀티턴 대화 파인튜닝")
        logger.info(f" Run: {self.config.run_name}")
        logger.info("="*80)
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # GPU 캐시 정리 및 메모리 최적화 (40GB VRAM 최적화)
        log_system_resources(logger, "학습 준비")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        if gpu_monitor.available:
            gpu_monitor.clear_cache()
            logger.info("[ COMPLETE ] GPU 캐시 정리 완료")
        
        # Python 가비지 컬렉션
        gc.collect()
        logger.info("[ COMPLETE ] Python GC 실행 완료")
        
        # 데이터 분할 (최소 eval 크기 보장)
        total_size = len(dataset)
        eval_size = max(int(0.05 * total_size), 100)  # 최소 100개 보장
        train_size = total_size - eval_size
        
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, total_size))
        
        logger.info(f"훈련: {train_size:,}개")
        logger.info(f"검증: {eval_size:,}개 (최소 100개 보장)")
        logger.info(f"Epoch: {self.config.num_train_epochs}")
        logger.info(f"배치 크기: {self.config.per_device_train_batch_size}")
        logger.info(f"효과적 배치: {self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps}")
        logger.info("="*80)
        
        # 데이터 포맷팅
        logger.info("[ INFO ] 데이터 포맷팅 중...")
        train_formatted = self.format_dataset(train_dataset)
        
        # 중간 메모리 정리
        del train_dataset
        gc.collect()
        
        eval_formatted = self.format_dataset(eval_dataset)
        
        # 메모리 정리
        del eval_dataset, dataset
        gc.collect()
        
        logger.info(f"[ COMPLETE ] 훈련: {len(train_formatted):,}개, 검증: {len(eval_formatted):,}개")
        log_system_resources(logger, "데이터 포맷팅 완료")
        
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
            per_device_eval_batch_size=8,  # 평가는 더 작은 배치 사용 (메모리 절약)
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            max_grad_norm=self.config.max_grad_norm,
            
            # 최적화 (H100)
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            optim="adamw_8bit",
            
            # 로깅/저장 (최적화됨)
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,
            eval_strategy="steps",
            load_best_model_at_end=False,  # True→False (메모리 절약, 필요 시 수동 로드)
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_only_model=False,  # optimizer state도 저장 (resume 위해)
            
            # 기타
            remove_unused_columns=False,
            report_to=[],
            seed=42,
            save_safetensors=True,
            
            # HuggingFace Hub 업로드
            push_to_hub=self.config.push_to_hub and self.config.hub_strategy == "end",  # end 전략일 때만 자동
            hub_model_id=self.config.hub_model_id if self.config.push_to_hub else None,
            hub_strategy="every_save" if (self.config.push_to_hub and self.config.hub_strategy == "checkpoint") else None,
        )
        
        # Callback 리스트 생성
        callbacks = [TrainingMonitorCallback()]
        
        # HuggingFace Hub 업로드 콜백 추가 (checkpoint 전략일 때)
        if self.config.push_to_hub and self.config.hub_model_id and self.config.hub_strategy == "checkpoint":
            try:
                hub_callback = HubUploadCallback(
                    hub_model_id=self.config.hub_model_id,
                    hub_token=self.config.hub_token,
                    upload_every_n_steps=self.config.save_steps
                )
                callbacks.append(hub_callback)
                logger.info(f"[ INFO ] Hub 자동 업로드 활성화: {self.config.hub_model_id}")
            except Exception as e:
                logger.warning(f"[ WARNING ] Hub 콜백 생성 실패: {e}")
        
        # Trainer 생성
        logger.info("[ INFO ] SFTTrainer 초기화 중...")
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_formatted,
            eval_dataset=eval_formatted,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            dataset_num_proc=4,  # RAM 압박 방지 (8→4, 3.8GB 데이터 안전하게 처리)
            packing=False,
            args=training_args,
            callbacks=callbacks
        )
        logger.info("[ COMPLETE ] Trainer 초기화 완료")
        
        # 훈련 시작
        logger.info("="*80)
        logger.info(" 학습 시작 ")
        logger.info(f" GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f" Flash Attention: {training_args.bf16}")
        logger.info("="*80)
        
        log_system_resources(logger, "훈련 시작")
        
        # Resume 체크포인트 자동 감지 (Fallback Logic)
        resume_checkpoint = self._find_resume_checkpoint(logger)
        
        trainer.train(resume_from_checkpoint=resume_checkpoint)
        
        # 최종 저장
        final_path = os.path.join(self.config.output_dir, "final")
        trainer.save_model(final_path)
        self.tokenizer.save_pretrained(final_path)
        
        logger.info("="*80)
        logger.info(" [ OK ] 훈련 완료!")
        logger.info("="*80)
        logger.info(f"모델 저장: {final_path}")
        
        # 최종 모델 Hub 업로드 (end 전략일 때)
        if self.config.push_to_hub and self.config.hub_model_id and self.config.hub_strategy == "end":
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=self.config.hub_token)
                logger.info(f"[ HUB ] 최종 모델 업로드 시작...")
                api.upload_folder(
                    folder_path=final_path,
                    repo_id=self.config.hub_model_id,
                    repo_type="model",
                    commit_message="Final trained model",
                )
                logger.info(f"[ HUB ] ✅ 최종 모델 업로드 완료: https://huggingface.co/{self.config.hub_model_id}")
            except Exception as e:
                logger.error(f"[ HUB ] ❌ 최종 모델 업로드 실패: {e}")
        
        logger.info("="*80)
        
        return final_path
    
    def _find_resume_checkpoint(self, logger) -> Optional[str]:
        """
        Checkpoint 찾기 (Fallback Logic)
        1. 로컬 checkpoint 확인
        2. 없으면 Hub에서 다운로드 시도
        3. 없으면 None 반환 (처음부터 시작)
        
        Returns:
            Optional[str]: checkpoint 경로 또는 None
        """
        logger.info("="*80)
        logger.info("[ CHECKPOINT ] 체크포인트 검색 중...")
        logger.info("="*80)
        
        # 1. 로컬 checkpoint 확인
        if os.path.exists(self.config.output_dir):
            checkpoints = [d for d in os.listdir(self.config.output_dir) 
                          if d.startswith("checkpoint-") and os.path.isdir(os.path.join(self.config.output_dir, d))]
            if checkpoints:
                # 가장 최근 체크포인트 찾기
                checkpoints.sort(key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else 0)
                latest_checkpoint = os.path.join(self.config.output_dir, checkpoints[-1])
                logger.info(f"[ CHECKPOINT ] ✅ 로컬 체크포인트 발견: {latest_checkpoint}")
                logger.info(f"[ CHECKPOINT ] 학습 재개: Step {checkpoints[-1].split('-')[1]}")
                return latest_checkpoint
        
        logger.info("[ CHECKPOINT ] 로컬에 체크포인트가 없습니다.")
        
        # 2. Hub에서 checkpoint 다운로드 시도
        if self.config.push_to_hub and self.config.hub_model_id and self.config.hub_token:
            try:
                logger.info(f"[ CHECKPOINT ] Hub에서 체크포인트 검색 중: {self.config.hub_model_id}")
                api = HfApi(token=self.config.hub_token)
                
                # Hub의 모든 파일 확인
                files = api.list_repo_files(self.config.hub_model_id, repo_type="model")
                
                # checkpoint 디렉토리 찾기
                checkpoint_dirs = set()
                for f in files:
                    if "checkpoint-" in f and "/" in f:
                        parts = f.split("/")
                        if len(parts) > 1 and parts[0].startswith("checkpoint-"):
                            checkpoint_dirs.add(parts[0])
                
                if checkpoint_dirs:
                    checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else 0)
                    latest_checkpoint_name = checkpoint_dirs[-1]
                    
                    logger.info(f"[ CHECKPOINT ] Hub에서 체크포인트 발견: {latest_checkpoint_name}")
                    logger.info(f"[ CHECKPOINT ] 다운로드 시작...")
                    
                    # checkpoint 다운로드
                    os.makedirs(self.config.output_dir, exist_ok=True)
                    snapshot_download(
                        repo_id=self.config.hub_model_id,
                        repo_type="model",
                        local_dir=self.config.output_dir,
                        token=self.config.hub_token,
                        allow_patterns=f"{latest_checkpoint_name}/*"
                    )
                    
                    latest_checkpoint = os.path.join(self.config.output_dir, latest_checkpoint_name)
                    logger.info(f"[ CHECKPOINT ] ✅ Hub에서 다운로드 완료: {latest_checkpoint}")
                    logger.info(f"[ CHECKPOINT ] 학습 재개: Step {latest_checkpoint_name.split('-')[1]}")
                    return latest_checkpoint
                else:
                    # checkpoint 디렉토리가 없지만 최종 모델이 있을 수 있음
                    logger.info("[ CHECKPOINT ] Hub에 checkpoint 디렉토리가 없습니다.")
                    
                    # trainer_state.json 확인
                    try:
                        from huggingface_hub import hf_hub_download
                        trainer_state_path = hf_hub_download(
                            repo_id=self.config.hub_model_id,
                            filename="trainer_state.json",
                            repo_type="model",
                            token=self.config.hub_token,
                            cache_dir="/tmp"
                        )
                        with open(trainer_state_path, 'r') as f:
                            state = json.load(f)
                        global_step = state.get('global_step', 0)
                        
                        if global_step > 0:
                            logger.info(f"[ CHECKPOINT ] Hub에 최종 모델 발견 (Step {global_step})")
                            logger.info("[ CHECKPOINT ] 최종 모델 다운로드 중...")
                            
                            # 모든 파일 다운로드 (adapter 포함)
                            snapshot_download(
                                repo_id=self.config.hub_model_id,
                                repo_type="model",
                                local_dir=self.config.output_dir,
                                token=self.config.hub_token,
                                ignore_patterns=["*.md", ".git*"]
                            )
                            
                            logger.info("[ CHECKPOINT ] ✅ 최종 모델 다운로드 완료")
                            logger.info("[ CHECKPOINT ] ⚠️  checkpoint 디렉토리가 없어 optimizer state는 없습니다.")
                            logger.info("[ CHECKPOINT ] adapter 가중치는 로드되어 학습을 계속할 수 있습니다.")
                            # adapter는 이미 load_model에서 로드됨
                            return None  # checkpoint가 없으므로 None 반환 (adapter는 이미 로드됨)
                    except Exception as e:
                        logger.warning(f"[ CHECKPOINT ] Hub에서 trainer_state.json 확인 실패: {e}")
                        
            except Exception as e:
                logger.warning(f"[ CHECKPOINT ] Hub에서 checkpoint 다운로드 실패: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        
        # 3. checkpoint 없음 - 처음부터 시작
        logger.info("[ CHECKPOINT ] 체크포인트를 찾을 수 없습니다. 처음부터 학습을 시작합니다.")
        logger.info("="*80)
        return None

