#!/usr/bin/env python3
"""
Qwen2.5-3B 멀티턴 대화 파인튜너
"""

import os
import gc
import json
import logging
import time
import torch
from datetime import datetime
from typing import Optional

# Unsloth를 먼저 import (최적화 활성화)
from unsloth import FastLanguageModel, is_bfloat16_supported

# 나머지 import
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from huggingface_hub import login, HfApi, snapshot_download

from .config import Qwen3BFineTuningConfig
from .dataset_loader import MultiTurnDatasetLoader
from .callbacks import TrainingMonitorCallback, HubUploadCallback
from .utils import log_system_resources, gpu_monitor


logger = logging.getLogger(__name__)


class SpeedLoggingSFTTrainer(SFTTrainer):
    """
    SFTTrainer + per-step 속도 로깅
    - log_interval마다 토큰 수 / 초, step 시간 찍어줌
    - PyTorch 2.6+ weights_only 문제 해결
    """

    def __init__(self, *args, log_interval: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_interval = log_interval
        self._window_start_time = None
        self._window_tokens = 0
        self._window_steps = 0

        # DDP / FSDP 대비 world_size 추정
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                self._world_size = dist.get_world_size()
            else:
                self._world_size = int(os.environ.get("WORLD_SIZE", 1))
        except Exception:
            self._world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    def _load_rng_state(self, checkpoint):
        """
        PyTorch 2.6+ weights_only 문제 해결을 위한 오버라이드
        """
        import os
        import torch
        
        if checkpoint is None:
            return
        
        checkpoint_path = checkpoint if os.path.isdir(checkpoint) else None
        if checkpoint_path is None:
            return
        
        rng_file = os.path.join(checkpoint_path, "rng_state.pth")
        if not os.path.exists(rng_file):
            return
        
        try:
            # PyTorch 2.6+ 호환: weights_only=False 명시
            checkpoint_rng_state = torch.load(rng_file, weights_only=False)
            torch.set_rng_state(checkpoint_rng_state["torch"])
            if torch.cuda.is_available():
                if "cuda" in checkpoint_rng_state:
                    torch.cuda.set_rng_state_all(checkpoint_rng_state["cuda"])
            if "numpy" in checkpoint_rng_state:
                import numpy as np
                np.random.set_state(checkpoint_rng_state["numpy"])
            if "python" in checkpoint_rng_state:
                import random
                random.setstate(checkpoint_rng_state["python"])
        except Exception as e:
            logger.warning(f"RNG state 로드 실패 (무시하고 계속 진행): {e}")

    def training_step(self, model, inputs, num_items_in_batch=None):
        # 윈도우 시작 시점 기록
        if self._window_start_time is None:
            self._window_start_time = time.time()
            self._window_tokens = 0
            self._window_steps = 0

        # 토큰 수 집계 (input_ids 기준)
        try:
            input_ids = inputs.get("input_ids", None)
            if input_ids is not None:
                batch_tokens = input_ids.numel() * self._world_size
                self._window_tokens += batch_tokens
        except Exception:
            pass

        # 실제 학습 스텝은 부모 클래스로 위임
        if num_items_in_batch is not None:
            loss = super().training_step(model, inputs, num_items_in_batch)
        else:
            loss = super().training_step(model, inputs)

        self._window_steps += 1

        # 일정 step마다 속도 로그 남기기
        if self._window_steps >= self.log_interval:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

            elapsed = time.time() - self._window_start_time if self._window_start_time else 0.0
            if elapsed > 0 and self._window_tokens > 0:
                toks_per_sec = self._window_tokens / elapsed
            else:
                toks_per_sec = 0.0

            self.log({
                "mh_window_steps": self._window_steps,
                "mh_window_tokens": int(self._window_tokens),
                "mh_tok_per_sec_window": float(toks_per_sec),
                "mh_step_time_avg_sec": float(elapsed / max(self._window_steps, 1)),
            })

            # 윈도우 초기화
            self._window_start_time = None
            self._window_tokens = 0
            self._window_steps = 0

        return loss


class Qwen3BFineTuner:
    """Qwen2.5-3B 멀티턴 대화 파인튜너"""
    
    def __init__(self, config: Qwen3BFineTuningConfig):
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
        logger.info("Qwen2.5-3B-Instruct 모델 로딩")
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
        
        try:
            # Flash Attention 3 시도
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.base_model,
                max_seq_length=self.config.max_seq_length,
                dtype=None,
                load_in_4bit=self.config.load_in_4bit,
                load_in_8bit=self.config.load_in_8bit,
                trust_remote_code=True,
                attn_implementation="flash_attention_3"
            )
            logger.info("[ INFO ] ✅ Flash Attention 3 로드 성공!")
        except Exception as e:
            logger.warning(f"[ WARNING ] Flash Attention 3 로드 실패: {e}")
            logger.info("[ INFO ] Flash Attention 2로 시도합니다...")
            try:
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.config.base_model,
                    max_seq_length=self.config.max_seq_length,
                    dtype=None,
                    load_in_4bit=self.config.load_in_4bit,
                    load_in_8bit=self.config.load_in_8bit,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2"
                )
                logger.info("[ INFO ] ✅ Flash Attention 2 로드 성공!")
            except Exception as e2:
                logger.warning(f"[ WARNING ] Flash Attention 2 로드 실패: {e2}")
                logger.info("[ INFO ] Unsloth가 자동으로 최적의 attention을 선택합니다...")
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.config.base_model,
                    max_seq_length=self.config.max_seq_length,
                    dtype=None,
                    load_in_4bit=self.config.load_in_4bit,
                    load_in_8bit=self.config.load_in_8bit,
                    trust_remote_code=True,
                )
        
        logger.info("[ COMPLETE ] 베이스 모델 로드 완료")
        log_system_resources(logger, "베이스 모델 로드 후")
        
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
        
        # max_samples 제한 적용
        if self.config.max_samples and len(dataset) > self.config.max_samples:
            logger.info(f"[ INFO ] 데이터 샘플링: {len(dataset):,} → {self.config.max_samples:,}")
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
        
        formatted_data = []
        failed_count = 0
        text_field_count = 0
        messages_field_count = 0
        batch = []
        batch_size = 5000
        
        for i, example in enumerate(dataset):
            try:
                # text 필드가 이미 있는 경우
                if 'text' in example and example['text'] is not None and '<|im_start|>' in example['text']:
                    batch.append({"text": example['text']})
                    text_field_count += 1
                
                # messages 필드가 있는 경우
                elif 'messages' in example and example.get('messages'):
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
                    
                    batch.append({"text": text})
                    messages_field_count += 1
                else:
                    failed_count += 1
                
                # 배치가 차면 추가하고 초기화
                if len(batch) >= batch_size:
                    formatted_data.extend(batch)
                    batch = []
                
            except Exception as e:
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
        logger.info(" Qwen2.5-3B 멀티턴 대화 파인튜닝")
        logger.info(f" Run: {self.config.run_name}")
        logger.info("="*80)
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        log_system_resources(logger, "학습 준비")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        if gpu_monitor.available:
            gpu_monitor.clear_cache()
            logger.info("[ COMPLETE ] GPU 캐시 정리 완료")
        
        gc.collect()
        logger.info("[ COMPLETE ] Python GC 실행 완료")
        
        # 데이터 포맷팅
        logger.info("[ INFO ] 데이터 포맷팅 중...")
        train_formatted = self.format_dataset(dataset)
        
        logger.info(f"[ COMPLETE ] 훈련: {len(train_formatted):,}개")
        logger.info(f"Epoch: {self.config.num_train_epochs}")
        logger.info(f"배치 크기: {self.config.per_device_train_batch_size}")
        logger.info(f"효과적 배치: {self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps}")
        logger.info("="*80)
        
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
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            max_grad_norm=self.config.max_grad_norm,
            
            # 최적화
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            optim="adamw_8bit",
            
            # 로깅/저장
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            save_only_model=False,
            
            # 기타
            remove_unused_columns=False,
            report_to=[],
            seed=42,
            save_safetensors=True,
            
            # HuggingFace Hub 업로드
            push_to_hub=self.config.push_to_hub and self.config.hub_strategy == "end",
            hub_model_id=self.config.hub_model_id if self.config.push_to_hub else None,
        )
        
        # Callback 리스트 생성
        callbacks = [TrainingMonitorCallback()]
        
        # Trainer 생성
        logger.info("[ INFO ] SFTTrainer 초기화 중...")
        trainer = SpeedLoggingSFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_formatted,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            dataset_num_proc=4,
            packing=False,
            args=training_args,
            callbacks=callbacks,
            log_interval=10,
        )
        logger.info("[ COMPLETE ] Trainer 초기화 완료")
        
        # 훈련 시작
        logger.info("="*80)
        logger.info(" 학습 시작 ")
        logger.info(f" GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f" BF16 enabled: {training_args.bf16}")
        logger.info("="*80)
        
        log_system_resources(logger, "훈련 시작")
        
        trainer.train()
        
        # 최종 저장
        final_path = os.path.join(self.config.output_dir, "final")
        trainer.save_model(final_path)
        self.tokenizer.save_pretrained(final_path)
        
        logger.info("="*80)
        logger.info(" [ OK ] 훈련 완료!")
        logger.info("="*80)
        logger.info(f"모델 저장: {final_path}")
        
        # 최종 모델 Hub 업로드
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
