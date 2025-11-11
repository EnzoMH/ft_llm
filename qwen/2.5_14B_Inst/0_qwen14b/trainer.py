#!/usr/bin/env python3
"""
Qwen2.5-14B 멀티턴 대화 파인튜너
"""

import os
import gc
import logging
import torch
from typing import Optional

# Unsloth를 먼저 import (최적화 활성화)
from unsloth import FastLanguageModel, is_bfloat16_supported

# 나머지 import
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from huggingface_hub import login

from .config import Qwen14BFineTuningConfig
from .dataset_loader import MultiTurnDatasetLoader
from .callbacks import TrainingMonitorCallback
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
        
        # GPU 캐시 정리 및 메모리 최적화
        log_system_resources(logger, "학습 준비")
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
            dataset_num_proc=4,  # RAM 압박 방지 (8→4, 3.8GB 데이터 안전하게 처리)
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
        
        log_system_resources(logger, "훈련 시작")
        
        # Resume 체크포인트 자동 감지
        resume_checkpoint = None
        if os.path.exists(self.config.output_dir):
            checkpoints = [d for d in os.listdir(self.config.output_dir) 
                          if d.startswith("checkpoint-") and os.path.isdir(os.path.join(self.config.output_dir, d))]
            if checkpoints:
                # 가장 최근 체크포인트 찾기
                checkpoints.sort(key=lambda x: int(x.split("-")[1]))
                resume_checkpoint = os.path.join(self.config.output_dir, checkpoints[-1])
                logger.info(f"[ INFO ] 체크포인트 발견: {resume_checkpoint}")
                logger.info(f"[ INFO ] 학습 재개 중...")
        
        trainer.train(resume_from_checkpoint=resume_checkpoint)
        
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

