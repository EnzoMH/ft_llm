#!/usr/bin/env python3
"""
Phase 1: Qwen3-VL 한국어 강화 파인튜닝
- Qwen3-VL-8B-Instruct 기반
- 한국어 능력 향상
- 중국어 차단
- LoRA 사용
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from datetime import datetime

from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments


def setup_model(
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    max_seq_length: int = 4096,
    load_in_4bit: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    """모델 및 LoRA 설정"""
    
    print("🚀 모델 로딩 중...")
    print(f"   모델: {model_name}")
    print(f"   최대 시퀀스 길이: {max_seq_length}")
    print(f"   4bit 양자화: {load_in_4bit}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=None,  # Auto detection
        trust_remote_code=True,
    )
    
    print("\n🔧 LoRA 적용 중...")
    print(f"   Rank: {lora_r}")
    print(f"   Alpha: {lora_alpha}")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )
    
    print("✅ 모델 준비 완료!\n")
    
    return model, tokenizer


def load_training_data(data_dir: str = "phase1_korean"):
    """학습 데이터 로드"""
    
    data_dir = Path(data_dir)
    
    print(f"📖 학습 데이터 로딩...")
    print(f"   디렉토리: {data_dir}")
    
    # JSONL 파일 찾기
    jsonl_files = list(data_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        raise FileNotFoundError(f"JSONL 파일을 찾을 수 없습니다: {data_dir}")
    
    print(f"   발견된 파일: {len(jsonl_files)}개")
    for f in jsonl_files:
        print(f"      - {f.name}")
    
    # 데이터셋 로드
    dataset = load_dataset(
        'json',
        data_files=[str(f) for f in jsonl_files],
        split='train'
    )
    
    print(f"\n✅ 데이터 로딩 완료!")
    print(f"   총 샘플: {len(dataset):,}개")
    
    # 샘플 확인
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n🔍 데이터 구조:")
        print(f"   키: {list(sample.keys())}")
        if 'messages' in sample:
            print(f"   메시지 수: {len(sample['messages'])}")
            print(f"   첫 메시지 역할: {sample['messages'][0]['role']}")
    
    return dataset


def format_chat_template(example, tokenizer):
    """채팅 템플릿 포맷팅"""
    messages = example['messages']
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}


def train_phase1(
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    data_dir: str = "phase1_korean",
    output_dir: str = "qwen3-vl-8b-korean-instruct",
    max_seq_length: int = 4096,
    num_train_epochs: int = 2,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    save_steps: int = 500,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    """Phase 1 학습 실행"""
    
    print("="*80)
    print("Phase 1: Qwen3-VL 한국어 강화 파인튜닝")
    print("="*80)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 모델 설정
    model, tokenizer = setup_model(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    )
    
    # 데이터 로드
    dataset = load_training_data(data_dir)
    
    # 데이터 포맷팅
    print("\n🔄 데이터 포맷팅 중...")
    dataset = dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=dataset.column_names
    )
    print("✅ 포맷팅 완료!\n")
    
    # 학습 설정
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("⚙️  학습 설정:")
    print(f"   출력 디렉토리: {output_dir}")
    print(f"   Epochs: {num_train_epochs}")
    print(f"   배치 크기: {per_device_train_batch_size}")
    print(f"   Gradient Accumulation: {gradient_accumulation_steps}")
    print(f"   효과적 배치 크기: {per_device_train_batch_size * gradient_accumulation_steps}")
    print(f"   학습률: {learning_rate}")
    print(f"   Warmup Steps: {warmup_steps}")
    print()
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=logging_steps,
        logging_dir=str(output_dir / "logs"),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        report_to=["tensorboard"],
        seed=42,
    )
    
    # Trainer 생성
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
        packing=False,
    )
    
    # 학습 시작
    print("🚀 학습 시작!")
    print("="*80)
    
    trainer.train()
    
    print("\n" + "="*80)
    print("✅ 학습 완료!")
    print("="*80)
    
    # 모델 저장
    print(f"\n💾 모델 저장 중: {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # LoRA 어댑터 병합 (선택적)
    print("\n💾 LoRA 어댑터 병합 중...")
    model.save_pretrained_merged(
        str(output_dir / "merged"),
        tokenizer,
        save_method="merged_16bit",
    )
    
    print(f"\n✅ 모든 작업 완료!")
    print(f"   모델 위치: {output_dir}")
    print(f"   병합 모델: {output_dir / 'merged'}")
    print(f"   종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 학습")
    
    parser.add_argument(
        '--model-name',
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help='기본 모델 이름'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default="phase1_korean",
        help='학습 데이터 디렉토리'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default="qwen3-vl-8b-korean-instruct",
        help='출력 디렉토리'
    )
    parser.add_argument(
        '--max-seq-length',
        type=int,
        default=4096,
        help='최대 시퀀스 길이'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=2,
        help='학습 에폭 수'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='배치 크기'
    )
    parser.add_argument(
        '--gradient-accumulation',
        type=int,
        default=4,
        help='Gradient Accumulation Steps'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-5,
        help='학습률'
    )
    parser.add_argument(
        '--lora-r',
        type=int,
        default=16,
        help='LoRA rank'
    )
    parser.add_argument(
        '--lora-alpha',
        type=int,
        default=32,
        help='LoRA alpha'
    )
    
    args = parser.parse_args()
    
    train_phase1(
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )


if __name__ == "__main__":
    main()

