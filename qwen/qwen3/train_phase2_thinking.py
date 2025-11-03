#!/usr/bin/env python3
"""
Phase 2: Qwen3-VL Thinking 능력 파인튜닝
- CoT 추론 능력 추가
- <think> 태그 활용
- Phase 1 모델 또는 Qwen3-VL-8B-Thinking 기반
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
    model_name: str,
    max_seq_length: int = 8192,
    load_in_4bit: bool = True,
    lora_r: int = 32,
    lora_alpha: int = 64,
):
    """모델 및 LoRA 설정 (Thinking용 더 큰 rank)"""
    
    print("🚀 모델 로딩 중...")
    print(f"   모델: {model_name}")
    print(f"   최대 시퀀스 길이: {max_seq_length} (CoT용 확장)")
    print(f"   4bit 양자화: {load_in_4bit}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=None,
        trust_remote_code=True,
    )
    
    print("\n🔧 LoRA 적용 중 (Thinking용 확장)...")
    print(f"   Rank: {lora_r} (Thinking은 더 큰 rank 필요)")
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


def load_training_data(data_dir: str = "phase2_thinking_exaone"):
    """CoT 학습 데이터 로드"""
    
    data_dir = Path(data_dir)
    
    print(f"📖 CoT 학습 데이터 로딩...")
    print(f"   디렉토리: {data_dir}")
    
    # JSONL 파일 찾기
    jsonl_files = list(data_dir.glob("*_cot_augmented.jsonl"))
    
    if not jsonl_files:
        # all_cot_augmented.jsonl 찾기
        jsonl_files = list(data_dir.glob("all_cot_augmented.jsonl"))
    
    if not jsonl_files:
        raise FileNotFoundError(f"CoT JSONL 파일을 찾을 수 없습니다: {data_dir}")
    
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
    
    # 타입별 분포 확인
    if 'type' in dataset.column_names:
        types = dataset['type']
        from collections import Counter
        type_counts = Counter(types)
        
        print(f"\n📊 데이터 타입 분포:")
        for dtype, count in type_counts.items():
            percentage = (count / len(dataset)) * 100
            print(f"   {dtype}: {count:,}개 ({percentage:.1f}%)")
    
    # 샘플 확인
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n🔍 데이터 구조:")
        print(f"   키: {list(sample.keys())}")
        if 'messages' in sample:
            print(f"   메시지 수: {len(sample['messages'])}")
            
            # <think> 태그 확인
            for msg in sample['messages']:
                if msg['role'] == 'assistant':
                    has_think = '<think>' in msg['content']
                    print(f"   <think> 태그: {'✅ 포함' if has_think else '❌ 없음'}")
                    break
    
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


def train_phase2(
    model_name: str,
    data_dir: str = "phase2_thinking_exaone",
    output_dir: str = "qwen3-vl-8b-korean-thinking",
    max_seq_length: int = 8192,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-5,
    warmup_steps: int = 200,
    logging_steps: int = 10,
    save_steps: int = 500,
    lora_r: int = 32,
    lora_alpha: int = 64,
):
    """Phase 2 학습 실행"""
    
    print("="*80)
    print("Phase 2: Qwen3-VL Thinking 능력 파인튜닝")
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
    
    print("⚙️  학습 설정 (CoT 특화):")
    print(f"   출력 디렉토리: {output_dir}")
    print(f"   Epochs: {num_train_epochs} (CoT는 더 많은 epoch 필요)")
    print(f"   배치 크기: {per_device_train_batch_size} (긴 시퀀스로 작게)")
    print(f"   Gradient Accumulation: {gradient_accumulation_steps}")
    print(f"   효과적 배치 크기: {per_device_train_batch_size * gradient_accumulation_steps}")
    print(f"   학습률: {learning_rate} (더 작은 LR)")
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
        max_grad_norm=1.0,  # CoT 학습 안정화
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
    print("🚀 학습 시작! (CoT 패턴 학습)")
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
    
    print(f"\n💡 테스트 방법:")
    print(f"   시스템 프롬프트에 '/think' 추가")
    print(f"   또는 enable_thinking=True 사용")


def main():
    parser = argparse.ArgumentParser(description="Phase 2 학습")
    
    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help='기본 모델 (Phase 1 결과 또는 Qwen3-VL-8B-Thinking)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default="phase2_thinking_exaone",
        help='CoT 학습 데이터 디렉토리'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default="qwen3-vl-8b-korean-thinking",
        help='출력 디렉토리'
    )
    parser.add_argument(
        '--max-seq-length',
        type=int,
        default=8192,
        help='최대 시퀀스 길이 (CoT용 확장)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='학습 에폭 수'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=2,
        help='배치 크기'
    )
    parser.add_argument(
        '--gradient-accumulation',
        type=int,
        default=8,
        help='Gradient Accumulation Steps'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-5,
        help='학습률'
    )
    parser.add_argument(
        '--lora-r',
        type=int,
        default=32,
        help='LoRA rank (Thinking은 더 큰 rank 권장)'
    )
    parser.add_argument(
        '--lora-alpha',
        type=int,
        default=64,
        help='LoRA alpha'
    )
    
    args = parser.parse_args()
    
    train_phase2(
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

