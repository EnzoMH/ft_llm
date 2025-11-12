"""
EEVE-VSS-SMH 모델을 WMS 데이터로 추가 파인튜닝
- 베이스: MyeongHo0621/eeve-vss-smh
- 데이터: wms_qa_dataset_v2_*.jsonl (EEVE 템플릿 형식)
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import argparse
from pathlib import Path


def load_wms_dataset(data_path: str):
    """EEVE 템플릿 형식의 WMS 데이터셋 로드"""
    
    # JSONL 파일 로드
    dataset = load_dataset('json', data_files=data_path, split='train')
    
    print(f"\n✓ 데이터셋 로드 완료: {len(dataset):,}개 샘플")
    print(f"✓ 샘플 예시:\n{dataset[0]['text'][:200]}...\n")
    
    return dataset


def create_training_dataset(dataset, tokenizer, max_length=4096):
    """토크나이징 및 훈련용 데이터셋 생성"""
    
    def tokenize_function(examples):
        # 'text' 필드를 그대로 토크나이징 (이미 EEVE 템플릿 형식)
        outputs = tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        
        # labels = input_ids (causal LM)
        outputs['labels'] = outputs['input_ids'].copy()
        
        return outputs
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    return tokenized_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True,
                        help='WMS 데이터셋 JSONL 파일 경로')
    parser.add_argument('--output-dir', type=str, default='./eeve-wms-output',
                        help='출력 디렉토리')
    parser.add_argument('--base-model', type=str, 
                        default='MyeongHo0621/eeve-vss-smh',
                        help='베이스 모델 경로')
    parser.add_argument('--epochs', type=int, default=3,
                        help='훈련 epoch 수')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='배치 크기 (per device)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='학습률')
    parser.add_argument('--max-seq-length', type=int, default=4096,
                        help='최대 시퀀스 길이')
    parser.add_argument('--use-4bit', action='store_true',
                        help='4bit 양자화 사용')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("EEVE-VSS-SMH WMS 파인튜닝")
    print("="*80)
    print(f"베이스 모델: {args.base_model}")
    print(f"데이터: {args.data_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print("="*80 + "\n")
    
    # 1. 데이터셋 로드
    print("[1/5] 데이터셋 로딩...")
    dataset = load_wms_dataset(args.data_path)
    
    # Train/Val 분할 (90/10)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    
    print(f"✓ Train: {len(train_dataset):,}개")
    print(f"✓ Eval: {len(eval_dataset):,}개\n")
    
    # 2. 토크나이저 로드
    print("[2/5] 토크나이저 로딩...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ Vocab size: {len(tokenizer)}\n")
    
    # 3. 모델 로드
    print("[3/5] 모델 로딩...")
    
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
    
    print(f"✓ 모델 로드 완료\n")
    
    # 4. LoRA 설정
    print("[4/5] LoRA 설정...")
    
    lora_config = LoraConfig(
        r=128,                    # 높은 rank (웹 검색 결과 기반)
        lora_alpha=256,           # alpha = 2 * r
        lora_dropout=0.0,         # Unsloth 최적화
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print()
    
    # 5. 데이터셋 토크나이징
    print("[5/5] 데이터셋 전처리...")
    train_dataset = create_training_dataset(train_dataset, tokenizer, args.max_seq_length)
    eval_dataset = create_training_dataset(eval_dataset, tokenizer, args.max_seq_length)
    print(f"✓ 전처리 완료\n")
    
    # 6. 훈련 설정
    print("="*80)
    print("훈련 시작")
    print("="*80 + "\n")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=2,  # effective batch = 16
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # 7. 훈련 실행
    trainer.train()
    
    # 8. 최종 모델 저장
    print("\n" + "="*80)
    print("훈련 완료! 모델 저장 중...")
    print("="*80 + "\n")
    
    final_output = output_dir / "final"
    trainer.save_model(str(final_output))
    tokenizer.save_pretrained(str(final_output))
    
    print(f"✓ 모델 저장: {final_output}")
    print("\n다음 단계:")
    print("1. 모델 병합 (선택):")
    print(f"   python merge_lora_to_base.py --base {args.base_model} --adapter {final_output}")
    print("2. 테스트:")
    print(f"   python test_eeve_wms.py --model {final_output}")
    print()


if __name__ == "__main__":
    main()

