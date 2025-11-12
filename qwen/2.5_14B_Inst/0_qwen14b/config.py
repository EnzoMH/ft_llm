#!/usr/bin/env python3
"""
Qwen2.5-14B 파인튜닝 설정
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List


@dataclass
class Qwen14BFineTuningConfig:
    """Qwen2.5-14B 파인튜닝 설정"""
    
    # 모델
    base_model: str = "Qwen/Qwen2.5-14B-Instruct"
    max_seq_length: int = 4096
    
    # 토크나이저 (기본 모델과 동일)
    tokenizer_name: Optional[str] = None  # None이면 base_model 사용
    
    # 데이터 (단일 파일 사용)
    korean_data_dir: str = "/home/work/tesseract/korean_large_data/cleaned_jsonl"
    data_files: List[str] = None  # ["smol_koreantalk_data.jsonl"]로 오버라이드 예정
    max_samples: int = 200000  # RAM 압박 방지 (460k 중 200k 사용)
    
    # 출력
    output_dir: str = "/home/work/tesseract/qwen/2.5_14B_Inst/output"
    run_name: str = f"qwen25-14b-KR-multiturn-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # LoRA
    lora_r: int = 64  # 14B는 64가 적당
    lora_alpha: int = 128
    lora_dropout: float = 0.0  # 0으로 설정 → Unsloth 최적화 활성화
    
    # 학습 설정 (H100 80GB 최적화 - VRAM 20GB 여유 활용)
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 28  # 24→28 (VRAM 20GB 여유, 약 17% 증가)
    gradient_accumulation_steps: int = 2  # 효과적 배치: 56
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    
    # 최적화
    use_gradient_checkpointing: str = "unsloth"
    load_in_8bit: bool = True
    load_in_4bit: bool = False
    
    # 저장 (체크포인트 100 step마다)
    save_steps: int = 100  # 500→100 (더 자주 저장, 약 30분마다)
    save_total_limit: int = 5  # 3→5 (최근 5개 체크포인트 유지)
    logging_steps: int = 10
    eval_steps: int = 100  # 500→100 (더 자주 평가)
    
    # HuggingFace Hub 업로드 (자동 백업)
    push_to_hub: bool = True  # True로 설정 시 자동 업로드
    hub_model_id: Optional[str] = ""  # 예: "username/qwen2.5-14b-korean-multiturn"
    hub_strategy: str = "checkpoint"  # "checkpoint" (매 checkpoint마다) 또는 "end" (훈련 완료 시만)
    hub_token: Optional[str] = ""  # None이면 ~/.huggingface/token 사용

