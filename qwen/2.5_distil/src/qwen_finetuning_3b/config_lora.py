#!/usr/bin/env python3
"""
Qwen2.5-3B LoRA (8bit) 파인튜닝 설정
- 기존 checkpoint-4250 스타일
- 8bit 양자화
- 460k 샘플 × 3 epoch
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from pathlib import Path
import os


def get_ft_llm_root() -> Path:
    """ft_llm 루트 디렉토리 경로 반환"""
    script_dir = Path(__file__).resolve().parent.parent.parent
    current = script_dir
    while current != current.parent:
        if current.name == 'ft_llm':
            return current
        current = current.parent
    
    # ft_llm을 찾지 못한 경우, .setting 디렉토리 사용
    if '.setting' in str(script_dir):
        return Path('/home/work/.setting')
    
    return script_dir.parent.parent

@dataclass
class Qwen3BFineTuningConfig:
    """Qwen2.5-3B LoRA (8bit) 파인튜닝 설정"""
    
    # 모델
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    max_seq_length: int = 2048
    
    # 토크나이저
    tokenizer_name: Optional[str] = None
    
    # 데이터 (전체 데이터셋 사용)
    korean_data_dir: str = field(default_factory=lambda: str(get_ft_llm_root() / "data"))
    data_files: List[str] = field(default_factory=lambda: ["smol_koreantalk_full.jsonl"])
    max_samples: Optional[int] = None  # 전체 사용 (460k)
    
    # 출력
    output_dir: str = field(default_factory=lambda: str(Path(__file__).resolve().parent.parent.parent / "outputs" / "checkpoints"))
    run_name: str = field(default_factory=lambda: f"qwen25-3b-KR-lora8bit-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # LoRA (8bit 용 - 기존 checkpoint-4250 스타일)
    lora_r: int = 32  # 8bit는 32 사용 (기존 값)
    lora_alpha: int = 64  # 8bit는 64 사용 (기존 값)
    lora_dropout: float = 0.0  # 0으로 설정 → Unsloth 최적화
    
    # 학습 설정
    num_train_epochs: int = 3  # 460k × 3 epoch
    per_device_train_batch_size: int = 32
    gradient_accumulation_steps: int = 4  # 효과적 배치: 128
    learning_rate: float = 1e-4  # 기존 학습률
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03  # 기존 값
    max_grad_norm: float = 1.0
    
    # 최적화 (LoRA 8bit)
    use_gradient_checkpointing: str = "unsloth"
    load_in_8bit: bool = True   # ✅ 8bit
    load_in_4bit: bool = False  # ❌ 4bit
    
    # 저장
    save_steps: int = 500
    save_total_limit: int = 3
    logging_steps: int = 10
    eval_steps: Optional[int] = None
    
    # 데이터셋 처리
    pre_tokenize_dataset: bool = True
    dataset_num_proc: int = 1
    
    # HuggingFace Hub 업로드
    push_to_hub: bool = True
    hub_model_id: Optional[str] = "MyeongHo0621/Qwen2.5-3B-Korean"  # 기존 모델
    hub_strategy: str = "end"
    hub_token: Optional[str] = None

