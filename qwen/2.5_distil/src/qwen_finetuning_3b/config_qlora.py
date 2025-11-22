#!/usr/bin/env python3
"""
Qwen2.5-3B QLoRA (4bit) 파인튜닝 설정
- 현재 학습 중인 설정
- 4bit 양자화
- 60k 샘플 × 1 epoch (시간 제약)
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
    """Qwen2.5-3B QLoRA (4bit) 파인튜닝 설정"""
    
    # 모델
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    max_seq_length: int = 2048
    
    # 토크나이저
    tokenizer_name: Optional[str] = None
    
    # 데이터 (60k 샘플 - 시간 제약)
    korean_data_dir: str = field(default_factory=lambda: str(get_ft_llm_root() / "data"))
    data_files: List[str] = field(default_factory=lambda: ["smol_koreantalk_full.jsonl"])
    max_samples: Optional[int] = 200000  # 60만 중 20만 샘플 사용
    
    # 출력
    output_dir: str = field(default_factory=lambda: str(Path(__file__).resolve().parent.parent.parent / "outputs" / "checkpoints"))
    run_name: str = field(default_factory=lambda: f"qwen25-3b-KR-qlora4bit-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # LoRA (QLoRA 4bit용 - 더 큰 rank)
    lora_r: int = 64  # 4bit는 64 사용 (더 높은 표현력)
    lora_alpha: int = 128  # 4bit는 128 사용
    lora_dropout: float = 0.05  # 과적합 방지
    
    # 학습 설정
    num_train_epochs: int = 3  # 200k 샘플 × 3 epochs = 600k 총 학습
    per_device_train_batch_size: int = 32  # QLoRA는 32 가능
    gradient_accumulation_steps: int = 4  # 효과적 배치: 128
    learning_rate: float = 2e-4  # 더 높은 LR (plateau 탈출)
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1  # 더 긴 warmup
    max_grad_norm: float = 1.0
    
    # 최적화 (QLoRA 4bit)
    use_gradient_checkpointing: str = "unsloth"
    load_in_8bit: bool = False  # ❌ 8bit
    load_in_4bit: bool = True   # ✅ 4bit QLoRA
    
    # 저장
    save_steps: int = 200  # 더 자주 저장 (시간 제약)
    save_total_limit: int = 2
    logging_steps: int = 10
    eval_steps: Optional[int] = None
    
    # 데이터셋 처리
    pre_tokenize_dataset: bool = True
    dataset_num_proc: int = 1
    
    # HuggingFace Hub 업로드
    push_to_hub: bool = True
    hub_model_id: Optional[str] = "MyeongHo0621/Qwen2.5-3B-Korean-QLoRA"  # QLoRA 전용
    hub_strategy: str = "end"
    hub_token: Optional[str] = None

