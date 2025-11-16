#!/usr/bin/env python3
"""
Qwen2.5-14B 파인튜닝 설정
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from pathlib import Path
import os


def get_ft_llm_root() -> Path:
    """ft_llm 루트 디렉토리 경로 반환"""
    # config.py가 0_qwen14b 디렉토리에 있으므로
    script_dir = Path(__file__).resolve().parent.parent
    
    # 상위 디렉토리로 올라가면서 ft_llm 찾기
    current = script_dir
    while current != current.parent:
        if current.name == 'ft_llm':
            return current
        current = current.parent
    
    return script_dir.parent.parent

@dataclass
class Qwen14BFineTuningConfig:
    """Qwen2.5-14B 파인튜닝 설정"""
    
    # 모델
    base_model: str = "Qwen/Qwen2.5-14B-Instruct"
    max_seq_length: int = 4096
    
    # 토크나이저 (기본 모델과 동일)
    tokenizer_name: Optional[str] = None  # None이면 base_model 사용
    
    # 데이터 (단일 파일 사용)
    korean_data_dir: str = field(default_factory=lambda: str(get_ft_llm_root() / "data"))
    data_files: List[str] = None  # ["smol_koreantalk_full.jsonl"]로 오버라이드 예정
    max_samples: int = 200000  # RAM 압박 방지 (460k 중 200k 사용)
    
    # 출력 (상대 경로 사용)
    output_dir: str = field(default_factory=lambda: str(Path(__file__).resolve().parent.parent / "output"))
    run_name: str = field(default_factory=lambda: f"qwen25-14b-KR-multiturn-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # LoRA
    lora_r: int = 64  # 14B는 64가 적당
    lora_alpha: int = 128
    lora_dropout: float = 0.0  # 0으로 설정 → Unsloth 최적화 활성화
    
    # 학습 설정 (H100 72GB 최적화 - VRAM 최대 활용)
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 22  # 72GB VRAM 기준 (40GB→72GB, 12→22로 증가)
    gradient_accumulation_steps: int = 4  # 효과적 배치: 88 (22×4)
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
    hub_model_id: Optional[str] = "MyeongHo0621/Qwen2.5-14B-Korean"  # 예: "username/qwen2.5-14b-korean-multiturn"
    hub_strategy: str = "checkpoint"  # "checkpoint" (매 checkpoint마다) 또는 "end" (훈련 완료 시만)
    hub_token: Optional[str] = ""  # None이면 ~/.huggingface/token 사용

