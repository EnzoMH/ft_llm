#!/usr/bin/env python3
"""
Qwen2.5-3B 파인튜닝 설정
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from pathlib import Path
import os


def get_ft_llm_root() -> Path:
    """ft_llm 루트 디렉토리 경로 반환"""
    # config.py가 src/qwen_finetuning_3b 디렉토리에 있으므로
    script_dir = Path(__file__).resolve().parent.parent.parent
    
    # 상위 디렉토리로 올라가면서 ft_llm 찾기
    current = script_dir
    while current != current.parent:
        if current.name == 'ft_llm':
            return current
        current = current.parent
    
    return script_dir.parent.parent

@dataclass
class Qwen3BFineTuningConfig:
    """Qwen2.5-3B 파인튜닝 설정"""
    
    # 모델
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    max_seq_length: int = 2048  # 3B 모델은 2048로 충분
    
    # 토크나이저 (기본 모델과 동일)
    tokenizer_name: Optional[str] = None  # None이면 base_model 사용
    
    # 데이터 (smol_koreantalk_full.jsonl 단일 파일 사용)
    korean_data_dir: str = field(default_factory=lambda: str(get_ft_llm_root() / "data"))
    data_files: List[str] = field(default_factory=lambda: ["smol_koreantalk_full.jsonl"])
    max_samples: Optional[int] = None  # 전체 사용 (460k 샘플)
    
    # 출력
    output_dir: str = field(default_factory=lambda: str(Path(__file__).resolve().parent.parent.parent / "outputs" / "checkpoints"))
    run_name: str = field(default_factory=lambda: f"qwen25-3b-KR-smoltalk-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # LoRA (3B 모델에 맞게 조정)
    lora_r: int = 32  # 3B는 32가 적당 (14B는 64)
    lora_alpha: int = 64  # 3B는 64가 적당 (14B는 128)
    lora_dropout: float = 0.0  # 0으로 설정 → Unsloth 최적화 활성화
    
    # 학습 설정 (H100 80GB 최적화)
    num_train_epochs: int = 3  # 46만개 샘플 × 3 epoch
    per_device_train_batch_size: int = 32  # 3B는 더 큰 배치 가능
    gradient_accumulation_steps: int = 4  # 효과적 배치: 128 (32×4)
    learning_rate: float = 1e-4  # 7.5e-5 → 1e-4 (plateau 탈출)
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    
    # 최적화
    use_gradient_checkpointing: str = "unsloth"
    load_in_8bit: bool = True
    load_in_4bit: bool = False
    
    # 저장
    save_steps: int = 500  # 500 step마다 저장 (약 5시간마다)
    save_total_limit: int = 3  # 최근 3개 체크포인트 유지
    logging_steps: int = 10
    eval_steps: Optional[int] = None  # 평가 없음
    
    # 데이터셋 처리 (시스템 리소스 기반 최적화)
    # CPU: 24 코어 → 8 코어로 제한, RAM: 63GB (51GB 사용 가능)
    # 멀티프로세싱 오류 방지를 위해 데이터셋을 미리 토크나이징
    pre_tokenize_dataset: bool = True  # True: format_dataset에서 토크나이징 완료, False: Unsloth에서 토크나이징
    dataset_num_proc: int = 1  # pre_tokenize_dataset=True일 때는 사용되지 않음
    
    # HuggingFace Hub 업로드
    push_to_hub: bool = True
    hub_model_id: Optional[str] = "MyeongHo0621/Qwen2.5-3B-Korean"
    hub_strategy: str = "end"  # 학습 완료 시만 업로드
    hub_token: Optional[str] = None  # None이면 ~/.huggingface/token 사용

