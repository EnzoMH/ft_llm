#!/usr/bin/env python3
"""
LoRA 어댑터를 base 모델에 병합하여 HuggingFace Hub에 업로드
- checkpoint-3700을 base 모델과 병합
- 병합된 전체 모델을 업로드 (독립 사용 가능)
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 환경 변수 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 모듈 경로 추가
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)


def get_ft_llm_root() -> Path:
    """ft_llm 루트 디렉토리 경로 반환"""
    script_dir = Path(__file__).resolve().parent
    
    # 상위 디렉토리로 올라가면서 ft_llm 찾기
    current = script_dir
    while current != current.parent:
        if current.name == 'ft_llm':
            return current
        current = current.parent
    
    # ft_llm을 찾지 못한 경우 스크립트 위치 기준으로 상대 경로 사용
    return script_dir.parent.parent


def find_and_load_env_file() -> str:
    """여러 경로에서 .env 파일을 찾아서 로드
    
    Returns:
        str: 로드된 .env 파일 경로, 없으면 None
    """
    script_dir = Path(__file__).resolve().parent
    ft_llm_root = get_ft_llm_root()
    
    # 우선순위 순서로 .env 파일 경로 리스트 생성
    env_paths = [
        script_dir / ".env",                          # 현재 스크립트 디렉토리
        ft_llm_root / "qwen" / "2.5_14B_Inst" / ".env",  # qwen/2.5_14B_Inst/.env
        ft_llm_root / "qwen" / ".env",               # qwen/.env
        ft_llm_root / ".env",                         # ft_llm/.env
    ]
    
    for env_path in env_paths:
        env_path_str = str(env_path)
        if os.path.exists(env_path_str):
            load_dotenv(env_path_str)
            print(f".env 파일 로드됨: {env_path_str}")
            return env_path_str
    
    print("경고: .env 파일을 찾을 수 없습니다.")
    return None


# .env 파일 찾기 및 로드
find_and_load_env_file()

from unsloth import FastLanguageModel
from huggingface_hub import HfApi, login
from peft import PeftModel
import torch


def merge_and_upload(
    checkpoint_path: str,
    base_model: str = "Qwen/Qwen2.5-14B-Instruct",
    hub_model_id: str = "MyeongHo0621/Qwen2.5-14B-Korean",
    hub_token: str = None,
    output_dir: str = None
):
    """
    LoRA 어댑터를 base 모델에 병합하고 HuggingFace Hub에 업로드
    
    Args:
        checkpoint_path: LoRA 어댑터 경로 (예: "output/checkpoint-3700")
        base_model: Base 모델 이름
        hub_model_id: HuggingFace Hub 모델 ID
        hub_token: HuggingFace Hub 토큰
        output_dir: 병합된 모델 저장 경로 (None이면 임시 디렉토리 사용)
    """
    print("\n" + "="*80)
    print(" LoRA 어댑터 병합 및 업로드")
    print("="*80)
    print(f"Base 모델: {base_model}")
    print(f"어댑터 경로: {checkpoint_path}")
    print(f"Hub 모델: {hub_model_id}")
    print("="*80)
    
    # 토큰 확인
    if hub_token is None:
        hub_token = os.getenv("HF_TOKEN")
        if not hub_token:
            print("\n오류: HF_TOKEN 환경변수가 설정되지 않았습니다.")
            sys.exit(1)
    
    # 로그인
    try:
        login(token=hub_token)
        print("\nHuggingFace Hub 로그인 성공")
    except Exception as e:
        print(f"\n오류: 로그인 실패 - {e}")
        sys.exit(1)
    
    # 출력 디렉토리 설정
    if output_dir is None:
        output_dir = os.path.join(_current_dir, "merged_model")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n[1/4] Base 모델 로드 중...")
    # Base 모델 로드 (8bit로 로드하여 메모리 절약)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=False,
        load_in_8bit=True,
        trust_remote_code=True,
    )
    
    print("[2/4] LoRA 어댑터 로드 중...")
    # LoRA 어댑터 로드
    model = PeftModel.from_pretrained(
        model,
        checkpoint_path,
        is_trainable=False
    )
    
    print("[3/4] 어댑터 병합 중...")
    # 어댑터를 base 모델에 병합
    model = model.merge_and_unload()
    
    # 추론 모드
    FastLanguageModel.for_inference(model)
    
    print("[4/4] 병합된 모델 저장 중...")
    
    # Base 모델의 config를 다시 로드하여 깨끗한 config 사용
    # (Unsloth가 추가한 직렬화 불가능한 항목 제거)
    from transformers import AutoConfig
    try:
        print("  Base 모델 config 로드 중...")
        clean_config = AutoConfig.from_pretrained(base_model)
        model.config = clean_config
        print("  Config 교체 완료")
    except Exception as e:
        print(f"  경고: Base config 로드 실패, 원본 config 사용: {e}")
    
    # 병합된 모델 저장
    try:
        model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
    except Exception as e:
        print(f"  오류: 모델 저장 실패: {e}")
        print("  재시도: config 없이 모델 가중치만 저장...")
        raise
    
    tokenizer.save_pretrained(output_dir)
    
    print("\n[5/5] HuggingFace Hub에 업로드 중...")
    # HuggingFace Hub에 업로드
    api = HfApi(token=hub_token)
    
    # 저장된 파일들을 업로드
    api.upload_folder(
        folder_path=output_dir,
        repo_id=hub_model_id,
        repo_type="model",
        token=hub_token,
        commit_message=f"Merge checkpoint-3700 (loss 0.8006) with base model"
    )
    
    print("\n" + "="*80)
    print(" 업로드 완료!")
    print("="*80)
    print(f"Hub 모델: https://huggingface.co/{hub_model_id}")
    print(f"로컬 저장 경로: {output_dir}")
    print("="*80)


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LoRA 어댑터 병합 및 업로드")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="output/checkpoint-3700",
        help="체크포인트 경로 (기본값: output/checkpoint-3700)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-14B-Instruct",
        help="Base 모델 이름"
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default="MyeongHo0621/Qwen2.5-14B-Korean",
        help="HuggingFace Hub 모델 ID"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="병합된 모델 저장 경로 (기본값: merged_model)"
    )
    
    args = parser.parse_args()
    
    # 경로 확인
    checkpoint_path = os.path.join(_current_dir, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        print(f"\n오류: 체크포인트 경로를 찾을 수 없습니다: {checkpoint_path}")
        sys.exit(1)
    
    merge_and_upload(
        checkpoint_path=checkpoint_path,
        base_model=args.base_model,
        hub_model_id=args.hub_model_id,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

