#!/usr/bin/env python3
"""
HuggingFace Hub에 checkpoint 업로드 스크립트
config.py의 설정을 사용하여 checkpoint-600까지 업로드
"""

import os
import sys
from pathlib import Path
from typing import Optional

# config.py가 0_qwen14b 디렉토리에 있으므로 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "0_qwen14b"))

from huggingface_hub import HfApi, login
from config import Qwen14BFineTuningConfig


def load_env_file(env_path: str) -> None:
    """.env 파일을 로드하여 환경변수에 설정"""
    if not os.path.exists(env_path):
        return
    
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value:
                    os.environ[key] = value


def upload_checkpoint_to_hub(
    checkpoint_dir: str,
    hub_model_id: str,
    hub_token: Optional[str] = None,
    checkpoint_name: Optional[str] = None
) -> None:
    """
    단일 checkpoint를 HuggingFace Hub에 업로드
    
    Args:
        checkpoint_dir: checkpoint 디렉토리 경로
        hub_model_id: HuggingFace Hub 모델 ID
        hub_token: HuggingFace Hub 토큰
        checkpoint_name: checkpoint 이름 (예: "checkpoint-600")
    """
    api = HfApi(token=hub_token)
    
    if checkpoint_name is None:
        checkpoint_name = os.path.basename(checkpoint_dir)
    
    # LoRA adapter만 업로드 (adapter_model.safetensors, adapter_config.json 등)
    # 전체 모델이 아닌 adapter만 업로드하는 경우
    repo_id = hub_model_id
    
    print(f"\n[{checkpoint_name}] 업로드 시작...")
    print(f"  디렉토리: {checkpoint_dir}")
    print(f"  Hub 모델: {repo_id}")
    
    try:
        # adapter 파일들만 업로드
        adapter_files = [
            "adapter_model.safetensors",
            "adapter_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "added_tokens.json",
            "chat_template.jinja",
            "README.md"
        ]
        
        uploaded_files = []
        for file_name in adapter_files:
            file_path = os.path.join(checkpoint_dir, file_name)
            if os.path.exists(file_path):
                print(f"  업로드 중: {file_name}")
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=f"{checkpoint_name}/{file_name}",
                    repo_id=repo_id,
                    token=hub_token,
                    repo_type="model"
                )
                uploaded_files.append(file_name)
        
        print(f"[{checkpoint_name}] 업로드 완료 ({len(uploaded_files)}개 파일)")
        return True
        
    except Exception as e:
        print(f"[{checkpoint_name}] 업로드 실패: {e}")
        return False


def main():
    """메인 함수: config.py 설정을 읽어서 checkpoint-600만 업로드"""
    # .env 파일 로드
    env_path = "/home/work/tes/ft_llm/qwen/.env"
    load_env_file(env_path)
    
    config = Qwen14BFineTuningConfig()
    
    # 설정 확인
    if not config.push_to_hub:
        print("경고: config.py에서 push_to_hub가 False로 설정되어 있습니다.")
        response = input("계속 진행하시겠습니까? (y/n): ")
        if response.lower() != 'y':
            print("업로드 취소됨")
            return
    
    if not config.hub_model_id:
        print("오류: hub_model_id가 설정되지 않았습니다.")
        sys.exit(1)
    
    # HuggingFace Hub 로그인 - os.getenv("HF_TOKEN") 우선 사용
    hub_token = os.getenv("HF_TOKEN")
    
    if not hub_token:
        print("오류: 환경변수 HF_TOKEN이 설정되지 않았습니다.")
        print(f".env 파일 경로: {env_path}")
        sys.exit(1)
    
    print("환경변수 HF_TOKEN 사용")
    
    # 로그인 시도
    try:
        login(token=hub_token)
        print("HuggingFace Hub 로그인 성공")
    except Exception as e:
        print(f"토큰으로 로그인 실패: {e}")
        sys.exit(1)
    
    # output 디렉토리에서 checkpoint-600만 찾기
    output_dir = Path(config.output_dir)
    if not output_dir.exists():
        print(f"오류: output 디렉토리가 존재하지 않습니다: {output_dir}")
        sys.exit(1)
    
    # checkpoint-600만 업로드
    checkpoint_dir = output_dir / "checkpoint-600"
    if not checkpoint_dir.exists():
        print(f"오류: checkpoint-600이 존재하지 않습니다: {checkpoint_dir}")
        sys.exit(1)
    
    print(f"\ncheckpoint-600 업로드 시작...")
    
    # checkpoint-600 업로드
    success = upload_checkpoint_to_hub(
        checkpoint_dir=str(checkpoint_dir),
        hub_model_id=config.hub_model_id,
        hub_token=hub_token,
        checkpoint_name="checkpoint-600"
    )
    
    if success:
        print(f"\n업로드 완료: checkpoint-600")
        print(f"Hub 모델: https://huggingface.co/{config.hub_model_id}")
    else:
        print(f"\n업로드 실패: checkpoint-600")
        sys.exit(1)


if __name__ == "__main__":
    main()

