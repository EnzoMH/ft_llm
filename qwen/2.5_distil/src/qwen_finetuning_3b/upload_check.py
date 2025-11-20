#!/usr/bin/env python3
"""
Checkpoint 3000을 HuggingFace Hub에 업로드
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, login

# 설정
HUB_MODEL_ID = "MyeongHo0621/Qwen2.5-3B-Korean"
CHECKPOINT_DIR = "/home/work/vss/ft_llm/qwen/2.5_distil/outputs/checkpoints/checkpoint-3000"
OUTPUT_DIR = "/home/work/vss/ft_llm/qwen/2.5_distil/outputs/checkpoints"
CHECKPOINT_NAME = "checkpoint-3000"
STEP = 3000

# HuggingFace 로그인
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    print("[ INFO ] HuggingFace 로그인 완료")
else:
    print("[ WARNING ] HF_TOKEN 환경 변수가 설정되지 않았습니다.")
    print("[ INFO ] ~/.huggingface/token을 사용합니다.")

# Checkpoint 존재 확인
if not os.path.exists(CHECKPOINT_DIR):
    print(f"[ ERROR ] Checkpoint 디렉토리가 없습니다: {CHECKPOINT_DIR}")
    exit(1)

print(f"[ INFO ] Checkpoint 경로: {CHECKPOINT_DIR}")
print(f"[ INFO ] Hub 모델 ID: {HUB_MODEL_ID}")

# API 초기화
api = HfApi(token=hf_token)

try:
    print(f"[ INFO ] Step {STEP} 체크포인트 업로드 시작...")
    
    # checkpoint-3000/** 패턴만 업로드
    api.upload_folder(
        folder_path=OUTPUT_DIR,  # output_dir를 기준으로
        repo_id=HUB_MODEL_ID,
        repo_type="model",
        commit_message=f"Checkpoint at step {STEP}",
        allow_patterns=[f"{CHECKPOINT_NAME}/**"],  # checkpoint-3000/** 패턴만 업로드
    )
    
    print(f"[ HUB ] ✅ Step {STEP} 체크포인트 업로드 완료!")
    print(f"[ HUB ] URL: https://huggingface.co/{HUB_MODEL_ID}/tree/main/{CHECKPOINT_NAME}")
    
except Exception as e:
    print(f"[ ERROR ] 업로드 실패: {e}")
    import traceback
    traceback.print_exc()
    exit(1)