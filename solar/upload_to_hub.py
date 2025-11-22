#!/usr/bin/env python3
"""
SOLAR-10.7B-Korean-QLora Checkpoint를 HuggingFace Hub에 업로드
Checkpoint-600 (Eval Loss: 0.6820, Epoch: 0.0505)
"""

import os
import json
from pathlib import Path
from huggingface_hub import HfApi, login, create_repo

# 설정
HUB_MODEL_ID = "MyeongHo0621/SOLAR-10.7B-Korean-QLora"
CHECKPOINT_DIR = "/home/work/.setting/solar/outputs/checkpoints/checkpoint-600"
OUTPUT_DIR = "/home/work/.setting/solar/outputs/checkpoints"
CHECKPOINT_NAME = "checkpoint-600"
STEP = 600
BASE_MODEL = "upstage/SOLAR-10.7B-Instruct-v1.0"
EVAL_LOSS = 0.6820
EPOCH = 0.0505

# 모델 카드 경로
MODEL_CARD_PATH = "/home/work/.setting/solar/MODEL_CARD.md"

print("=" * 80)
print("SOLAR-10.7B-Korean-QLora HuggingFace Hub 업로드")
print("=" * 80)

# HuggingFace 로그인
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    print("✅ HuggingFace 로그인 완료 (HF_TOKEN 환경 변수)")
else:
    print("⚠️  HF_TOKEN 환경 변수가 설정되지 않았습니다.")
    print("ℹ️  ~/.huggingface/token을 사용합니다.")
    try:
        login()
        print("✅ HuggingFace 로그인 완료 (~/.huggingface/token)")
    except Exception as e:
        print(f"❌ 로그인 실패: {e}")
        exit(1)

# Checkpoint 존재 확인
if not os.path.exists(CHECKPOINT_DIR):
    print(f"❌ Checkpoint 디렉토리가 없습니다: {CHECKPOINT_DIR}")
    exit(1)

# 필수 파일 확인
required_files = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]

print("\n📦 필수 파일 확인:")
for file in required_files:
    file_path = os.path.join(CHECKPOINT_DIR, file)
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"  ✅ {file:30s} ({file_size:.2f} MB)")
    else:
        print(f"  ❌ {file:30s} (없음)")
        exit(1)

print(f"\n📊 모델 정보:")
print(f"  • Base Model    : {BASE_MODEL}")
print(f"  • Hub Model ID  : {HUB_MODEL_ID}")
print(f"  • Checkpoint    : {CHECKPOINT_NAME}")
print(f"  • Step          : {STEP}")
print(f"  • Eval Loss     : {EVAL_LOSS:.4f}")
print(f"  • Epoch         : {EPOCH:.4f}")
print(f"  • License       : Apache 2.0")

# API 초기화
api = HfApi(token=hf_token)

# 리포지토리 생성 (이미 존재하면 무시)
try:
    print(f"\n🔧 리포지토리 생성 중...")
    create_repo(
        repo_id=HUB_MODEL_ID,
        repo_type="model",
        private=False,  # Public
        exist_ok=True
    )
    print(f"  ✅ 리포지토리 준비 완료")
except Exception as e:
    print(f"  ⚠️  리포지토리 생성 중 경고: {e}")
    print(f"  ℹ️  기존 리포지토리를 사용합니다.")

# 모델 카드 업로드 (있는 경우)
if os.path.exists(MODEL_CARD_PATH):
    print(f"\n📄 모델 카드 업로드 중...")
    try:
        api.upload_file(
            path_or_fileobj=MODEL_CARD_PATH,
            path_in_repo="README.md",
            repo_id=HUB_MODEL_ID,
            repo_type="model",
            commit_message="Add model card"
        )
        print(f"  ✅ 모델 카드 업로드 완료")
    except Exception as e:
        print(f"  ⚠️  모델 카드 업로드 실패: {e}")
else:
    print(f"\n⚠️  모델 카드를 찾을 수 없습니다: {MODEL_CARD_PATH}")
    print(f"  ℹ️  먼저 MODEL_CARD.md를 생성해주세요.")

# Checkpoint 업로드
try:
    print(f"\n🚀 Checkpoint 업로드 시작...")
    print(f"  ℹ️  업로드 경로: {CHECKPOINT_NAME}/**")
    
    # checkpoint-600/** 패턴만 업로드
    api.upload_folder(
        folder_path=OUTPUT_DIR,  # checkpoints 디렉토리
        repo_id=HUB_MODEL_ID,
        repo_type="model",
        commit_message=f"Upload {CHECKPOINT_NAME} (eval_loss: {EVAL_LOSS:.4f}, epoch: {EPOCH:.4f})",
        allow_patterns=[f"{CHECKPOINT_NAME}/**"],  # checkpoint-600/** 만
        ignore_patterns=["*.pt", "*.pth", "*.bin"],  # optimizer, scheduler 제외
    )
    
    print(f"\n{'=' * 80}")
    print(f"🎉 업로드 완료!")
    print(f"{'=' * 80}")
    print(f"\n📍 모델 URL:")
    print(f"  🔗 https://huggingface.co/{HUB_MODEL_ID}")
    print(f"\n📂 Checkpoint URL:")
    print(f"  🔗 https://huggingface.co/{HUB_MODEL_ID}/tree/main/{CHECKPOINT_NAME}")
    print(f"\n💡 사용 방법:")
    print(f"""
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# LoRA 어댑터 로드
config = PeftConfig.from_pretrained("{HUB_MODEL_ID}")
base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(base_model, "{HUB_MODEL_ID}", subfolder="{CHECKPOINT_NAME}")
tokenizer = AutoTokenizer.from_pretrained("{HUB_MODEL_ID}", subfolder="{CHECKPOINT_NAME}")
    """)
    print(f"{'=' * 80}\n")
    
except Exception as e:
    print(f"\n❌ 업로드 실패: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

