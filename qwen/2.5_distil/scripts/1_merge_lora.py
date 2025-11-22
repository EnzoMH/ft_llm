#!/usr/bin/env python3
"""
Step 1: LoRA 어댑터를 Base 모델과 Merge
- PEFT 어댑터를 베이스 모델과 병합
- 완전한 모델로 저장 (모든 프레임워크 호환)
"""

import os
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 설정
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LORA_ADAPTER = "MyeongHo0621/Qwen2.5-3B-Korean-QLoRA"
LORA_CHECKPOINT = None  # None = 루트 경로 사용, "final" = final 폴더 사용
OUTPUT_DIR = "/home/work/.setting/qwen/2.5_distil/outputs/merged"

print("=" * 80)
print("LoRA 어댑터 Merge 시작")
print("=" * 80)

# 출력 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Base 모델 로딩
print(f"\n[ 1/4 ] Base 모델 로딩: {BASE_MODEL}")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,  # 메모리 절약
    device_map="auto",
    trust_remote_code=True
)
print(f"  ✅ Base 모델 로딩 완료")
print(f"  ℹ️  파라미터 수: {base_model.num_parameters() / 1e9:.2f}B")

# 2. LoRA 어댑터 로딩
if LORA_CHECKPOINT:
    print(f"\n[ 2/4 ] LoRA 어댑터 로딩: {LORA_ADAPTER}/{LORA_CHECKPOINT}")
    model = PeftModel.from_pretrained(
        base_model,
        LORA_ADAPTER,
        subfolder=LORA_CHECKPOINT
    )
else:
    print(f"\n[ 2/4 ] LoRA 어댑터 로딩: {LORA_ADAPTER} (루트)")
    model = PeftModel.from_pretrained(
        base_model,
        LORA_ADAPTER
    )
print(f"  ✅ LoRA 어댑터 로딩 완료 (step 4689, 최종 모델)")

# 3. Merge
print(f"\n[ 3/4 ] LoRA 어댑터를 Base 모델과 Merge 중...")
print(f"  ℹ️  메모리 사용량: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
merged_model = model.merge_and_unload()
print(f"  ✅ Merge 완료!")

# 4. 저장
print(f"\n[ 4/4 ] Merged 모델 저장 중: {OUTPUT_DIR}")
merged_model.save_pretrained(
    OUTPUT_DIR,
    safe_serialization=True,  # safetensors 사용
    max_shard_size="2GB"
)
print(f"  ✅ 모델 저장 완료")

# 5. 토크나이저 저장
print(f"\n[ 5/4 ] 토크나이저 저장 중...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"  ✅ 토크나이저 저장 완료")

# 6. 저장된 파일 확인
print(f"\n📂 저장된 파일:")
for file in sorted(Path(OUTPUT_DIR).glob("*")):
    if file.is_file():
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  - {file.name:40s} ({size_mb:>8.2f} MB)")

print(f"\n{'=' * 80}")
print(f"🎉 Merge 완료!")
print(f"{'=' * 80}")
print(f"\n📍 출력 디렉토리: {OUTPUT_DIR}")
print(f"\n💡 다음 단계: GGUF 변환")
print(f"  python 2_convert_to_gguf.py")

