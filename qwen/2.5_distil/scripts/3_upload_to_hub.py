#!/usr/bin/env python3
"""
Step 3: Merged 모델과 GGUF 파일들을 HuggingFace Hub에 업로드
- Merged 모델 → MyeongHo0621/Qwen2.5-3B-Korean (별도 리포)
- GGUF 파일 → MyeongHo0621/Qwen2.5-3B-Korean (gguf/ 폴더)
- PEFT 어댑터는 기존 리포에 유지: MyeongHo0621/Qwen2.5-3B-Korean-QLoRA
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, login, create_repo

# 설정
MERGED_REPO_ID = "MyeongHo0621/Qwen2.5-3B-Korean"  # 별도 리포 (Merged + GGUF)
PEFT_REPO_ID = "MyeongHo0621/Qwen2.5-3B-Korean-QLoRA"  # 기존 리포 (PEFT만)
MERGED_DIR = "/home/work/.setting/qwen/2.5_distil/outputs/merged"
GGUF_DIR = "/home/work/.setting/qwen/2.5_distil/outputs/gguf"

print("=" * 80)
print("HuggingFace Hub 업로드")
print("=" * 80)

# 1. HuggingFace 로그인
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    print("✅ HuggingFace 로그인 완료 (HF_TOKEN)")
else:
    print("⚠️  HF_TOKEN 환경 변수가 설정되지 않았습니다.")
    print("ℹ️  ~/.huggingface/token을 사용합니다.")
    try:
        login()
        print("✅ HuggingFace 로그인 완료 (~/.huggingface/token)")
    except Exception as e:
        print(f"❌ 로그인 실패: {e}")
        exit(1)

# 2. API 초기화
api = HfApi(token=hf_token)

# 3. 별도 리포지토리 생성 (Merged 모델용)
print(f"\n[ 1/3 ] 별도 리포지토리 생성")
print(f"  ℹ️  리포: {MERGED_REPO_ID}")

try:
    create_repo(
        repo_id=MERGED_REPO_ID,
        repo_type="model",
        private=False,
        exist_ok=True
    )
    print(f"  ✅ 리포지토리 준비 완료 (또는 기존 리포 사용)")
except Exception as e:
    print(f"  ⚠️  리포 생성 중 경고: {e}")

# 4. Merged 모델 업로드 (루트에 직접)
print(f"\n[ 2/3 ] Merged 모델 업로드")
print(f"  ℹ️  경로: {MERGED_DIR}")
print(f"  ℹ️  Hub: {MERGED_REPO_ID}/ (루트)")

if not Path(MERGED_DIR).exists():
    print(f"  ⚠️  Merged 모델을 찾을 수 없습니다, 스킵")
else:
    # 파일 목록 출력
    merged_files = list(Path(MERGED_DIR).glob("*"))
    print(f"  ℹ️  파일 수: {len(merged_files)}")
    
    try:
        print(f"  🚀 업로드 중... (루트에 직접)")
        api.upload_folder(
            folder_path=MERGED_DIR,
            repo_id=MERGED_REPO_ID,
            repo_type="model",
            commit_message="Add merged model (Transformers, vLLM, SGLang compatible)"
        )
        print(f"  ✅ Merged 모델 업로드 완료!")
    except Exception as e:
        print(f"  ❌ 업로드 실패: {e}")

# 5. GGUF 파일 업로드 (같은 리포, gguf/ 폴더)
print(f"\n[ 3/3 ] GGUF 파일 업로드")
print(f"  ℹ️  경로: {GGUF_DIR}")
print(f"  ℹ️  Hub: {MERGED_REPO_ID}/gguf/")

if not Path(GGUF_DIR).exists():
    print(f"  ⚠️  GGUF 파일을 찾을 수 없습니다, 스킵")
else:
    # GGUF 파일 목록
    gguf_files = list(Path(GGUF_DIR).glob("*.gguf"))
    print(f"  ℹ️  파일 수: {len(gguf_files)}")
    
    if not gguf_files:
        print(f"  ⚠️  GGUF 파일이 없습니다, 스킵")
    else:
        for gguf_file in gguf_files:
            size_mb = gguf_file.stat().st_size / (1024 * 1024)
            print(f"    - {gguf_file.name:40s} ({size_mb:>8.2f} MB)")
        
        try:
            print(f"  🚀 업로드 중...")
            api.upload_folder(
                folder_path=GGUF_DIR,
                repo_id=MERGED_REPO_ID,
                repo_type="model",
                path_in_repo="gguf",
                commit_message="Add GGUF files (Q4_K_M, Q5_K_M, Q8_0, F16 for Ollama, Llama.cpp)"
            )
            print(f"  ✅ GGUF 파일 업로드 완료!")
        except Exception as e:
            print(f"  ❌ 업로드 실패: {e}")

# 5. 완료
print(f"\n{'=' * 80}")
print(f"🎉 업로드 완료!")
print(f"{'=' * 80}")

print(f"\n📍 모델 URL:")
print(f"  🔗 Merged 모델: https://huggingface.co/{MERGED_REPO_ID}")
print(f"  🔗 PEFT 어댑터: https://huggingface.co/{PEFT_REPO_ID}")

print(f"\n📂 업로드된 구조:")
print(f"""
✅ {MERGED_REPO_ID}/
├── config.json                    # 모델 설정
├── model.safetensors             # Merged 모델 (~6GB)
├── tokenizer.json                # 토크나이저
└── gguf/                         # GGUF 파일들
    ├── qwen25-3b-korean-Q4_K_M.gguf  (~2GB)
    ├── qwen25-3b-korean-Q5_K_M.gguf  (~2.5GB)
    ├── qwen25-3b-korean-Q8_0.gguf    (~3.5GB)
    └── qwen25-3b-korean-F16.gguf     (~6GB)

✅ {PEFT_REPO_ID}/
├── adapter_model.safetensors     # LoRA 어댑터 (~479MB)
├── adapter_config.json           # LoRA 설정
└── final/                        # 저장본
""")

print(f"\n💡 다음 단계: 모델 카드 업데이트")
print(f"  - {MERGED_REPO_ID} README 업데이트:")
print(f"    huggingface-cli upload {MERGED_REPO_ID} \\")
print(f"      /home/work/.setting/qwen/2.5_distil/MODEL_CARD_MERGED.md \\")
print(f"      README.md")
print(f"")
print(f"  - 'Use this model' 버튼에 모든 프레임워크 표시됨")

