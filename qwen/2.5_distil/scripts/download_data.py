#!/usr/bin/env python3
"""
Private HuggingFace 데이터셋 다운로드 스크립트
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset

# .env 파일 로드
# /home/work/.setting/qwen/2.5_distil/scripts/download_data.py
# -> /home/work/.setting/.env
env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ .env 파일 로드: {env_path}")
else:
    # 절대 경로로도 시도
    env_path = Path("/home/work/.setting/.env")
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ .env 파일 로드 (절대 경로): {env_path}")

# HF_TOKEN 확인
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print("❌ HF_TOKEN이 .env 파일에 없습니다!")
    exit(1)

print(f"✅ HF_TOKEN 발견: {hf_token[:10]}...")

# 데이터셋 다운로드
dataset_name = "MyeongHo0621/Qwen2.5-14B-Korean-Data"
output_dir = Path("/home/work/.setting/data")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "smol_koreantalk_full.jsonl"

print(f"\n📥 데이터셋 다운로드 시작: {dataset_name}")
print(f"   파일: smol_koreantalk_full.jsonl")
print(f"   저장 경로: {output_path}")

try:
    # Private 데이터셋에서 특정 파일 다운로드 (토큰 사용)
    # 데이터셋이 아닌 파일인 경우 hf_hub_download 사용
    try:
        from huggingface_hub import hf_hub_download
        import json
        
        print("📥 HuggingFace Hub에서 파일 다운로드 중...")
        # 파일 직접 다운로드
        downloaded_file = hf_hub_download(
            repo_id=dataset_name,
            filename="smol_koreantalk_full.jsonl",
            token=hf_token,
            repo_type="dataset"
        )
        
        print(f"✅ 다운로드 완료: {downloaded_file}")
        print(f"📝 파일 복사 중...")
        
        # 다운로드된 파일을 목적지로 복사
        import shutil
        shutil.copy2(downloaded_file, output_path)
        
        # 파일에서 샘플 수 확인
        count = 0
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    count += 1
        
        print(f"\n✅ 저장 완료!")
        print(f"   파일: {output_path}")
        print(f"   샘플 수: {count:,}개")
        
    except Exception as e:
        # 파일 다운로드 실패 시 데이터셋으로 시도
        print(f"⚠️  파일 다운로드 실패, 데이터셋으로 시도: {e}")
        dataset = load_dataset(
            dataset_name,
            split="train",
            token=hf_token  # Private 데이터셋 접근을 위한 토큰
        )
    
        print(f"✅ 다운로드 완료: {len(dataset):,}개 샘플")
        print(f"📝 JSONL 파일로 저장 중...")
        
        # JSONL로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, item in enumerate(dataset):
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                if (i + 1) % 10000 == 0:
                    print(f"   진행: {i+1:,}/{len(dataset):,}")
        
        print(f"\n✅ 저장 완료!")
        print(f"   파일: {output_path}")
        print(f"   샘플 수: {len(dataset):,}개")
    
    # 파일 크기 확인
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   파일 크기: {size_mb:.2f} MB")
    
except Exception as e:
    print(f"\n❌ 에러 발생: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

