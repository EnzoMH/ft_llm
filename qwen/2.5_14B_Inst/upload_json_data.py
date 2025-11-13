#!/usr/bin/env python3
"""
JSON 데이터 파일들을 HuggingFace Hub에 업로드
"""

import os
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from typing import List, Optional

# 설정
HF_TOKEN = "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"
REPO_ID = "MyeongHo0621/Qwen2.5-14B-Korean-Data"  # 데이터 전용 repo
REPO_TYPE = "dataset"  # dataset 타입으로 생성

# 업로드할 JSON/JSONL 파일 경로들
JSON_FILES = [
    # expanded_data
    "/home/work/tes/ft_llm/datageneration/instruction/expanded_data/0_topics_20_practical.json",
    "/home/work/tes/ft_llm/datageneration/instruction/expanded_data/0_topics_20_technical.json",
    "/home/work/tes/ft_llm/datageneration/instruction/expanded_data/personas_10.json",
    "/home/work/tes/ft_llm/datageneration/instruction/expanded_data/personas_25.json",
    "/home/work/tes/ft_llm/datageneration/instruction/expanded_data/topics_30_practical.json",
    
    # output
    "/home/work/tes/ft_llm/datageneration/instruction/output/wms_qa_dataset_20251013_051415.json",
    "/home/work/tes/ft_llm/datageneration/instruction/output/wms_qa_dataset_20251013_052605.json",
    
    # test
    "/home/work/tes/ft_llm/datageneration/test/wms_qa_demo_result.json",
    "/home/work/tes/ft_llm/datageneration/test/wms_qa_optimized_result.json",
    
    # korean_large_data (JSONL)
    "/home/work/tes/korean_large_data/cleaned_jsonl/smol_koreantalk_full.jsonl",
]


def validate_json_file(file_path: str) -> bool:
    """JSON/JSONL 파일 유효성 검사"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.jsonl'):
                # JSONL 파일: 각 줄이 유효한 JSON인지 확인 (처음 10줄만)
                for i, line in enumerate(f):
                    if i >= 10:  # 처음 10줄만 검사
                        break
                    line = line.strip()
                    if line:
                        json.loads(line)
            else:
                # JSON 파일: 전체 파일 파싱
                json.load(f)
        return True
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON 파싱 실패: {e}")
        return False
    except Exception as e:
        print(f"  ❌ 파일 읽기 실패: {e}")
        return False


def upload_json_files():
    """JSON 파일들을 Hub에 업로드"""
    api = HfApi(token=HF_TOKEN)
    
    print("="*80)
    print("JSON/JSONL 데이터 파일 Hub 업로드")
    print("="*80)
    print(f"Repo: {REPO_ID} ({REPO_TYPE})")
    print(f"총 {len(JSON_FILES)}개 파일")
    print("="*80)
    
    # Repo 생성 (없으면)
    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            token=HF_TOKEN,
            exist_ok=True,
            private=True
        )
        print(f"[ INFO ] Repo 확인/생성 완료: {REPO_ID}")
    except Exception as e:
        print(f"[ WARNING ] Repo 생성 실패 (이미 존재할 수 있음): {e}")
    
    # 각 파일 업로드
    success_count = 0
    fail_count = 0
    
    for file_path in JSON_FILES:
        file_name = os.path.basename(file_path)
        print(f"\n[ 파일 ] {file_name}")
        print(f"  경로: {file_path}")
        
        # 파일 존재 확인
        if not os.path.exists(file_path):
            print(f"  ⚠️  파일이 존재하지 않습니다. 스킵합니다.")
            fail_count += 1
            continue
        
        # 파일 크기 확인
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"  크기: {file_size:.2f} MB")
        
        # JSON 유효성 검사
        if not validate_json_file(file_path):
            fail_count += 1
            continue
        
        # Hub에 업로드
        try:
            print(f"  업로드 중...")
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_name,  # 루트에 파일명으로 저장
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                commit_message=f"Upload {file_name}",
            )
            print(f"  ✅ 업로드 완료: https://huggingface.co/datasets/{REPO_ID}/blob/main/{file_name}")
            success_count += 1
        except Exception as e:
            print(f"  ❌ 업로드 실패: {e}")
            fail_count += 1
    
    # 결과 요약
    print("\n" + "="*80)
    print("업로드 완료")
    print("="*80)
    print(f"성공: {success_count}개")
    print(f"실패: {fail_count}개")
    print(f"Repo URL: https://huggingface.co/datasets/{REPO_ID}")
    print("="*80)


if __name__ == "__main__":
    upload_json_files()

