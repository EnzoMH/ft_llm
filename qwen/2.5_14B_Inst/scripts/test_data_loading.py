#!/usr/bin/env python3
"""
데이터 로딩 테스트
- 한국어 멀티턴 데이터셋 확인
- 데이터 포맷 검증
- 샘플 출력
"""

import os
import sys
import json
from typing import List

sys.path.insert(0, os.path.dirname(__file__))
from datasets import Dataset


def test_jsonl_file(file_path: str, max_samples: int = 5):
    """JSONL 파일 테스트"""
    print(f"\n{'='*80}")
    print(f"파일: {os.path.basename(file_path)}")
    print('='*80)
    
    if not os.path.exists(file_path):
        print(f"[ ERROR ] 파일이 존재하지 않습니다: {file_path}")
        return
    
    total_count = 0
    valid_count = 0
    multiturn_count = 0
    samples = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                total_count += 1
                
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # messages 필드가 있는 경우
                    if 'messages' in data and isinstance(data['messages'], list):
                        valid_count += 1
                        
                        # 멀티턴 확인 (2개 이상)
                        if len(data['messages']) >= 2:
                            multiturn_count += 1
                            
                            # 샘플 수집
                            if len(samples) < max_samples:
                                samples.append(data)
                    
                    # text 필드만 있는 경우 (이미 포맷팅됨)
                    elif 'text' in data and isinstance(data['text'], str):
                        valid_count += 1
                        
                        # text에 assistant가 2번 이상 나오면 멀티턴으로 간주
                        if data['text'].count('<|im_start|>assistant') >= 2:
                            multiturn_count += 1
                        
                        # 샘플 수집
                        if len(samples) < max_samples:
                            samples.append(data)
                
                except json.JSONDecodeError:
                    continue
    
    except Exception as e:
        print(f"[ ERROR ] 파일 읽기 실패: {e}")
        return
    
    print(f"총 라인: {total_count:,}")
    print(f"유효한 데이터: {valid_count:,} ({100*valid_count/total_count:.1f}%)")
    if valid_count > 0:
        print(f"멀티턴 대화: {multiturn_count:,} ({100*multiturn_count/valid_count:.1f}%)")
    else:
        print(f"멀티턴 대화: {multiturn_count:,} (N/A)")
    
    # 샘플 출력
    if samples:
        print(f"\n{'='*80}")
        print("샘플 데이터:")
        print('='*80)
        
        for i, sample in enumerate(samples[:3], 1):
            print(f"\n[ 샘플 {i} ]")
            
            # messages 필드가 있는 경우
            if 'messages' in sample:
                print(f"메시지 수: {len(sample.get('messages', []))}")
                print(f"소스: {sample.get('source', 'unknown')}")
                
                messages = sample.get('messages', [])[:3]  # 처음 3개만
                for msg in messages:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    preview = content[:100] + '...' if len(content) > 100 else content
                    print(f"  [{role}] {preview}")
            
            # text 필드만 있는 경우
            elif 'text' in sample:
                text = sample['text']
                print(f"포맷: ChatML (text 필드)")
                print(f"길이: {len(text)} chars")
                
                # 첫 300자만 출력
                preview = text[:300] + '...' if len(text) > 300 else text
                print(f"내용:\n{preview}")
    
    print('='*80)


def test_directory(directory: str):
    """디렉토리 전체 테스트"""
    print(f"\n{'='*80}")
    print(f"디렉토리: {directory}")
    print('='*80)
    
    if not os.path.exists(directory):
        print(f"[ ERROR ] 디렉토리가 존재하지 않습니다: {directory}")
        return
    
    import glob
    jsonl_files = glob.glob(os.path.join(directory, '*.jsonl'))
    
    print(f"발견된 JSONL 파일: {len(jsonl_files)}개\n")
    
    summary = []
    
    for file_path in sorted(jsonl_files):
        filename = os.path.basename(file_path)
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        # 라인 수 확인
        line_count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in f:
                line_count += 1
        
        summary.append({
            'filename': filename,
            'size_mb': file_size_mb,
            'lines': line_count
        })
    
    # 요약 테이블
    print(f"\n{'='*80}")
    print("파일 요약")
    print('='*80)
    print(f"{'파일명':<40} {'크기':>10} {'라인 수':>15}")
    print('-'*80)
    
    total_size = 0
    total_lines = 0
    
    for item in summary:
        print(f"{item['filename']:<40} {item['size_mb']:>8.1f}MB {item['lines']:>15,}")
        total_size += item['size_mb']
        total_lines += item['lines']
    
    print('-'*80)
    print(f"{'합계':<40} {total_size:>8.1f}MB {total_lines:>15,}")
    print('='*80)


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print(" 한국어 멀티턴 데이터셋 테스트")
    print("="*80)
    
    # 데이터 디렉토리
    data_dir = "/home/work/tesseract/korean_large_data/cleaned_jsonl"
    
    # 1. 디렉토리 전체 요약
    test_directory(data_dir)
    
    # 2. 주요 파일 상세 테스트
    test_files = [
        "kowiki_qa_data.jsonl",
        "kullm_v2_full_data.jsonl",
        "smol_koreantalk_data.jsonl",
    ]
    
    print(f"\n\n{'='*80}")
    print(" 주요 파일 상세 분석")
    print('='*80)
    
    for filename in test_files:
        file_path = os.path.join(data_dir, filename)
        test_jsonl_file(file_path, max_samples=3)
    
    print(f"\n{'='*80}")
    print(" 테스트 완료")
    print('='*80)


if __name__ == "__main__":
    main()

