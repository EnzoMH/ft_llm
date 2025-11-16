#!/usr/bin/env python3
"""
lemon-mint/smol-koreantalk 전체 데이터셋 다운로드
- 460K 멀티턴 대화
- JSONL 포맷으로 저장
- ChatML 포맷 호환
"""

import os
import json
from pathlib import Path
from datasets import load_dataset
from datetime import datetime


def get_ft_llm_data_dir() -> Path:
    """ft_llm 디렉토리 내부의 data 디렉토리 경로 반환"""
    # 현재 스크립트 위치 기준으로 ft_llm 디렉토리 찾기
    script_dir = Path(__file__).resolve().parent
    
    # 상위 디렉토리로 올라가면서 ft_llm 찾기
    current = script_dir
    while current != current.parent:
        if current.name == 'ft_llm':
            # ft_llm/data 디렉토리 반환
            data_dir = current / 'data'
            data_dir.mkdir(exist_ok=True)
            return data_dir
        current = current.parent
    
    # ft_llm을 찾지 못한 경우 스크립트 위치 기준으로 상대 경로 사용
    return script_dir.parent.parent / 'data'


def download_full_dataset(output_path: str):
    """전체 데이터셋 다운로드"""
    print("\n" + "="*80)
    print(" lemon-mint/smol-koreantalk 전체 다운로드")
    print("="*80)
    print(f"출력 경로: {output_path}")
    
    # 1. 데이터셋 로드
    print("\n[ INFO ] 데이터셋 로딩 중...")
    print("  예상 크기: ~460K 샘플")
    print("  예상 시간: 5-10분")
    
    try:
        dataset = load_dataset(
            "lemon-mint/smol-koreantalk",
            split="train"
        )
        print(f"[ COMPLETE ] 로드 완료: {len(dataset):,}개")
    except Exception as e:
        print(f"[ ERROR ] 로드 실패: {e}")
        return False
    
    # 2. JSONL 포맷으로 변환 및 저장
    print(f"\n[ INFO ] JSONL 변환 및 저장 중...")
    
    turn_distribution = {}
    saved_count = 0
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, example in enumerate(dataset):
            try:
                # messages 필드 확인
                if 'messages' not in example:
                    continue
                
                messages = example['messages']
                
                # 턴 수 계산
                turn_count = sum(1 for msg in messages if msg.get('role') in ['user', 'assistant'])
                turn_distribution[turn_count] = turn_distribution.get(turn_count, 0) + 1
                
                # JSONL 저장
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
                saved_count += 1
                
                if (i + 1) % 10000 == 0:
                    print(f"  진행: {i+1:,}/{len(dataset):,}")
            
            except Exception as e:
                print(f"[ WARNING ] 샘플 {i} 실패: {e}")
                continue
    
    print(f"[ COMPLETE ] 저장 완료: {saved_count:,}개")
    
    # 3. 통계 출력
    print(f"\n{'='*80}")
    print("턴 수 분포")
    print('='*80)
    
    for turns in sorted(turn_distribution.keys()):
        count = turn_distribution[turns]
        percentage = (count / saved_count * 100) if saved_count > 0 else 0
        bar = '█' * int(percentage / 2)
        print(f"  {turns:2d}턴: {count:8,}개 ({percentage:5.1f}%) {bar}")
    
    # 멀티턴 통계
    turns_3plus = sum(count for turns, count in turn_distribution.items() if turns >= 3)
    turns_5plus = sum(count for turns, count in turn_distribution.items() if turns >= 5)
    percentage_3plus = (turns_3plus / saved_count * 100) if saved_count > 0 else 0
    percentage_5plus = (turns_5plus / saved_count * 100) if saved_count > 0 else 0
    
    print(f"\n{'='*80}")
    print("멀티턴 대화 통계")
    print('='*80)
    print(f"3턴 이상: {turns_3plus:,}개 ({percentage_3plus:.1f}%)")
    print(f"5턴 이상: {turns_5plus:,}개 ({percentage_5plus:.1f}%)")
    print(f"최대 턴 수: {max(turn_distribution.keys()) if turn_distribution else 0}")
    
    # 파일 크기
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n파일 크기: {file_size_mb:.1f}MB")
    print('='*80)
    
    return True


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print(" smol-koreantalk 전체 다운로드")
    print(" 460K 멀티턴 대화 데이터셋")
    print("="*80)
    
    # 출력 경로 (ft_llm 내부의 data 디렉토리 사용)
    data_dir = get_ft_llm_data_dir()
    output_path = str(data_dir / 'smol_koreantalk_full.jsonl')
    
    print(f"\n출력 파일: {output_path}")
    
    # 기존 파일 확인
    if os.path.exists(output_path):
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n[ WARNING ] 파일이 이미 존재합니다 ({file_size_mb:.1f}MB)")
        
        response = input("덮어쓰시겠습니까? (y/n): ")
        if response.lower() != 'y':
            print("[ CANCEL ] 작업 취소")
            return
    
    # 다운로드 시작
    start_time = datetime.now()
    
    success = download_full_dataset(output_path)
    
    if success:
        elapsed_time = datetime.now() - start_time
        print(f"\n{'='*80}")
        print(" [ OK ] 다운로드 완료!")
        print('='*80)
        print(f"소요 시간: {str(elapsed_time).split('.')[0]}")
        print(f"저장 위치: {output_path}")
        print('='*80)
        
        print("\n[ 다음 단계 ]")
        print(f"데이터가 저장되었습니다: {output_path}")
        print(f"데이터 디렉토리: {data_dir}")
    else:
        print("\n[ ERROR ] 다운로드 실패")


if __name__ == "__main__":
    main()

