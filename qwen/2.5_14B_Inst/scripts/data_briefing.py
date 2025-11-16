#!/usr/bin/env python3
"""
smol_koreantalk_full.jsonl 데이터 브리핑
- 총 샘플 수
- 턴 수 분포 분석
- 평균 턴 수
- 최소/최대 턴 수
"""

import os
import json
from collections import Counter, defaultdict
from typing import Dict, List


def analyze_turns(file_path: str, max_samples: int = None):
    """
    데이터 파일의 턴 수 분석
    
    Args:
        file_path: JSONL 파일 경로
        max_samples: 분석할 최대 샘플 수 (None이면 전체)
    """
    print("="*80)
    print(" smol_koreantalk_full.jsonl 데이터 브리핑")
    print("="*80)
    
    if not os.path.exists(file_path):
        print(f"[ ERROR ] 파일이 존재하지 않습니다: {file_path}")
        return
    
    # 파일 크기 확인
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"\n파일 정보:")
    print(f"  경로: {file_path}")
    print(f"  크기: {file_size_mb:.2f} MB")
    
    # 통계 변수
    total_samples = 0
    valid_samples = 0
    turn_counts = []
    role_distribution = defaultdict(int)
    samples_by_turns = defaultdict(list)
    
    # 샘플 수집 (각 턴 수별로 최대 2개씩)
    sample_collection = defaultdict(lambda: [])
    
    print(f"\n데이터 분석 중...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if max_samples and line_num > max_samples:
                    break
                
                total_samples += 1
                
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # messages 필드 확인
                    if 'messages' in data and isinstance(data['messages'], list):
                        messages = data['messages']
                        num_turns = len(messages)
                        
                        if num_turns > 0:
                            valid_samples += 1
                            turn_counts.append(num_turns)
                            
                            # 턴 수별 샘플 수집 (최대 2개)
                            if len(sample_collection[num_turns]) < 2:
                                sample_collection[num_turns].append({
                                    'turn_count': num_turns,
                                    'messages': messages[:4],  # 처음 4개만
                                    'custom_id': data.get('custom_id', 'N/A')
                                })
                            
                            # 역할 분포 확인
                            for msg in messages:
                                role = msg.get('role', 'unknown')
                                role_distribution[role] += 1
                            
                            # 턴 수별 그룹화
                            samples_by_turns[num_turns].append(data)
                
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"  경고: 라인 {line_num} 처리 실패: {e}")
                    continue
                
                # 진행 상황 출력 (10만개마다)
                if line_num % 100000 == 0:
                    print(f"  진행: {line_num:,}개 라인 처리됨...")
    
    except Exception as e:
        print(f"[ ERROR ] 파일 읽기 실패: {e}")
        return
    
    # 통계 계산
    if not turn_counts:
        print("\n[ ERROR ] 유효한 데이터를 찾을 수 없습니다.")
        return
    
    turn_counter = Counter(turn_counts)
    avg_turns = sum(turn_counts) / len(turn_counts)
    min_turns = min(turn_counts)
    max_turns = max(turn_counts)
    median_turns = sorted(turn_counts)[len(turn_counts) // 2]
    
    # 결과 출력
    print(f"\n{'='*80}")
    print(" 기본 통계")
    print(f"{'='*80}")
    print(f"총 라인 수: {total_samples:,}")
    print(f"유효한 샘플: {valid_samples:,} ({100*valid_samples/total_samples:.1f}%)")
    print(f"\n턴 수 통계:")
    print(f"  평균: {avg_turns:.2f}턴")
    print(f"  중앙값: {median_turns}턴")
    print(f"  최소: {min_turns}턴")
    print(f"  최대: {max_turns}턴")
    
    # 턴 수 분포
    print(f"\n{'='*80}")
    print(" 턴 수 분포")
    print(f"{'='*80}")
    print(f"{'턴 수':<10} {'샘플 수':>15} {'비율':>10} {'누적 비율':>12}")
    print("-"*80)
    
    cumulative = 0
    for turn_num in sorted(turn_counter.keys()):
        count = turn_counter[turn_num]
        cumulative += count
        percentage = 100 * count / valid_samples
        cum_percentage = 100 * cumulative / valid_samples
        print(f"{turn_num:<10} {count:>15,} {percentage:>9.2f}% {cum_percentage:>11.2f}%")
    
    print("-"*80)
    print(f"{'합계':<10} {valid_samples:>15,} {'100.00':>10}% {'100.00':>12}%")
    
    # 역할 분포
    print(f"\n{'='*80}")
    print(" 역할 분포 (전체 메시지 기준)")
    print(f"{'='*80}")
    total_messages = sum(role_distribution.values())
    for role, count in sorted(role_distribution.items(), key=lambda x: -x[1]):
        percentage = 100 * count / total_messages if total_messages > 0 else 0
        print(f"  {role:<15} {count:>10,} ({percentage:>5.2f}%)")
    
    # 턴 수별 샘플 예시
    print(f"\n{'='*80}")
    print(" 턴 수별 샘플 예시")
    print(f"{'='*80}")
    
    # 가장 많은 턴 수부터 최대 5개 턴 수 그룹만 표시
    top_turn_counts = sorted(turn_counter.keys(), key=lambda x: turn_counter[x], reverse=True)[:5]
    
    for turn_num in top_turn_counts:
        if turn_num in sample_collection and sample_collection[turn_num]:
            sample = sample_collection[turn_num][0]
            print(f"\n[{turn_num}턴 대화] (총 {turn_counter[turn_num]:,}개)")
            print(f"  Custom ID: {sample['custom_id']}")
            print(f"  메시지 구조:")
            for i, msg in enumerate(sample['messages'][:4], 1):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                content_preview = content[:80] + '...' if len(content) > 80 else content
                print(f"    {i}. [{role}] {content_preview}")
            if len(sample['messages']) > 4:
                print(f"    ... 외 {len(sample['messages']) - 4}개 메시지")
    
    # 턴 수 히스토그램 (간단한 텍스트)
    print(f"\n{'='*80}")
    print(" 턴 수 히스토그램 (간단)")
    print(f"{'='*80}")
    
    # 10개 구간으로 나누기
    bins = 10
    bin_size = (max_turns - min_turns) / bins if max_turns > min_turns else 1
    histogram = defaultdict(int)
    
    for turn_count in turn_counts:
        if bin_size > 0:
            bin_idx = min(int((turn_count - min_turns) / bin_size), bins - 1)
        else:
            bin_idx = 0
        histogram[bin_idx] += 1
    
    max_bar_length = 50
    max_count = max(histogram.values())
    
    for bin_idx in sorted(histogram.keys()):
        count = histogram[bin_idx]
        bar_length = int((count / max_count) * max_bar_length) if max_count > 0 else 0
        bar = '█' * bar_length
        
        bin_start = int(min_turns + bin_idx * bin_size)
        bin_end = int(min_turns + (bin_idx + 1) * bin_size) if bin_idx < bins - 1 else max_turns
        
        print(f"  {bin_start:>3}-{bin_end:<3}턴: {bar} {count:>8,}")
    
    print(f"\n{'='*80}")
    print(" 브리핑 완료")
    print(f"{'='*80}")


def main():
    """메인 함수"""
    file_path = "/home/work/vss/ft_llm/data/smol_koreantalk_full.jsonl"
    
    # 전체 분석 (max_samples=None이면 전체)
    analyze_turns(file_path, max_samples=None)


if __name__ == "__main__":
    main()

