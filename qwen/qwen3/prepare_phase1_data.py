#!/usr/bin/env python3
"""
Phase 1 한국어 강화 데이터 준비
- 중국어 필터링
- Messages 형식 통일
- 한국어 시스템 프롬프트 추가
- 256K 샘플 생성
"""

import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import re


def has_chinese(text: str) -> bool:
    """중국어 포함 여부 확인"""
    return bool(re.search(r'[\u4e00-\u9fff]+', text))


def parse_chatml(text: str) -> list[dict]:
    """ChatML 형식 파싱"""
    messages = []
    parts = text.split('<|im_start|>')
    
    for part in parts[1:]:
        if '<|im_end|>' in part:
            split_part = part.split('\n', 1)
            if len(split_part) < 2:
                continue
            role = split_part[0].strip()
            content = split_part[1].split('<|im_end|>')[0].strip()
            
            if role in ['system', 'user', 'assistant']:
                messages.append({"role": role, "content": content})
    
    return messages


def extract_messages(data_item: dict) -> list[dict] | None:
    """데이터에서 messages 추출"""
    if 'messages' in data_item:
        return data_item['messages']
    elif 'text' in data_item:
        return parse_chatml(data_item['text'])
    else:
        return None


def prepare_phase1_data(
    input_dir: str = 'korean_large_data/cleaned_jsonl',
    output_dir: str = 'phase1_korean',
    target_samples: int = 256000,
    exclude_datasets: list[str] = ['identity_training_data.jsonl'],
    seed: int = 42
):
    """Phase 1용 한국어 강화 데이터 준비"""
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_data = []
    stats = {}
    
    print("📖 데이터 로딩 및 중국어 필터링 시작...")
    print(f"   입력 디렉토리: {input_dir}")
    print(f"   제외 파일: {exclude_datasets}")
    print()
    
    # 모든 JSONL 파일 로드
    jsonl_files = sorted(input_dir.glob('*.jsonl'))
    
    for jsonl_file in jsonl_files:
        if jsonl_file.name in exclude_datasets:
            print(f"⏭️  건너뜀: {jsonl_file.name}")
            continue
        
        print(f"📖 처리 중: {jsonl_file.name}")
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            data = []
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        # 중국어 필터링
        filtered = []
        chinese_count = 0
        empty_count = 0
        
        for item in tqdm(data, desc="  중국어 필터링"):
            messages = extract_messages(item)
            
            if not messages:
                empty_count += 1
                continue
            
            # 모든 메시지 내용 검사
            has_chinese_content = False
            for msg in messages:
                if has_chinese(msg.get('content', '')):
                    has_chinese_content = True
                    break
            
            if not has_chinese_content:
                filtered.append({
                    'messages': messages,
                    'source': jsonl_file.stem
                })
            else:
                chinese_count += 1
        
        stats[jsonl_file.name] = {
            'original': len(data),
            'filtered': len(filtered),
            'removed_chinese': chinese_count,
            'removed_empty': empty_count
        }
        
        all_data.extend(filtered)
        print(f"   결과: {len(data):,} → {len(filtered):,} (중국어: {chinese_count:,}, 빈 데이터: {empty_count:,})")
    
    print(f"\n📊 총 수집: {len(all_data):,}개")
    
    # 타겟 샘플 수에 맞게 샘플링
    random.seed(seed)
    if len(all_data) > target_samples:
        print(f"🎲 샘플링: {len(all_data):,} → {target_samples:,}")
        all_data = random.sample(all_data, target_samples)
    else:
        print(f"✅ 전체 데이터 사용: {len(all_data):,}개")
    
    # Messages 형식 통일 + 한국어 시스템 프롬프트
    korean_system = {
        "role": "system",
        "content": "당신은 한국어 전용 AI 어시스턴트입니다. 모든 응답은 반드시 한국어로만 작성하세요."
    }
    
    processed = []
    source_distribution = {}
    
    print("\n🔄 형식 변환 및 시스템 프롬프트 추가 중...")
    
    for item in tqdm(all_data, desc="형식 변환"):
        messages = item['messages']
        source = item.get('source', 'unknown')
        
        # 소스별 분포 집계
        source_distribution[source] = source_distribution.get(source, 0) + 1
        
        # system 메시지 제거 후 한국어 시스템 메시지 추가
        user_assistant_msgs = [m for m in messages if m['role'] in ['user', 'assistant']]
        
        if not user_assistant_msgs:
            continue
        
        new_messages = [korean_system] + user_assistant_msgs
        
        processed.append({
            "messages": new_messages,
            "source": source
        })
    
    # 저장
    output_file = output_dir / f'phase1_korean_{len(processed)}samples.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n✅ 데이터 준비 완료: {output_file}")
    print(f"   최종 샘플: {len(processed):,}개")
    
    # 소스별 분포 출력
    print(f"\n📊 소스별 분포:")
    for source, count in sorted(source_distribution.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(processed)) * 100
        print(f"   {source}: {count:,}개 ({percentage:.1f}%)")
    
    # 통계 저장
    stats_file = output_dir / 'phase1_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            'file_stats': stats,
            'source_distribution': source_distribution,
            'total_samples': len(processed),
            'target_samples': target_samples,
            'seed': seed
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n📈 통계 저장: {stats_file}")
    
    # 샘플 미리보기
    print(f"\n🔍 샘플 미리보기 (처음 2개):")
    for i, sample in enumerate(processed[:2], 1):
        print(f"\n[샘플 {i}] - 소스: {sample['source']}")
        for msg in sample['messages']:
            content_preview = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
            print(f"  {msg['role']}: {content_preview}")
    
    return processed


def main():
    parser = argparse.ArgumentParser(description="Phase 1 데이터 준비")
    parser.add_argument(
        '--input-dir',
        type=str,
        default='../../korean_large_data/cleaned_jsonl',
        help='입력 JSONL 디렉토리'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='phase1_korean',
        help='출력 디렉토리'
    )
    parser.add_argument(
        '--target-samples',
        type=int,
        default=256000,
        help='목표 샘플 수'
    )
    parser.add_argument(
        '--exclude',
        type=str,
        nargs='+',
        default=['identity_training_data.jsonl'],
        help='제외할 파일 목록'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='랜덤 시드'
    )
    
    args = parser.parse_args()
    
    prepare_phase1_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_samples=args.target_samples,
        exclude_datasets=args.exclude,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

