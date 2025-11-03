#!/usr/bin/env python3
"""
CoT 증강 데이터 품질 검증
- <think> 태그 확인
- 중국어 포함 여부 확인
- 답변 길이 검증
- 샘플 출력
"""

import json
import random
import argparse
import re
from pathlib import Path
from typing import Optional


def has_chinese(text: str) -> bool:
    """중국어 포함 여부 확인"""
    return bool(re.search(r'[\u4e00-\u9fff]+', text))


def has_think_tags(text: str) -> bool:
    """<think> 태그 포함 여부 확인"""
    return '<think>' in text and '</think>' in text


def extract_think_content(text: str) -> Optional[str]:
    """<think> 태그 내용 추출"""
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_answer_content(text: str) -> str:
    """</think> 이후 최종 답변 추출"""
    parts = text.split('</think>')
    if len(parts) > 1:
        return parts[1].strip()
    return text


def validate_augmented_data(
    jsonl_path: str,
    num_samples: int = 20,
    show_full: bool = False,
    output_bad_samples: bool = False
):
    """생성된 데이터 품질 확인"""
    
    jsonl_path = Path(jsonl_path)
    
    if not jsonl_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {jsonl_path}")
        return
    
    print(f"\n🔍 품질 검증: {jsonl_path.name}")
    print(f"{'='*80}\n")
    
    # 데이터 로드
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    print(f"📊 총 샘플 수: {len(data):,}개")
    
    # 샘플링
    if num_samples > len(data):
        num_samples = len(data)
        print(f"   (전체 샘플 검증)")
    else:
        print(f"   ({num_samples}개 샘플 검증)")
    
    samples = random.sample(data, num_samples)
    
    # 품질 이슈 추적
    issues = {
        'no_think_tag': 0,
        'has_chinese': 0,
        'too_short': 0,
        'no_steps': 0,
        'good': 0
    }
    
    bad_samples = []
    
    # 각 샘플 검증
    for i, sample in enumerate(samples, 1):
        # assistant 메시지 찾기
        assistant_msg = None
        for msg in sample.get('messages', []):
            if msg['role'] == 'assistant':
                assistant_msg = msg
                break
        
        if not assistant_msg:
            print(f"[샘플 {i}] ❌ assistant 메시지 없음")
            issues['no_think_tag'] += 1
            bad_samples.append(sample)
            continue
        
        content = assistant_msg['content']
        sample_type = sample.get('type', 'unknown')
        
        print(f"\n{'─'*80}")
        print(f"[샘플 {i}] - 타입: {sample_type}")
        print(f"{'─'*80}")
        
        # 검증
        has_think = has_think_tags(content)
        has_chinese_content = has_chinese(content)
        is_short = len(content) < 100
        
        # <think> 태그 내부 검사
        has_steps = False
        if has_think:
            think_content = extract_think_content(content)
            if think_content:
                # "1단계:", "2단계:" 등의 패턴 확인
                has_steps = bool(re.search(r'\d+단계:', think_content))
        
        # 내용 출력
        if show_full or not has_think or has_chinese_content:
            print(content)
        else:
            # 요약 출력
            think_content = extract_think_content(content)
            answer_content = extract_answer_content(content)
            
            print("\n[추론 과정]")
            if think_content:
                think_lines = think_content.split('\n')[:5]  # 처음 5줄만
                for line in think_lines:
                    print(f"  {line}")
                if len(think_content.split('\n')) > 5:
                    print("  ...")
            
            print("\n[최종 답변]")
            answer_preview = answer_content[:200] + "..." if len(answer_content) > 200 else answer_content
            print(f"  {answer_preview}")
        
        # 품질 평가
        print(f"\n[품질 평가]")
        
        if not has_think:
            issues['no_think_tag'] += 1
            print("  ❌ <think> 태그 없음")
            bad_samples.append(sample)
        elif has_chinese_content:
            issues['has_chinese'] += 1
            print("  ⚠️  중국어 포함")
            bad_samples.append(sample)
        elif is_short:
            issues['too_short'] += 1
            print("  ⚠️  답변이 너무 짧음 (100자 미만)")
        elif not has_steps:
            issues['no_steps'] += 1
            print("  ⚠️  단계별 구분 없음 (1단계:, 2단계: 등)")
        else:
            issues['good'] += 1
            print("  ✅ 품질 양호")
    
    # 통계 출력
    print(f"\n{'='*80}")
    print(f"📊 품질 통계 ({num_samples}개 샘플)")
    print(f"{'='*80}")
    
    for issue, count in issues.items():
        percentage = (count / num_samples) * 100
        emoji = "✅" if issue == 'good' else "⚠️" if count > 0 else "✅"
        issue_name = {
            'no_think_tag': '<think> 태그 없음',
            'has_chinese': '중국어 포함',
            'too_short': '답변 너무 짧음',
            'no_steps': '단계별 구분 없음',
            'good': '품질 양호'
        }[issue]
        
        print(f"{emoji} {issue_name}: {count}/{num_samples} ({percentage:.1f}%)")
    
    # 전체 품질 점수
    quality_score = (issues['good'] / num_samples) * 100
    print(f"\n🎯 전체 품질 점수: {quality_score:.1f}%")
    
    if quality_score >= 80:
        print("   ✅ 우수 (80% 이상)")
    elif quality_score >= 60:
        print("   ⚠️  양호 (60-80%)")
    else:
        print("   ❌ 개선 필요 (60% 미만)")
    
    # 문제 샘플 저장
    if output_bad_samples and bad_samples:
        bad_samples_file = jsonl_path.parent / f"{jsonl_path.stem}_bad_samples.jsonl"
        with open(bad_samples_file, 'w', encoding='utf-8') as f:
            for sample in bad_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"\n📝 문제 샘플 저장: {bad_samples_file}")
        print(f"   ({len(bad_samples)}개)")
    
    return {
        'total_samples': len(data),
        'validated_samples': num_samples,
        'issues': issues,
        'quality_score': quality_score
    }


def validate_directory(
    dir_path: str,
    num_samples: int = 20,
    output_report: bool = True
):
    """디렉토리 내 모든 JSONL 파일 검증"""
    
    dir_path = Path(dir_path)
    
    if not dir_path.exists():
        print(f"❌ 디렉토리를 찾을 수 없습니다: {dir_path}")
        return
    
    jsonl_files = list(dir_path.glob('*.jsonl'))
    
    if not jsonl_files:
        print(f"❌ JSONL 파일이 없습니다: {dir_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"📁 디렉토리 검증: {dir_path}")
    print(f"{'='*80}")
    print(f"발견된 파일: {len(jsonl_files)}개\n")
    
    all_results = {}
    
    for jsonl_file in sorted(jsonl_files):
        result = validate_augmented_data(
            jsonl_path=str(jsonl_file),
            num_samples=num_samples,
            show_full=False,
            output_bad_samples=False
        )
        
        all_results[jsonl_file.name] = result
        print("\n")
    
    # 전체 리포트
    if output_report:
        print(f"\n{'='*80}")
        print(f"📋 전체 요약 리포트")
        print(f"{'='*80}\n")
        
        total_samples = 0
        total_good = 0
        total_validated = 0
        
        for filename, result in all_results.items():
            total_samples += result['total_samples']
            total_validated += result['validated_samples']
            total_good += result['issues']['good']
            
            print(f"📄 {filename}")
            print(f"   총 샘플: {result['total_samples']:,}개")
            print(f"   품질 점수: {result['quality_score']:.1f}%")
            print()
        
        overall_quality = (total_good / total_validated) * 100 if total_validated > 0 else 0
        
        print(f"🎯 전체 통계:")
        print(f"   총 파일: {len(all_results)}개")
        print(f"   총 샘플: {total_samples:,}개")
        print(f"   검증 샘플: {total_validated:,}개")
        print(f"   전체 품질 점수: {overall_quality:.1f}%")
        
        # 리포트 저장
        report_file = dir_path / 'quality_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'total_files': len(all_results),
                    'total_samples': total_samples,
                    'validated_samples': total_validated,
                    'overall_quality': overall_quality
                },
                'files': all_results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n📊 리포트 저장: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="CoT 데이터 품질 검증")
    parser.add_argument(
        'path',
        type=str,
        help='검증할 JSONL 파일 또는 디렉토리'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=20,
        help='검증할 샘플 수'
    )
    parser.add_argument(
        '--show-full',
        action='store_true',
        help='전체 내용 출력'
    )
    parser.add_argument(
        '--output-bad',
        action='store_true',
        help='문제 샘플 별도 저장'
    )
    parser.add_argument(
        '--directory',
        action='store_true',
        help='디렉토리 모드 (모든 JSONL 검증)'
    )
    
    args = parser.parse_args()
    
    if args.directory or Path(args.path).is_dir():
        validate_directory(
            dir_path=args.path,
            num_samples=args.num_samples,
            output_report=True
        )
    else:
        validate_augmented_data(
            jsonl_path=args.path,
            num_samples=args.num_samples,
            show_full=args.show_full,
            output_bad_samples=args.output_bad
        )


if __name__ == "__main__":
    main()

