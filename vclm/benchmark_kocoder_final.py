#!/usr/bin/env python3
"""
vclm-KoCoder-7B 종합 벤치마크 평가 (수정 버전)
- KMMLU: 한국어 지식
- KoBEST: 한국어 이해  
- GSM8K: 수학 추론
- MMLU: 일반 지식
- ARC: 과학 추론
"""

import subprocess
import sys
import os
from datetime import datetime

MODEL_PATH = "/home/work/tesseract/vclm/vclm-korean-7b-coder-merged"
OUTPUT_DIR = "/home/work/tesseract/vclm/benchmark_results/kocoder"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# 코드 실행 권한 설정 (HumanEval/MBPP용)
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

# 벤치마크 설정 (현실적인 것만)
ALL_BENCHMARKS = [
    # 한국어
    ("kmmlu", "KMMLU - 한국어 MMLU (45개 과목)", "HIGH", 40),
    ("kobest", "KoBEST - 한국어 이해", "HIGH", 20),
    
    # 수학 (GSM8K만, MATH는 너무 오래 걸림)
    ("gsm8k", "GSM8K - 수학 추론", "MEDIUM", 30),
    
    # 일반 지식
    ("mmlu", "MMLU - 일반 지식 (57개 과목)", "MEDIUM", 60),
    
    # 과학
    ("arc_challenge", "ARC Challenge - 과학 추론", "LOW", 20),
    ("arc_easy", "ARC Easy - 기본 과학", "LOW", 15),
]

def run_benchmark(task, description, est_min):
    """단일 벤치마크 실행"""
    print(f"\n{'='*80}")
    print(f"🚀 {description} (예상: {est_min}분)")
    print(f"{'='*80}\n")
    
    # 배치 크기 조정 (task별 최적화)
    if task in ["kmmlu", "mmlu"]:
        batch_size = 16
    elif task == "gsm8k":
        batch_size = 8
    else:
        batch_size = 8
    
    cmd = f"lm_eval --model hf --model_args pretrained={MODEL_PATH},trust_remote_code=True --tasks {task} --device cuda --batch_size {batch_size} --output_path {OUTPUT_DIR}/{task}_{TIMESTAMP}.json"
    
    print(f"📝 명령어: {cmd}\n")
    
    start = datetime.now()
    result = subprocess.run(cmd, shell=True)
    duration = (datetime.now() - start).total_seconds()
    
    if result.returncode == 0:
        print(f"\n✅ {description} 완료 ({duration/60:.1f}분)")
        return True, duration
    else:
        print(f"\n❌ {description} 실패")
        return False, duration

def main():
    # 명령줄 인수 파싱
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    print("="*80)
    print(f"🔬 vclm-KoCoder-7B 종합 벤치마크 평가")
    print("="*80)
    print(f"📅 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 모델: {MODEL_PATH}")
    print(f"💾 결과: {OUTPUT_DIR}")
    print(f"🎯 모드: {mode}")
    print(f"🔒 코드 실행 권한: {'설정됨' if os.environ.get('HF_ALLOW_CODE_EVAL') == '1' else '미설정'}")
    print("="*80)
    
    # 출력 디렉토리 생성
    subprocess.run(f"mkdir -p {OUTPUT_DIR}", shell=True, check=True)
    
    # 벤치마크 선택
    if mode == "korean":
        selected = [b for b in ALL_BENCHMARKS if b[0] in ["kmmlu", "kobest"]]
        print("\n📋 실행: 한국어 능력만 (1시간 예상)")
    elif mode == "math":
        selected = [b for b in ALL_BENCHMARKS if b[0] == "gsm8k"]
        print("\n📋 실행: 수학 능력만 (30분 예상)")
    elif mode == "high":
        selected = [b for b in ALL_BENCHMARKS if b[2] == "HIGH"]
        print("\n📋 실행: HIGH 우선순위 (1시간 예상)")
    else:  # all
        selected = ALL_BENCHMARKS
        print("\n📋 실행: 전체 벤치마크 (3시간 예상)")
    
    total_est = sum(b[3] for b in selected)
    print(f"\n선택된 벤치마크: {len(selected)}개 (예상 {total_est}분)")
    for task, desc, priority, est in selected:
        print(f"   [{priority:>6}] {desc} ({est}분)")
    
    print("\n" + "="*80)
    print("⚠️  참고: HumanEval/MBPP는 제외됨 (실행 시간 과다)")
    print("⚠️  참고: Hendrycks MATH는 제외됨 (8시간+ 소요)")
    print("="*80 + "\n")
    
    # 실행
    results = {}
    total_duration = 0
    
    for i, (task, desc, priority, est) in enumerate(selected, 1):
        print(f"\n\n{'#'*80}")
        print(f"진행률: {i}/{len(selected)}")
        print(f"{'#'*80}")
        success, duration = run_benchmark(task, desc, est)
        results[task] = {"success": success, "duration": duration, "desc": desc}
        total_duration += duration
    
    # 결과 요약
    print("\n\n" + "="*80)
    print("📊 최종 결과")
    print("="*80)
    success_count = sum(1 for v in results.values() if v["success"])
    print(f"성공: {success_count}/{len(results)}")
    print(f"총 소요 시간: {total_duration/60:.1f}분\n")
    
    for task, data in results.items():
        status = "✅" if data["success"] else "❌"
        print(f"  {status} {data['desc']} ({data['duration']/60:.1f}분)")
    
    print(f"\n💾 결과 저장: {OUTPUT_DIR}/")
    for task in results.keys():
        print(f"   - {task}_{TIMESTAMP}.json")
    
    print("\n" + "="*80)
    print(f"🎉 벤치마크 완료! ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print("="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자 중단")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
