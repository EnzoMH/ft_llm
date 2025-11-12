#!/usr/bin/env python3
"""
vclm-KoCoder-7B 벤치마크 평가
- HumanEval: 코드 생성 능력 (Python)
- GSM8K: 수학 추론 능력
- MMLU: 일반 지식
- KoBEST: 한국어 이해
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path

# 설정
MODEL_PATH = "/home/work/tesseract/vclm/vclm-korean-7b-coder-merged"
MODEL_NAME = "vclm-KoCoder-7B"
OUTPUT_DIR = "/home/work/tesseract/vclm/benchmark_results/kocoder"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# 벤치마크 설정
BENCHMARKS = {
    "humaneval": {
        "name": "HumanEval",
        "description": "Python 코드 생성 (pass@1, pass@10)",
        "command": "{lm_eval_cmd} --model hf --model_args pretrained={model_path},trust_remote_code=True --tasks humaneval --device cuda --batch_size 4 --output_path {output_dir}/humaneval_{timestamp}.json",
        "priority": "HIGH"  # 코드 능력 핵심 지표
    },
    "mbpp": {
        "name": "MBPP",
        "description": "Python 코드 생성 (기본 프로그래밍)",
        "command": "{lm_eval_cmd} --model hf --model_args pretrained={model_path},trust_remote_code=True --tasks mbpp --device cuda --batch_size 4 --output_path {output_dir}/mbpp_{timestamp}.json",
        "priority": "HIGH"
    },
    "gsm8k": {
        "name": "GSM8K",
        "description": "수학 문제 풀이 (초등 수준)",
        "command": "{lm_eval_cmd} --model hf --model_args pretrained={model_path},trust_remote_code=True --tasks gsm8k --device cuda --batch_size 8 --num_fewshot 5 --output_path {output_dir}/gsm8k_{timestamp}.json",
        "priority": "MEDIUM"  # 기존 능력 유지 확인
    },
    "mmlu": {
        "name": "MMLU",
        "description": "일반 지식 (57개 과목)",
        "command": "{lm_eval_cmd} --model hf --model_args pretrained={model_path},trust_remote_code=True --tasks mmlu --device cuda --batch_size 8 --num_fewshot 5 --output_path {output_dir}/mmlu_{timestamp}.json",
        "priority": "MEDIUM"
    },
    "kobest": {
        "name": "KoBEST",
        "description": "한국어 이해 (BoolQ, COPA, HellaSwag, SentiNeg, WiC)",
        "command": "{lm_eval_cmd} --model hf --model_args pretrained={model_path},trust_remote_code=True --tasks kobest --device cuda --batch_size 8 --output_path {output_dir}/kobest_{timestamp}.json",
        "priority": "MEDIUM"  # 한국어 능력 유지 확인
    },
    "arc_challenge": {
        "name": "ARC Challenge",
        "description": "과학 문제 (중고등 수준)",
        "command": "{lm_eval_cmd} --model hf --model_args pretrained={model_path},trust_remote_code=True --tasks arc_challenge --device cuda --batch_size 8 --num_fewshot 25 --output_path {output_dir}/arc_challenge_{timestamp}.json",
        "priority": "LOW"  # 참고용
    }
}

def print_header():
    print("=" * 80)
    print(f"🔬 {MODEL_NAME} 벤치마크 평가")
    print("=" * 80)
    print(f"📅 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 모델 경로: {MODEL_PATH}")
    print(f"💾 결과 저장: {OUTPUT_DIR}")
    print("=" * 80)
    print()

def check_dependencies():
    """lm-evaluation-harness 설치 확인"""
    print("[1/6] 의존성 확인...")
    
    # lm_eval 또는 lm-eval 명령어 확인
    for cmd in ["/home/work/.local/bin/lm_eval", "lm_eval", "lm-eval"]:
        try:
            result = subprocess.run(
                [cmd, "--help"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"✅ lm-evaluation-harness 설치됨: {cmd}")
                return cmd
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    
    # python -m lm_eval 시도
    try:
        result = subprocess.run(
            ["python", "-m", "lm_eval", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✅ lm-evaluation-harness 설치됨: python -m lm_eval")
            return "python -m lm_eval"
    except:
        pass
    
    print("❌ lm-evaluation-harness가 설치되지 않았습니다.")
    print("설치 명령어:")
    print("  pip install lm-eval")
    return None

def check_model():
    """모델 파일 존재 확인"""
    print("\n[2/6] 모델 확인...")
    model_path = Path(MODEL_PATH)
    
    required_files = [
        "config.json",
        "modeling_soka.py",
        "tokenizer.json"
    ]
    
    missing = []
    for file in required_files:
        if not (model_path / file).exists():
            missing.append(file)
    
    if missing:
        print(f"❌ 누락된 파일: {', '.join(missing)}")
        return False
    
    print(f"✅ 모델 파일 확인 완료: {MODEL_PATH}")
    return True

def create_output_dir():
    """결과 저장 디렉토리 생성"""
    print("\n[3/6] 출력 디렉토리 생성...")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"✅ {OUTPUT_DIR}")

def select_benchmarks():
    """실행할 벤치마크 선택"""
    print("\n[4/6] 벤치마크 선택...")
    print("\n사용 가능한 벤치마크:")
    for i, (key, config) in enumerate(BENCHMARKS.items(), 1):
        print(f"  {i}. [{config['priority']:>6}] {config['name']:<20} - {config['description']}")
    
    print("\n옵션:")
    print("  all    - 모든 벤치마크 실행 (약 2-3시간 소요)")
    print("  high   - HIGH 우선순위만 (코드 테스트, 약 30분)")
    print("  custom - 개별 선택")
    print()
    
    choice = input("선택 (all/high/custom/번호): ").strip().lower()
    
    if choice == "all":
        return list(BENCHMARKS.keys())
    elif choice == "high":
        return [k for k, v in BENCHMARKS.items() if v["priority"] == "HIGH"]
    elif choice == "custom":
        selected = []
        for key in BENCHMARKS.keys():
            ans = input(f"  {BENCHMARKS[key]['name']} 실행? (y/n): ").strip().lower()
            if ans == 'y':
                selected.append(key)
        return selected
    else:
        # 번호로 선택
        try:
            idx = int(choice) - 1
            keys = list(BENCHMARKS.keys())
            if 0 <= idx < len(keys):
                return [keys[idx]]
        except:
            pass
        print("❌ 잘못된 입력. HIGH 우선순위만 실행합니다.")
        return [k for k, v in BENCHMARKS.items() if v["priority"] == "HIGH"]

def run_benchmark(benchmark_key, lm_eval_cmd):
    """벤치마크 실행"""
    config = BENCHMARKS[benchmark_key]
    
    print("\n" + "=" * 80)
    print(f"🚀 {config['name']} 실행 중...")
    print(f"   {config['description']}")
    print("=" * 80)
    
    cmd = config["command"].format(
        lm_eval_cmd=lm_eval_cmd,
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        timestamp=TIMESTAMP
    )
    
    print(f"📝 명령어: {cmd}\n")
    
    start_time = datetime.now()
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=False,  # 실시간 출력
            text=True
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if result.returncode == 0:
            print(f"\n✅ {config['name']} 완료 (소요 시간: {duration:.1f}초)")
            return True, duration
        else:
            print(f"\n❌ {config['name']} 실패 (exit code: {result.returncode})")
            return False, duration
            
    except Exception as e:
        print(f"\n❌ {config['name']} 오류: {e}")
        return False, 0

def generate_report(results):
    """결과 리포트 생성"""
    print("\n" + "=" * 80)
    print("📊 벤치마크 결과 요약")
    print("=" * 80)
    
    report = {
        "model": MODEL_NAME,
        "model_path": MODEL_PATH,
        "timestamp": TIMESTAMP,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": results
    }
    
    total_time = sum(r["duration"] for r in results.values())
    success_count = sum(1 for r in results.values() if r["success"])
    total_count = len(results)
    
    print(f"\n총 실행: {total_count}개")
    print(f"성공: {success_count}개")
    print(f"실패: {total_count - success_count}개")
    print(f"총 소요 시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
    
    # 상세 결과
    print("\n상세 결과:")
    for key, result in results.items():
        status = "✅" if result["success"] else "❌"
        name = BENCHMARKS[key]["name"]
        duration = result["duration"]
        print(f"  {status} {name:<20} ({duration:.1f}초)")
    
    # JSON 저장
    report_path = Path(OUTPUT_DIR) / f"benchmark_summary_{TIMESTAMP}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 리포트 저장: {report_path}")
    
    # 결과 파일 위치
    print(f"\n📁 결과 파일:")
    print(f"   {OUTPUT_DIR}/")
    for key in results.keys():
        result_file = f"{key}_{TIMESTAMP}.json"
        print(f"   - {result_file}")
    
    print("\n" + "=" * 80)
    print("🎉 벤치마크 평가 완료!")
    print("=" * 80)

def main():
    print_header()
    
    # 1. 의존성 확인
    lm_eval_cmd = check_dependencies()
    if not lm_eval_cmd:
        print("\n❌ lm-evaluation-harness를 먼저 설치하세요:")
        print("   pip install lm-eval")
        sys.exit(1)
    
    # 2. 모델 확인
    if not check_model():
        print(f"\n❌ 모델을 찾을 수 없습니다: {MODEL_PATH}")
        sys.exit(1)
    
    # 3. 출력 디렉토리 생성
    create_output_dir()
    
    # 4. 벤치마크 선택
    selected = select_benchmarks()
    
    if not selected:
        print("❌ 선택된 벤치마크가 없습니다.")
        sys.exit(1)
    
    print(f"\n✅ 선택된 벤치마크: {', '.join([BENCHMARKS[k]['name'] for k in selected])}")
    print(f"⏱️  예상 소요 시간: ", end="")
    if len(selected) >= 5:
        print("2-3시간")
    elif len(selected) >= 3:
        print("1-2시간")
    else:
        print("30분-1시간")
    
    input("\n계속하려면 Enter를 누르세요...")
    
    # 5. 벤치마크 실행
    print("\n[5/6] 벤치마크 실행 중...")
    results = {}
    
    for i, benchmark_key in enumerate(selected, 1):
        print(f"\n진행률: {i}/{len(selected)}")
        success, duration = run_benchmark(benchmark_key, lm_eval_cmd)
        results[benchmark_key] = {
            "success": success,
            "duration": duration,
            "name": BENCHMARKS[benchmark_key]["name"]
        }
    
    # 6. 리포트 생성
    print("\n[6/6] 리포트 생성 중...")
    generate_report(results)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

