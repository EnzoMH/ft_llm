#!/usr/bin/env python3
"""
Qwen 학습이 끝나면 자동으로 Solar 학습 시작
"""
import os
import time
import subprocess
import psutil
from datetime import datetime

def wait_for_qwen_completion():
    """Qwen 학습 프로세스가 끝날 때까지 대기"""
    print("=" * 80)
    print(" 자동 학습 스케줄러")
    print("=" * 80)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Qwen 프로세스 찾기
    qwen_process = None
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and 'train_3b.py' in ' '.join(cmdline):
                qwen_process = proc
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if qwen_process is None:
        print("⚠️  Qwen 학습 프로세스를 찾을 수 없습니다!")
        print("Solar 학습을 바로 시작합니다.\n")
        return
    
    print(f"✓ Qwen 학습 프로세스 발견: PID {qwen_process.pid}")
    print(f"  명령어: {' '.join(qwen_process.cmdline())}")
    print()
    print("Qwen 학습이 끝날 때까지 대기 중...")
    print("(10분마다 상태 체크)")
    print()
    
    # 프로세스가 끝날 때까지 대기
    check_count = 0
    while True:
        try:
            if not qwen_process.is_running():
                print(f"\n✅ Qwen 학습 완료! ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
                break
            
            # 10분마다 상태 출력
            if check_count % 60 == 0:  # 10분 = 600초 / 10초 간격 = 60회
                elapsed = check_count * 10 // 60
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] 대기 중... (경과: {elapsed}분)")
            
            time.sleep(10)  # 10초마다 체크
            check_count += 1
            
        except psutil.NoSuchProcess:
            print(f"\n✅ Qwen 학습 완료! ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
            break
        except KeyboardInterrupt:
            print("\n\n⚠️  사용자가 중단했습니다.")
            exit(0)
    
    # 잠시 대기 (GPU 메모리 정리 시간)
    print("\n⏳ GPU 메모리 정리 대기 중... (30초)")
    time.sleep(30)


def start_solar_training():
    """Solar 학습 시작"""
    print("\n" + "=" * 80)
    print(" SOLAR-10.7B 학습 시작")
    print("=" * 80)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Solar 학습 경로
    solar_script = "/home/work/.setting/solar/solar_v2.py"
    log_file = "/home/work/.setting/solar/solar_train.log"
    
    # 로그 디렉토리 생성
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    print(f"스크립트: {solar_script}")
    print(f"로그 파일: {log_file}")
    print()
    
    # nohup으로 백그라운드 실행
    cmd = f"cd /home/work/.setting/solar && nohup python -u {solar_script} > {log_file} 2>&1 &"
    
    print(f"실행 명령어: {cmd}")
    print()
    
    # 실행
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Solar 학습 시작 성공!")
        print()
        print("=" * 80)
        print(" 모니터링 명령어")
        print("=" * 80)
        print(f"  tail -f {log_file}")
        print(f"  ps aux | grep solar_v2.py")
        print(f"  nvidia-smi")
        print("=" * 80)
    else:
        print(f"❌ Solar 학습 시작 실패!")
        print(f"에러: {result.stderr}")
        exit(1)


def main():
    try:
        # 1. Qwen 학습 완료 대기
        wait_for_qwen_completion()
        
        # 2. Solar 학습 시작
        start_solar_training()
        
        print("\n✅ 모든 작업 완료!")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()

