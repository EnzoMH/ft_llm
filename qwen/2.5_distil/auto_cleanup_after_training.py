#!/usr/bin/env python3
"""
학습 완료 후 자동 삭제 스크립트
- 학습 프로세스(PID 426723) 완료 대기
- 업로드 완료 확인
- 지정된 디렉토리 자동 삭제
"""

import os
import time
import shutil
import psutil
import logging
from pathlib import Path
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/work/vss/ft_llm/qwen/2.5_distil/cleanup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 설정
TRAINING_PID = 426723
TRAINING_LOG = "/home/work/vss/ft_llm/qwen/2.5_distil/train_QLoRA.log"
DELETE_DIRS = [
    "/home/work/flash-attention",
    "/home/work/miniconda3",
    "/home/work/vss",
    "/home/work/unsloth_compiled_cache",
]

def check_process_exists(pid: int) -> bool:
    """프로세스 존재 여부 확인"""
    try:
        return psutil.pid_exists(pid)
    except Exception as e:
        logger.warning(f"프로세스 확인 실패: {e}")
        return False

def check_training_completed(log_path: str) -> bool:
    """학습 완료 확인 (로그에서 '완료' 또는 'Upload' 키워드 확인)"""
    try:
        if not os.path.exists(log_path):
            return False
        
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # 완료 키워드 확인
            completion_keywords = [
                "완료!",
                "Upload complete",
                "Training completed",
                "Successfully uploaded",
            ]
            
            for keyword in completion_keywords:
                if keyword in content:
                    logger.info(f"✅ 완료 키워드 발견: '{keyword}'")
                    return True
        
        return False
    except Exception as e:
        logger.warning(f"로그 확인 실패: {e}")
        return False

def get_dir_size(path: str) -> float:
    """디렉토리 크기 계산 (GB)"""
    try:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    if os.path.exists(fp):
                        total_size += os.path.getsize(fp)
                except:
                    pass
        return total_size / (1024**3)
    except Exception as e:
        logger.warning(f"크기 계산 실패 ({path}): {e}")
        return 0.0

def delete_directory(path: str) -> bool:
    """디렉토리 삭제"""
    try:
        if not os.path.exists(path):
            logger.warning(f"⚠️  디렉토리 없음: {path}")
            return True
        
        size_gb = get_dir_size(path)
        logger.info(f"🗑️  삭제 시작: {path} ({size_gb:.2f} GB)")
        
        shutil.rmtree(path)
        logger.info(f"✅ 삭제 완료: {path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 삭제 실패 ({path}): {e}")
        return False

def main():
    logger.info("="*70)
    logger.info("🤖 학습 완료 후 자동 삭제 스크립트 시작")
    logger.info("="*70)
    logger.info(f"모니터링 PID: {TRAINING_PID}")
    logger.info(f"로그 파일: {TRAINING_LOG}")
    logger.info(f"삭제 대상: {len(DELETE_DIRS)}개 디렉토리")
    logger.info("="*70)
    
    # 1단계: 프로세스 종료 대기
    logger.info("\n[ 1단계 ] 학습 프로세스 완료 대기...")
    check_interval = 60  # 60초마다 확인
    
    while check_process_exists(TRAINING_PID):
        logger.info(f"⏳ 프로세스 {TRAINING_PID} 실행 중... ({datetime.now().strftime('%H:%M:%S')})")
        time.sleep(check_interval)
    
    logger.info(f"✅ 프로세스 {TRAINING_PID} 종료됨")
    
    # 2단계: 업로드 완료 대기 (추가 10분 대기)
    logger.info("\n[ 2단계 ] 업로드 완료 확인...")
    logger.info("⏳ 10분 대기 (업로드 완료 시간 확보)...")
    time.sleep(600)  # 10분
    
    # 로그에서 완료 확인
    if check_training_completed(TRAINING_LOG):
        logger.info("✅ 학습 및 업로드 완료 확인됨")
    else:
        logger.warning("⚠️  완료 키워드 미확인 (계속 진행)")
    
    # 3단계: 디렉토리 삭제
    logger.info("\n[ 3단계 ] 디렉토리 삭제 시작...")
    logger.info("="*70)
    
    total_size = 0
    success_count = 0
    
    for dir_path in DELETE_DIRS:
        if os.path.exists(dir_path):
            size = get_dir_size(dir_path)
            total_size += size
        
        if delete_directory(dir_path):
            success_count += 1
        
        time.sleep(2)  # 2초 대기
    
    # 완료 보고
    logger.info("\n" + "="*70)
    logger.info("📊 삭제 작업 완료")
    logger.info("="*70)
    logger.info(f"삭제 성공: {success_count}/{len(DELETE_DIRS)}개")
    logger.info(f"확보된 공간: {total_size:.2f} GB")
    logger.info(f"완료 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n❌ 사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"\n\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

