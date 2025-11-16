#!/usr/bin/env python3
"""
유틸리티 함수들
"""

import os
import sys
import logging

# 모니터링 모듈 경로 추가
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
_util_dir = os.path.join(_parent_dir, 'util')
if _util_dir not in sys.path:
    sys.path.insert(0, _util_dir)

from cpu_mntrg import CPUMonitor
from gpu_mnrtg import GPUMonitor


# 글로벌 모니터
cpu_monitor = CPUMonitor()
gpu_monitor = GPUMonitor()


def log_system_resources(logger: logging.Logger, stage: str):
    """시스템 리소스 상태 로깅
    
    Args:
        logger: 로거 인스턴스
        stage: 현재 단계 이름
    """
    logger.info("\n" + "="*80)
    logger.info(f"[{stage}] 시스템 리소스 모니터링")
    logger.info("="*80)
    
    # CPU 상세
    cpu_monitor.log_snapshot(logger, stage)
    
    # GPU 상세
    if gpu_monitor.available:
        gpu_monitor.log_all_gpus(logger, stage)
        
        # 메모리 압박 확인
        for i in range(gpu_monitor.device_count):
            if not gpu_monitor.check_memory_available(i, required_gb=5.0):
                logger.warning(f"[ WARNING ]  GPU{i} 메모리 부족! (5GB 미만)")
    
    # RAM 압박 확인
    if cpu_monitor.check_memory_pressure(threshold=85.0):
        logger.warning(f"[ WARNING ]  RAM 메모리 압박! (85% 이상)")
    
    logger.info("="*80 + "\n")

