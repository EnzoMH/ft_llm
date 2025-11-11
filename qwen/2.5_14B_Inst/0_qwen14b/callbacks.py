#!/usr/bin/env python3
"""
학습 모니터링 콜백
"""

import logging
import torch
from datetime import datetime
from transformers import TrainerCallback


logger = logging.getLogger(__name__)


class TrainingMonitorCallback(TrainerCallback):
    """학습 모니터링 콜백 (개선됨)"""
    
    def __init__(self):
        self.start_time = None
        self.step_times = []
        self.last_cleanup_step = 0
    
    def on_train_begin(self, args, state, control, **kwargs):
        """학습 시작"""
        self.start_time = datetime.now()
        logger.info("="*80)
        logger.info("[ START ] 학습 시작")
        logger.info(f"  시작 시간: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*80)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """로그 출력 시 (개선된 통계)"""
        if logs and state.global_step % 50 == 0:
            elapsed = datetime.now() - self.start_time
            
            # 평균 step 시간 계산
            if len(self.step_times) > 10:
                avg_step_time = sum(self.step_times[-10:]) / len(self.step_times[-10:])
                remaining_steps = state.max_steps - state.global_step
                eta_seconds = remaining_steps * avg_step_time
                eta_hours = eta_seconds / 3600
                
                logger.info(f"Step {state.global_step}/{state.max_steps}: "
                           f"loss={logs.get('loss', 0):.4f}, "
                           f"lr={logs.get('learning_rate', 0):.2e}, "
                           f"elapsed={str(elapsed).split('.')[0]}, "
                           f"ETA={eta_hours:.1f}h")
            else:
                logger.info(f"Step {state.global_step}: loss={logs.get('loss', 0):.4f}, "
                           f"lr={logs.get('learning_rate', 0):.2e}, "
                           f"elapsed={str(elapsed).split('.')[0]}")
        
        # 매 100 step마다 메모리 정리
        if state.global_step - self.last_cleanup_step >= 100:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.last_cleanup_step = state.global_step
    
    def on_step_end(self, args, state, control, **kwargs):
        """Step 종료 시 시간 기록"""
        if hasattr(self, '_step_start_time'):
            step_time = (datetime.now() - self._step_start_time).total_seconds()
            self.step_times.append(step_time)
            if len(self.step_times) > 100:  # 최근 100개만 유지
                self.step_times.pop(0)
        self._step_start_time = datetime.now()
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """평가 시"""
        if metrics:
            logger.info("="*80)
            logger.info(f"[ EVAL ] Step {state.global_step}/{state.max_steps}")
            logger.info(f"  Loss: {metrics.get('eval_loss', 0):.4f}")
            logger.info(f"  Runtime: {metrics.get('eval_runtime', 0):.1f}s")
            logger.info("="*80)
            
            # 평가 후 GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def on_save(self, args, state, control, **kwargs):
        """체크포인트 저장 시"""
        logger.info(f"[ SAVE ] Step {state.global_step} 체크포인트 저장 완료")
        
        # 저장 후 GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

