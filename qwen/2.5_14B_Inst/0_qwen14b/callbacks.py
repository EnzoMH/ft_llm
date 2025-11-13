#!/usr/bin/env python3
"""
학습 모니터링 콜백
"""

import os
import json
import logging
import torch
import threading
from datetime import datetime
from transformers import TrainerCallback
from huggingface_hub import HfApi, upload_folder


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


class HubUploadCallback(TrainerCallback):
    """HuggingFace Hub 자동 업로드 콜백 (비동기 업로드)"""
    
    def __init__(self, hub_model_id: str, hub_token: str = None, upload_every_n_steps: int = 100):
        """
        Args:
            hub_model_id: Hub 모델 ID (예: "username/model-name")
            hub_token: HuggingFace 토큰 (None이면 자동 감지)
            upload_every_n_steps: N step마다 업로드 (기본: 100, checkpoint 저장 주기와 맞춤)
        """
        self.hub_model_id = hub_model_id
        self.hub_token = hub_token
        self.upload_every_n_steps = upload_every_n_steps
        self.api = HfApi(token=hub_token)
        self.upload_thread = None  # 백그라운드 업로드 스레드
        self.upload_lock = threading.Lock()  # 동시 업로드 방지
        self.uploaded_checkpoints_file = None  # Hub 업로드 완료 추적 파일 경로
        
        # Hub 로그인 확인
        try:
            if hub_token:
                from huggingface_hub import login
                login(token=hub_token, add_to_git_credential=True)
            logger.info(f"[ HUB ] Hub 업로드 활성화: {hub_model_id}")
        except Exception as e:
            logger.warning(f"[ HUB ] 로그인 실패: {e}")
    
    def _mark_checkpoint_uploaded(self, output_dir: str, step: int):
        """Hub 업로드 완료된 checkpoint를 마커 파일에 기록"""
        marker_file = os.path.join(output_dir, ".hub_uploaded_checkpoints.json")
        
        # 기존 마커 파일 읽기
        uploaded_steps = set()
        if os.path.exists(marker_file):
            try:
                with open(marker_file, 'r') as f:
                    data = json.load(f)
                    uploaded_steps = set(data.get('uploaded_steps', []))
            except Exception as e:
                logger.warning(f"[ HUB ] 마커 파일 읽기 실패: {e}")
        
        # 현재 step 추가
        uploaded_steps.add(step)
        
        # 마커 파일 저장
        try:
            with open(marker_file, 'w') as f:
                json.dump({
                    'uploaded_steps': sorted(list(uploaded_steps)),
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
            logger.debug(f"[ HUB ] Step {step} Hub 업로드 완료 마커 저장")
        except Exception as e:
            logger.warning(f"[ HUB ] 마커 파일 저장 실패: {e}")
    
    def _upload_checkpoint_async(self, output_dir: str, checkpoint_name: str, step: int):
        """백그라운드에서 체크포인트 업로드 (학습 블로킹 방지)"""
        checkpoint_dir = os.path.join(output_dir, checkpoint_name)
        
        if not os.path.exists(checkpoint_dir):
            return
        
        with self.upload_lock:
            try:
                logger.info(f"[ HUB ] Step {step} 체크포인트 업로드 시작 (백그라운드)...")
                logger.info(f"[ HUB ] 체크포인트 경로: {checkpoint_dir}")
                
                # checkpoint-XXX 디렉토리 구조를 유지하기 위해 output_dir를 기준으로 업로드
                # allow_patterns로 checkpoint-XXX/**만 업로드
                self.api.upload_folder(
                    folder_path=output_dir,  # output_dir를 기준으로
                    repo_id=self.hub_model_id,
                    repo_type="model",
                    commit_message=f"Checkpoint at step {step}",
                    allow_patterns=[f"{checkpoint_name}/**"],  # checkpoint-XXX/** 패턴만 업로드
                )
                
                # Hub 업로드 완료 마커 저장
                self._mark_checkpoint_uploaded(output_dir, step)
                
                logger.info(f"[ HUB ] ✅ Step {step} 체크포인트 업로드 완료 (Hub에 저장됨)")
                logger.info(f"[ HUB ] URL: https://huggingface.co/{self.hub_model_id}/tree/main/{checkpoint_name}")
            except Exception as e:
                logger.error(f"[ HUB ] ❌ Step {step} 업로드 실패: {e}")
                import traceback
                logger.debug(traceback.format_exc())
    
    def on_save(self, args, state, control, **kwargs):
        """체크포인트 저장 시 Hub에 비동기 업로드 (학습 블로킹 방지)"""
        if state.global_step % self.upload_every_n_steps == 0:
            checkpoint_name = f"checkpoint-{state.global_step}"
            
            # 이전 업로드 스레드가 완료될 때까지 대기 (최대 1초)
            if self.upload_thread and self.upload_thread.is_alive():
                logger.info(f"[ HUB ] 이전 업로드 완료 대기 중...")
                self.upload_thread.join(timeout=1.0)
            
            # 백그라운드 스레드에서 업로드 실행 (학습 블로킹 방지)
            self.upload_thread = threading.Thread(
                target=self._upload_checkpoint_async,
                args=(args.output_dir, checkpoint_name, state.global_step),
                daemon=True
            )
            self.upload_thread.start()
            logger.info(f"[ HUB ] Step {state.global_step} 체크포인트 업로드 백그라운드 시작 (학습 계속 진행)")
    
    def on_train_end(self, args, state, control, **kwargs):
        """훈련 완료 시 최종 모델 업로드"""
        final_dir = os.path.join(args.output_dir, "final")
        
        if os.path.exists(final_dir):
            try:
                logger.info(f"[ HUB ] 최종 모델 업로드 시작...")
                
                self.api.upload_folder(
                    folder_path=final_dir,
                    repo_id=self.hub_model_id,
                    repo_type="model",
                    commit_message="Final trained model",
                )
                
                logger.info(f"[ HUB ] ✅ 최종 모델 업로드 완료: https://huggingface.co/{self.hub_model_id}")
            except Exception as e:
                logger.error(f"[ HUB ] ❌ 최종 모델 업로드 실패: {e}")

