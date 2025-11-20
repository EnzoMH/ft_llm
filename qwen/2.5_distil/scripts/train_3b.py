#!/usr/bin/env python3
"""
Qwen2.5-3B-Instruct 한국어 멀티턴 대화 파인튜닝
- H100 80GB 최적화
- Flash Attention 3
- smol_koreantalk_full.jsonl 데이터셋
- LoRA + 8bit 양자화
"""

import os
import sys
import logging
import time
import torch
import argparse

# 멀티프로세싱 최적 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# CUDA 메모리 최적화
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# CPU 코어 수 제한 (RAM 압박 방지 및 멀티프로세싱 안정성 향상)
# 시스템: 24 코어, 63GB RAM → 8 코어로 제한하여 안정성 확보
import multiprocessing
_original_mp_cpu_count = multiprocessing.cpu_count
_original_os_cpu_count = os.cpu_count
LIMIT_CPU_CORES = 8  # RAM 51GB 사용 가능, 안정적인 토크나이징을 위해 8개로 제한
multiprocessing.cpu_count = lambda: LIMIT_CPU_CORES
os.cpu_count = lambda: LIMIT_CPU_CORES

# psutil도 오버라이드
try:
    import psutil
    _original_psutil_cpu_count = psutil.cpu_count
    psutil.cpu_count = lambda logical=True: LIMIT_CPU_CORES
except ImportError:
    pass

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 모듈 경로 추가
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)  # scripts의 상위 디렉토리
_src_dir = os.path.join(_project_root, 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# 모듈 임포트
from qwen_finetuning_3b import Qwen3BFineTuningConfig, Qwen3BFineTuner


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Qwen2.5-3B 파인튜닝")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="재개할 checkpoint 경로 (Hub 모델 ID 또는 로컬 경로). 예: MyeongHo0621/Qwen2.5-3B-Korean 또는 outputs/checkpoints/checkpoint-2500"
    )
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(" Qwen2.5-3B-Instruct 한국어 멀티턴 대화 파인튜닝")
    print(" H100 80GB 최적화 | Flash Attention 2.8.3")
    print("="*80)
    
    # GPU 확인
    if not torch.cuda.is_available():
        print("\n[ ERROR ] CUDA를 사용할 수 없습니다!")
        sys.exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"\nGPU: {gpu_name}")
    print(f"메모리: {gpu_memory_gb:.1f}GB")
    
    # 설정
    config = Qwen3BFineTuningConfig()
    
    # Checkpoint 재개 정보 출력
    if args.resume_from_checkpoint:
        print(f"\n[ INFO ] Checkpoint에서 재개: {args.resume_from_checkpoint}")
    
    print(f"\n{'='*80}")
    print(" 설정 요약")
    print(f"{'='*80}")
    print(f"모델: {config.base_model}")
    print(f"데이터: {config.korean_data_dir}")
    print(f"  파일: {', '.join(config.data_files)}")
    print(f"  최대 샘플: {config.max_samples if config.max_samples else '전체'}")
    print(f"출력: {config.output_dir}")
    print(f"LoRA: r={config.lora_r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
    print(f"Epoch: {config.num_train_epochs}")
    print(f"배치: {config.per_device_train_batch_size} × {config.gradient_accumulation_steps} = {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"학습률: {config.learning_rate}")
    print(f"Max Seq: {config.max_seq_length}")
    print(f"체크포인트: {config.save_steps} step마다")
    if config.pre_tokenize_dataset:
        print(f"데이터셋 처리: 미리 토크나이징 (멀티프로세싱 오류 방지)")
    else:
        print(f"데이터셋 처리: {config.dataset_num_proc} 프로세스 (CPU 코어 제한: 8개)")
    print(f"Hub 업로드: {config.hub_model_id}")
    print(f"{'='*80}\n")
    
    # 파인튜너
    finetuner = Qwen3BFineTuner(config)
    
    try:
        # 1. 모델 로드
        finetuner.load_model()
        
        # 2. 데이터 로드
        dataset = finetuner.load_data()
        num_samples = len(dataset)
        
        # 3. 훈련
        train_start = time.perf_counter()
        model_path = finetuner.train(dataset, resume_from_checkpoint=args.resume_from_checkpoint)
        train_end = time.perf_counter()
        
        total_time = train_end - train_start
        
        # 대략적인 총 토큰 수 추정
        total_tokens = (
            num_samples
            * config.max_seq_length
            * config.num_train_epochs
        )
        
        tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
        
        print(f"\n{'='*80}")
        print(" 완료!")
        print(f"{'='*80}")
        print(f"모델: {model_path}")
        print(f"총 샘플 수: {num_samples:,}")
        print(f"총 토큰 수(대략): {total_tokens:,}")
        print(f"총 학습 시간: {total_time/3600:.2f} 시간")
        print(f"평균 처리 속도: {tokens_per_sec:,.0f} tokens/sec")
        print(f"{'='*80}\n")
        
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

