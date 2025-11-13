#!/usr/bin/env python3
"""
Qwen2.5-14B-Instruct 한국어 멀티턴 대화 파인튜닝
- H100 80GB 최적화
- Flash Attention 2
- 멀티턴 대화 데이터셋
- LoRA + 8bit 양자화
"""

import os
import sys
import logging
import torch

# 멀티프로세싱 최적 설정 (20 CPU cores → 12 cores)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# CUDA 메모리 최적화 (40GB VRAM)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# CPU 코어 수 제한 (RAM 압박 방지)
import multiprocessing
_original_mp_cpu_count = multiprocessing.cpu_count
_original_os_cpu_count = os.cpu_count
multiprocessing.cpu_count = lambda: 12
os.cpu_count = lambda: 12

# psutil도 오버라이드
try:
    import psutil
    _original_psutil_cpu_count = psutil.cpu_count
    psutil.cpu_count = lambda logical=True: 12
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
_module_dir = os.path.join(_current_dir, '0_qwen14b')
if _module_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# 모듈 임포트
import importlib
qwen14b_module = importlib.import_module('0_qwen14b')
Qwen14BFineTuningConfig = qwen14b_module.Qwen14BFineTuningConfig
Qwen14BFineTuner = qwen14b_module.Qwen14BFineTuner


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print(" Qwen2.5-14B-Instruct 한국어 멀티턴 대화 파인튜닝")
    print(" H100 80GB 최적화 | Flash Attention 2")
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
    config = Qwen14BFineTuningConfig()
    
    # 단일 데이터셋 파일 사용 (smol_koreantalk_full.jsonl)
    config.data_files = ["smol_koreantalk_full.jsonl"]
    
    print(f"\n{'='*80}")
    print(" 설정 요약")
    print(f"{'='*80}")
    print(f"모델: {config.base_model}")
    print(f"데이터: {config.korean_data_dir}")
    print(f"  파일: {', '.join(config.data_files)}")
    print(f"  최대 샘플: {config.max_samples:,}개")
    print(f"출력: {config.output_dir}")
    print(f"LoRA: r={config.lora_r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
    print(f"Epoch: {config.num_train_epochs}")
    print(f"배치: {config.per_device_train_batch_size} × {config.gradient_accumulation_steps} = {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"학습률: {config.learning_rate}")
    print(f"Max Seq: {config.max_seq_length}")
    print(f"체크포인트: {config.save_steps} step마다")
    print(f"{'='*80}\n")
    
    # 파인튜너
    finetuner = Qwen14BFineTuner(config)
    
    try:
        # 1. 모델 로드
        finetuner.load_model()
        
        # 2. 데이터 로드
        dataset = finetuner.load_data()
        
        # 3. 훈련
        model_path = finetuner.train(dataset)
        
        print(f"\n{'='*80}")
        print(" 완료!")
        print(f"{'='*80}")
        print(f"모델: {model_path}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
