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

# 멀티프로세싱 최적 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# CUDA 메모리 최적화
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    print("\n" + "="*80)
    print(" Qwen2.5-3B-Instruct 한국어 멀티턴 대화 파인튜닝")
    print(" H100 80GB 최적화 | Flash Attention 3")
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
        model_path = finetuner.train(dataset)
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

