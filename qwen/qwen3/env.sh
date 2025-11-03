#!/bin/bash
# Qwen3-VL H100 환경 변수 설정 스크립트
# CUDA 12.8 + PyTorch 2.8.0 + Flash-Attention 3 최적화
# 사용법: source env.sh

echo "=========================================================================="
echo "Qwen3-VL H100 환경 변수 설정"
echo "CUDA 12.8 + Flash-Attention 3 최적화"
echo "=========================================================================="

# ============================================================================
# GPU 설정
# ============================================================================
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# H100 최적화 설정
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# ============================================================================
# PyTorch 최적화
# ============================================================================
# 멀티스레딩 설정 (H100 PCIe - 16 cores)
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# cuDNN 최적화
export CUDNN_BENCHMARK=1
export TORCH_CUDNN_V8_API_ENABLED=1

# TF32 활성화 (H100 Tensor Core 최적화)
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# ============================================================================
# Transformers 설정
# ============================================================================
export TRANSFORMERS_CACHE=/home/work/.cache/huggingface/transformers
export HF_HOME=/home/work/.cache/huggingface
export HF_DATASETS_CACHE=/home/work/.cache/huggingface/datasets

# Offline 모드 (필요시 주석 해제)
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

# ============================================================================
# Tokenizers 병렬화
# ============================================================================
export TOKENIZERS_PARALLELISM=true

# ============================================================================
# vLLM 최적화 (CoT 증강용)
# ============================================================================
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_LOGGING_LEVEL=INFO

# Flash-Attention 3 활성화 (H100 최적화)
export VLLM_ATTENTION_BACKEND=FLASHINFER

# GPU 메모리 관리
export VLLM_GPU_MEMORY_UTILIZATION=0.90

# ============================================================================
# Unsloth 최적화
# ============================================================================
export UNSLOTH_DISABLE_WARNINGS=1

# ============================================================================
# 학습 최적화
# ============================================================================
# Gradient Checkpointing
export TORCH_USE_CUDA_DSA=1

# Mixed Precision (BF16 우선)
export ACCELERATE_MIXED_PRECISION=bf16

# ============================================================================
# 로깅 설정
# ============================================================================
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# Weights & Biases (선택적 - 필요시 주석 해제)
# export WANDB_PROJECT=qwen3-vl-korean-vla
# export WANDB_LOG_MODEL=false
# export WANDB_DISABLED=false

# TensorBoard
export TENSORBOARD_PORT=6006

# ============================================================================
# 디버깅 (개발 시에만 활성화 - 주석 해제)
# ============================================================================
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_DEBUG=INFO
# export CUDA_LAUNCH_BLOCKING=1

# ============================================================================
# 프로젝트 경로
# ============================================================================
export PROJECT_ROOT=/home/work/tesseract/qwen/qwen3
export DATA_DIR=/home/work/tesseract/korean_large_data/cleaned_jsonl

# ============================================================================
# 모델 설정
# ============================================================================
# Phase 1
export PHASE1_MODEL=Qwen/Qwen3-VL-8B-Instruct
export PHASE1_OUTPUT=qwen3-vl-8b-korean-instruct

# Phase 2
export PHASE2_MODEL=Qwen/Qwen3-VL-8B-Thinking
export PHASE2_OUTPUT=qwen3-vl-8b-korean-thinking

# CoT 증강 모델
export COT_MODEL=LGAI-EXAONE/EXAONE-4.0-1.2B-Instruct

echo ""
echo "환경 변수 설정 완료!"
echo ""
echo "주요 설정:"
echo "  - CUDA: 12.8"
echo "  - GPU: $CUDA_VISIBLE_DEVICES"
echo "  - OMP Threads: $OMP_NUM_THREADS"
echo "  - vLLM Backend: $VLLM_ATTENTION_BACKEND (Flash-Attention 3)"
echo "  - Mixed Precision: $ACCELERATE_MIXED_PRECISION"
echo "  - Project Root: $PROJECT_ROOT"
echo ""
echo "환경 확인: python check_env.py"
echo "=========================================================================="

