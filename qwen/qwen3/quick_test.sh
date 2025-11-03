#!/bin/bash
# 빠른 테스트 (소규모 샘플링)

set -e

echo "======================================================================"
echo "Qwen3-VL 빠른 테스트 (1% 샘플링)"
echo "======================================================================"
echo ""

# 환경 변수
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

BASE_DIR=$(pwd)
DATA_DIR="../../korean_large_data/cleaned_jsonl"

# =============================================================================
# Step 1: Phase 1 데이터 (10K 샘플)
# =============================================================================
echo "Step 1: Phase 1 데이터 준비 (10K 샘플)"
echo "----------------------------------------------------------------------"

python prepare_phase1_data.py \
    --input-dir "$DATA_DIR" \
    --output-dir phase1_korean_test \
    --target-samples 10000

echo "✅ 완료!"
echo ""

# =============================================================================
# Step 2: CoT 데이터 (1% 샘플링)
# =============================================================================
echo "Step 2: CoT 데이터 증강 (1% 샘플링)"
echo "----------------------------------------------------------------------"

python augment_cot_exaone.py \
    --input-dir "$DATA_DIR" \
    --output-dir phase2_thinking_test \
    --sample-ratio 0.01 \
    --singleturn-ratio 0.7 \
    --datasets orca_math_ko_data.jsonl kullm_v2_full_data.jsonl

echo "✅ 완료!"
echo ""

# =============================================================================
# Step 3: 품질 검증
# =============================================================================
echo "Step 3: 품질 검증"
echo "----------------------------------------------------------------------"

python validate_cot_quality.py \
    phase2_thinking_test \
    --directory \
    --num-samples 5

echo "✅ 완료!"
echo ""

# =============================================================================
# Step 4: Phase 1 학습 (1 epoch)
# =============================================================================
echo "Step 4: Phase 1 학습 (1 epoch, 빠른 테스트)"
echo "----------------------------------------------------------------------"

python train_phase1_korean_instruct.py \
    --model-name Qwen/Qwen3-VL-8B-Instruct \
    --data-dir phase1_korean_test \
    --output-dir qwen3-vl-8b-korean-instruct-test \
    --epochs 1 \
    --batch-size 4 \
    --gradient-accumulation 2 \
    --learning-rate 2e-5

echo "✅ 완료!"
echo ""

# =============================================================================
# Step 5: 모델 테스트
# =============================================================================
echo "Step 5: 모델 테스트"
echo "----------------------------------------------------------------------"

python model_load.py qwen3-vl-8b-korean-instruct-test/merged

echo ""
echo "======================================================================"
echo "✅ 빠른 테스트 완료!"
echo "======================================================================"
echo ""
echo "Phase 2 테스트를 계속하려면:"
echo "  python train_phase2_thinking.py \\"
echo "      --model-name qwen3-vl-8b-korean-instruct-test/merged \\"
echo "      --data-dir phase2_thinking_test \\"
echo "      --output-dir qwen3-vl-8b-korean-thinking-test \\"
echo "      --epochs 1"
echo ""
echo "대화형 테스트:"
echo "  python model_load.py qwen3-vl-8b-korean-instruct-test/merged --interactive"

