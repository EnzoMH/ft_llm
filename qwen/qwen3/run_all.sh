#!/bin/bash
# Qwen3-VL 한국어 파인튜닝 전체 파이프라인 실행

set -e  # 오류 발생 시 중단

echo "======================================================================"
echo "Qwen3-VL 한국어 VLA 파인튜닝 전체 파이프라인"
echo "======================================================================"
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# 기본 디렉토리
BASE_DIR=$(pwd)
DATA_DIR="../../korean_large_data/cleaned_jsonl"

echo "작업 디렉토리: $BASE_DIR"
echo "데이터 디렉토리: $DATA_DIR"
echo ""

# =============================================================================
# Phase 0: 환경 확인
# =============================================================================
echo "======================================================================"
echo "Phase 0: 환경 확인"
echo "======================================================================"

echo "Python 버전:"
python --version

echo ""
echo "GPU 상태:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

echo ""
echo "필수 패키지 확인:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import unsloth; print('Unsloth: OK')" 2>/dev/null || echo "Unsloth: 설치 필요"
python -c "import vllm; print('vLLM: OK')" 2>/dev/null || echo "vLLM: 설치 필요"

echo ""
read -p "계속 진행하시겠습니까? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "중단되었습니다."
    exit 1
fi

# =============================================================================
# Step 1: Phase 1 데이터 준비
# =============================================================================
echo ""
echo "======================================================================"
echo "Step 1: Phase 1 데이터 준비 (256K 샘플)"
echo "======================================================================"

if [ -d "phase1_korean" ] && [ -n "$(ls -A phase1_korean 2>/dev/null)" ]; then
    echo "⚠️  phase1_korean 디렉토리가 이미 존재합니다."
    read -p "데이터 재생성하시겠습니까? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf phase1_korean
        python prepare_phase1_data.py \
            --input-dir "$DATA_DIR" \
            --output-dir phase1_korean \
            --target-samples 256000
    fi
else
    python prepare_phase1_data.py \
        --input-dir "$DATA_DIR" \
        --output-dir phase1_korean \
        --target-samples 256000
fi

echo "✅ Phase 1 데이터 준비 완료!"

# =============================================================================
# Step 2: CoT 데이터 증강 (5% 샘플링)
# =============================================================================
echo ""
echo "======================================================================"
echo "Step 2: CoT 데이터 증강 (EXAONE 4.0 1.2B)"
echo "======================================================================"

if [ -d "phase2_thinking_exaone" ] && [ -n "$(ls -A phase2_thinking_exaone 2>/dev/null)" ]; then
    echo "⚠️  phase2_thinking_exaone 디렉토리가 이미 존재합니다."
    read -p "데이터 재생성하시겠습니까? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf phase2_thinking_exaone
        python augment_cot_exaone.py \
            --input-dir "$DATA_DIR" \
            --output-dir phase2_thinking_exaone \
            --sample-ratio 0.05 \
            --singleturn-ratio 0.7
    fi
else
    python augment_cot_exaone.py \
        --input-dir "$DATA_DIR" \
        --output-dir phase2_thinking_exaone \
        --sample-ratio 0.05 \
        --singleturn-ratio 0.7
fi

echo "✅ CoT 데이터 증강 완료!"

# =============================================================================
# Step 3: 품질 검증
# =============================================================================
echo ""
echo "======================================================================"
echo "Step 3: CoT 데이터 품질 검증"
echo "======================================================================"

python validate_cot_quality.py \
    phase2_thinking_exaone \
    --directory \
    --num-samples 20

echo ""
read -p "품질이 만족스럽습니까? 계속 진행하시겠습니까? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "중단되었습니다. 데이터를 확인하고 다시 실행하세요."
    exit 1
fi

# =============================================================================
# Step 4: Phase 1 학습 (한국어 강화)
# =============================================================================
echo ""
echo "======================================================================"
echo "Step 4: Phase 1 학습 (한국어 강화)"
echo "======================================================================"

if [ -d "qwen3-vl-8b-korean-instruct" ]; then
    echo "⚠️  qwen3-vl-8b-korean-instruct 디렉토리가 이미 존재합니다."
    read -p "재학습하시겠습니까? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Phase 1 학습 건너뜀."
    else
        python train_phase1_korean_instruct.py \
            --model-name "Qwen/Qwen3-VL-8B-Instruct" \
            --data-dir phase1_korean \
            --output-dir qwen3-vl-8b-korean-instruct \
            --epochs 2 \
            --batch-size 4 \
            --gradient-accumulation 4
    fi
else
    python train_phase1_korean_instruct.py \
        --model-name "Qwen/Qwen3-VL-8B-Instruct" \
        --data-dir phase1_korean \
        --output-dir qwen3-vl-8b-korean-instruct \
        --epochs 2 \
        --batch-size 4 \
        --gradient-accumulation 4
fi

echo "✅ Phase 1 학습 완료!"

# =============================================================================
# Step 5: Phase 2 학습 (Thinking)
# =============================================================================
echo ""
echo "======================================================================"
echo "Step 5: Phase 2 학습 (Thinking 능력 추가)"
echo "======================================================================"

# Phase 1 결과 사용 여부 선택
echo "Phase 2의 기본 모델을 선택하세요:"
echo "1) Phase 1 결과 사용 (qwen3-vl-8b-korean-instruct)"
echo "2) Qwen3-VL-8B-Thinking 사용"
read -p "선택 (1 또는 2): " -n 1 -r
echo ""

if [[ $REPLY == "1" ]]; then
    BASE_MODEL="qwen3-vl-8b-korean-instruct/merged"
    if [ ! -d "$BASE_MODEL" ]; then
        echo "❌ Phase 1 병합 모델이 없습니다. Phase 1을 먼저 완료하세요."
        exit 1
    fi
elif [[ $REPLY == "2" ]]; then
    BASE_MODEL="Qwen/Qwen3-VL-8B-Thinking"
else
    echo "잘못된 선택입니다."
    exit 1
fi

echo "선택된 기본 모델: $BASE_MODEL"

python train_phase2_thinking.py \
    --model-name "$BASE_MODEL" \
    --data-dir phase2_thinking_exaone \
    --output-dir qwen3-vl-8b-korean-thinking \
    --epochs 3 \
    --batch-size 2 \
    --gradient-accumulation 8

echo "✅ Phase 2 학습 완료!"

# =============================================================================
# Step 6: 최종 테스트
# =============================================================================
echo ""
echo "======================================================================"
echo "Step 6: 최종 모델 테스트"
echo "======================================================================"

echo "테스트 중..."
python model_load.py qwen3-vl-8b-korean-thinking/merged

echo ""
echo "======================================================================"
echo "✅ 전체 파이프라인 완료!"
echo "======================================================================"
echo "종료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "결과 모델:"
echo "  - Phase 1: qwen3-vl-8b-korean-instruct/"
echo "  - Phase 2: qwen3-vl-8b-korean-thinking/"
echo ""
echo "대화형 테스트:"
echo "  python model_load.py qwen3-vl-8b-korean-thinking/merged --interactive --thinking"

