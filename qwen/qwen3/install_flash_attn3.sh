#!/bin/bash
# Flash-Attention 3 기반 H100 최적화 환경 설치
# CUDA 12.8 + PyTorch 2.6+ 호환

set -e

echo "=========================================================================="
echo "Flash-Attention 3 기반 H100 환경 설치"
echo "=========================================================================="
echo ""

# 현재 환경 확인
echo "Step 0: 현재 환경 확인"
echo "----------------------------------------------------------------------"
python --version
nvcc --version | grep "release"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
echo ""

# 환경 변수 설정
echo "Step 1: 환경 변수 로드"
echo "----------------------------------------------------------------------"
source env.sh
echo ""

# 기존 Flash-Attention 제거
echo "Step 2: 기존 Flash-Attention 제거"
echo "----------------------------------------------------------------------"
pip uninstall flash-attn -y 2>/dev/null || echo "Flash-Attention 미설치"
echo ""

# Flash-Attention 3 설치 (Beta)
echo "Step 3: Flash-Attention 3 설치 (H100 최적화)"
echo "----------------------------------------------------------------------"
echo "주의: Flash-Attention 3는 Beta 버전입니다."
echo "      H100에서 1.5~2배 성능 향상이 예상됩니다."
echo ""

# Flash-Attention 3 설치 옵션
read -p "Flash-Attention 3를 설치하시겠습니까? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Flash-Attention 3 설치 중..."
    
    # 방법 1: PyPI에서 설치 (사용 가능한 경우)
    pip install flash-attn-3 --no-build-isolation 2>/dev/null || \
    # 방법 2: GitHub에서 빌드
    pip install ninja packaging wheel && \
    pip install git+https://github.com/Dao-AILab/flash-attention-3.git --no-build-isolation
    
    echo "✅ Flash-Attention 3 설치 완료"
else
    echo "Flash-Attention 2 최신 버전으로 설치..."
    pip install flash-attn>=2.7.0 --no-build-isolation
    echo "✅ Flash-Attention 2 설치 완료"
fi
echo ""

# FlashInfer 설치 (vLLM용)
echo "Step 4: FlashInfer 설치 (vLLM 최적화)"
echo "----------------------------------------------------------------------"
pip install flashinfer --extra-index-url https://flashinfer.ai/whl/cu128/torch2.6/
echo "✅ FlashInfer 설치 완료"
echo ""

# PyTorch 업그레이드 (선택적)
echo "Step 5: PyTorch 업그레이드 (선택적)"
echo "----------------------------------------------------------------------"
echo "현재 PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "최신 PyTorch 2.8.0으로 업그레이드하려면 아래 명령 실행:"
echo "  pip install torch>=2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
echo ""
read -p "지금 업그레이드하시겠습니까? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "PyTorch 2.8.0 설치 중..."
    pip install torch>=2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    echo "✅ PyTorch 업그레이드 완료"
else
    echo "현재 PyTorch 유지 (2.6.0은 Flash-Attention 3와 호환됨)"
fi
echo ""

# Transformers 업그레이드
echo "Step 6: Transformers 생태계 업그레이드"
echo "----------------------------------------------------------------------"
pip install --upgrade \
    transformers>=4.56.0 \
    tokenizers>=0.22.0 \
    accelerate>=1.5.0 \
    datasets>=3.0.0
echo "✅ Transformers 업그레이드 완료"
echo ""

# PEFT & TRL
echo "Step 7: 파인튜닝 라이브러리 업그레이드"
echo "----------------------------------------------------------------------"
pip install --upgrade \
    peft>=0.14.0 \
    trl>=0.24.0 \
    bitsandbytes>=0.45.0
echo "✅ 파인튜닝 라이브러리 업그레이드 완료"
echo ""

# vLLM 설치
echo "Step 8: vLLM 설치 (CoT 증강용)"
echo "----------------------------------------------------------------------"
pip install vllm>=0.8.0
echo "✅ vLLM 설치 완료"
echo ""

# Unsloth 설치
echo "Step 9: Unsloth 설치 (학습 최적화)"
echo "----------------------------------------------------------------------"
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
echo "✅ Unsloth 설치 완료"
echo ""

# 추가 최적화 라이브러리
echo "Step 10: 추가 최적화 라이브러리"
echo "----------------------------------------------------------------------"
pip install --upgrade \
    triton>=3.2.0 \
    sentence-transformers>=2.8.0
echo "✅ 추가 라이브러리 설치 완료"
echo ""

# 환경 검증
echo "Step 11: 최종 환경 검증"
echo "----------------------------------------------------------------------"
python check_env.py
echo ""

# 설치 요약
echo "=========================================================================="
echo "설치 완료!"
echo "=========================================================================="
echo ""
echo "설치된 주요 라이브러리:"
python -c "
import sys
libs = [
    'torch', 'transformers', 'flash_attn', 'vllm', 
    'unsloth', 'peft', 'trl', 'accelerate'
]
for lib in libs:
    try:
        mod = __import__(lib)
        ver = getattr(mod, '__version__', 'N/A')
        print(f'  ✅ {lib:20} {ver}')
    except ImportError:
        print(f'  ❌ {lib:20} 설치 안 됨')
"
echo ""
echo "다음 단계:"
echo "  1. source env.sh              # 환경 변수 로드"
echo "  2. python check_env.py        # 환경 재확인"
echo "  3. ./quick_test.sh            # 빠른 테스트"
echo ""
echo "=========================================================================="

