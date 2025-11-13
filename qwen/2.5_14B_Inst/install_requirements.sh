#!/bin/bash
# 필요한 패키지만 선택적으로 설치하는 스크립트

echo "=========================================="
echo "필수 패키지 선택적 설치"
echo "=========================================="

# 현재 환경 확인
echo "현재 환경 확인 중..."
pip list | grep -E "(torch|transformers|flash|unsloth|peft|bitsandbytes|accelerate|trl|datasets)" | head -10

echo ""
echo "=========================================="
echo "필수 패키지 설치 시작"
echo "=========================================="

# 1. Flash Attention 2.8.3 (가장 중요)
echo "[1/8] Flash Attention 2.8.3 설치 중..."
pip install --user flash-attn==2.8.3 --no-build-isolation || echo "Flash Attention 설치 실패 (이미 설치되어 있을 수 있음)"

# 2. Transformers 업그레이드
echo "[2/8] Transformers 업그레이드 중..."
pip install --user --upgrade "transformers>=4.57.0" || echo "Transformers 업그레이드 실패"

# 3. Datasets 업그레이드
echo "[3/8] Datasets 업그레이드 중..."
pip install --user --upgrade "datasets>=3.4.0" || echo "Datasets 업그레이드 실패"

# 4. Accelerate 업그레이드
echo "[4/8] Accelerate 업그레이드 중..."
pip install --user --upgrade "accelerate>=1.11.0" || echo "Accelerate 업그레이드 실패"

# 5. TRL 설치
echo "[5/8] TRL 설치 중..."
pip install --user "trl>=0.23.0" || echo "TRL 설치 실패"

# 6. BitsAndBytes 업그레이드
echo "[6/8] BitsAndBytes 업그레이드 중..."
pip install --user --upgrade "bitsandbytes>=0.48.0" || echo "BitsAndBytes 업그레이드 실패"

# 7. Unsloth 설치
echo "[7/8] Unsloth 설치 중..."
pip install --user "unsloth>=2025.11.2" || echo "Unsloth 설치 실패"
pip install --user "unsloth-zoo>=2025.11.3" || echo "Unsloth-zoo 설치 실패"

# 8. 기타 필수 패키지
echo "[8/8] 기타 필수 패키지 설치 중..."
pip install --user --upgrade "huggingface-hub>=0.36.0" || echo "HuggingFace Hub 업그레이드 실패"
pip install --user --upgrade "safetensors>=0.6.0" || echo "Safetensors 업그레이드 실패"
pip install --user --upgrade "scipy>=1.14.0" || echo "Scipy 업그레이드 실패"
pip install --user "sentencepiece>=0.2.0" || echo "Sentencepiece 설치 실패"
pip install --user "protobuf>=3.20.0" || echo "Protobuf 설치 실패"
pip install --user "psutil>=7.1.0" || echo "Psutil 설치 실패"

echo ""
echo "=========================================="
echo "설치 완료!"
echo "=========================================="
echo ""
echo "설치된 버전 확인:"
pip list | grep -E "(torch|transformers|flash|unsloth|peft|bitsandbytes|accelerate|trl|datasets|safetensors|huggingface-hub)" | head -15

echo ""
echo "다음 명령어로 훈련을 재개하세요:"
echo "cd /home/work/tes/ft_llm/qwen/2.5_14B_Inst"
echo "nohup python 0_qwen14b_multiturn_ft.py > train.log 2>&1 &"

