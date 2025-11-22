#!/bin/bash
###############################################################################
# 전체 파이프라인 실행: Merge → GGUF 변환 → Hub 업로드
# 사용법: bash run_all.sh
###############################################################################

set -e  # 에러 발생 시 중단

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "Qwen2.5-3B-Korean-QLoRA 전체 프레임워크 지원 파이프라인"
echo "================================================================================"
echo ""
echo "작업 순서:"
echo "  1. LoRA 어댑터 Merge"
echo "  2. GGUF 변환 (Q4_K_M, Q5_K_M, Q8_0, F16)"
echo "  3. HuggingFace Hub 업로드 (Merged + GGUF)"
echo "  4. 모델 카드 업로드"
echo ""
echo "예상 소요 시간: 30-60분"
echo "필요 디스크 용량: ~20GB"
echo ""

read -p "계속하시겠습니까? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "취소되었습니다."
    exit 1
fi

echo ""
echo "================================================================================"
echo "Step 1/3: LoRA 어댑터 Merge"
echo "================================================================================"
python 1_merge_lora.py
if [ $? -ne 0 ]; then
    echo "❌ Merge 실패"
    exit 1
fi

echo ""
echo "================================================================================"
echo "Step 2/3: GGUF 변환"
echo "================================================================================"
python 2_convert_to_gguf.py
if [ $? -ne 0 ]; then
    echo "❌ GGUF 변환 실패"
    exit 1
fi

echo ""
echo "================================================================================"
echo "Step 3/4: HuggingFace Hub 업로드"
echo "================================================================================"
python 3_upload_to_hub.py
if [ $? -ne 0 ]; then
    echo "❌ Hub 업로드 실패"
    exit 1
fi

echo ""
echo "================================================================================"
echo "Step 4/4: 모델 카드 업로드"
echo "================================================================================"
echo "업로드 중: MODEL_CARD_MERGED.md → MyeongHo0621/Qwen2.5-3B-Korean/README.md"
huggingface-cli upload \
    MyeongHo0621/Qwen2.5-3B-Korean \
    ../MODEL_CARD_MERGED.md \
    README.md \
    --commit-message "Update model card with all frameworks support"
if [ $? -ne 0 ]; then
    echo "❌ 모델 카드 업로드 실패"
    exit 1
fi
echo "✅ 모델 카드 업로드 완료"

echo ""
echo "================================================================================"
echo "🎉 전체 파이프라인 완료!"
echo "================================================================================"
echo ""
echo "✅ 완료된 작업:"
echo "  1. LoRA 어댑터 Merge"
echo "  2. GGUF 변환 (4개 레벨)"
echo "  3. HuggingFace Hub 업로드"
echo "  4. 모델 카드 업로드"
echo ""
echo "📍 모델 URL:"
echo "  🔗 Merged 모델: https://huggingface.co/MyeongHo0621/Qwen2.5-3B-Korean"
echo "  🔗 PEFT 어댑터: https://huggingface.co/MyeongHo0621/Qwen2.5-3B-Korean-QLoRA"
echo ""
echo "💡 이제 다음 프레임워크에서 사용 가능합니다:"
echo "  - Transformers (MyeongHo0621/Qwen2.5-3B-Korean)"
echo "  - vLLM (MyeongHo0621/Qwen2.5-3B-Korean)"
echo "  - SGLang (MyeongHo0621/Qwen2.5-3B-Korean)"
echo "  - Ollama (MyeongHo0621/Qwen2.5-3B-Korean/gguf/)"
echo "  - Llama.cpp (MyeongHo0621/Qwen2.5-3B-Korean/gguf/)"
echo "  - PEFT (MyeongHo0621/Qwen2.5-3B-Korean-QLoRA)"
echo ""

