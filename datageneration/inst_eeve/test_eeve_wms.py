"""
훈련된 EEVE-WMS 모델 테스트
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse


def generate_answer(model, tokenizer, question: str, max_tokens: int = 512):
    """EEVE 템플릿 형식으로 답변 생성"""
    
    # EEVE 공식 템플릿
    prompt = f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: {question}
Assistant: """
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,      # 웹 검색 결과 권장값
            top_p=0.85,
            repetition_penalty=1.0,  # 중요!
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    answer = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    ).strip()
    
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='모델 경로 (LoRA 어댑터 또는 병합 모델)')
    parser.add_argument('--base-model', type=str,
                        default='MyeongHo0621/eeve-vss-smh',
                        help='베이스 모델 (LoRA 사용 시)')
    parser.add_argument('--merged', action='store_true',
                        help='병합된 모델 사용')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("EEVE-WMS 모델 테스트")
    print("="*80 + "\n")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 모델 로드
    if args.merged:
        # 병합된 모델
        print(f"병합 모델 로딩: {args.model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    else:
        # LoRA 어댑터
        print(f"베이스 모델 로딩: {args.base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        print(f"LoRA 어댑터 로딩: {args.model}")
        model = PeftModel.from_pretrained(
            model,
            args.model,
            is_trainable=False
        )
    
    model.eval()
    print("✓ 모델 로드 완료\n")
    
    # 테스트 질문들
    test_questions = [
        "WMS 도입 비용은 얼마나 드나요?",
        "재고 실사 시간을 줄이는 방법은?",
        "바코드와 RFID 중 어떤 것을 선택해야 하나요?",
        "피킹 효율을 높이는 방법은?",
        "WMS와 ERP 연동 시 주의사항은?",
    ]
    
    print("="*80)
    print("테스트 시작")
    print("="*80 + "\n")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[질문 {i}]")
        print(f"Q: {question}")
        print("-" * 80)
        
        answer = generate_answer(model, tokenizer, question)
        
        print(f"A: {answer}")
        print("="*80)
    
    print("\n✓ 테스트 완료!")


if __name__ == "__main__":
    main()

