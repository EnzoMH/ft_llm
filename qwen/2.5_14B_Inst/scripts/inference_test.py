#!/usr/bin/env python3
"""
학습된 모델 추론 테스트
- 모델 로드
- 간단한 대화 테스트
- 멀티턴 대화 테스트
"""

import os
import sys
import torch
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'util'))
from gpu_mnrtg import GPUMonitor


def load_model(model_path: str):
    """모델 로드"""
    print("\n" + "="*80)
    print("모델 로딩")
    print("="*80)
    print(f"경로: {model_path}")
    
    from unsloth import FastLanguageModel
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=False,
        trust_remote_code=True,
    )
    
    # 추론 모드
    FastLanguageModel.for_inference(model)
    
    print("[ COMPLETE ] 모델 로드 완료")
    print("="*80)
    
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> str:
    """응답 생성"""
    
    # ChatML 포맷 적용
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    # 생성
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 디코딩
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # 응답 부분만 추출
    # (입력 부분을 제외한 생성된 부분)
    generated_text = tokenizer.decode(
        outputs[0][inputs.shape[1]:], 
        skip_special_tokens=True
    )
    
    return generated_text


def test_single_turn(model, tokenizer):
    """단일턴 대화 테스트"""
    print("\n" + "="*80)
    print("[ 테스트 1 ] 단일턴 대화")
    print("="*80)
    
    test_cases = [
        "안녕하세요! 자기소개 부탁드립니다.",
        "Python에서 리스트와 튜플의 차이점은 무엇인가요?",
        "서울의 유명한 관광지 3곳을 추천해주세요.",
    ]
    
    for i, user_input in enumerate(test_cases, 1):
        print(f"\n[ 케이스 {i} ]")
        print(f"입력: {user_input}")
        print("-"*80)
        
        messages = [
            {"role": "user", "content": user_input}
        ]
        
        response = generate_response(
            model, tokenizer, messages,
            max_new_tokens=256,
            temperature=0.7
        )
        
        print(f"응답: {response}")
        print("-"*80)


def test_multi_turn(model, tokenizer):
    """멀티턴 대화 테스트"""
    print("\n" + "="*80)
    print("[ 테스트 2 ] 멀티턴 대화")
    print("="*80)
    
    # 대화 히스토리
    conversation = [
        {"role": "user", "content": "머신러닝이 뭔가요?"},
    ]
    
    print(f"\n턴 1:")
    print(f"사용자: {conversation[0]['content']}")
    
    # 첫 번째 응답
    response_1 = generate_response(
        model, tokenizer, conversation,
        max_new_tokens=256
    )
    print(f"AI: {response_1}")
    
    # 대화 히스토리 업데이트
    conversation.append({"role": "assistant", "content": response_1})
    conversation.append({
        "role": "user", 
        "content": "그럼 딥러닝과는 어떻게 다른가요?"
    })
    
    print(f"\n턴 2:")
    print(f"사용자: {conversation[-1]['content']}")
    
    # 두 번째 응답
    response_2 = generate_response(
        model, tokenizer, conversation,
        max_new_tokens=256
    )
    print(f"AI: {response_2}")
    
    print("\n" + "="*80)


def test_korean_specific(model, tokenizer):
    """한국어 특화 테스트"""
    print("\n" + "="*80)
    print("[ 테스트 3 ] 한국어 특화")
    print("="*80)
    
    test_cases = [
        "한국의 전통 음식인 김치에 대해 설명해주세요.",
        "서울에서 부산까지 가는 방법을 알려주세요.",
        "한국어 존댓말과 반말의 차이는 무엇인가요?",
    ]
    
    for i, user_input in enumerate(test_cases, 1):
        print(f"\n[ 케이스 {i} ]")
        print(f"입력: {user_input}")
        print("-"*80)
        
        messages = [
            {"role": "user", "content": user_input}
        ]
        
        response = generate_response(
            model, tokenizer, messages,
            max_new_tokens=300,
            temperature=0.7
        )
        
        print(f"응답: {response}")
        print("-"*80)


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print(" Qwen2.5-14B 한국어 모델 추론 테스트")
    print("="*80)
    
    # GPU 확인
    gpu_monitor = GPUMonitor()
    if not gpu_monitor.available:
        print("\n[ ERROR ] CUDA를 사용할 수 없습니다!")
        sys.exit(1)
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    
    # 모델 경로
    model_path = "/home/work/tesseract/qwen/2.5_14B_Inst/output/final"
    
    if not os.path.exists(model_path):
        print(f"\n[ ERROR ] 모델을 찾을 수 없습니다: {model_path}")
        print("먼저 학습을 완료해주세요.")
        sys.exit(1)
    
    # 모델 로드
    model, tokenizer = load_model(model_path)
    
    # 테스트 실행
    try:
        # 1. 단일턴 대화
        test_single_turn(model, tokenizer)
        
        # 2. 멀티턴 대화
        test_multi_turn(model, tokenizer)
        
        # 3. 한국어 특화
        test_korean_specific(model, tokenizer)
        
        print("\n" + "="*80)
        print(" [ OK ] 모든 테스트 완료!")
        print("="*80)
        
    except Exception as e:
        print(f"\n[ ERROR ] 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

