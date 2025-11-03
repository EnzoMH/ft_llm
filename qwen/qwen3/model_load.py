#!/usr/bin/env python3
"""
Qwen3-VL 모델 로드 및 추론 테스트
- Phase 1/2 모델 테스트
- Thinking 모드 활성화
- 배치 추론 지원
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str, device: str = "auto"):
    """모델 로드"""
    
    print(f"🚀 모델 로딩: {model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    print("✅ 모델 로딩 완료!\n")
    
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    enable_thinking: bool = False,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """응답 생성"""
    
    # 메시지 구성
    messages = [
        {
            "role": "system",
            "content": "/think" if enable_thinking else "당신은 한국어 전용 AI 어시스턴트입니다."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    # 토크나이징
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 디코딩
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 입력 제거하고 응답만 추출
    response = response.split("assistant\n")[-1] if "assistant\n" in response else response
    
    return response


def interactive_mode(model, tokenizer, enable_thinking: bool = False):
    """대화형 모드"""
    
    print("\n" + "="*80)
    print("대화형 모드 시작")
    print("="*80)
    print(f"Thinking 모드: {'✅ 활성화' if enable_thinking else '❌ 비활성화'}")
    print("종료: 'quit' 또는 'exit' 입력")
    print("Thinking 전환: '/think' 또는 '/no_think' 입력")
    print("="*80 + "\n")
    
    thinking_enabled = enable_thinking
    
    while True:
        try:
            user_input = input("사용자: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', '종료']:
                print("\n👋 종료합니다.")
                break
            
            if user_input == '/think':
                thinking_enabled = True
                print("💭 Thinking 모드 활성화\n")
                continue
            
            if user_input == '/no_think':
                thinking_enabled = False
                print("💬 일반 모드 활성화\n")
                continue
            
            print("\n어시스턴트: ", end="", flush=True)
            
            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=user_input,
                enable_thinking=thinking_enabled,
            )
            
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}\n")


def test_model(model_path: str, interactive: bool = False, enable_thinking: bool = False):
    """모델 테스트"""
    
    # 모델 로드
    model, tokenizer = load_model(model_path)
    
    if interactive:
        # 대화형 모드
        interactive_mode(model, tokenizer, enable_thinking)
    else:
        # 테스트 프롬프트
        test_prompts = [
            "AGV가 앞에 장애물이 있을 때 어떻게 해야 하나요?",
            "3x + 5 = 14 를 풀어주세요.",
            "한국어로 파인튜닝이 잘 되었는지 확인하고 싶습니다. 자기소개를 해주세요.",
        ]
        
        print(f"🧪 테스트 프롬프트 ({len(test_prompts)}개)")
        print(f"Thinking 모드: {'✅ 활성화' if enable_thinking else '❌ 비활성화'}")
        print("="*80 + "\n")
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"[테스트 {i}/{len(test_prompts)}]")
            print(f"질문: {prompt}\n")
            
            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                enable_thinking=enable_thinking,
            )
            
            print(f"응답:\n{response}\n")
            print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL 모델 테스트")
    
    parser.add_argument(
        'model_path',
        type=str,
        help='모델 경로 (Phase 1/2 결과 또는 HF 모델)'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='대화형 모드'
    )
    parser.add_argument(
        '--thinking',
        action='store_true',
        help='Thinking 모드 활성화'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='디바이스 (auto, cuda, cpu)'
    )
    
    args = parser.parse_args()
    
    test_model(
        model_path=args.model_path,
        interactive=args.interactive,
        enable_thinking=args.thinking,
    )


if __name__ == "__main__":
    main()

