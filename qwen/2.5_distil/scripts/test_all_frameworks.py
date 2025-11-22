#!/usr/bin/env python3
"""
모든 프레임워크 테스트 스크립트
- vLLM (Merged 모델)
- SGLang (Merged 모델)
- Transformers (Merged 모델)
- PEFT + Transformers (LoRA 어댑터)
- Ollama (GGUF)
- Llama.cpp (GGUF)
"""

import os
import sys
import subprocess
from pathlib import Path

# 테스트 프롬프트
TEST_PROMPTS = [
    "한국의 수도는 어디인가요?",
    "김치찌개 레시피를 간단히 알려주세요",
    "파이썬으로 피보나치 수열을 구현하는 방법은?"
]

print("=" * 80)
print("Qwen2.5-3B-Korean-QLoRA 프레임워크 테스트")
print("=" * 80)

# 1. vLLM 테스트 (Merged 모델)
print("\n[ 1/6 ] vLLM 테스트 (Merged 모델)")
print("-" * 80)
try:
    from vllm import LLM, SamplingParams
    
    print("  ℹ️  vLLM 로딩 중...")
    llm = LLM(
        model="MyeongHo0621/Qwen2.5-3B-Korean",
        quantization="bitsandbytes",
        gpu_memory_utilization=0.6
    )
    
    params = SamplingParams(temperature=0.7, max_tokens=256)
    
    print("  ✅ vLLM 로딩 완료")
    
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n  질문 {i}: {prompt}")
        outputs = llm.generate([prompt], params)
        response = outputs[0].outputs[0].text
        print(f"  답변: {response[:200]}...")
    
    print("\n  ✅ vLLM 테스트 완료")
    
except Exception as e:
    print(f"  ❌ vLLM 테스트 실패: {e}")

# 2. SGLang 테스트 (Merged 모델)
print("\n[ 2/6 ] SGLang 테스트 (Merged 모델)")
print("-" * 80)
try:
    import sglang as sgl
    
    print("  ℹ️  SGLang 로딩 중...")
    runtime = sgl.Runtime(
        model_path="MyeongHo0621/Qwen2.5-3B-Korean",
        quantization="bitsandbytes"
    )
    sgl.set_default_backend(runtime)
    
    @sgl.function
    def chat(s, prompt):
        s += sgl.user(prompt)
        s += sgl.assistant(sgl.gen("response", max_tokens=256, temperature=0.7))
    
    print("  ✅ SGLang 로딩 완료")
    
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n  질문 {i}: {prompt}")
        state = chat.run(prompt=prompt)
        response = state["response"]
        print(f"  답변: {response[:200]}...")
    
    print("\n  ✅ SGLang 테스트 완료")
    
except Exception as e:
    print(f"  ❌ SGLang 테스트 실패: {e}")

# 3. Transformers 테스트 (Merged 모델)
print("\n[ 3/6 ] Transformers 테스트 (Merged 모델)")
print("-" * 80)
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    print("  ℹ️  모델 로딩 중...")
    model = AutoModelForCausalLM.from_pretrained(
        "MyeongHo0621/Qwen2.5-3B-Korean",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained("MyeongHo0621/Qwen2.5-3B-Korean")
    
    print("  ✅ 모델 로딩 완료")
    
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n  질문 {i}: {prompt}")
        
        messages = [
            {"role": "system", "content": "You are a helpful Korean assistant."},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        
        print(f"  답변: {response[:200]}...")
    
    print("\n  ✅ Transformers 테스트 완료")
    
except Exception as e:
    print(f"  ❌ Transformers 테스트 실패: {e}")

# 4. PEFT + Transformers 테스트 (LoRA 어댑터)
print("\n[ 4/6 ] PEFT + Transformers 테스트 (LoRA 어댑터)")
print("-" * 80)
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch
    
    print("  ℹ️  모델 로딩 중...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    model = PeftModel.from_pretrained(
        base_model,
        "MyeongHo0621/Qwen2.5-3B-Korean-QLoRA"  # 루트 = 최종 모델
    )
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    
    print("  ✅ 모델 로딩 완료")
    
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n  질문 {i}: {prompt}")
        
        messages = [
            {"role": "system", "content": "You are a helpful Korean assistant."},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 어시스턴트 응답만 추출
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        
        print(f"  답변: {response[:200]}...")
    
    print("\n  ✅ PEFT 테스트 완료")
    
except Exception as e:
    print(f"  ❌ PEFT 테스트 실패: {e}")

# 5. Ollama 테스트 (GGUF)
print("\n[ 5/6 ] Ollama 테스트 (GGUF)")
print("-" * 80)
try:
    # Ollama 설치 확인
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        check=False
    )
    
    if result.returncode != 0:
        print("  ⚠️  Ollama가 설치되지 않았습니다")
        print("  ℹ️  설치: curl -fsSL https://ollama.com/install.sh | sh")
    else:
        # 모델 확인
        if "qwen25-korean" not in result.stdout:
            print("  ⚠️  qwen25-korean 모델이 없습니다")
            print("  ℹ️  먼저 GGUF 변환 및 Ollama 모델 생성이 필요합니다")
        else:
            print("  ✅ Ollama 모델 확인됨")
            
            for i, prompt in enumerate(TEST_PROMPTS, 1):
                print(f"\n  질문 {i}: {prompt}")
                result = subprocess.run(
                    ["ollama", "run", "qwen25-korean", prompt],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                response = result.stdout.strip()
                print(f"  답변: {response[:200]}...")
            
            print("\n  ✅ Ollama 테스트 완료")
    
except Exception as e:
    print(f"  ❌ Ollama 테스트 실패: {e}")

# 6. Llama.cpp 테스트 (GGUF)
print("\n[ 6/6 ] Llama.cpp 테스트 (GGUF)")
print("-" * 80)
try:
    llama_cpp_main = "/home/work/llama.cpp/main"
    gguf_file = "/home/work/.setting/qwen/2.5_distil/outputs/gguf/qwen25-3b-korean-Q4_K_M.gguf"
    
    if not Path(llama_cpp_main).exists():
        print(f"  ⚠️  Llama.cpp를 찾을 수 없습니다: {llama_cpp_main}")
        print(f"  ℹ️  먼저 Llama.cpp를 빌드해주세요")
    elif not Path(gguf_file).exists():
        print(f"  ⚠️  GGUF 파일을 찾을 수 없습니다: {gguf_file}")
        print(f"  ℹ️  먼저 GGUF 변환을 실행해주세요")
    else:
        print("  ✅ Llama.cpp 및 GGUF 확인됨")
        
        for i, prompt in enumerate(TEST_PROMPTS, 1):
            print(f"\n  질문 {i}: {prompt}")
            
            full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            result = subprocess.run(
                [
                    llama_cpp_main,
                    "-m", gguf_file,
                    "-p", full_prompt,
                    "-n", "256",
                    "--temp", "0.7",
                    "-ngl", "99"
                ],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # 응답 추출 (assistant 이후)
            output = result.stdout
            if "assistant" in output:
                response = output.split("assistant")[-1].strip()
            else:
                response = output
            
            print(f"  답변: {response[:200]}...")
        
        print("\n  ✅ Llama.cpp 테스트 완료")
    
except Exception as e:
    print(f"  ❌ Llama.cpp 테스트 실패: {e}")

# 완료
print("\n" + "=" * 80)
print("🎉 전체 테스트 완료!")
print("=" * 80)

