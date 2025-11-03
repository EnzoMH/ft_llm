#!/usr/bin/env python3
"""
EXAONE 4.0 1.2B를 사용한 CoT 데이터 증강
- 한국어 특화 Chain-of-Thought 생성
- 싱글턴/멀티턴 지원
- H100 80GB 최적화
"""

import json
import random
import argparse
from pathlib import Path
from typing import Optional
from vllm import LLM, SamplingParams
from tqdm import tqdm
import re


class CoTDataAugmenter:
    def __init__(self, model_name: str = "LGAI-EXAONE/EXAONE-4.0-1.2B-Instruct"):
        print("EXAONE 4.0 1.2B 모델 로딩 중...")
        self.llm = LLM(
            model=model_name,
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            max_model_len=8192,
            tensor_parallel_size=1,
            trust_remote_code=True,
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048,
            stop=["</s>", "<|im_end|>"],
        )
        
        print("✅ 모델 로딩 완료!")
    
    def sample_dataset(self, input_jsonl: str, sample_ratio: float = 0.05, seed: int = 42) -> list:
        """데이터셋에서 일정 비율만 샘플링"""
        print(f"📖 데이터 샘플링: {input_jsonl} (비율: {sample_ratio*100}%)")
        
        all_data = []
        with open(input_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_data.append(json.loads(line))
        
        if sample_ratio >= 1.0:
            print(f"✅ 전체 데이터 사용: {len(all_data):,}개")
            return all_data
        
        random.seed(seed)
        num_samples = int(len(all_data) * sample_ratio)
        sampled = random.sample(all_data, num_samples)
        
        print(f"✅ 전체 {len(all_data):,}개 중 {len(sampled):,}개 샘플링됨")
        return sampled
    
    def create_singleturn_prompt(self, question: str, answer: str) -> str:
        """싱글턴 CoT 생성 프롬프트"""
        return f"""당신은 단계별 추론을 하는 AI 어시스턴트입니다.
다음 문제에 대한 답변을 <think> 태그를 사용하여 단계별로 작성하세요.

[규칙]
1. <think> 태그 안에 추론 과정을 한국어로 명확하게 작성
2. 각 단계는 "1단계:", "2단계:" 형식으로 구분
3. </think> 태그 밖에 최종 답변만 간결하게 작성
4. 절대 중국어를 사용하지 마세요

[질문]
{question}

[원본 답변 참고]
{answer}

[출력 형식]
<think>
1단계: [문제 분석 및 이해]
2단계: [해결 방법 결정]
3단계: [단계별 계산/추론]
4단계: [답변 검증]
</think>

최종 답변: [간결한 답]"""

    def create_multiturn_prompt(self, messages: list[dict]) -> str:
        """멀티턴 CoT 생성 프롬프트"""
        # 대화 히스토리 구성
        conversation = []
        for msg in messages:
            if msg['role'] == 'user':
                conversation.append(f"사용자: {msg['content']}")
            elif msg['role'] == 'assistant':
                conversation.append(f"어시스턴트: {msg['content']}")
        
        history = "\n".join(conversation[:-1])  # 마지막 답변 제외
        last_question = messages[-2]['content'] if len(messages) >= 2 else ""
        original_answer = messages[-1]['content'] if messages else ""
        
        return f"""당신은 단계별 추론을 하는 AI 어시스턴트입니다.
다음은 사용자와의 대화입니다. 마지막 질문에 대해 <think> 태그를 사용하여 단계별로 답변하세요.

[이전 대화]
{history}

[현재 질문]
{last_question}

[원본 답변 참고]
{original_answer}

[규칙]
1. <think> 태그 안에 추론 과정을 한국어로 명확하게 작성
2. 이전 대화 맥락을 고려하여 답변
3. </think> 태그 밖에 최종 답변만 간결하게 작성
4. 절대 중국어를 사용하지 마세요

[출력 형식]
<think>
1단계: [이전 대화 맥락 파악]
2단계: [현재 질문 분석]
3단계: [답변 구성]
4단계: [일관성 검증]
</think>

최종 답변: [간결한 답]"""

    def parse_chatml(self, text: str) -> list[dict]:
        """ChatML 파싱"""
        messages = []
        parts = text.split('<|im_start|>')
        
        for part in parts[1:]:
            if '<|im_end|>' in part:
                split_part = part.split('\n', 1)
                if len(split_part) < 2:
                    continue
                role = split_part[0].strip()
                content = split_part[1].split('<|im_end|>')[0].strip()
                
                if role in ['system', 'user', 'assistant']:
                    messages.append({"role": role, "content": content})
        
        return messages
    
    def extract_qa_pairs(self, data_item: dict) -> Optional[dict]:
        """데이터에서 Q&A 추출"""
        if 'messages' in data_item:
            messages = data_item['messages']
        else:
            messages = self.parse_chatml(data_item.get('text', ''))
        
        # system 메시지 제외
        messages = [m for m in messages if m['role'] != 'system']
        
        user_msgs = [m for m in messages if m['role'] == 'user']
        assistant_msgs = [m for m in messages if m['role'] == 'assistant']
        
        if not user_msgs or not assistant_msgs:
            return None
        
        is_multiturn = len(user_msgs) > 1 and len(assistant_msgs) > 1
        
        return {
            'messages': messages,
            'is_multiturn': is_multiturn,
            'last_question': user_msgs[-1]['content'],
            'last_answer': assistant_msgs[-1]['content']
        }
    
    def generate_cot_batch(self, prompts: list[str], batch_size: int = 128) -> list[str]:
        """배치 단위 CoT 생성"""
        all_results = []
        
        for i in tqdm(range(0, len(prompts), batch_size), desc="🤖 CoT 생성"):
            batch = prompts[i:i+batch_size]
            outputs = self.llm.generate(batch, self.sampling_params)
            
            for output in outputs:
                generated = output.outputs[0].text.strip()
                all_results.append(generated)
        
        return all_results
    
    def augment_singleturn(self, data_items: list[dict]) -> list[dict]:
        """싱글턴 데이터 증강"""
        print("\n📝 싱글턴 CoT 생성 중...")
        
        prompts = []
        valid_items = []
        
        for item in data_items:
            qa = self.extract_qa_pairs(item)
            if qa and not qa['is_multiturn']:
                prompt = self.create_singleturn_prompt(
                    qa['last_question'],
                    qa['last_answer']
                )
                prompts.append(prompt)
                valid_items.append(qa)
        
        print(f"✅ {len(prompts):,}개 싱글턴 프롬프트 생성됨")
        
        if not prompts:
            print("⚠️  싱글턴 데이터 없음")
            return []
        
        cot_results = self.generate_cot_batch(prompts, batch_size=128)
        
        augmented = []
        for qa, cot_text in zip(valid_items, cot_results):
            augmented.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "당신은 단계별로 사고하는 한국어 AI 어시스턴트입니다."
                    },
                    {
                        "role": "user",
                        "content": qa['last_question']
                    },
                    {
                        "role": "assistant",
                        "content": cot_text
                    }
                ],
                "type": "singleturn",
                "augmented_by": "EXAONE-4.0-1.2B"
            })
        
        return augmented
    
    def augment_multiturn(self, data_items: list[dict]) -> list[dict]:
        """멀티턴 데이터 증강"""
        print("\n💬 멀티턴 CoT 생성 중...")
        
        prompts = []
        valid_items = []
        
        for item in data_items:
            qa = self.extract_qa_pairs(item)
            if qa and qa['is_multiturn']:
                prompt = self.create_multiturn_prompt(qa['messages'])
                prompts.append(prompt)
                valid_items.append(qa)
        
        print(f"✅ {len(prompts):,}개 멀티턴 프롬프트 생성됨")
        
        if not prompts:
            print("⚠️  멀티턴 데이터 없음")
            return []
        
        cot_results = self.generate_cot_batch(prompts, batch_size=64)
        
        augmented = []
        for qa, cot_text in zip(valid_items, cot_results):
            # 원본 대화에서 마지막 답변만 CoT로 교체
            new_messages = [
                {
                    "role": "system",
                    "content": "당신은 단계별로 사고하는 한국어 AI 어시스턴트입니다."
                }
            ] + qa['messages'][:-1] + [
                {
                    "role": "assistant",
                    "content": cot_text
                }
            ]
            
            augmented.append({
                "messages": new_messages,
                "type": "multiturn",
                "augmented_by": "EXAONE-4.0-1.2B"
            })
        
        return augmented
    
    def process_dataset(
        self, 
        input_jsonl: str, 
        output_dir: str, 
        sample_ratio: float = 0.05, 
        singleturn_ratio: float = 0.7
    ) -> list[dict]:
        """전체 데이터셋 처리"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_name = Path(input_jsonl).stem
        
        # 샘플링
        sampled_data = self.sample_dataset(input_jsonl, sample_ratio)
        
        # 싱글턴/멀티턴 분할
        random.shuffle(sampled_data)
        split_idx = int(len(sampled_data) * singleturn_ratio)
        
        singleturn_data = sampled_data[:split_idx]
        multiturn_data = sampled_data[split_idx:]
        
        print(f"\n📊 데이터 분포:")
        print(f"   싱글턴: {len(singleturn_data):,}개 ({singleturn_ratio*100}%)")
        print(f"   멀티턴: {len(multiturn_data):,}개 ({(1-singleturn_ratio)*100}%)")
        
        results = []
        
        # 싱글턴 처리
        if singleturn_data:
            singleturn_results = self.augment_singleturn(singleturn_data)
            results.extend(singleturn_results)
        
        # 멀티턴 처리
        if multiturn_data:
            multiturn_results = self.augment_multiturn(multiturn_data)
            results.extend(multiturn_results)
        
        # 저장
        output_file = output_dir / f"{dataset_name}_cot_augmented.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"\n✅ 완료! {len(results):,}개 샘플 → {output_file}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="EXAONE CoT 데이터 증강")
    parser.add_argument(
        '--input-dir', 
        type=str, 
        default='../../korean_large_data/cleaned_jsonl',
        help='입력 JSONL 디렉토리'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='phase2_thinking_exaone',
        help='출력 디렉토리'
    )
    parser.add_argument(
        '--sample-ratio', 
        type=float, 
        default=0.05,
        help='샘플링 비율 (0.05 = 5%)'
    )
    parser.add_argument(
        '--singleturn-ratio', 
        type=float, 
        default=0.7,
        help='싱글턴 비율 (0.7 = 70%)'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=None,
        help='처리할 데이터셋 목록 (기본: 전체)'
    )
    
    args = parser.parse_args()
    
    # 기본 데이터셋 목록
    if args.datasets is None:
        datasets = [
            'orca_math_ko_data.jsonl',
            'kullm_v2_full_data.jsonl',
            'smol_koreantalk_data.jsonl',
            'won_instruct_data.jsonl',
            'koalpaca_data.jsonl',
            'kovicuna_data.jsonl',
            'kowiki_qa_data.jsonl',
            'kullm_v2_data.jsonl',
            'ko_evol_writing_data.jsonl',
        ]
    else:
        datasets = args.datasets
    
    # EXAONE 모델 로드
    augmenter = CoTDataAugmenter()
    
    all_results = []
    
    # 각 데이터셋 처리
    for dataset_name in datasets:
        dataset_path = Path(args.input_dir) / dataset_name
        
        if not dataset_path.exists():
            print(f"⚠️  파일 없음: {dataset_path}")
            continue
        
        print(f"\n{'='*80}")
        print(f"📦 처리 중: {dataset_name}")
        print(f"{'='*80}")
        
        try:
            results = augmenter.process_dataset(
                input_jsonl=str(dataset_path),
                output_dir=args.output_dir,
                sample_ratio=args.sample_ratio,
                singleturn_ratio=args.singleturn_ratio
            )
            all_results.extend(results)
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            continue
    
    # 전체 결과 저장
    if all_results:
        combined_file = Path(args.output_dir) / 'all_cot_augmented.jsonl'
        with open(combined_file, 'w', encoding='utf-8') as f:
            for item in all_results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"\n{'='*80}")
        print(f"🎉 전체 완료!")
        print(f"   총 샘플: {len(all_results):,}개")
        print(f"   통합 파일: {combined_file}")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()

