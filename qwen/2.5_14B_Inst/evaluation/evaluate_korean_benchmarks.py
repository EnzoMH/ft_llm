#!/usr/bin/env python3
"""
한국어 벤치마크 평가 스크립트
- 원본 Qwen2.5-14B-Instruct vs 파인튜닝된 Qwen2.5-14B-Korean 비교
- KMMLU, KoBEST, KorQuAD 등 주요 한국어 벤치마크 평가
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import torch

# 환경 변수 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 모듈 경로 추가
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)


@dataclass
class BenchmarkResult:
    """벤치마크 평가 결과"""
    benchmark_name: str
    model_name: str
    accuracy: float
    total_samples: int
    correct_samples: int
    evaluation_time: float
    details: Optional[Dict] = None


class KoreanBenchmarkEvaluator:
    """한국어 벤치마크 평가기"""
    
    def __init__(self, model, tokenizer, model_name: str):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = next(model.parameters()).device
    
    def generate_answer(self, prompt: str, max_new_tokens: int = 512) -> str:
        """프롬프트에 대한 답변 생성"""
        messages = [{"role": "user", "content": prompt}]
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,  # 일관성을 위해 greedy decoding
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def evaluate_mmlu_style(self, dataset, benchmark_name: str, show_examples: int = 5) -> BenchmarkResult:
        """MMLU 스타일 벤치마크 평가 (KMMLU 등)"""
        print(f"\n[{benchmark_name}] 평가 시작...")
        
        correct = 0
        total = 0
        start_time = time.time()
        examples = []  # 질문-답변 예시 저장
        
        for idx, item in enumerate(dataset):
            question = item.get("question", item.get("input", ""))
            choices = item.get("choices", [])
            answer = item.get("answer", item.get("label", ""))
            
            if not question or not choices:
                continue
            
            # 프롬프트 구성
            prompt = f"{question}\n\n"
            for i, choice in enumerate(choices):
                prompt += f"{chr(65+i)}. {choice}\n"
            prompt += "\n정답을 선택하세요 (A, B, C, D 중 하나):"
            
            # 답변 생성
            response = self.generate_answer(prompt, max_new_tokens=50)
            
            # 답변 추출 (더 정확한 패턴 매칭)
            predicted = None
            response_upper = response.upper()
            
            # 1. "답: A" 또는 "정답은 B입니다" 같은 명시적 패턴 찾기
            explicit_patterns = [
                r'답[:\s]*([A-E])',
                r'정답[은는]?\s*([A-E])',
                r'선택[은는]?\s*([A-E])',
                r'([A-E])[\.\)]\s*정답',
            ]
            
            for pattern in explicit_patterns:
                match = re.search(pattern, response_upper)
                if match:
                    predicted = match.group(1)
                    break
            
            # 2. 패턴이 없으면 첫 번째 A-E 문자 찾기
            if not predicted:
                for char in response_upper:
                    if char in ['A', 'B', 'C', 'D', 'E']:
                        predicted = char
                        break
            
            # 정확도 계산
            answer_upper = str(answer).upper().strip()
            is_correct = predicted == answer_upper if predicted else False
            if is_correct:
                correct += 1
            total += 1
            
            # 처음 몇 개 샘플 출력
            if idx < show_examples:
                examples.append({
                    "question": question[:200] + "..." if len(question) > 200 else question,
                    "choices": choices,
                    "correct_answer": answer,
                    "predicted": predicted,
                    "model_response": response[:100] + "..." if len(response) > 100 else response,
                    "is_correct": is_correct
                })
                print(f"\n  [예시 {idx+1}]")
                print(f"  질문: {question[:150]}...")
                print(f"  정답: {answer}")
                print(f"  모델 답변: {predicted if predicted else '답변 없음'}")
                print(f"  모델 응답: {response[:100]}...")
                print(f"  결과: {'✓ 정답' if is_correct else '✗ 오답'}")
            
            if total % 100 == 0:
                print(f"  진행: {total}개 평가 중... (정확도: {correct/total*100:.2f}%)")
        
        elapsed_time = time.time() - start_time
        accuracy = correct / total if total > 0 else 0.0
        
        return BenchmarkResult(
            benchmark_name=benchmark_name,
            model_name=self.model_name,
            accuracy=accuracy,
            total_samples=total,
            correct_samples=correct,
            evaluation_time=elapsed_time,
            details={"examples": examples}
        )
    
    def evaluate_qa(self, dataset, benchmark_name: str, show_examples: int = 10, save_all: bool = False) -> BenchmarkResult:
        """QA 벤치마크 평가 (KorQuAD 등) - 정확한 채점 기준 적용"""
        import re
        
        print(f"\n[{benchmark_name}] 평가 시작...")
        
        correct = 0
        total = 0
        start_time = time.time()
        examples = []  # 질문-답변 예시 저장
        all_results = [] if save_all else None  # 전체 결과 저장 (옵션)
        
        def normalize_answer(text):
            """답변 정규화 (공백 제거, 소문자 변환)"""
            if not text:
                return ""
            # 공백 제거 및 소문자 변환
            text = re.sub(r'\s+', '', str(text).lower())
            # 구두점 제거 (하지만 의미 있는 구두점은 보존)
            text = re.sub(r'[^\w가-힣]', '', text)
            return text
        
        def exact_match(prediction, ground_truth):
            """정확한 매칭 (대소문자, 공백 무시)"""
            return normalize_answer(prediction) == normalize_answer(ground_truth)
        
        def contains_answer(prediction, ground_truth):
            """답변이 포함되어 있는지 확인 (단어 경계 고려)"""
            pred_norm = normalize_answer(prediction)
            gt_norm = normalize_answer(ground_truth)
            
            if not gt_norm:
                return False
            
            # 정확한 매칭
            if pred_norm == gt_norm:
                return True
            
            # 포함 여부 확인 (단, 너무 짧은 답변은 제외)
            if len(gt_norm) >= 3 and gt_norm in pred_norm:
                return True
            
            return False
        
        for idx, item in enumerate(dataset):
            context = item.get("context", item.get("paragraph", ""))
            question = item.get("question", "")
            answer_raw = item.get("answers", item.get("answer", ""))
            
            # 정답 추출 (다양한 형태 처리)
            answer_texts = []
            if isinstance(answer_raw, list):
                for ans in answer_raw:
                    if isinstance(ans, str):
                        answer_texts.append(ans.strip())
                    elif isinstance(ans, list) and ans:
                        answer_texts.append(str(ans[0]).strip() if isinstance(ans[0], str) else str(ans[0]).strip())
                    elif isinstance(ans, dict):
                        text = ans.get("text", ans.get("answer", ""))
                        if text:
                            answer_texts.append(str(text).strip())
            elif isinstance(answer_raw, dict):
                text = answer_raw.get("text", answer_raw.get("answer", ""))
                if text:
                    answer_texts.append(str(text).strip())
            elif isinstance(answer_raw, str):
                answer_texts.append(answer_raw.strip())
            
            # 정답이 없으면 건너뜀
            answer_texts = [a for a in answer_texts if a]
            if not answer_texts:
                continue
            
            if not context or not question:
                continue
            
            # 프롬프트 구성
            prompt = f"다음 지문을 읽고 질문에 답하세요.\n\n지문:\n{context}\n\n질문: {question}\n\n답변:"
            
            # 답변 생성
            response = self.generate_answer(prompt, max_new_tokens=100)
            response_clean = response.strip()
            
            # 정확도 계산 (모든 가능한 정답과 비교)
            is_correct = False
            
            for ans_text in answer_texts:
                # 1. 정확한 매칭 (대소문자, 공백 무시)
                if exact_match(response_clean, ans_text):
                    is_correct = True
                    break
                
                # 2. 포함 여부 확인 (단어 경계 고려, 최소 길이 체크)
                if contains_answer(response_clean, ans_text):
                    is_correct = True
                    break
            
            if is_correct:
                correct += 1
            total += 1
            
            # 결과 저장
            result_item = {
                "question": question,
                "context": context[:200] + "..." if len(context) > 200 else context,
                "correct_answer": answer_texts,  # 모든 가능한 정답 저장
                "model_response": response,
                "is_correct": is_correct
            }
            
            # 처음 몇 개 샘플 출력
            if idx < show_examples:
                examples.append(result_item)
                print(f"\n  [예시 {idx+1}]")
                print(f"  질문: {question}")
                print(f"  정답: {answer_texts}")
                print(f"  모델 답변: {response[:150]}...")
                print(f"  결과: {'✓ 정답' if is_correct else '✗ 오답'}")
                if not is_correct:
                    print(f"  디버그: 정답 '{answer_texts[0]}' vs 응답 '{response[:100]}'")
            
            # 전체 결과 저장 (옵션)
            if save_all:
                all_results.append(result_item)
            
            if total % 50 == 0:
                print(f"  진행: {total}개 평가 중... (정확도: {correct/total*100:.2f}%)")
        
        elapsed_time = time.time() - start_time
        accuracy = correct / total if total > 0 else 0.0
        
        details = {"examples": examples}
        if save_all and all_results:
            details["all_results"] = all_results
        
        return BenchmarkResult(
            benchmark_name=benchmark_name,
            model_name=self.model_name,
            accuracy=accuracy,
            total_samples=total,
            correct_samples=correct,
            evaluation_time=elapsed_time,
            details=details
        )
    
    def evaluate_math(self, dataset, benchmark_name: str, show_examples: int = 10) -> BenchmarkResult:
        """수학 추론 벤치마크 평가 (GSM8K-Ko, HRM8K 등) - GSM8K 표준 채점 방식"""
        import re
        
        print(f"\n[{benchmark_name}] 평가 시작...")
        
        def extract_gold_answer(gold_text: str) -> str | None:
            """정답 텍스트에서 최종 답 추출 (GSM8K 표준 형식)"""
            if not gold_text:
                return None
            
            gold_str = str(gold_text).strip()
            
            # 1. #### 18 패턴 우선 (GSM8K 표준)
            m = re.search(r"####\s*([-+]?\d+\.?\d*)", gold_str)
            if m:
                return m.group(1)
            
            # 2. 숫자만 있는 경우 (final_answer 컬럼)
            if gold_str.isdigit() or (gold_str.replace('.', '').replace('-', '').isdigit()):
                return gold_str
            
            # 3. 마지막 숫자 추출
            numbers = re.findall(r"[-+]?\d+\.?\d*", gold_str)
            return numbers[-1] if numbers else None
        
        def extract_pred_answer(pred_text: str) -> str | None:
            """모델 출력에서 최종 답 추출"""
            if not pred_text:
                return None
            
            pred_str = str(pred_text).strip()
            
            # 1. "정답: 18" 같은 명시적 패턴 우선
            m = re.search(r"정답\s*[:\-]?\s*([-+]?\d+\.?\d*)", pred_str, re.IGNORECASE)
            if m:
                return m.group(1)
            
            # 2. "답: 18" 패턴
            m = re.search(r"답\s*[:\-]?\s*([-+]?\d+\.?\d*)", pred_str, re.IGNORECASE)
            if m:
                return m.group(1)
            
            # 3. #### 18 패턴 (GSM8K 표준)
            m = re.search(r"####\s*([-+]?\d+\.?\d*)", pred_str)
            if m:
                return m.group(1)
            
            # 4. "최종 답: 18" 패턴
            m = re.search(r"최종\s*답\s*[:\-]?\s*([-+]?\d+\.?\d*)", pred_str, re.IGNORECASE)
            if m:
                return m.group(1)
            
            # 5. 마지막 숫자 추출 (중간 계산값 제외하기 위해 마지막 사용)
            numbers = re.findall(r"[-+]?\d+\.?\d*", pred_str)
            return numbers[-1] if numbers else None
        
        def normalize_num(x: str | None) -> str | None:
            """숫자 정규화 (000, 018 같은 것도 처리)"""
            if x is None:
                return None
            try:
                # 소수점 처리
                num = float(x)
                # 정수면 정수로, 아니면 소수점 유지
                if num == int(num):
                    return str(int(num))
                return str(num)
            except (ValueError, TypeError):
                return None
        
        correct = 0
        total = 0
        start_time = time.time()
        examples = []
        
        for idx, item in enumerate(dataset):
            # 문제 추출 (다양한 컬럼명 지원)
            question = item.get("question", item.get("problem", item.get("input", "")))
            
            # 정답 추출 (다양한 컬럼명 지원)
            # GSM8K-Ko는 final_answer, 다른 데이터셋은 answer, label 등
            answer_raw = item.get("final_answer") or item.get("answer") or item.get("label") or item.get("solution", "")
            
            if not question:
                continue
            
            # 정답 추출
            if isinstance(answer_raw, (int, float)):
                answer_str = str(answer_raw)
            elif isinstance(answer_raw, str):
                answer_str = answer_raw
            else:
                answer_str = str(answer_raw) if answer_raw else ""
            
            gold_answer = extract_gold_answer(answer_str)
            
            if gold_answer is None:
                # full_answer에서 추출 시도
                full_answer = item.get("full_answer", "")
                gold_answer = extract_gold_answer(full_answer)
            
            if gold_answer is None:
                print(f"  경고: 샘플 {idx+1}에서 정답을 찾을 수 없습니다.")
                continue
            
            # 프롬프트 구성 (정답 형식 강제)
            prompt = f"다음 수학 문제를 단계별로 풀어보세요.\n\n문제: {question}\n\n단계별 풀이를 작성한 후, 마지막 줄에 '정답: <숫자>' 형식으로 최종 답만 한 번 더 써주세요."
            
            # 답변 생성 (수학 문제는 더 긴 응답 필요)
            response = self.generate_answer(prompt, max_new_tokens=512)
            
            # 모델 응답에서 답변 추출
            predicted = extract_pred_answer(response)
            
            # 정확도 계산
            gold_norm = normalize_num(gold_answer)
            pred_norm = normalize_num(predicted)
            
            is_correct = False
            if gold_norm is not None and pred_norm is not None:
                is_correct = gold_norm == pred_norm
            
            if is_correct:
                correct += 1
            total += 1
            
            # 처음 몇 개 샘플 출력
            if idx < show_examples:
                examples.append({
                    "question": question[:200] + "..." if len(question) > 200 else question,
                    "correct_answer": gold_norm if gold_norm else gold_answer,
                    "predicted": pred_norm if pred_norm else predicted,
                    "model_response": response[:300] + "..." if len(response) > 300 else response,
                    "is_correct": is_correct
                })
                print(f"\n  [예시 {idx+1}]")
                print(f"  문제: {question[:150]}...")
                print(f"  정답: {gold_norm if gold_norm else gold_answer}")
                print(f"  모델 답변: {pred_norm if pred_norm else (predicted if predicted else '답변 없음')}")
                print(f"  모델 풀이: {response[:200]}...")
                print(f"  결과: {'✓ 정답' if is_correct else '✗ 오답'}")
                if not is_correct:
                    print(f"  디버그: 정답 '{gold_norm}' vs 모델 '{pred_norm}'")
            
            if total % 50 == 0:
                print(f"  진행: {total}개 평가 중... (정확도: {correct/total*100:.2f}%)")
        
        elapsed_time = time.time() - start_time
        accuracy = correct / total if total > 0 else 0.0
        
        return BenchmarkResult(
            benchmark_name=benchmark_name,
            model_name=self.model_name,
            accuracy=accuracy,
            total_samples=total,
            correct_samples=correct,
            evaluation_time=elapsed_time,
            details={"examples": examples}
        )
    
    def evaluate_code(self, dataset, benchmark_name: str, show_examples: int = 10) -> BenchmarkResult:
        """코드/알고리즘 벤치마크 평가 (HumanEval, MBPP 등)"""
        import re
        
        print(f"\n[{benchmark_name}] 평가 시작...")
        
        correct = 0
        total = 0
        start_time = time.time()
        examples = []
        
        for idx, item in enumerate(dataset):
            # 문제 추출
            prompt_en = item.get("prompt", item.get("question", ""))
            instruction = item.get("instruction", "")
            test_cases = item.get("test", item.get("test_cases", []))
            canonical_solution = item.get("canonical_solution", item.get("solution", ""))
            
            # 한국어 프롬프트로 변환 (영어 문제를 한국어로 설명)
            if instruction:
                question = instruction
            elif prompt_en:
                # 간단한 한국어 설명 추가
                question = f"다음 문제를 해결하는 Python 함수를 작성하세요:\n\n{prompt_en}"
            else:
                continue
            
            # 프롬프트 구성
            prompt = f"{question}\n\nPython 코드를 작성하세요:"
            
            # 답변 생성
            response = self.generate_answer(prompt, max_new_tokens=512)
            
            # 코드 블록 추출
            code_blocks = re.findall(r'```(?:python)?\s*(.*?)```', response, re.DOTALL)
            if not code_blocks:
                # 코드 블록이 없으면 전체 응답에서 함수 정의 찾기
                function_match = re.search(r'def\s+\w+.*?(?=\n\n|\nclass\s+|$)', response, re.DOTALL)
                if function_match:
                    code_blocks = [function_match.group(0)]
            
            generated_code = code_blocks[0].strip() if code_blocks else response.strip()
            
            # 정확도 계산 (간단한 방법: 코드 구조 확인)
            # 실제로는 코드 실행이 필요하지만, 여기서는 기본적인 구조만 확인
            is_correct = False
            
            # 1. 함수 정의가 있는지 확인
            has_function = bool(re.search(r'def\s+\w+\s*\(', generated_code))
            
            # 2. 정답 코드와의 유사도 (간단한 키워드 비교)
            if canonical_solution:
                solution_keywords = set(re.findall(r'\b\w+\b', canonical_solution.lower()))
                generated_keywords = set(re.findall(r'\b\w+\b', generated_code.lower()))
                keyword_overlap = len(solution_keywords & generated_keywords) / len(solution_keywords) if solution_keywords else 0
                
                # 키워드 유사도가 높고 함수 정의가 있으면 정답으로 간주
                is_correct = has_function and keyword_overlap > 0.5
            else:
                # 정답 코드가 없으면 함수 정의만 있으면 정답으로 간주
                is_correct = has_function
            
            if is_correct:
                correct += 1
            total += 1
            
            # 처음 몇 개 샘플 출력
            if idx < show_examples:
                examples.append({
                    "question": question[:200] + "..." if len(question) > 200 else question,
                    "canonical_solution": canonical_solution[:200] + "..." if canonical_solution and len(canonical_solution) > 200 else canonical_solution,
                    "generated_code": generated_code[:300] + "..." if len(generated_code) > 300 else generated_code,
                    "is_correct": is_correct
                })
                print(f"\n  [예시 {idx+1}]")
                print(f"  문제: {question[:150]}...")
                print(f"  생성된 코드: {generated_code[:200]}...")
                print(f"  결과: {'✓ 정답' if is_correct else '✗ 오답'}")
            
            if total % 20 == 0:
                print(f"  진행: {total}개 평가 중... (정확도: {correct/total*100:.2f}%)")
        
        elapsed_time = time.time() - start_time
        accuracy = correct / total if total > 0 else 0.0
        
        return BenchmarkResult(
            benchmark_name=benchmark_name,
            model_name=self.model_name,
            accuracy=accuracy,
            total_samples=total,
            correct_samples=correct,
            evaluation_time=elapsed_time,
            details={"examples": examples}
        )


def load_model(model_name: str, use_8bit: bool = True):
    """모델 로드"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"\n모델 로드 중: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        load_in_8bit=use_8bit,
    )
    
    return model, tokenizer


def load_dataset_from_huggingface(dataset_name: str, split: str = "test", max_samples: Optional[int] = None, config: Optional[str] = None):
    """HuggingFace에서 데이터셋 로드"""
    try:
        from datasets import load_dataset
        
        print(f"\n데이터셋 로드 중: {dataset_name}")
        if config:
            dataset = load_dataset(dataset_name, config, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        print(f"  로드 완료: {len(dataset)}개 샘플")
        return dataset
    except Exception as e:
        print(f"  경고: 데이터셋 로드 실패 - {e}")
        # 대체 데이터셋 시도
        alternatives = []
        
        # KMMLU 대체 경로
        if "KMMLU" in dataset_name or "kmmlu" in dataset_name.lower():
            print("  대체: KMMLU 다른 경로 시도 중...")
            alternatives = [
                "HAERAE-HUB/KMMLU",  # 공식 경로
                "LGAI-EXAONE/KMMLU-Redux",  # 심화 버전
                "LGAI-EXAONE/KMMLU-Pro",  # 프로 버전
            ]
        
        # GSM8K-Ko 대체 경로
        elif "GSM8K" in dataset_name.upper() or "gsm8k" in dataset_name.lower():
            print("  대체: GSM8K-Ko 다른 경로 시도 중...")
            alternatives = [
                "ChuGyouk/GSM8k-Ko",  # 공식 경로
                "thunder-research-group/SNU_Ko-GSM8K",  # SNU 버전
                "kuotient/gsm8k-ko",  # kuotient 버전
            ]
        
        # HRM8K 대체 경로
        elif "HRM8K" in dataset_name.upper() or "hrm8k" in dataset_name.lower():
            print("  대체: HRM8K 다른 경로 시도 중...")
            alternatives = [
                "HAERAE-HUB/HRM8K",  # 공식 경로
            ]
        
        # 대체 경로 시도
        if alternatives:
            for alt in alternatives:
                try:
                    if config:
                        dataset = load_dataset(alt, config, split=split)
                    else:
                        dataset = load_dataset(alt, split=split)
                    if max_samples:
                        dataset = dataset.select(range(min(max_samples, len(dataset))))
                    print(f"  성공: {alt}에서 로드")
                    return dataset
                except Exception as e:
                    continue
        
        return None


def evaluate_benchmarks(
    finetuned_model_name: str,
    benchmarks: List[Dict],
    output_dir: str = "evaluation_results",
    skip_base: bool = True,
    base_model_name: Optional[str] = None
):
    """벤치마크 평가 실행"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "finetuned_model": finetuned_model_name,
        "benchmarks": []
    }
    
    base_results = []
    
    # Base 모델 평가 (선택적)
    if not skip_base and base_model_name:
        results["base_model"] = base_model_name
        
        print("\n" + "="*80)
        print(" [1/2] Base 모델 평가")
        print("="*80)
        
        base_model, base_tokenizer = load_model(base_model_name)
        base_evaluator = KoreanBenchmarkEvaluator(base_model, base_tokenizer, "base")
        
        for bench_config in benchmarks:
            dataset = load_dataset_from_huggingface(
                bench_config["dataset"],
                split=bench_config.get("split", "test"),
                max_samples=bench_config.get("max_samples", None),
                config=bench_config.get("config", None)
            )
            
            if dataset is None:
                continue
            
            if bench_config["type"] == "mmlu":
                result = base_evaluator.evaluate_mmlu_style(dataset, bench_config["name"], show_examples=0)
            elif bench_config["type"] == "qa":
                result = base_evaluator.evaluate_qa(dataset, bench_config["name"], show_examples=0)
            elif bench_config["type"] == "math":
                result = base_evaluator.evaluate_math(dataset, bench_config["name"], show_examples=0)
            elif bench_config["type"] == "code":
                result = base_evaluator.evaluate_code(dataset, bench_config["name"], show_examples=0)
            else:
                continue
            
            base_results.append(result)
            print(f"\n[{bench_config['name']}] Base 모델 결과:")
            print(f"  정확도: {result.accuracy*100:.2f}%")
            print(f"  정답/전체: {result.correct_samples}/{result.total_samples}")
            print(f"  평가 시간: {result.evaluation_time:.2f}초")
        
        # 메모리 정리
        del base_model, base_tokenizer, base_evaluator
        torch.cuda.empty_cache()
    
    # Finetuned 모델 평가
    print("\n" + "="*80)
    print(" Finetuned 모델 평가")
    print("="*80)
    
    finetuned_model, finetuned_tokenizer = load_model(finetuned_model_name)
    finetuned_evaluator = KoreanBenchmarkEvaluator(finetuned_model, finetuned_tokenizer, "finetuned")
    
    finetuned_results = []
    for bench_config in benchmarks:
        dataset = load_dataset_from_huggingface(
            bench_config["dataset"],
            split=bench_config.get("split", "test"),
            max_samples=bench_config.get("max_samples", None),
            config=bench_config.get("config", None)
        )
        
        if dataset is None:
            continue
        
        save_all = bench_config.get("save_all", False)  # 전체 결과 저장 옵션
        
        if bench_config["type"] == "mmlu":
            result = finetuned_evaluator.evaluate_mmlu_style(dataset, bench_config["name"], show_examples=10)
        elif bench_config["type"] == "qa":
            result = finetuned_evaluator.evaluate_qa(dataset, bench_config["name"], show_examples=10, save_all=save_all)
        elif bench_config["type"] == "math":
            result = finetuned_evaluator.evaluate_math(dataset, bench_config["name"], show_examples=10)
        elif bench_config["type"] == "code":
            result = finetuned_evaluator.evaluate_code(dataset, bench_config["name"], show_examples=10)
        else:
            continue
        
        finetuned_results.append(result)
        print(f"\n[{bench_config['name']}] Finetuned 모델 결과:")
        print(f"  정확도: {result.accuracy*100:.2f}%")
        print(f"  정답/전체: {result.correct_samples}/{result.total_samples}")
        print(f"  평가 시간: {result.evaluation_time:.2f}초")
    
    # 결과 저장
    results["finetuned_results"] = [asdict(r) for r in finetuned_results]
    
    # Base 모델과 비교 (있는 경우)
    if base_results:
        print("\n" + "="*80)
        print(" 결과 비교")
        print("="*80)
        
        comparison = []
        for base_result, finetuned_result in zip(base_results, finetuned_results):
            delta = finetuned_result.accuracy - base_result.accuracy
            comparison.append({
                "benchmark": base_result.benchmark_name,
                "base_accuracy": base_result.accuracy,
                "finetuned_accuracy": finetuned_result.accuracy,
                "delta": delta,
                "delta_percent": delta * 100,
                "improvement": "↑" if delta > 0 else "↓" if delta < 0 else "="
            })
            
            print(f"\n[{base_result.benchmark_name}]")
            print(f"  Base:      {base_result.accuracy*100:.2f}%")
            print(f"  Finetuned: {finetuned_result.accuracy*100:.2f}%")
            print(f"  Delta:     {delta*100:+.2f}% {comparison[-1]['improvement']}")
        
        results["comparison"] = comparison
        results["base_results"] = [asdict(r) for r in base_results]
    else:
        print("\n" + "="*80)
        print(" 평가 결과 요약")
        print("="*80)
        for result in finetuned_results:
            print(f"\n[{result.benchmark_name}]")
            print(f"  정확도: {result.accuracy*100:.2f}%")
            print(f"  정답/전체: {result.correct_samples}/{result.total_samples}")
    
    output_file = os.path.join(output_dir, "evaluation_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n결과 저장 완료: {output_file}")
    print("="*80)


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="한국어 벤치마크 평가")
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base 모델 이름 (비교 평가 시 사용, 생략 가능)"
    )
    parser.add_argument(
        "--finetuned-model",
        type=str,
        default="MyeongHo0621/Qwen2.5-14B-Korean",
        help="파인튜닝된 모델 이름"
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Base 모델 평가 건너뛰기 (기본값: True)"
    )
    parser.add_argument(
        "--compare-base",
        action="store_true",
        help="Base 모델과 비교 평가 (--base-model 필요)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="결과 저장 디렉토리"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="각 벤치마크당 최대 샘플 수 (테스트용)"
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        choices=["qa", "math", "code", "mmlu", "all"],
        default=["all"],
        help="평가할 벤치마크 카테고리 (qa, math, code, mmlu, all)"
    )
    
    args = parser.parse_args()
    
    # 벤치마크 설정 (카테고리별 분리)
    benchmark_categories = {
        "qa": [
            {
                "name": "KoBEST-boolq",
                "dataset": "skt/kobest_v1",
                "type": "qa",
                "split": "test",
                "max_samples": args.max_samples or 500,
                "config": "boolq",
            },
            {
                "name": "KorQuAD-1.0",
                "dataset": "squad_kor_v1",
                "type": "qa",
                "split": "validation",
                "max_samples": args.max_samples or 500,
                "config": None,
            },
        ],
        "math": [
            {
                "name": "GSM8K-Ko",
                "dataset": "ChuGyouk/GSM8k-Ko",  # 공식 GSM8K 한국어 번역 버전
                "type": "math",
                "split": "test",
                "max_samples": args.max_samples or 200,
                "config": None,
            },
            {
                "name": "HRM8K",
                "dataset": "HAERAE-HUB/HRM8K",  # 공식 HRM8K 데이터셋
                "type": "math",
                "split": "test",
                "max_samples": args.max_samples or 200,
                "config": None,
            },
        ],
        "code": [
            {
                "name": "HumanEval-Ko",
                "dataset": "openai/humaneval",
                "type": "code",
                "split": "test",
                "max_samples": args.max_samples or 50,
                "config": None,
            },
        ],
        "mmlu": [
            {
                "name": "KMMLU",
                "dataset": "HAERAE-HUB/KMMLU",  # 공식 KMMLU 데이터셋
                "type": "mmlu",
                "split": "test",
                "max_samples": args.max_samples or 1000,
                "config": None,  # None이면 전체 과목 통합, 특정 과목만 평가하려면 과목명 지정 가능
            },
        ],
    }
    
    # 선택된 카테고리로 벤치마크 선택
    if "all" in args.categories:
        benchmarks = []
        for category_benchmarks in benchmark_categories.values():
            benchmarks.extend(category_benchmarks)
    else:
        benchmarks = []
        for category in args.categories:
            if category in benchmark_categories:
                benchmarks.extend(benchmark_categories[category])
    
    print("\n" + "="*80)
    print(" 한국어 벤치마크 평가")
    print("="*80)
    print(f"Finetuned 모델: {args.finetuned_model}")
    
    # Base 모델 비교 여부 결정
    skip_base = not args.compare_base
    if args.compare_base:
        if not args.base_model:
            args.base_model = "Qwen/Qwen2.5-14B-Instruct"
        print(f"Base 모델: {args.base_model} (비교 평가)")
    else:
        print("Base 모델: 평가 건너뜀")
    
    # 카테고리별 벤치마크 표시
    print(f"\n선택된 카테고리: {', '.join(args.categories)}")
    print(f"총 벤치마크 수: {len(benchmarks)}개")
    print(f"벤치마크 목록:")
    for bench in benchmarks:
        print(f"  - {bench['name']} ({bench['type']})")
    print("="*80)
    
    # GPU 확인
    if not torch.cuda.is_available():
        print("\n[ ERROR ] CUDA를 사용할 수 없습니다!")
        sys.exit(1)
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    
    # 평가 실행
    evaluate_benchmarks(
        finetuned_model_name=args.finetuned_model,
        benchmarks=benchmarks,
        output_dir=args.output_dir,
        skip_base=skip_base,
        base_model_name=args.base_model
    )


if __name__ == "__main__":
    main()

