#!/usr/bin/env python3
"""
4가지 모델 설정 비교 테스트
1. EXAONE-4.0 (RAG 없음)
2. EXAONE-4.0 + RAG
3. Qwen2.5-7B (RAG 없음)
4. Qwen2.5-7B + RAG
"""

import torch
import json
import random
import faiss
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import time


def load_topics(topics_file: str) -> list:
    """토픽 로드"""
    with open(topics_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('topics', [])


def convert_topic_to_question(topic: str) -> str:
    """토픽을 자연스러운 질문으로 변환"""
    if topic.endswith('?') or '어떻게' in topic or '방법' in topic:
        return topic
    
    question_patterns = [
        f"{topic}에 대해 자세히 설명해주세요.",
        f"{topic}를 효과적으로 구현하는 방법은?",
        f"{topic} 도입 시 고려사항과 실무 팁을 알려주세요.",
        f"{topic}의 장단점과 실제 적용 사례를 설명해주세요."
    ]
    
    if len(topic) < 10:
        return question_patterns[0]
    elif '최적화' in topic or '관리' in topic:
        return question_patterns[1]
    elif '도입' in topic or '구축' in topic:
        return question_patterns[2]
    else:
        return question_patterns[3]


def evaluate_answer_quality(answer: str, topic: str) -> float:
    """답변 품질 평가 (0-10점)"""
    score = 5.0
    
    if len(answer) < 100:
        score -= 2
    elif 200 <= len(answer) <= 800:
        score += 1
    elif len(answer) > 1000:
        score -= 0.5
    
    if any(word in answer for word in ['첫째', '둘째', '셋째', '먼저', '다음으로', '마지막으로']):
        score += 1
    
    if '\n' in answer or '  ' in answer:
        score += 0.5
    
    technical_terms = ['시스템', '프로세스', '최적화', '효율', '관리', '자동화', 
                       'WMS', '재고', '물류', '데이터', '실시간', '통합']
    term_count = sum(1 for term in technical_terms if term in answer)
    score += min(term_count * 0.2, 2)
    
    topic_words = topic.split()
    relevance = sum(1 for word in topic_words if word in answer and len(word) > 1)
    score += min(relevance * 0.3, 1.5)
    
    if '모르겠' in answer or '잘 모름' in answer:
        score -= 2
    if len(answer) < 50:
        score -= 3
    if answer.count('.') < 2:
        score -= 1
    
    return max(0, min(10, score))


class RAGSystem:
    """FAISS 기반 RAG 시스템"""
    
    def __init__(self, faiss_path: str = "/home/work/tesseract/faiss_storage"):
        print(f"  RAG 시스템 로딩 중... ({faiss_path})")
        
        faiss_dir = Path(faiss_path)
        
        # FAISS 인덱스 로드
        index_file = faiss_dir / "warehouse_automation_knowledge.index"
        self.index = faiss.read_index(str(index_file))
        
        # 문서 로드
        with open(faiss_dir / "documents.json", 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        
        with open(faiss_dir / "metadata.json", 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # 임베딩 모델
        self.embedding_model = SentenceTransformer(
            "jhgan/ko-sroberta-multitask",
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        print(f"  ✓ RAG 로드 완료: {len(self.documents)}개 문서")
    
    def retrieve_context(self, question: str, k: int = 3) -> list:
        """질문과 관련된 컨텍스트 검색"""
        # 임베딩 생성
        query_embedding = self.embedding_model.encode(
            question,
            convert_to_numpy=True
        ).reshape(1, -1).astype('float32')
        
        # FAISS 검색
        distances, indices = self.index.search(query_embedding, k)
        
        contexts = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                contexts.append({
                    'content': self.documents[idx][:500],  # 길이 제한
                    'distance': float(distance)
                })
        
        return contexts
    
    def create_rag_prompt(self, question: str, contexts: list) -> str:
        """RAG 프롬프트 생성"""
        context_text = "\n\n".join([
            f"참고정보 {i+1}:\n{ctx['content']}"
            for i, ctx in enumerate(contexts)
        ])
        
        prompt = f"""다음 참고정보를 바탕으로 질문에 답변하세요.

{context_text}

질문: {question}

위 정보를 활용하되, "참고자료" 같은 표현은 사용하지 말고 자연스럽게 설명하세요.
답변:"""
        
        return prompt


def test_single_model(
    model_name: str,
    topics: list,
    use_rag: bool = False,
    rag_system: RAGSystem = None,
    num_samples: int = 5,
    seed: int = 42
):
    """단일 모델 테스트"""
    
    config_name = f"{model_name.split('/')[-1]} {'+ RAG' if use_rag else '(RAG 없음)'}"
    
    print(f"\n{'='*80}")
    print(f"테스트: {config_name}")
    print(f"{'='*80}\n")
    
    # 모델 로딩
    print("모델 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"✓ 모델 로드 완료")
    print(f"✓ 메모리: {model.get_memory_footprint() / 1024**3:.2f} GB\n")
    
    # 샘플링 (동일한 질문)
    random.seed(seed)
    sampled_topics = random.sample(topics, min(num_samples, len(topics)))
    
    generation_config = {
        "max_new_tokens": 800,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }
    
    results = []
    total_time = 0
    
    for i, topic in enumerate(sampled_topics, 1):
        question = convert_topic_to_question(topic)
        
        print(f"질문 {i}/{num_samples}: {topic}")
        
        # RAG 사용 여부에 따라 프롬프트 생성
        if use_rag and rag_system:
            contexts = rag_system.retrieve_context(question)
            rag_prompt = rag_system.create_rag_prompt(question, contexts)
            
            # 디버깅: 프롬프트 출력
            if i == 1:  # 첫 번째 질문만
                print(f"\n{'='*80}")
                print("RAG 프롬프트 확인:")
                print(f"{'='*80}")
                print(rag_prompt[:500] + "..." if len(rag_prompt) > 500 else rag_prompt)
                print(f"{'='*80}\n")
            
            # Chat template에 RAG 프롬프트 넣기 시도
            messages = [
                {
                    "role": "system",
                    "content": "당신은 10년 경력의 WMS 및 물류 자동화 전문가입니다."
                },
                {"role": "user", "content": rag_prompt}
            ]
            
            try:
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(model.device)
                input_text = None
            except:
                # Chat template 실패 시 직접 텍스트 사용
                input_text = rag_prompt
        else:
            # Chat template 사용
            messages = [
                {
                    "role": "system",
                    "content": "당신은 10년 경력의 WMS 및 물류 자동화 전문가입니다."
                },
                {"role": "user", "content": question}
            ]
            
            try:
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(model.device)
                input_text = None
                
                # 디버깅: 첫 번째 질문의 프롬프트 출력
                if i == 1:
                    decoded_prompt = tokenizer.decode(input_ids[0])
                    print(f"\n{'='*80}")
                    print("Chat Template 프롬프트 확인:")
                    print(f"{'='*80}")
                    print(decoded_prompt[:500] + "..." if len(decoded_prompt) > 500 else decoded_prompt)
                    print(f"{'='*80}\n")
            except:
                input_text = f"질문: {question}\n\n답변:"
        
        # 토큰화
        if input_text:
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        
        # 생성
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(input_ids, **generation_config)
        
        elapsed = time.time() - start_time
        total_time += elapsed
        
        # 디코딩
        answer = tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        quality_score = evaluate_answer_quality(answer, topic)
        
        print(f"  ✓ 완료 - 길이: {len(answer)}자, 시간: {elapsed:.1f}s, 점수: {quality_score:.1f}/10\n")
        
        results.append({
            'topic': topic,
            'question': question,
            'answer': answer,
            'length': len(answer),
            'time': elapsed,
            'quality_score': quality_score,
            'use_rag': use_rag
        })
    
    # 통계
    avg_length = sum(r['length'] for r in results) / len(results)
    avg_time = total_time / len(results)
    avg_quality = sum(r['quality_score'] for r in results) / len(results)
    
    print(f"{'='*80}")
    print(f"[{config_name}] 통계")
    print(f"{'='*80}")
    print(f"평균 답변 길이: {avg_length:.0f}자")
    print(f"평균 생성 시간: {avg_time:.2f}초")
    print(f"평균 품질 점수: {avg_quality:.2f}/10")
    print(f"{'='*80}\n")
    
    # 메모리 정리
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        'config_name': config_name,
        'model_name': model_name,
        'use_rag': use_rag,
        'avg_length': avg_length,
        'avg_time': avg_time,
        'avg_quality': avg_quality,
        'results': results
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='4가지 모델 설정 비교')
    parser.add_argument('--topics-file', type=str,
                        default='expanded_data/topics_200_mixed.json')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='각 설정당 테스트 질문 수')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='model_comparison.json')
    parser.add_argument('--skip-qwen', action='store_true',
                        help='Qwen 테스트 스킵 (시간 절약)')
    
    args = parser.parse_args()
    
    # 토픽 로드
    topics_path = Path(args.topics_file)
    if not topics_path.exists():
        topics_path = Path(__file__).parent / args.topics_file
    
    topics = load_topics(str(topics_path))
    print(f"\n✓ {len(topics)}개 토픽 로드됨\n")
    
    # RAG 시스템 초기화 (재사용)
    print("="*80)
    print("RAG 시스템 초기화")
    print("="*80)
    rag_system = RAGSystem()
    
    all_results = []
    
    # 1. EXAONE-4.0 (RAG 없음)
    print(f"\n{'#'*80}")
    print("1/4: EXAONE-4.0 (RAG 없음)")
    print(f"{'#'*80}")
    result1 = test_single_model(
        model_name="LGAI-EXAONE/EXAONE-4.0-1.2B",
        topics=topics,
        use_rag=False,
        num_samples=args.num_samples,
        seed=args.seed
    )
    all_results.append(result1)
    
    # 2. EXAONE-4.0 + RAG
    print(f"\n{'#'*80}")
    print("2/4: EXAONE-4.0 + RAG")
    print(f"{'#'*80}")
    result2 = test_single_model(
        model_name="LGAI-EXAONE/EXAONE-4.0-1.2B",
        topics=topics,
        use_rag=True,
        rag_system=rag_system,
        num_samples=args.num_samples,
        seed=args.seed
    )
    all_results.append(result2)
    
    if not args.skip_qwen:
        # 3. Qwen2.5-7B (RAG 없음)
        print(f"\n{'#'*80}")
        print("3/4: Qwen2.5-7B (RAG 없음)")
        print(f"{'#'*80}")
        result3 = test_single_model(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            topics=topics,
            use_rag=False,
            num_samples=args.num_samples,
            seed=args.seed
        )
        all_results.append(result3)
        
        # 4. Qwen2.5-7B + RAG
        print(f"\n{'#'*80}")
        print("4/4: Qwen2.5-7B + RAG")
        print(f"{'#'*80}")
        result4 = test_single_model(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            topics=topics,
            use_rag=True,
            rag_system=rag_system,
            num_samples=args.num_samples,
            seed=args.seed
        )
        all_results.append(result4)
    
    # 최종 비교 리포트
    print(f"\n\n{'#'*80}")
    print("최종 비교 결과")
    print(f"{'#'*80}\n")
    
    print(f"{'설정':<30} {'평균 길이':<12} {'평균 시간':<12} {'평균 품질':<12}")
    print("="*80)
    
    for result in all_results:
        print(f"{result['config_name']:<30} "
              f"{result['avg_length']:<12.0f} "
              f"{result['avg_time']:<12.2f} "
              f"{result['avg_quality']:<12.2f}")
    
    print("="*80)
    
    # 승자 결정
    best_quality = max(all_results, key=lambda x: x['avg_quality'])
    best_speed = min(all_results, key=lambda x: x['avg_time'])
    
    print(f"\n🏆 최고 품질: {best_quality['config_name']} ({best_quality['avg_quality']:.2f}/10)")
    print(f"⚡ 최고 속도: {best_speed['config_name']} ({best_speed['avg_time']:.2f}초)")
    
    # 결과 저장
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 결과 저장: {args.output}\n")


if __name__ == "__main__":
    main()

