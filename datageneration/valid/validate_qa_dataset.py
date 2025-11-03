#!/usr/bin/env python3
"""
WMS QA 데이터셋 전용 품질 검증기
0_gen_qa_dtset_v2.py로 생성된 JSONL 데이터셋 검증
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any
import re

class WMSQADatasetValidator:
    """WMS QA 데이터셋 품질 검증기"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.data = []
        
    def load_dataset(self):
        """JSONL 형식 데이터셋 로드"""
        print(f"\n{'='*80}")
        print(f"데이터셋 로딩: {self.dataset_path}")
        print(f"{'='*80}\n")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        print(f"✓ {len(self.data):,}개 샘플 로드 완료\n")
    
    def validate_structure(self) -> Dict[str, Any]:
        """데이터 구조 검증"""
        print(f"\n{'='*80}")
        print("1. 데이터 구조 검증")
        print(f"{'='*80}\n")
        
        issues = {
            'missing_messages': 0,
            'wrong_message_count': 0,
            'missing_system': 0,
            'missing_user': 0,
            'missing_assistant': 0,
            'missing_metadata': 0
        }
        
        for idx, item in enumerate(self.data):
            # messages 존재 확인
            if 'messages' not in item:
                issues['missing_messages'] += 1
                continue
            
            messages = item['messages']
            
            # messages 개수 확인 (system + user + assistant = 3)
            if len(messages) != 3:
                issues['wrong_message_count'] += 1
            
            # 각 role 확인
            roles = [msg.get('role') for msg in messages]
            if 'system' not in roles:
                issues['missing_system'] += 1
            if 'user' not in roles:
                issues['missing_user'] += 1
            if 'assistant' not in roles:
                issues['missing_assistant'] += 1
            
            # metadata 확인
            if 'metadata' not in item:
                issues['missing_metadata'] += 1
        
        print("구조 검증 결과:")
        for issue, count in issues.items():
            status = "✓" if count == 0 else "⚠️"
            print(f"  {status} {issue}: {count}건")
        
        return issues
    
    def analyze_content_quality(self, sample_size: int = 1000) -> Dict[str, Any]:
        """내용 품질 분석"""
        print(f"\n{'='*80}")
        print("2. 내용 품질 분석")
        print(f"{'='*80}\n")
        
        sample_data = self.data[:sample_size]
        
        questions = []
        answers = []
        
        for item in sample_data:
            messages = item.get('messages', [])
            for msg in messages:
                if msg.get('role') == 'user':
                    questions.append(msg.get('content', ''))
                elif msg.get('role') == 'assistant':
                    answers.append(msg.get('content', ''))
        
        # 질문 분석
        q_lengths = [len(q) for q in questions]
        q_korean_ratio = self._calculate_korean_ratio(questions)
        
        # 답변 분석
        a_lengths = [len(a) for a in answers]
        a_korean_ratio = self._calculate_korean_ratio(answers)
        
        # WMS 관련 키워드 분석
        wms_keywords = self._analyze_wms_keywords(questions + answers)
        
        # 답변 품질 지표
        quality_issues = self._detect_quality_issues(answers)
        
        metrics = {
            'sample_size': len(sample_data),
            'question_stats': {
                'avg_length': np.mean(q_lengths),
                'min_length': min(q_lengths),
                'max_length': max(q_lengths),
                'korean_ratio': q_korean_ratio
            },
            'answer_stats': {
                'avg_length': np.mean(a_lengths),
                'min_length': min(a_lengths),
                'max_length': max(a_lengths),
                'korean_ratio': a_korean_ratio
            },
            'wms_keywords': wms_keywords,
            'quality_issues': quality_issues
        }
        
        print("질문 통계:")
        print(f"  평균 길이: {metrics['question_stats']['avg_length']:.1f} 자")
        print(f"  범위: {metrics['question_stats']['min_length']} ~ {metrics['question_stats']['max_length']} 자")
        print(f"  한국어 비율: {metrics['question_stats']['korean_ratio']:.1%}")
        
        print("\n답변 통계:")
        print(f"  평균 길이: {metrics['answer_stats']['avg_length']:.1f} 자")
        print(f"  범위: {metrics['answer_stats']['min_length']} ~ {metrics['answer_stats']['max_length']} 자")
        print(f"  한국어 비율: {metrics['answer_stats']['korean_ratio']:.1%}")
        
        print("\nWMS 관련 키워드 빈도 (Top 10):")
        for keyword, count in wms_keywords.most_common(10):
            print(f"  {keyword}: {count}회")
        
        print("\n품질 이슈:")
        for issue, count in quality_issues.items():
            print(f"  {issue}: {count}건")
        
        return metrics
    
    def analyze_diversity(self) -> Dict[str, Any]:
        """데이터 다양성 분석"""
        print(f"\n{'='*80}")
        print("3. 데이터 다양성 분석")
        print(f"{'='*80}\n")
        
        # Topic 분포
        topics = []
        personas = []
        
        for item in self.data:
            metadata = item.get('metadata', {})
            if 'topic' in metadata:
                topics.append(metadata['topic'])
            if 'persona' in metadata:
                personas.append(metadata['persona'])
        
        topic_dist = Counter(topics)
        persona_dist = Counter(personas)
        
        # 질문 중복도
        questions = []
        for item in self.data:
            for msg in item.get('messages', []):
                if msg.get('role') == 'user':
                    questions.append(msg.get('content', ''))
        
        unique_questions = len(set(questions))
        duplicate_ratio = 1 - (unique_questions / len(questions)) if questions else 0
        
        metrics = {
            'total_topics': len(topic_dist),
            'total_personas': len(persona_dist),
            'top_10_topics': topic_dist.most_common(10),
            'top_10_personas': persona_dist.most_common(10),
            'unique_questions': unique_questions,
            'total_questions': len(questions),
            'duplicate_ratio': duplicate_ratio
        }
        
        print(f"토픽 다양성: {metrics['total_topics']}개")
        print(f"페르소나 다양성: {metrics['total_personas']}개")
        print(f"\n토픽 분포 (Top 10):")
        for topic, count in metrics['top_10_topics']:
            print(f"  {topic[:60]}: {count}건")
        
        print(f"\n질문 중복도: {duplicate_ratio:.2%}")
        print(f"  고유 질문: {unique_questions:,}개 / 전체: {len(questions):,}개")
        
        return metrics
    
    def analyze_instruction_tuning_quality(self) -> Dict[str, Any]:
        """Instruction Tuning 품질 분석"""
        print(f"\n{'='*80}")
        print("4. Instruction Tuning 품질 분석")
        print(f"{'='*80}\n")
        
        system_prompts = []
        qa_pairs = []
        
        for item in self.data:
            messages = item.get('messages', [])
            
            system_msg = None
            user_msg = None
            assistant_msg = None
            
            for msg in messages:
                if msg.get('role') == 'system':
                    system_msg = msg.get('content', '')
                elif msg.get('role') == 'user':
                    user_msg = msg.get('content', '')
                elif msg.get('role') == 'assistant':
                    assistant_msg = msg.get('content', '')
            
            if system_msg:
                system_prompts.append(system_msg)
            if user_msg and assistant_msg:
                qa_pairs.append((user_msg, assistant_msg))
        
        # System prompt 일관성
        unique_systems = len(set(system_prompts))
        
        # 답변 스타일 분석
        formal_count = 0
        informal_count = 0
        
        for _, answer in qa_pairs[:1000]:
            if any(word in answer for word in ['입니다', '습니다', '합니다']):
                formal_count += 1
            if any(word in answer for word in ['이야', '거야', '이지', '해봐']):
                informal_count += 1
        
        # 답변 길이 일관성
        answer_lengths = [len(a) for _, a in qa_pairs]
        length_std = np.std(answer_lengths)
        
        metrics = {
            'total_system_prompts': len(system_prompts),
            'unique_system_prompts': unique_systems,
            'formal_style_ratio': formal_count / len(qa_pairs[:1000]) if qa_pairs else 0,
            'informal_style_ratio': informal_count / len(qa_pairs[:1000]) if qa_pairs else 0,
            'answer_length_std': length_std,
            'avg_answer_length': np.mean(answer_lengths) if answer_lengths else 0
        }
        
        print(f"System Prompt 일관성:")
        print(f"  총 {metrics['total_system_prompts']:,}개 중 {metrics['unique_system_prompts']}개 고유")
        
        print(f"\n답변 스타일:")
        print(f"  격식체(~습니다): {metrics['formal_style_ratio']:.1%}")
        print(f"  구어체(~이야): {metrics['informal_style_ratio']:.1%}")
        
        print(f"\n답변 길이 일관성:")
        print(f"  평균: {metrics['avg_answer_length']:.1f} 자")
        print(f"  표준편차: {metrics['answer_length_std']:.1f}")
        
        return metrics
    
    def _calculate_korean_ratio(self, texts: List[str]) -> float:
        """한국어 비율 계산"""
        korean_chars = 0
        total_chars = 0
        
        for text in texts:
            for char in text:
                total_chars += 1
                if '\uAC00' <= char <= '\uD7A3':
                    korean_chars += 1
        
        return korean_chars / total_chars if total_chars > 0 else 0
    
    def _analyze_wms_keywords(self, texts: List[str]) -> Counter:
        """WMS 관련 키워드 분석"""
        wms_keywords = [
            'WMS', '창고', '재고', '물류', '입고', '출고', '피킹', 'AGV',
            '로봇', '자동화', 'ERP', 'TMS', 'RFID', '바코드', '실시간',
            '통합', '최적화', '효율', '관리', 'AI', '머신러닝', '예측'
        ]
        
        keyword_counts = Counter()
        
        for text in texts:
            for keyword in wms_keywords:
                keyword_counts[keyword] += text.count(keyword)
        
        return keyword_counts
    
    def _detect_quality_issues(self, answers: List[str]) -> Dict[str, int]:
        """답변 품질 이슈 감지"""
        issues = {
            'too_short': 0,  # <50자
            'too_long': 0,   # >2000자
            'repetitive': 0,  # 반복 패턴
            'no_korean': 0,   # 한국어 없음
            'has_reference': 0  # "참고자료", "문서" 언급
        }
        
        for answer in answers:
            if len(answer) < 50:
                issues['too_short'] += 1
            if len(answer) > 2000:
                issues['too_long'] += 1
            if re.search(r'(.{10,})\1{2,}', answer):
                issues['repetitive'] += 1
            if self._calculate_korean_ratio([answer]) < 0.3:
                issues['no_korean'] += 1
            if any(word in answer for word in ['참고자료', '참고 자료', '문서', '[참고']):
                issues['has_reference'] += 1
        
        return issues
    
    def generate_report(self, output_file: str = "qa_validation_report.txt"):
        """종합 리포트 생성"""
        print(f"\n{'='*80}")
        print("5. 종합 리포트 생성")
        print(f"{'='*80}\n")
        
        report = f"""
WMS QA 데이터셋 품질 검증 리포트
데이터셋: {self.dataset_path}
총 샘플 수: {len(self.data):,}개
생성 일시: {Path(self.dataset_path).stat().st_mtime}

{'='*80}
검증 요약
{'='*80}

✓ 데이터 구조 검증 완료
✓ 내용 품질 분석 완료
✓ 다양성 분석 완료
✓ Instruction Tuning 품질 분석 완료

권장사항:
1. 한국어 비율이 낮은 샘플 필터링 검토
2. 중복 질문 제거로 다양성 향상
3. 답변 길이 표준편차 관리로 일관성 확보
4. "참고자료" 언급 샘플 제거 또는 수정

상세 분석 결과는 위 섹션을 참고하세요.
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✓ 리포트 저장: {output_file}")
        
        return report


def main():
    """실행"""
    import sys
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = "/home/work/tesseract/datageneration/instruction/output/checkpoint_20000_20251013_133415.jsonl"
    
    print(f"\n{'#'*80}")
    print("WMS QA 데이터셋 품질 검증기")
    print(f"{'#'*80}")
    
    validator = WMSQADatasetValidator(dataset_path)
    validator.load_dataset()
    
    # 검증 실행
    validator.validate_structure()
    validator.analyze_content_quality(sample_size=1000)
    validator.analyze_diversity()
    validator.analyze_instruction_tuning_quality()
    validator.generate_report()
    
    print(f"\n{'#'*80}")
    print("검증 완료!")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()

