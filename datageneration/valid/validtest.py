#!/usr/bin/env python3
"""
한국어 데이터셋 품질 검증 도구
Hugging Face 한국어 언어 데이터셋을 위한 종합적인 검증 도구
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import json
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

try:
    import seaborn as sns
except (ImportError, AttributeError) as e:
    print(f"Seaborn 로드 실패 (호환성 문제): {e}")
    print("시각화 기능 일부가 제한될 수 있습니다.")
    sns = None

# 필요한 패키지 설치 명령어
# pip install datasets pandas matplotlib seaborn wordcloud konlpy evaluate

try:
    from konlpy.tag import Okt
    from konlpy.utils import pprint
except ImportError:
    print("KoNLPy가 설치되지 않았습니다. 다음 명령어로 설치하세요: pip install konlpy")
    Okt = None

try:
    import evaluate
except ImportError:
    print("Evaluate 라이브러리가 설치되지 않았습니다. 다음 명령어로 설치하세요: pip install evaluate")
    evaluate = None

class KoreanDatasetValidator:
    """
    한국어 언어 데이터셋을 위한 종합적인 검증 도구
    """
    
    def __init__(self, dataset_name: str):
        """
        Hugging Face 데이터셋으로 검증기를 초기화합니다
        
        Args:
            dataset_name: Hugging Face Hub의 데이터셋 이름
        """
        self.dataset_name = dataset_name
        self.dataset = None
        self.okt = Okt() if Okt else None
        self.validation_results = {}
        
    def load_dataset(self, split: str = None) -> None:
        """Hugging Face Hub에서 데이터셋을 불러옵니다"""
        try:
            if split:
                self.dataset = load_dataset(self.dataset_name, split=split)
            else:
                self.dataset = load_dataset(self.dataset_name)
            print(f"데이터셋 로드 성공: {self.dataset_name}")
        except Exception as e:
            print(f"데이터셋 로드 오류: {e}")
            return
    
    def basic_statistics(self) -> Dict[str, Any]:
        """
        데이터셋의 기본 통계 정보를 생성합니다
        """
        print("\n=== 데이터셋 기본 통계 ===")
        
        if isinstance(self.dataset, dict):
            # 여러 분할이 있는 데이터셋
            stats = {}
            for split_name, split_data in self.dataset.items():
                stats[split_name] = {
                    'num_examples': len(split_data),
                    'num_features': len(split_data.features),
                    'feature_names': list(split_data.features.keys()),
                    'feature_types': {k: str(v) for k, v in split_data.features.items()}
                }
                print(f"\n{split_name.upper()} 분할:")
                print(f"  총 예제 수: {stats[split_name]['num_examples']:,}")
                print(f"  특성 목록: {stats[split_name]['feature_names']}")
        else:
            # 단일 분할 데이터셋
            stats = {
                'num_examples': len(self.dataset),
                'num_features': len(self.dataset.features),
                'feature_names': list(self.dataset.features.keys()),
                'feature_types': {k: str(v) for k, v in self.dataset.features.items()}
            }
            print(f"총 예제 수: {stats['num_examples']:,}")
            print(f"특성 목록: {stats['feature_names']}")
        
        self.validation_results['basic_stats'] = stats
        return stats
    
    def text_quality_analysis(self, text_column: str, sample_size: int = 1000) -> Dict[str, Any]:
        """
        한국어 텍스트의 품질 지표를 분석합니다
        
        Args:
            text_column: 분석할 텍스트 열의 이름
            sample_size: 분석할 샘플 수 (성능을 위해)
        """
        print(f"\n=== 텍스트 품질 분석: {text_column} ===")
        
        # 텍스트 데이터 가져오기
        if isinstance(self.dataset, dict):
            # train 분할이 있으면 사용, 없으면 첫 번째 분할 사용
            split_name = 'train' if 'train' in self.dataset else list(self.dataset.keys())[0]
            texts = self.dataset[split_name][text_column][:sample_size]
        else:
            texts = self.dataset[text_column][:sample_size]
        
        # 기본 텍스트 통계
        text_lengths = [len(text) for text in texts]
        word_counts = [len(text.split()) for text in texts]
        
        # 한국어 특화 분석
        korean_ratio = self._calculate_korean_ratio(texts)
        encoding_issues = self._detect_encoding_issues(texts)
        
        quality_metrics = {
            'total_samples_analyzed': len(texts),
            'avg_character_length': np.mean(text_lengths),
            'avg_word_count': np.mean(word_counts),
            'min_length': min(text_lengths),
            'max_length': max(text_lengths),
            'std_length': np.std(text_lengths),
            'korean_character_ratio': korean_ratio,
            'encoding_issues_count': encoding_issues,
            'empty_texts': sum(1 for text in texts if not text.strip()),
            'duplicate_texts': len(texts) - len(set(texts))
        }
        
        # 결과 출력
        for key, value in quality_metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        
        # KoNLPy가 사용 가능한 경우 어휘 분석
        if self.okt:
            vocab_analysis = self._analyze_vocabulary(texts[:100])  # 성능을 위해 샘플링
            quality_metrics.update(vocab_analysis)
        
        self.validation_results['text_quality'] = quality_metrics
        return quality_metrics
    
    def _calculate_korean_ratio(self, texts: List[str]) -> float:
        """텍스트에서 한국어 문자의 비율을 계산합니다"""
        korean_chars = 0
        total_chars = 0
        
        for text in texts:
            for char in text:
                total_chars += 1
                if '\uAC00' <= char <= '\uD7A3':  # 한글 음절
                    korean_chars += 1
        
        return korean_chars / total_chars if total_chars > 0 else 0
    
    def _detect_encoding_issues(self, texts: List[str]) -> int:
        """텍스트에서 잠재적인 인코딩 문제를 감지합니다"""
        issues = 0
        problematic_patterns = ['�', '\\u', '\\x']
        
        for text in texts:
            for pattern in problematic_patterns:
                if pattern in text:
                    issues += 1
                    break
        
        return issues
    
    def _analyze_vocabulary(self, texts: List[str]) -> Dict[str, Any]:
        """KoNLPy를 사용하여 어휘를 분석합니다"""
        print("\n어휘 분석 (KoNLPy 사용)...")
        
        all_morphs = []
        pos_counts = Counter()
        
        for text in texts:
            try:
                morphs = self.okt.morphs(text)
                pos_tagged = self.okt.pos(text)
                
                all_morphs.extend(morphs)
                for word, pos in pos_tagged:
                    pos_counts[pos] += 1
            except:
                continue
        
        vocab_stats = {
            'unique_morphemes': len(set(all_morphs)),
            'total_morphemes': len(all_morphs),
            'vocabulary_richness': len(set(all_morphs)) / len(all_morphs) if all_morphs else 0,
            'top_pos_tags': dict(pos_counts.most_common(10)),
            'most_common_words': dict(Counter(all_morphs).most_common(20))
        }
        
        print(f"고유 형태소: {vocab_stats['unique_morphemes']}")
        print(f"어휘 풍부도: {vocab_stats['vocabulary_richness']:.3f}")
        
        return vocab_stats
    
    def data_distribution_analysis(self, label_column: str = None) -> Dict[str, Any]:
        """
        데이터 분포와 균형을 분석합니다
        
        Args:
            label_column: 분류 작업을 위한 라벨 열의 이름
        """
        print(f"\n=== 데이터 분포 분석 ===")
        
        distribution_stats = {}
        
        if isinstance(self.dataset, dict):
            for split_name, split_data in self.dataset.items():
                if label_column and label_column in split_data.features:
                    labels = split_data[label_column]
                    label_counts = Counter(labels)
                    
                    distribution_stats[split_name] = {
                        'label_distribution': dict(label_counts),
                        'num_classes': len(label_counts),
                        'most_common_class': label_counts.most_common(1)[0],
                        'least_common_class': label_counts.most_common()[-1],
                        'balance_ratio': min(label_counts.values()) / max(label_counts.values())
                    }
                    
                    print(f"\n{split_name.upper()} 분할 라벨 분포:")
                    for label, count in label_counts.most_common():
                        percentage = (count / len(labels)) * 100
                        print(f"  {label}: {count} ({percentage:.1f}%)")
        
        self.validation_results['distribution'] = distribution_stats
        return distribution_stats
    
    def detect_quality_issues(self, text_column: str) -> Dict[str, List]:
        """
        데이터셋에서 잠재적인 품질 문제를 감지합니다
        """
        print(f"\n=== 품질 문제 감지 ===")
        
        if isinstance(self.dataset, dict):
            split_name = 'train' if 'train' in self.dataset else list(self.dataset.keys())[0]
            texts = self.dataset[split_name][text_column]
        else:
            texts = self.dataset[text_column]
        
        issues = {
            'too_short': [],
            'too_long': [],
            'non_korean': [],
            'suspicious_patterns': [],
            'potential_duplicates': []
        }
        
        # 임계값 정의
        MIN_LENGTH = 10
        MAX_LENGTH = 2000
        MIN_KOREAN_RATIO = 0.3
        
        for idx, text in enumerate(texts):
            if len(text) < MIN_LENGTH:
                issues['too_short'].append(idx)
            
            if len(text) > MAX_LENGTH:
                issues['too_long'].append(idx)
            
            korean_ratio = self._calculate_korean_ratio([text])
            if korean_ratio < MIN_KOREAN_RATIO:
                issues['non_korean'].append(idx)
            
            # 의심스러운 패턴 검사
            if re.search(r'(.)\1{5,}', text):  # 반복되는 문자
                issues['suspicious_patterns'].append(idx)
        
        # 잠재적 중복 찾기 (단순화된 방법)
        text_hashes = {}
        for idx, text in enumerate(texts):
            text_hash = hash(text.strip().lower())
            if text_hash in text_hashes:
                issues['potential_duplicates'].append(idx)
            text_hashes[text_hash] = idx
        
        # 요약 출력
        for issue_type, indices in issues.items():
            print(f"{issue_type}: {len(indices)}건")
        
        self.validation_results['quality_issues'] = issues
        return issues
    
    def generate_validation_report(self, output_file: str = None) -> str:
        """
        종합적인 검증 리포트를 생성합니다
        """
        report = f"""
한국어 데이터셋 검증 리포트
데이터셋: {self.dataset_name}
생성일시: {pd.Timestamp.now()}

{'='*50}
요약
{'='*50}
"""
        
        if 'basic_stats' in self.validation_results:
            stats = self.validation_results['basic_stats']
            if isinstance(stats, dict) and 'train' in stats:
                report += f"총 훈련 예제 수: {stats['train']['num_examples']:,}\n"
            
        if 'text_quality' in self.validation_results:
            quality = self.validation_results['text_quality']
            report += f"평균 텍스트 길이: {quality['avg_character_length']:.1f} 문자\n"
            report += f"한국어 문자 비율: {quality['korean_character_ratio']:.1%}\n"
            report += f"빈 텍스트: {quality['empty_texts']}건\n"
            report += f"중복 텍스트: {quality['duplicate_texts']}건\n"
        
        if 'quality_issues' in self.validation_results:
            issues = self.validation_results['quality_issues']
            total_issues = sum(len(issue_list) for issue_list in issues.values())
            report += f"감지된 총 품질 문제: {total_issues}건\n"
        
        report += "\n"
        report += "권장사항:\n"
        
        # 발견된 내용을 바탕으로 권장사항 추가
        if 'text_quality' in self.validation_results:
            quality = self.validation_results['text_quality']
            
            if quality['korean_character_ratio'] < 0.8:
                report += "- 한국어 문자 비율이 낮은 텍스트 필터링을 고려하세요\n"
            
            if quality['duplicate_texts'] > 0:
                report += "- 데이터 품질 향상을 위해 중복 텍스트를 제거하세요\n"
            
            if quality['empty_texts'] > 0:
                report += "- 빈 텍스트 또는 공백만 있는 텍스트를 제거하세요\n"
        
        report += "\n"
        report += json.dumps(self.validation_results, indent=2, ensure_ascii=False)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"리포트가 다음 파일에 저장되었습니다: {output_file}")
        
        return report
    
    def visualize_results(self, text_column: str = None):
        """
        검증 결과를 시각화합니다
        """
        print("\n=== 시각화 생성 중 ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'데이터셋 품질 분석: {self.dataset_name}', fontsize=16)
        
        # 텍스트 길이 분포
        if 'text_quality' in self.validation_results and text_column:
            if isinstance(self.dataset, dict):
                split_name = 'train' if 'train' in self.dataset else list(self.dataset.keys())[0]
                texts = self.dataset[split_name][text_column][:1000]
            else:
                texts = self.dataset[text_column][:1000]
            
            text_lengths = [len(text) for text in texts]
            axes[0, 0].hist(text_lengths, bins=50, alpha=0.7)
            axes[0, 0].set_title('텍스트 길이 분포')
            axes[0, 0].set_xlabel('문자 수')
            axes[0, 0].set_ylabel('빈도')
        
        # 라벨 분포 (가능한 경우)
        if 'distribution' in self.validation_results:
            dist_data = self.validation_results['distribution']
            if 'train' in dist_data:
                labels = list(dist_data['train']['label_distribution'].keys())
                counts = list(dist_data['train']['label_distribution'].values())
                
                axes[0, 1].bar(labels, counts)
                axes[0, 1].set_title('라벨 분포')
                axes[0, 1].set_xlabel('라벨')
                axes[0, 1].set_ylabel('개수')
                plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # 품질 문제 분석
        if 'quality_issues' in self.validation_results:
            issues = self.validation_results['quality_issues']
            issue_names = list(issues.keys())
            issue_counts = [len(issues[name]) for name in issue_names]
            
            axes[1, 0].bar(issue_names, issue_counts)
            axes[1, 0].set_title('감지된 품질 문제')
            axes[1, 0].set_xlabel('문제 유형')
            axes[1, 0].set_ylabel('개수')
            plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # 텍스트 품질 지표
        if 'text_quality' in self.validation_results:
            quality = self.validation_results['text_quality']
            metrics = ['korean_character_ratio', 'vocabulary_richness']
            values = [quality.get(metric, 0) for metric in metrics]
            
            axes[1, 1].bar(['한국어 비율', '어휘 풍부도'], values)
            axes[1, 1].set_title('텍스트 품질 지표')
            axes[1, 1].set_ylabel('비율 (0-1)')
            axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()

def main():
    """
    한국어 데이터셋 검증기 사용 예제
    """
    # 데이터셋으로 검증기 초기화
    validator = KoreanDatasetValidator("MyeongHo0621/korean-quality-cleaned")
    
    # 데이터셋 로드
    validator.load_dataset()
    
    # 기본 통계 실행
    validator.basic_statistics()
    
    # 대화 형식 데이터셋을 위한 텍스트 추출
    print("\n=== 대화 형식 데이터셋 처리 ===")
    if isinstance(validator.dataset, dict):
        split_name = 'train' if 'train' in validator.dataset else list(validator.dataset.keys())[0]
        dataset = validator.dataset[split_name]
    else:
        dataset = validator.dataset
    
    # messages에서 모든 content를 하나의 텍스트로 합치기
    print("messages 필드에서 텍스트 추출 중...")
    all_texts = []
    for example in dataset:
        if 'messages' in example and example['messages']:
            # 모든 메시지의 content를 결합
            combined_text = ' '.join([msg['content'] for msg in example['messages'] if 'content' in msg])
            all_texts.append(combined_text)
    
    print(f"✓ {len(all_texts)}개의 대화 텍스트 추출 완료\n")
    
    # 추출한 텍스트로 임시 데이터셋 생성
    from datasets import Dataset
    temp_dataset = Dataset.from_dict({'text': all_texts})
    validator.dataset = {'train': temp_dataset}
    
    # 텍스트 품질 분석
    text_column = "text"
    validator.text_quality_analysis(text_column)
    
    # 데이터 분포 분석 (라벨이 있는 경우)
    # label_column = "label"  # 라벨이 있는 경우 주석 해제하고 조정
    # validator.data_distribution_analysis(label_column)
    
    # 품질 문제 감지
    validator.detect_quality_issues(text_column)
    
    # 종합 리포트 생성
    report = validator.generate_validation_report("validation_report.txt")
    
    # 시각화 생성 (matplotlib가 display를 지원하는 경우에만)
    try:
        validator.visualize_results(text_column)
    except Exception as e:
        print(f"\n시각화 생성 실패 (서버 환경에서는 정상): {e}")
    
    print("\n검증이 완료되었습니다! 상세한 결과는 'validation_report.txt' 파일을 확인하세요.")

if __name__ == "__main__":
    main()