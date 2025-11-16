#!/usr/bin/env python3
"""
한국어 멀티턴 대화 데이터셋 로더
"""

import os
import json
import logging
from typing import List, Optional
from datasets import Dataset


logger = logging.getLogger(__name__)


class MultiTurnDatasetLoader:
    """한국어 멀티턴 대화 데이터셋 로더"""
    
    def __init__(self, data_dirs: List[str]):
        """
        Args:
            data_dirs: JSONL 데이터 디렉토리 리스트
        """
        self.data_dirs = data_dirs
    
    def load_jsonl_files(self, file_paths: List[str]) -> Dataset:
        """JSONL 파일들을 로드 (메모리 효율적)
        
        Args:
            file_paths: 로드할 JSONL 파일 경로 리스트
            
        Returns:
            Dataset: HuggingFace Dataset 객체
        """
        all_data = []
        
        for file_path in file_paths:
            logger.info(f"  로딩: {os.path.basename(file_path)}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    count = 0
                    batch = []
                    batch_size = 10000  # 1만개씩 배치로 처리 (RAM 압박 방지)
                    
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                
                                # messages 필드가 있는 경우
                                if 'messages' in data and isinstance(data['messages'], list):
                                    # 멀티턴 대화인지 확인 (최소 2개 이상)
                                    if len(data['messages']) >= 2:
                                        batch.append(data)
                                        count += 1
                                
                                # text 필드만 있는 경우 (이미 ChatML 포맷팅됨)
                                elif 'text' in data and isinstance(data['text'], str):
                                    # assistant가 최소 1번 이상 있으면 유효한 대화
                                    if '<|im_start|>assistant' in data['text']:
                                        batch.append(data)
                                        count += 1
                                
                                # 배치가 차면 all_data에 추가하고 배치 초기화
                                if len(batch) >= batch_size:
                                    all_data.extend(batch)
                                    batch = []
                                    logger.debug(f"    진행: {count:,}개 로드됨")
                                    
                            except json.JSONDecodeError as e:
                                logger.debug(f"JSON 파싱 실패: {e}")
                                continue
                    
                    # 남은 배치 처리
                    if batch:
                        all_data.extend(batch)
                    
                    logger.info(f"    추가됨: {count:,}개")
            except Exception as e:
                logger.error(f"  파일 로드 실패 ({file_path}): {e}")
                continue
        
        logger.info(f"[ COMPLETE ] 총 {len(all_data):,}개 데이터 로드")
        return Dataset.from_list(all_data)
    
    def load_from_directory(self, directory: str, patterns: Optional[List[str]] = None) -> Dataset:
        """디렉토리에서 JSONL 파일들을 로드
        
        Args:
            directory: 탐색할 디렉토리
            patterns: 파일 패턴 리스트 (기본: ['*.jsonl'])
            
        Returns:
            Dataset: HuggingFace Dataset 객체
        """
        import glob
        
        if patterns is None:
            patterns = ['*.jsonl']
        
        file_paths = []
        for pattern in patterns:
            file_paths.extend(glob.glob(os.path.join(directory, pattern)))
        
        logger.info(f"발견된 파일: {len(file_paths)}개")
        return self.load_jsonl_files(file_paths)

