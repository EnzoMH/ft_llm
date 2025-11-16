# Qwen2.5-14B-Instruct 한국어 파인튜닝 프로젝트

H100 80GB 환경에서 Qwen2.5-14B-Instruct 모델을 한국어 멀티턴 대화 데이터로 파인튜닝하는 프로젝트입니다.

## 📁 프로젝트 구조

```
2.5_14B_Inst/
├── scripts/              # 실행 스크립트
│   ├── train.py         # 학습 스크립트
│   ├── inference_test.py
│   ├── merge_and_upload.py
│   └── ...
├── src/
│   └── qwen_finetuning/ # 핵심 모듈
│       ├── config.py
│       ├── trainer.py
│       ├── dataset_loader.py
│       └── ...
├── configs/             # 설정 파일
│   └── config.py
├── evaluation/          # 평가 관련
│   ├── evaluate_korean_benchmarks.py
│   └── evaluation_results/
├── docs/                # 문서
│   ├── README.md
│   ├── README_EVALUATION.md
│   └── ...
├── outputs/            # 출력 파일
│   ├── checkpoints/    # 모델 체크포인트
│   └── logs/          # 로그 파일
├── utils/              # 유틸리티
├── requirements.txt
└── README.md
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 활성화 (fa3)
source /home/work/miniconda3/etc/profile.d/conda.sh
conda activate fa3

# 의존성 설치
pip install -r requirements.txt

# HuggingFace 로그인
huggingface-cli login
# 또는 .env 파일에 HF_TOKEN 설정
```

### 2. 학습 실행

```bash
cd /home/work/vss/ft_llm/qwen/2.5_14B_Inst

# 기본 학습
python scripts/train.py

# 백그라운드 실행 (로그 저장)
nohup python scripts/train.py > outputs/logs/train.log 2>&1 &
```

### 3. 학습 모니터링

```bash
# 로그 확인
tail -f outputs/logs/train.log

# GPU 모니터링
watch -n 1 nvidia-smi
```

## 📊 평가

한국어 벤치마크 평가:

```bash
# 전체 평가
python evaluation/evaluate_korean_benchmarks.py

# 특정 카테고리만 평가
python evaluation/evaluate_korean_benchmarks.py --categories qa math

# 카테고리 옵션: qa, math, code, mmlu, all
```

## 🔧 설정 커스터마이징

`configs/config.py` 또는 `src/qwen_finetuning/config.py`에서 설정 변경:

```python
@dataclass
class Qwen14BFineTuningConfig:
    # 모델
    max_seq_length: int = 4096
    
    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    
    # 학습
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 22  # 72GB VRAM 기준
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
```

## 📦 모델 정보

- **베이스 모델**: [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)
- **파인튜닝 모델**: [MyeongHo0621/Qwen2.5-14B-Korean](https://huggingface.co/MyeongHo0621/Qwen2.5-14B-Korean)
- **파라미터**: 14.7B (13.1B non-embedding)
- **Context Length**: 131,072 tokens (학습 시 4,096 사용)

## 🎯 최적화 기술

1. **Flash Attention 3**: H100에서 최적화된 어텐션 연산
2. **LoRA**: 메모리 효율적인 파인튜닝 (r=64, alpha=128)
3. **8-bit 양자화**: bitsandbytes를 통한 메모리 절약
4. **Gradient Checkpointing**: Unsloth 최적화 버전
5. **BF16**: H100 네이티브 지원

## 📚 데이터셋

한국어 멀티턴 대화 데이터셋 (약 68만개):
- `kowiki_qa_data.jsonl` (48,699)
- `kullm_v2_full_data.jsonl` (146,963)
- `orca_math_ko_data.jsonl` (192,807)
- `smol_koreantalk_data.jsonl` (88,752)
- `won_instruct_data.jsonl` (86,007)

**데이터 포맷**: ChatML 형식의 `messages` 필드

```json
{
  "messages": [
    {"role": "user", "content": "질문"},
    {"role": "assistant", "content": "답변"}
  ],
  "source": "dataset_name"
}
```

## 📈 평가 벤치마크

- **KMMLU**: 한국어 MMLU 스타일 벤치마크
- **KoBEST**: 한국어 고급 언어 현상/추론
- **KorQuAD**: 한국어 위키 기반 MRC
- **GSM8K-Ko**: 한국어 수학 추론
- **HRM8K**: 한국 수학 추론 벤치마크
- **HumanEval-Ko**: 코드 생성 평가

## 🔍 주요 스크립트

- `scripts/train.py`: 학습 실행
- `scripts/merge_and_upload.py`: LoRA 어댑터와 베이스 모델 병합 후 HuggingFace Hub 업로드
- `scripts/test_hub_model.py`: Hub에서 다운로드한 모델 테스트
- `evaluation/evaluate_korean_benchmarks.py`: 한국어 벤치마크 평가

## 📝 문서

- [평가 가이드](docs/README_EVALUATION.md)
- [학습 상태 분석](docs/TRAINING_STATUS_ANALYSIS.md)
- [Flash Attention 3 설정 가이드](docs/FLASH_ATTENTION_3_SETUP_GUIDE.md)

## 🐛 문제 해결

### OOM (Out of Memory) 에러
```python
# 배치 크기 줄이기
per_device_train_batch_size: int = 2
gradient_accumulation_steps: int = 8
```

### 학습 속도가 느림
```python
# Gradient accumulation 줄이기
gradient_accumulation_steps: int = 2
```

### Flash Attention 에러
```python
# Flash Attention 비활성화
attn_implementation="eager"
```

## 📄 라이선스

- Qwen2.5-14B-Instruct: Apache 2.0

## 🔗 참고 자료

- [Qwen2.5 공식 문서](https://qwenlm.github.io/blog/qwen2.5/)
- [Unsloth 문서](https://github.com/unslothai/unsloth)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [LoRA 논문](https://arxiv.org/abs/2106.09685)

