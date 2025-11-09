# Qwen2.5-14B-Instruct 한국어 멀티턴 대화 파인튜닝

H100 80GB 환경에서 Qwen2.5-14B-Instruct 모델을 한국어 멀티턴 대화 데이터로 파인튜닝합니다.

## 환경 요구사항

- **GPU**: NVIDIA H100 80GB (또는 80GB+ VRAM)
- **Python**: 3.10+
- **CUDA**: 12.0+
- **라이브러리**:
  - `torch >= 2.0`
  - `transformers >= 4.40`
  - `unsloth >= 2025.10`
  - `flash-attn >= 2.4`
  - `peft >= 0.14`
  - `trl`
  - `datasets`
  - `bitsandbytes`
  - `accelerate`

## 모델 정보

- **베이스 모델**: [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)
- **파라미터**: 14.7B (13.1B non-embedding)
- **Context Length**: 131,072 tokens (학습 시 4,096 사용)
- **아키텍처**: Transformers with RoPE, SwiGLU, RMSNorm, GQA

## 최적화 기술

1. **Flash Attention 2**: H100에서 최적화된 어텐션 연산
2. **LoRA**: 메모리 효율적인 파인튜닝 (r=64, alpha=128)
3. **8-bit 양자화**: bitsandbytes를 통한 메모리 절약
4. **Gradient Checkpointing**: Unsloth 최적화 버전
5. **BF16**: H100 네이티브 지원

## 데이터셋

한국어 멀티턴 대화 데이터셋 (약 68만개):
- `kowiki_qa_data.jsonl` (48,699)
- `kullm_v2_full_data.jsonl` (146,963)
- `orca_math_ko_data.jsonl` (192,807)
- `smol_koreantalk_data.jsonl` (88,752)
- `won_instruct_data.jsonl` (86,007)
- 기타 데이터셋

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

## 사용 방법

### 1. 환경 설정

```bash
# HuggingFace 로그인
huggingface-cli login

# 또는 .env 파일 설정
cp .env.example .env
# .env 파일에 HF_TOKEN 입력
```

### 2. 학습 실행

```bash
cd /home/work/tesseract/qwen/2.5_14B_Inst

# 기본 학습
python 0_qwen14b_multiturn_ft.py

# 백그라운드 실행 (로그 저장)
nohup python 0_qwen14b_multiturn_ft.py > train.log 2>&1 &
```

### 3. 학습 모니터링

```bash
# 로그 확인
tail -f train.log

# GPU 모니터링
watch -n 1 nvidia-smi
```

## 설정 커스터마이징

`0_qwen14b_multiturn_ft.py`의 `Qwen14BFineTuningConfig` 클래스에서 설정 변경:

```python
@dataclass
class Qwen14BFineTuningConfig:
    # 모델
    max_seq_length: int = 4096  # 최대 시퀀스 길이
    
    # LoRA
    lora_r: int = 64            # LoRA rank
    lora_alpha: int = 128       # LoRA alpha
    lora_dropout: float = 0.05  # Dropout
    
    # 학습
    num_train_epochs: int = 3                    # Epoch 수
    per_device_train_batch_size: int = 4         # 배치 크기
    gradient_accumulation_steps: int = 4         # Gradient 누적
    learning_rate: float = 2e-4                  # 학습률
    
    # 데이터 (특정 파일만 사용하려면)
    data_files: List[str] = None  # 예: ["kowiki_qa_data.jsonl"]
```

## 예상 학습 시간

- **H100 80GB**: 약 12-18시간 (68만개 데이터, 3 epoch)
- **메모리 사용량**: 약 60-70GB VRAM
- **효과적 배치 크기**: 16 (4 × 4)

## 출력

학습 완료 후 다음 위치에 모델 저장:

```
/home/work/tesseract/qwen/2.5_14B_Inst/output/
├── checkpoint-500/
├── checkpoint-1000/
├── checkpoint-1500/
└── final/              # 최종 모델
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── tokenizer*
```

## 추론 예제

```python
from unsloth import FastLanguageModel

# 모델 로드
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/work/tesseract/qwen/2.5_14B_Inst/output/final",
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=False,
)

# 추론 모드
FastLanguageModel.for_inference(model)

# 대화 생성
messages = [
    {"role": "user", "content": "안녕하세요!"}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 문제 해결

### OOM (Out of Memory) 에러
```python
# 배치 크기 줄이기
per_device_train_batch_size: int = 2
gradient_accumulation_steps: int = 8

# 또는 시퀀스 길이 줄이기
max_seq_length: int = 2048
```

### 학습 속도가 느림
```python
# Gradient accumulation 줄이기
gradient_accumulation_steps: int = 2

# Dataset num_proc 조정
dataset_num_proc=4  # CPU 코어 수에 따라
```

### Flash Attention 에러
```python
# Flash Attention 비활성화
attn_implementation="eager"
```

## 참고 자료

- [Qwen2.5 공식 문서](https://qwenlm.github.io/blog/qwen2.5/)
- [Unsloth 문서](https://github.com/unslothai/unsloth)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [LoRA 논문](https://arxiv.org/abs/2106.09685)

## 라이선스

- Qwen2.5-14B-Instruct: Apache 2.0

