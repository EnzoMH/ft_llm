# Qwen3-VL 한국어 VLA 파인튜닝

> AGV/AMR용 한국어 Vision-Language-Action 모델 개발
> 환경: H100 80GB GPU
> 증강 모델: EXAONE 4.0 1.2B

## 📁 프로젝트 구조

```
qwen3/
├── augment_cot_exaone.py           # CoT 데이터 증강 (EXAONE)
├── prepare_phase1_data.py          # Phase 1 데이터 준비
├── validate_cot_quality.py         # CoT 품질 검증
├── train_phase1_korean_instruct.py # Phase 1 학습
├── train_phase2_thinking.py        # Phase 2 학습
├── model_load.py                   # 모델 테스트/추론
├── run_all.sh                      # 전체 파이프라인 실행
├── guied.md                        # 상세 가이드 문서
└── README.md                       # 이 파일
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 필수 패키지 설치
pip install torch transformers datasets trl accelerate
pip install unsloth  # 파인튜닝 최적화
pip install vllm     # CoT 증강용

# GPU 확인
nvidia-smi
python --version  # Python 3.10+ 필요
```

### 2. 데이터 준비

```bash
# 데이터 위치 확인
ls ../../korean_large_data/cleaned_jsonl/

# 총 681K 샘플:
# - orca_math_ko_data.jsonl (192K)
# - kullm_v2_full_data.jsonl (147K)
# - smol_koreantalk_data.jsonl (89K)
# 등...
```

### 3. 전체 파이프라인 실행

```bash
# 실행 권한 부여
chmod +x run_all.sh

# 전체 실행 (대화형)
./run_all.sh

# 또는 단계별 실행 (아래 참조)
```

## 📋 단계별 실행

### Step 1: Phase 1 데이터 준비 (30분)

```bash
python prepare_phase1_data.py \
    --input-dir ../../korean_large_data/cleaned_jsonl \
    --output-dir phase1_korean \
    --target-samples 256000
```

**출력:**
- `phase1_korean/phase1_korean_256000samples.jsonl`
- 중국어 필터링 완료
- 한국어 시스템 프롬프트 추가

### Step 2: CoT 데이터 증강 (1-2시간)

```bash
# 5% 샘플링 (테스트)
python augment_cot_exaone.py \
    --input-dir ../../korean_large_data/cleaned_jsonl \
    --output-dir phase2_thinking_exaone \
    --sample-ratio 0.05 \
    --singleturn-ratio 0.7

# 전체 데이터 (승인 후)
python augment_cot_exaone.py \
    --input-dir ../../korean_large_data/cleaned_jsonl \
    --output-dir phase2_thinking_full \
    --sample-ratio 1.0 \
    --singleturn-ratio 0.7
```

**처리 시간 (H100 80GB):**
- 5% (28K 샘플): 1-2시간
- 100% (565K 샘플): 20-40시간

### Step 3: 품질 검증 (10분)

```bash
# 단일 파일 검증
python validate_cot_quality.py \
    phase2_thinking_exaone/orca_math_ko_data_cot_augmented.jsonl \
    --num-samples 20 \
    --show-full

# 디렉토리 전체 검증
python validate_cot_quality.py \
    phase2_thinking_exaone \
    --directory \
    --num-samples 20
```

**확인 항목:**
- ✅ `<think>` 태그 포함 여부
- ✅ 중국어 포함 여부
- ✅ 단계별 구분 (1단계:, 2단계:)
- ✅ 답변 길이

### Step 4: Phase 1 학습 (4-6시간)

```bash
python train_phase1_korean_instruct.py \
    --model-name Qwen/Qwen3-VL-8B-Instruct \
    --data-dir phase1_korean \
    --output-dir qwen3-vl-8b-korean-instruct \
    --epochs 2 \
    --batch-size 4 \
    --gradient-accumulation 4 \
    --learning-rate 2e-5 \
    --lora-r 16 \
    --lora-alpha 32
```

**출력:**
- `qwen3-vl-8b-korean-instruct/`: LoRA 어댑터
- `qwen3-vl-8b-korean-instruct/merged/`: 병합 모델

### Step 5: Phase 2 학습 (6-8시간)

```bash
# 옵션 1: Phase 1 결과 사용 (권장)
python train_phase2_thinking.py \
    --model-name qwen3-vl-8b-korean-instruct/merged \
    --data-dir phase2_thinking_exaone \
    --output-dir qwen3-vl-8b-korean-thinking \
    --epochs 3 \
    --batch-size 2 \
    --gradient-accumulation 8 \
    --learning-rate 1e-5 \
    --lora-r 32 \
    --lora-alpha 64

# 옵션 2: Thinking 베이스 사용
python train_phase2_thinking.py \
    --model-name Qwen/Qwen3-VL-8B-Thinking \
    --data-dir phase2_thinking_exaone \
    --output-dir qwen3-vl-8b-korean-thinking \
    --epochs 3 \
    --batch-size 2 \
    --gradient-accumulation 8
```

**출력:**
- `qwen3-vl-8b-korean-thinking/`: LoRA 어댑터
- `qwen3-vl-8b-korean-thinking/merged/`: 병합 모델

### Step 6: 모델 테스트

```bash
# 자동 테스트
python model_load.py qwen3-vl-8b-korean-thinking/merged

# Thinking 모드 테스트
python model_load.py qwen3-vl-8b-korean-thinking/merged --thinking

# 대화형 모드
python model_load.py qwen3-vl-8b-korean-thinking/merged --interactive --thinking
```

**대화형 명령어:**
- `/think`: Thinking 모드 활성화
- `/no_think`: 일반 모드
- `quit` 또는 `exit`: 종료

## 📊 데이터 구성

### Phase 1: 한국어 강화 (256K)

| 데이터셋 | 샘플 수 | 용도 |
|---------|---------|------|
| kullm_v2_full | 147K | 전문 지식 |
| smol_koreantalk | 89K | 일반 대화 |
| won_instruct | 86K | 전문 지식 |
| 기타 | ~34K | 다양한 도메인 |

### Phase 2: Thinking CoT (28K @ 5%)

| 데이터셋 | 5% 샘플 | 용도 |
|---------|---------|------|
| orca_math_ko | 9,640 | 수학 추론 |
| kullm_v2_full | 7,348 | 일반 추론 |
| smol_koreantalk | 4,438 | 대화 추론 |
| won_instruct | 4,300 | 전문 추론 |
| 기타 | ~2,531 | 다양한 추론 |

## ⚙️ 하이퍼파라미터

### Phase 1 (한국어 강화)

```python
max_seq_length = 4096        # 일반 대화 길이
num_train_epochs = 2         # 적은 epoch으로 빠르게
batch_size = 4               # GPU 메모리에 맞게
gradient_accumulation = 4    # 효과적 배치 = 16
learning_rate = 2e-5         # 표준 파인튜닝 LR
lora_r = 16                  # 적당한 rank
lora_alpha = 32              # r의 2배
```

### Phase 2 (Thinking)

```python
max_seq_length = 8192        # CoT는 긴 시퀀스 필요
num_train_epochs = 3         # CoT 패턴 학습에 더 많은 epoch
batch_size = 2               # 긴 시퀀스로 배치 줄임
gradient_accumulation = 8    # 효과적 배치 = 16
learning_rate = 1e-5         # 더 작은 LR (안정성)
lora_r = 32                  # 더 큰 rank (복잡한 패턴)
lora_alpha = 64              # r의 2배
```

## 🔧 트러블슈팅

### CUDA Out of Memory

```bash
# 배치 사이즈 감소
--batch-size 2
--gradient-accumulation 8

# 시퀀스 길이 감소
--max-seq-length 4096  # (Phase 2)
```

### CoT 품질 낮음

1. **Temperature 조정**
```bash
# augment_cot_exaone.py 수정
temperature=0.7 → 0.5  # 더 결정적
```

2. **프롬프트 개선**
```python
# create_singleturn_prompt() 수정
# 더 명확한 지시사항 추가
```

3. **모델 업그레이드**
```bash
# EXAONE 3B 시도
model_name="LGAI-EXAONE/EXAONE-4.0-3B-Instruct"
```

### 학습 불안정

```bash
# Learning rate 감소
--learning-rate 1e-5

# Warmup 증가
--warmup-steps 500

# Gradient clipping
# training_args에 max_grad_norm=1.0 추가됨
```

### vLLM 설치 실패

```bash
# CUDA 버전 확인
nvcc --version

# 호환되는 버전 설치
pip install vllm==0.5.0  # CUDA 11.8
pip install vllm==0.6.0  # CUDA 12.1
```

## 📈 예상 소요 시간 (H100 80GB)

| 단계 | 5% 샘플링 | 100% (전체) |
|-----|-----------|-------------|
| **Phase 1 데이터 준비** | 30분 | 30분 |
| **CoT 데이터 증강** | 1-2시간 | 20-40시간 |
| **품질 검증** | 10분 | 30분 |
| **Phase 1 학습** | 4-6시간 | 4-6시간 |
| **Phase 2 학습** | 2-3시간 | 20-30시간 |
| **총 소요 시간** | **8-12시간** | **45-107시간** |

## 💾 디스크 공간

- 원본 데이터: ~1.4GB
- Phase 1 데이터: ~1GB
- Phase 2 데이터 (5%): ~100MB
- Phase 2 데이터 (100%): ~2GB
- 모델 체크포인트: ~20GB per phase
- **총 필요 공간**: ~50GB (5% 기준), ~100GB (100% 기준)

## 🎯 성능 평가

### 평가 항목

1. **한국어 능력**
   - 중국어 출력 여부
   - 문법 정확성
   - 자연스러운 표현

2. **Thinking 능력**
   - `<think>` 태그 사용
   - 단계별 추론
   - 최종 답변 정확성

3. **도메인 특화**
   - AGV/AMR 관련 질의 응답
   - 한손로봇 제어 명령

### 테스트 예시

```bash
# 수학 추론
python model_load.py qwen3-vl-8b-korean-thinking/merged --thinking
> 3x + 5 = 14를 풀어주세요.

# AGV 제어
python model_load.py qwen3-vl-8b-korean-thinking/merged --thinking
> AGV가 앞에 장애물이 있을 때 어떻게 해야 하나요?

# 한국어 검증
python model_load.py qwen3-vl-8b-korean-thinking/merged
> 자기소개를 해주세요. (중국어 출력 여부 확인)
```

## 📚 참고 자료

- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Qwen3-VL GitHub](https://github.com/QwenLM/Qwen3-VL)
- [EXAONE 4.0](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B-Instruct)
- [Unsloth Documentation](https://docs.unsloth.ai/)
- [상세 가이드 문서](guied.md)

## 🤝 기여

이슈 및 개선 제안은 GitHub Issues에 등록해주세요.

## 📝 라이선스

이 프로젝트는 각 모델의 라이선스를 따릅니다:
- Qwen3-VL: Apache 2.0
- EXAONE 4.0: Apache 2.0

---

**작성일**: 2025-11-03  
**환경**: H100 80GB GPU  
**프레임워크**: vLLM, Unsloth, Transformers

