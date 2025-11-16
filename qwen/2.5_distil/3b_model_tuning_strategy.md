# Qwen2.5-3B-Instruct – Korean Instruction Tuning Plan

## 1. 목표

- **Qwen2.5-3B-Instruct**를 `smol_koreantalk_full.jsonl`로 직접 학습
- 일반 Instruction Tuning (SFT) 방식으로 진행
- 학습 완료 후 HuggingFace Hub에 업로드

---

## 2. 베이스 & 데이터

- **Base 모델**
  - `Qwen/Qwen2.5-3B-Instruct`

- **튜닝 데이터**
  - 파일: `/home/work/vss/ft_llm/data/smol_koreantalk_full.jsonl`
  - 포맷: `messages = [{role: system/user/assistant, content: ...}, ...]`
  - 전체 데이터셋 사용 (약 460k 샘플)

---

## 3. 학습 전략

- **방법:** 일반 Instruction Tuning (SFT)
- **Distillation:** 사용 안 함

- **LoRA vs Full FT**
  - **LoRA**로 진행 (리소스 절약 & 실험 회전 속도 우선)

- **권장 기본 설정 (3B 모델 최적화)**
  - max_seq_len: 2048
  - batch_size (effective): 128 전후
    - per_device_train_batch_size: 32 (H100 80GB 기준)
    - gradient_accumulation_steps: 4
  - lr: 2e-4 (LoRA 기준)
  - epoch: 1~2 (460k 샘플 기준)
  - LoRA rank: 32 (3B 모델이므로 14B보다 작게)
  - LoRA alpha: 64
  - optimizer: AdamW
  - scheduler: cosine / linear warmup (warmup_ratio ~ 0.03)

---

## 4. 배포 계획

- 학습 완료 후:
  1. LoRA 어댑터와 베이스 모델 병합
  2. HuggingFace Hub에 업로드
     - 모델 ID: `MyeongHo0621/Qwen2.5-3B-Korean` (가칭)
  3. README.md 작성 및 업로드

---

## 5. 확인 사항

### 데이터
- [x] 데이터 위치 확인: `/home/work/vss/ft_llm/data/smol_koreantalk_full.jsonl`
- [x] 데이터 포맷 확인: `messages` 필드에 `[{role: user/assistant, content: ..., content_en: ...}]`
- [x] 샘플 수 확인: 약 460,281개

### 학습 설정
- [x] LoRA rank/alpha 결정: r=32, alpha=64
- [x] 배치 크기 최적화: per_device=32, grad_accum=4 (effective 128)
- [x] Epoch 수 결정: 1 epoch로 시작

### 코드
- [x] 기존 14B 학습 코드 재사용 가능 여부 확인
- [x] 3B 모델용 설정 파일 생성: `src/qwen_finetuning_3b/config.py`
- [x] 학습 스크립트 작성: `scripts/train_3b.py`
- [x] Trainer 모듈 생성: `src/qwen_finetuning_3b/trainer.py`
- [x] 데이터 로더 복사: `src/qwen_finetuning_3b/dataset_loader.py`
- [x] 유틸리티 복사: `src/qwen_finetuning_3b/utils.py`, `callbacks.py`

### 배포
- [x] HuggingFace Hub 모델 ID 결정: `MyeongHo0621/Qwen2.5-3B-Korean`
- [ ] README.md 작성 (학습 완료 후)
- [x] 업로드 스크립트 확인 (trainer에 통합됨)

---

## 6. 실행 방법

```bash
# fa3 가상환경 활성화
conda activate fa3

# 학습 실행
cd /home/work/vss/ft_llm/qwen/2.5_distil
python scripts/train_3b.py
```

학습이 완료되면:
- LoRA 어댑터가 `outputs/checkpoints/final/`에 저장됨
- HuggingFace Hub에 자동 업로드됨 (`hub_strategy="end"`)
