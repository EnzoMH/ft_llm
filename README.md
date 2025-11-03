# Korean LLM Fine-tuning & Evaluation Suite

**한국어 특화 대규모 언어모델(LLM) 파인튜닝 및 평가 통합 프로젝트**

이 프로젝트는 다양한 오픈소스 한국어 LLM의 파인튜닝, 평가, 벤치마킹을 위한 통합 파이프라인입니다.

## Supported Models

| Model | Status | Description |
|-------|--------|-------------|
| **EEVE-Korean-10.8B** | ✅ Complete | Instruction tuning, HuggingFace 배포 완료 |
| **Qwen2.5** | 🔄 In Progress | Unsupervised & Checkpoint training |
| **VCLM-Korean-7B** | ✅ Complete | Benchmarking & Evaluation |
| **SOLAR-10.7B** | ✅ Complete | Legacy project (archived) |

## Key Features

- 🚀 **다중 모델 지원**: EEVE, Qwen, VCLM, SOLAR
- 🔧 **최적화**: Unsloth, LoRA, 4-bit quantization
- 📊 **벤치마킹**: KoCoder, HumanEval 평가
- 🗃️ **데이터 생성**: RAG 기반 instruction 데이터 자동 생성
- 🔍 **검증 도구**: 데이터셋 품질 검증, 리소스 모니터링
- 💾 **효율적 학습**: Gradient checkpointing, Mixed precision 

## Deployed Model

**HuggingFace**: [MyeongHo0621/eeve-vss-smh](https://huggingface.co/MyeongHo0621/eeve-vss-smh)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "MyeongHo0621/eeve-vss-smh",
    device_map="auto",
    torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained("MyeongHo0621/eeve-vss-smh")
```

## Model Information

- **Base Model**: [yanolja/EEVE-Korean-Instruct-10.8B-v1.0](https://huggingface.co/yanolja/EEVE-Korean-Instruct-10.8B-v1.0)
- **How to fine-tune**: LoRA (r=128, alpha=256) + Unsloth
- **Data**: 고품질 한국어 instruction 데이터 (~100K 샘플)

## Train envrionment & configuration

### H/W info
- **GPU**: NVIDIA H100 80GB HBM3
- **CPU**: 24 cores
- **RAM**: 192GB
- **Framework**: Unsloth + PyTorch 2.8, Transformers 4.56.2

### LoRA configuration 
- **r**: 128 
- **alpha**: 256 (alpha = 2 * r)
- **dropout**: 0.0 (Only 0.0)
- **target_modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **use_rslora**: false

### Training Hyper Parameter 
- **Framework**: Unsloth 
- **Epochs**: 3 
- **Batch Size**: 8 
- **Gradient Accumulation**: 2 
- **Learning Rate**: 1e-4
- **Max Sequence Length**: 4096 tokens
- **Warmup Ratio**: 0.05
- **Weight Decay**: 0.01

### Memory Optimization
- **Full Precision Training**
- **Unsloth Gradient Checkpointing**
- **BF16 Training**
- **Peak VRAM**

## Directory Structure

```
tesseract/
├── eeve/                           # EEVE-Korean-10.8B Fine-tuning
│   ├── 0_unsl_ft.py               # Main training script
│   ├── 1_cp_ft.py                 # Checkpoint resume training
│   ├── 2_merg_uplod.py            # Model merge & HuggingFace upload
│   ├── 3_test_checkpoint.py       # Checkpoint testing
│   ├── UNSLOTH_GUIDE.md           # Unsloth optimization guide
│   └── quant/                     # Quantization scripts
│
├── qwen/                           # Qwen2.5 Fine-tuning
│   ├── 0_qwen_ft_us_cp.py         # Qwen training with checkpoint
│   ├── util/                      # Training utilities
│   │   ├── cpu_mntrg.py           # CPU monitoring
│   │   ├── gpu_mnrtg.py           # GPU monitoring
│   │   ├── local_dataset_loader.py # Dataset loader
│   │   └── monitoring_callback.py  # Training callback
│   └── 4_credential/              # Credentials (empty)
│
├── vclm/                           # VCLM-Korean-7B Evaluation
│   ├── benchmark_vclm_kocoder.py  # KoCoder benchmarking
│   ├── benchmark_kocoder_final.py # Final evaluation
│   └── .gitattributes             # LFS configuration
│
├── solar/                          # SOLAR-10.7B (Legacy)
│   └── ...                        # Archived fine-tuning scripts
│
├── datageneration/                 # Data Generation Pipeline
│   ├── inst_eeve/                 # EEVE instruction data
│   │   ├── train_eeve_wms.py      # WMS training data
│   │   ├── train_eeve_wms_fp8.py  # FP8 training
│   │   └── test_eeve_wms.py       # Testing
│   ├── instruction/               # Instruction generation
│   │   ├── compare_all_models.py  # Model comparison
│   │   └── convert_to_eeve.py     # Format conversion
│   └── valid/                     # Validation tools
│       ├── validtest.py           # General validation
│       └── validate_qa_dataset.py # QA validation
│
├── eval_computing_resource/        # Resource Evaluation
│   └── eval.py                    # Computing resource profiling
│
├── faiss_storage/                  # Vector Database (gitignored)
│   └── ...                        # FAISS index for RAG
│
└── korean_large_data/              # Large Datasets (gitignored)
    └── ...                        # Training datasets
```

---

## 1. EEVE-Korean-10.8B Fine-tuning

### Model Information

- **Base Model**: [yanolja/EEVE-Korean-Instruct-10.8B-v1.0](https://huggingface.co/yanolja/EEVE-Korean-Instruct-10.8B-v1.0)
- **Deployed Model**: [MyeongHo0621/eeve-vss-smh](https://huggingface.co/MyeongHo0621/eeve-vss-smh)
- **Status**: ✅ Training Complete & HuggingFace Deployment Complete
- **Vocab Size**: 40,960 (한영 balanced)
- **Context Length**: 8K tokens
- **Method**: LoRA (r=128, alpha=256) + Unsloth

### Training Configuration

#### Hardware
- **GPU**: NVIDIA H100 80GB HBM3
- **CPU**: 24 cores
- **RAM**: 192GB
- **Framework**: Unsloth + PyTorch 2.8, Transformers 4.56.2

#### LoRA Settings
- **r**: 128 
- **alpha**: 256 (alpha = 2 * r)
- **dropout**: 0.0
- **target_modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **use_rslora**: false

#### Training Hyperparameters
- **Epochs**: 3 
- **Batch Size**: 8 
- **Gradient Accumulation**: 2 
- **Learning Rate**: 1e-4
- **Max Sequence Length**: 4096 tokens
- **Warmup Ratio**: 0.05
- **Weight Decay**: 0.01

#### Results
- **Training time**: ~3 hours (6,250 steps)
- **Peak VRAM**: ~26GB
- **Checkpoint Interval**: 250 steps

### How to Use

#### 1. HuggingFace (Recommended)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# model load
model = AutoModelForCausalLM.from_pretrained(
    "MyeongHo0621/eeve-vss-smh",
    device_map="auto",
    torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained("MyeongHo0621/eeve-vss-smh")

# prompt Template
def create_prompt(user_input):
    return f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: {user_input}
Assistant: """

# generating response
prompt = create_prompt("한국의 수도가 어디야?")
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.3,
    top_p=0.85,
    do_sample=True
)
response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
print(response)
```

#### 2. Re-Training from Checkpoint

```bash
cd eeve

# Train from scratch
python 0_unsl_ft.py

# Resume from checkpoint
python 1_cp_ft.py

# Merge LoRA and upload to HuggingFace
python 2_merg_uplod.py

# Test checkpoints
python 3_test_checkpoint.py --compare \
  /path/to/checkpoint-1 \
  /path/to/checkpoint-2
```

#### 3. Model Load (Python API)

#### 기본 로드 (4-bit 양자화)
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 4bit Quantization Configuration 
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Base Model Load
base_model = AutoModelForCausalLM.from_pretrained(
    "yanolja/EEVE-Korean-Instruct-10.8B-v1.0",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

# LoRA Adaptor load
model = PeftModel.from_pretrained(
    base_model, 
    "/home/work/eeve-korean-output/final",
    is_trainable=False
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "/home/work/eeve-korean-output/final",
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

#### Text Generation (EEVE Prompt Template)
```python
def generate_response(user_input, max_tokens=512):
    # EEVE Official Prompt Template
    prompt = f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: {user_input}
Assistant: """
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096
    )
    
    input_length = inputs.input_ids.shape[1]
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,           # 자연스러운 다양성
            top_p=0.9,                # Nucleus sampling
            top_k=50,
            repetition_penalty=1.1,    # 반복 방지
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(
        outputs[0][input_length:], 
        skip_special_tokens=True
    ).strip()
    
    return response

# example
print(generate_response("한국의 수도가 어디야?"))
print(generate_response("피보나치 수열 설명해봐"))
```

### Training Strategy

#### Label Masking
```python
# 프롬프트 부분은 -100으로 마스킹 (loss 계산 제외)
labels = input_ids.clone()
labels[:prompt_length] = -100  # 프롬프트 마스킹
labels[labels == pad_token_id] = -100  # 패딩 마스킹
```

**Why Label Masking?**
- 사용자 질문은 학습하지 않음
- 어시스턴트 답변만 학습
- 자연스러운 대화 스타일 형성

#### Memory Optimization
1. **4-bit Quantization (NF4)**: 모델 크기 1/4로 축소
2. **Gradient Checkpointing**: 메모리 사용량 감소
3. **LoRA**: 전체 파라미터의 ~0.5%만 학습
4. **BF16 Training**: H100 하드웨어 최적화

**결과**: 80GB GPU에서 26GB만 사용

---

## 2. Qwen2.5 Fine-tuning

### Model Information

- **Base Model**: Qwen2.5 series
- **Status**: 🔄 In Progress
- **Method**: Unsupervised learning with checkpoint support
- **Features**: CPU/GPU monitoring, custom dataset loader

### Training Scripts

```bash
cd qwen

# Main training script with checkpoint support
python 0_qwen_ft_us_cp.py
```

### Utilities

#### Monitoring Tools
- `util/cpu_mntrg.py`: CPU usage monitoring
- `util/gpu_mnrtg.py`: GPU usage monitoring (NVIDIA-SMI)
- `util/monitoring_callback.py`: Training callback with resource tracking

#### Dataset Loader
- `util/local_dataset_loader.py`: Custom dataset loading utilities

### Key Features
- ✅ Checkpoint save/resume
- ✅ Real-time resource monitoring
- ✅ Custom dataset pipeline
- ✅ Distributed training support

---

## 3. VCLM-Korean-7B Benchmarking

### Model Information

- **Model**: VCLM-Korean-7B (Quantized GGUF format)
- **Status**: ✅ Benchmarking Complete
- **Benchmark**: KoCoder evaluation

### Evaluation Scripts

```bash
cd vclm

# Run KoCoder benchmark
python benchmark_vclm_kocoder.py

# Final evaluation
python benchmark_kocoder_final.py
```

### Benchmark Results

VCLM-Korean-7B의 코드 생성 능력을 KoCoder 벤치마크로 평가합니다.

- **평가 항목**: 한국어 코드 생성, 주석 작성, 디버깅
- **모델 형식**: GGUF (Q4_K_M quantization)
- **실행 환경**: llama.cpp 기반

---

## 4. Data Generation Pipeline

### Instruction Data Generation

#### EEVE Instruction Data (`datageneration/inst_eeve/`)

```bash
cd datageneration/inst_eeve

# Generate training data for WMS domain
python train_eeve_wms.py

# FP8 precision training data
python train_eeve_wms_fp8.py

# Test data generation
python test_eeve_wms.py
```

#### General Instruction Tools (`datageneration/instruction/`)

```bash
cd datageneration/instruction

# Compare outputs from multiple models
python compare_all_models.py

# Convert datasets to EEVE format
python convert_to_eeve.py
```

### Data Validation (`datageneration/valid/`)

```bash
cd datageneration/valid

# General dataset validation
python validtest.py

# QA dataset validation
python validate_qa_dataset.py
```

**Features**:
- RAG 기반 자동 데이터 생성
- WMS(창고관리) 도메인 특화
- 데이터 품질 검증
- 다중 모델 출력 비교

---

## 5. Resource Evaluation

### Computing Resource Profiling

```bash
cd eval_computing_resource

# Profile system resources during training
python eval.py
```

**Metrics**:
- GPU utilization & memory
- CPU usage & load
- Memory consumption
- Training throughput

---

## 6. Training Environment

### Hardware Requirements

| Component | Recommended | Minimum |
|-----------|-------------|---------|
| **GPU** | H100 80GB | RTX 3090 24GB |
| **CPU** | 24+ cores | 8+ cores |
| **RAM** | 192GB | 64GB |
| **Storage** | 1TB NVMe | 500GB SSD |

### Software Dependencies

```bash
# PyTorch & Transformers
pip install torch transformers accelerate

# Optimization libraries
pip install unsloth bitsandbytes peft

# Monitoring & utilities
pip install psutil nvidia-ml-py3 tqdm

# Data processing
pip install datasets faiss-cpu pandas
```

--- 


## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/EnzoMH/ft_llm.git
cd ft_llm
```

### 2. Install Dependencies

```bash
pip install torch transformers accelerate unsloth bitsandbytes peft
pip install psutil nvidia-ml-py3 datasets faiss-cpu
```

### 3. Run Training

```bash
# EEVE Fine-tuning
cd eeve
python 0_unsl_ft.py

# Qwen Fine-tuning
cd qwen
python 0_qwen_ft_us_cp.py

# VCLM Benchmarking
cd vclm
python benchmark_vclm_kocoder.py
```

---

## Best Practices

### Memory Optimization Tips
1. **4-bit Quantization**: 메모리 사용량 75% 감소
2. **Gradient Checkpointing**: 추가 30% 메모리 절약
3. **LoRA**: Full fine-tuning 대비 99.5% 파라미터 감소
4. **Mixed Precision (BF16)**: 학습 속도 2배 향상

### Training Tips
- Checkpoint 자주 저장 (250-500 steps)
- Learning rate warmup 사용 (5-10%)
- Gradient accumulation으로 effective batch size 증가
- Label masking으로 instruction tuning 품질 향상

### Data Quality
- 데이터 검증 도구로 품질 확인 (`datageneration/valid/`)
- 중복 데이터 제거
- Instruction-response 형식 일관성 유지
- 도메인 특화 데이터로 성능 향상

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Contribution Guidelines
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

이 프로젝트는 각 베이스 모델의 라이선스를 따릅니다:

- **EEVE-Korean-10.8B**: [Apache 2.0](https://huggingface.co/yanolja/EEVE-Korean-Instruct-10.8B-v1.0)
- **Qwen2.5**: [Apache 2.0](https://huggingface.co/Qwen)
- **VCLM-Korean-7B**: 해당 모델 라이선스 참조
- **SOLAR-10.7B**: [Apache 2.0](https://huggingface.co/upstage/SOLAR-10.7B-v1.0)

---

## Acknowledgments

### Model Providers
- **[Yanolja (EEVE Team)](https://huggingface.co/yanolja)**: EEVE-Korean-Instruct-10.8B
- **[Alibaba Cloud (Qwen Team)](https://huggingface.co/Qwen)**: Qwen2.5 series
- **[VCLM Team](https://huggingface.co/VCLM)**: VCLM-Korean-7B
- **[Upstage](https://huggingface.co/upstage)**: SOLAR-10.7B

### Infrastructure
- **KT Cloud**: H100 GPU 인프라 제공

### Libraries & Tools
- **[Unsloth](https://github.com/unslothai/unsloth)**: 2배 빠른 학습 속도, 메모리 최적화
- **[Hugging Face](https://huggingface.co)**: Transformers, PEFT, Datasets, TRL
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)**: GGUF 모델 추론

### Datasets
- **한국어 데이터셋 기여자들**: KoAlpaca, Kullm-v2, Smol Korean Talk, KoWiki QA 등

---

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{korean-llm-finetuning-suite,
  author = {MyeongHo},
  title = {Korean LLM Fine-tuning & Evaluation Suite},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/EnzoMH/ft_llm}
}
```

---

## Contact

- **GitHub**: [@EnzoMH](https://github.com/EnzoMH)
- **HuggingFace**: [MyeongHo0621](https://huggingface.co/MyeongHo0621)

---

**Made with 🔥 for Korean NLP Community**
