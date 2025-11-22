#!/usr/bin/env python3
"""
Step 2: Merged 모델을 GGUF로 변환
- Llama.cpp, Ollama 호환 포맷
- 여러 양자화 레벨 (Q4_K_M, Q5_K_M, Q8_0, F16)
"""

import os
import subprocess
from pathlib import Path

# 설정
MERGED_MODEL_DIR = "/home/work/.setting/qwen/2.5_distil/outputs/merged"
OUTPUT_DIR = "/home/work/.setting/qwen/2.5_distil/outputs/gguf"
LLAMA_CPP_DIR = "/home/work/llama.cpp"  # llama.cpp 경로

# Step 1: HF → F16 GGUF 변환 (convert_hf_to_gguf.py)
# Step 2: F16 → 양자화 레벨 (llama-quantize)
QUANTIZATION_LEVELS = [
    ("Q4_K_M", "4-bit 중간 품질 (권장, 빠름)"),
    ("Q5_K_M", "5-bit 중간 품질 (균형)"),
    ("Q8_0", "8-bit 고품질"),
    ("F16", "16-bit 원본 (변환만)")
]

print("=" * 80)
print("GGUF 변환 시작")
print("=" * 80)

# 1. llama.cpp 확인
if not Path(LLAMA_CPP_DIR).exists():
    print(f"\n❌ llama.cpp를 찾을 수 없습니다: {LLAMA_CPP_DIR}")
    print(f"\n📥 llama.cpp 클론 중...")
    subprocess.run([
        "git", "clone", "https://github.com/ggerganov/llama.cpp",
        LLAMA_CPP_DIR
    ], check=True)
    print(f"  ✅ llama.cpp 클론 완료")
    
    print(f"\n🔧 llama.cpp 빌드 중 (CMake)...")
    # CMake 빌드 (최신 llama.cpp)
    build_dir = Path(LLAMA_CPP_DIR) / "build"
    build_dir.mkdir(exist_ok=True)
    
    # CMake 설정
    subprocess.run([
        "cmake", "-B", str(build_dir), "-S", LLAMA_CPP_DIR,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGGML_CUDA=ON"  # GPU 지원
    ], check=True)
    
    # 빌드
    subprocess.run([
        "cmake", "--build", str(build_dir), "--config", "Release", "-j"
    ], check=True)
    
    print(f"  ✅ llama.cpp 빌드 완료")
else:
    print(f"✅ llama.cpp 확인됨: {LLAMA_CPP_DIR}")
    
    # 빌드가 안 되어 있으면 빌드
    build_dir = Path(LLAMA_CPP_DIR) / "build"
    if not build_dir.exists():
        print(f"\n🔧 llama.cpp 빌드 중 (CMake)...")
        build_dir.mkdir(exist_ok=True)
        
        subprocess.run([
            "cmake", "-B", str(build_dir), "-S", LLAMA_CPP_DIR,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_CUDA=ON"
        ], check=True)
        
        subprocess.run([
            "cmake", "--build", str(build_dir), "--config", "Release", "-j"
        ], check=True)
        
        print(f"  ✅ llama.cpp 빌드 완료")

# 2. 변환 스크립트 확인 (여러 가능한 이름 시도)
possible_scripts = [
    Path(LLAMA_CPP_DIR) / "convert_hf_to_gguf.py",
    Path(LLAMA_CPP_DIR) / "convert-hf-to-gguf.py",
    Path(LLAMA_CPP_DIR) / "convert.py"
]

convert_script = None
for script in possible_scripts:
    if script.exists():
        convert_script = script
        break

if not convert_script:
    print(f"❌ 변환 스크립트를 찾을 수 없습니다")
    print(f"  시도한 경로:")
    for script in possible_scripts:
        print(f"    - {script}")
    exit(1)

print(f"✅ 변환 스크립트 확인됨: {convert_script.name}")

# 3. Merged 모델 확인
if not Path(MERGED_MODEL_DIR).exists():
    print(f"\n❌ Merged 모델을 찾을 수 없습니다: {MERGED_MODEL_DIR}")
    print(f"  💡 먼저 1_merge_lora.py를 실행하세요")
    exit(1)

print(f"✅ Merged 모델 확인됨: {MERGED_MODEL_DIR}")

# 4. 출력 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 5. Step 1: HF → F16 GGUF 변환
f16_file = Path(OUTPUT_DIR) / "qwen25-3b-korean-F16.gguf"

print(f"\n[ Step 1 ] HF → F16 GGUF 변환")
print(f"  ℹ️  출력: {f16_file.name}")

if f16_file.exists():
    print(f"  ⚠️  F16 파일이 이미 존재합니다, 스킵")
    size_mb = f16_file.stat().st_size / (1024 * 1024)
    print(f"  ℹ️  크기: {size_mb:.2f} MB")
else:
    print(f"  🔄 변환 중...")
    cmd = [
        "python", str(convert_script),
        MERGED_MODEL_DIR,
        "--outtype", "f16",
        "--outfile", str(f16_file)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"  ✅ F16 GGUF 변환 완료")
        size_mb = f16_file.stat().st_size / (1024 * 1024)
        print(f"  ℹ️  크기: {size_mb:.2f} MB")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ 변환 실패: {e}")
        exit(1)

# 6. Step 2: F16 → 양자화 레벨 (llama-quantize)
quantize_bin = Path(LLAMA_CPP_DIR) / "build" / "bin" / "llama-quantize"
if not quantize_bin.exists():
    # 대안 경로
    quantize_bin = Path(LLAMA_CPP_DIR) / "llama-quantize"

if not quantize_bin.exists():
    print(f"\n❌ llama-quantize를 찾을 수 없습니다")
    print(f"  시도한 경로:")
    print(f"    - {Path(LLAMA_CPP_DIR) / 'build' / 'bin' / 'llama-quantize'}")
    print(f"    - {Path(LLAMA_CPP_DIR) / 'llama-quantize'}")
    exit(1)

print(f"\n[ Step 2 ] F16 → 양자화 레벨")
print(f"  ℹ️  도구: {quantize_bin.name}")

# F16 제외하고 양자화 진행
for i, (quant_type, description) in enumerate([q for q in QUANTIZATION_LEVELS if q[0] != "F16"], 1):
    quant_type, description = quant_type, description
    output_file = Path(OUTPUT_DIR) / f"qwen25-3b-korean-{quant_type}.gguf"
    
    print(f"\n[ {i}/3 ] {quant_type}: {description}")
    print(f"  ℹ️  출력: {output_file.name}")
    
    if output_file.exists():
        print(f"  ⚠️  파일이 이미 존재합니다, 스킵")
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"  ℹ️  크기: {size_mb:.2f} MB")
        continue
    
    # llama-quantize 실행
    cmd = [
        str(quantize_bin),
        str(f16_file),
        str(output_file),
        quant_type
    ]
    
    print(f"  🔄 양자화 중...")
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"  ✅ 양자화 완료")
        
        # 파일 크기 출력
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"  ℹ️  크기: {size_mb:.2f} MB")
        
    except subprocess.CalledProcessError as e:
        print(f"  ❌ 양자화 실패: {e}")
        if e.stderr:
            print(f"  stderr: {e.stderr.decode()}")
        continue

# 6. 결과 요약
print(f"\n{'=' * 80}")
print(f"🎉 GGUF 변환 완료!")
print(f"{'=' * 80}")

print(f"\n📂 생성된 파일:")
gguf_files = sorted(Path(OUTPUT_DIR).glob("*.gguf"))
total_size = 0
for gguf_file in gguf_files:
    size_mb = gguf_file.stat().st_size / (1024 * 1024)
    total_size += size_mb
    print(f"  - {gguf_file.name:40s} ({size_mb:>8.2f} MB)")

print(f"\n📊 총 크기: {total_size:.2f} MB ({total_size / 1024:.2f} GB)")

print(f"\n💡 다음 단계: HuggingFace Hub 업로드")
print(f"  python 3_upload_to_hub.py")

# 7. 테스트 명령어 출력
print(f"\n🧪 테스트 명령어 (Llama.cpp):")
if gguf_files:
    test_file = gguf_files[0]  # 첫 번째 파일 (보통 Q4_K_M)
    print(f"""
{LLAMA_CPP_DIR}/main \\
    -m {test_file} \\
    -p "<|im_start|>user\\n한국의 수도는?<|im_end|>\\n<|im_start|>assistant\\n" \\
    -n 512 \\
    --temp 0.7 \\
    -ngl 99
""")

