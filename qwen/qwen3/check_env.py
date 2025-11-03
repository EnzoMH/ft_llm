#!/usr/bin/env python3
"""
환경 설정 검증 스크립트
- CUDA, PyTorch, GPU 확인
- 주요 라이브러리 버전 체크
- H100 최적화 설정 검증
"""

import sys
import os
from importlib import import_module


def print_header(title):
    """섹션 헤더 출력"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n")


def check_python():
    """Python 버전 확인"""
    print_header("Python 환경")
    
    version = sys.version_info
    print(f"Python 버전: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 10:
        print("  ✅ Python 3.10+ (권장)")
    else:
        print("  ⚠️  Python 3.10+ 권장 (현재: {}.{})".format(version.major, version.minor))
    
    print(f"실행 경로: {sys.executable}")


def check_cuda():
    """CUDA 및 GPU 확인"""
    print_header("CUDA 및 GPU")
    
    try:
        import torch
        
        # CUDA 사용 가능 여부
        cuda_available = torch.cuda.is_available()
        print(f"CUDA 사용 가능: {'✅ Yes' if cuda_available else '❌ No'}")
        
        if cuda_available:
            # CUDA 버전
            cuda_version = torch.version.cuda
            print(f"CUDA 버전: {cuda_version}")
            
            # 권장 버전 확인
            if cuda_version and cuda_version.startswith("12.8"):
                print("  ✅ CUDA 12.8 (H100 최적)")
            elif cuda_version and cuda_version.startswith("12."):
                print("  ⚠️  CUDA 12.x (작동 가능, 12.8 권장)")
            
            # GPU 정보
            num_gpus = torch.cuda.device_count()
            print(f"\nGPU 개수: {num_gpus}")
            
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"\nGPU {i}:")
                print(f"  이름: {gpu_name}")
                print(f"  메모리: {gpu_memory:.1f} GB")
                
                # H100 확인
                if "H100" in gpu_name:
                    print("  ✅ H100 GPU 감지 (최고 성능)")
                elif "A100" in gpu_name:
                    print("  ✅ A100 GPU (고성능)")
                
                # Compute Capability
                capability = torch.cuda.get_device_capability(i)
                print(f"  Compute Capability: {capability[0]}.{capability[1]}")
                
                if capability[0] >= 9:  # Hopper (H100)
                    print("  ✅ Hopper 아키텍처 (SM 9.x) - Flash-Attention 3 최적")
                elif capability[0] >= 8:  # Ampere (A100)
                    print("  ✅ Ampere 아키텍처 (SM 8.x)")
            
            # cuDNN
            if torch.backends.cudnn.is_available():
                cudnn_version = torch.backends.cudnn.version()
                print(f"\ncuDNN 버전: {cudnn_version}")
                print(f"cuDNN 활성화: ✅ Yes")
            
            # TF32
            tf32_enabled = torch.backends.cuda.matmul.allow_tf32
            print(f"TF32 활성화: {'✅ Yes' if tf32_enabled else '❌ No (권장: 활성화)'}")
            
            # BF16 지원
            bf16_support = torch.cuda.is_bf16_supported()
            print(f"BF16 지원: {'✅ Yes' if bf16_support else '❌ No'}")
            
        else:
            print("❌ CUDA를 사용할 수 없습니다.")
            print("   GPU 드라이버 및 CUDA 설치를 확인하세요.")
    
    except ImportError:
        print("❌ PyTorch가 설치되지 않았습니다.")


def check_libraries():
    """주요 라이브러리 버전 확인"""
    print_header("라이브러리 버전")
    
    libraries = [
        ("torch", "PyTorch", "2.8.0"),
        ("transformers", "Transformers", "4.56.0"),
        ("tokenizers", "Tokenizers", "0.22.0"),
        ("accelerate", "Accelerate", "1.5.0"),
        ("datasets", "Datasets", "3.0.0"),
        ("peft", "PEFT", "0.14.0"),
        ("trl", "TRL", "0.24.0"),
        ("bitsandbytes", "BitsAndBytes", "0.45.0"),
        ("vllm", "vLLM", "0.8.0"),
        ("flash_attn", "Flash-Attention", "2.7.0"),
        ("unsloth", "Unsloth", None),
        ("sentence_transformers", "Sentence-Transformers", "2.8.0"),
    ]
    
    installed = []
    missing = []
    
    for module_name, display_name, min_version in libraries:
        try:
            module = import_module(module_name)
            version = getattr(module, "__version__", "알 수 없음")
            
            status = "✅"
            note = ""
            
            # 버전 비교 (간단 버전)
            if min_version and version != "알 수 없음":
                try:
                    from packaging import version as pkg_version
                    if pkg_version.parse(version) < pkg_version.parse(min_version):
                        status = "⚠️"
                        note = f" (권장: {min_version}+)"
                except:
                    pass
            
            print(f"{status} {display_name:20} {version:15} {note}")
            installed.append(display_name)
            
        except ImportError:
            print(f"❌ {display_name:20} 설치 안 됨")
            missing.append(display_name)
    
    print(f"\n설치됨: {len(installed)}/{len(libraries)}")
    if missing:
        print(f"누락됨: {', '.join(missing)}")


def check_flash_attention():
    """Flash-Attention 상세 확인"""
    print_header("Flash-Attention 확인")
    
    try:
        import flash_attn
        version = getattr(flash_attn, "__version__", "알 수 없음")
        
        print(f"Flash-Attention 버전: {version}")
        
        # 버전별 추천
        if version.startswith("3."):
            print("  ✅ Flash-Attention 3 (H100 최적)")
        elif version.startswith("2."):
            print("  ✅ Flash-Attention 2 (안정적)")
        
        # FlashInfer 확인
        try:
            import flashinfer
            print(f"FlashInfer 설치: ✅ (vLLM 최적화)")
        except ImportError:
            print(f"FlashInfer 설치: ❌ (선택적)")
        
    except ImportError:
        print("❌ Flash-Attention이 설치되지 않았습니다.")
        print("   설치: pip install flash-attn --no-build-isolation")


def check_environment_variables():
    """환경 변수 확인"""
    print_header("환경 변수")
    
    important_vars = [
        ("CUDA_VISIBLE_DEVICES", "0 (기본값)"),
        ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,..."),
        ("OMP_NUM_THREADS", "16 (권장)"),
        ("TOKENIZERS_PARALLELISM", "true"),
        ("VLLM_ATTENTION_BACKEND", "FLASHINFER (H100 최적)"),
        ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1 (H100 최적)"),
        ("HF_HOME", "캐시 경로"),
    ]
    
    for var, description in important_vars:
        value = os.environ.get(var)
        if value:
            print(f"✅ {var:35} = {value}")
        else:
            print(f"⚠️  {var:35} = (미설정) - {description}")
    
    print("\n환경 변수 설정: source env.sh")


def check_disk_space():
    """디스크 공간 확인"""
    print_header("디스크 공간")
    
    import shutil
    
    paths = [
        ("/home/work/tesseract/qwen/qwen3", "프로젝트 디렉토리"),
        ("/home/work/.cache/huggingface", "Hugging Face 캐시"),
    ]
    
    for path, description in paths:
        if os.path.exists(path):
            total, used, free = shutil.disk_usage(path)
            
            free_gb = free / (1024**3)
            total_gb = total / (1024**3)
            
            print(f"{description}:")
            print(f"  경로: {path}")
            print(f"  여유 공간: {free_gb:.1f} GB / {total_gb:.1f} GB")
            
            if free_gb < 50:
                print(f"  ⚠️  공간 부족 (50GB 이상 권장)")
            else:
                print(f"  ✅ 충분")
        else:
            print(f"{description}: ❌ 경로 없음 ({path})")


def main():
    """메인 검증 함수"""
    print("\n" + "="*80)
    print("Qwen3-VL 환경 검증")
    print("="*80)
    
    check_python()
    check_cuda()
    check_libraries()
    check_flash_attention()
    check_environment_variables()
    check_disk_space()
    
    print_header("권장 사항")
    
    print("H100 GPU 최적 설정:")
    print("  1. source env.sh          # 환경 변수 로드")
    print("  2. CUDA 12.8 사용")
    print("  3. Flash-Attention 3 설치 (Beta)")
    print("  4. TF32 활성화")
    print("  5. BF16 Mixed Precision")
    print("")
    print("설치 명령:")
    print("  pip install -r requirements-cu128.txt")
    print("")
    print("="*80)


if __name__ == "__main__":
    main()

