import psutil
import cpuinfo
import subprocess
from typing import Dict, Any


def get_system_resources() -> Dict[str, Any]:
    """시스템 리소스 정보를 수집합니다."""
    resources = {}
    
    # CPU 정보
    try:
        result = subprocess.run(['lscpu'], capture_output=True, text=True, check=True)
        cpu_name = 'Unknown'
        for line in result.stdout.split('\n'):
            if 'Model name:' in line:
                cpu_name = line.split(':', 1)[1].strip()
                break
        resources['cpu_name'] = cpu_name
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            cpu_info = cpuinfo.get_cpu_info()
            resources['cpu_name'] = cpu_info.get('brand_raw', 'Unknown')
        except Exception:
            resources['cpu_name'] = 'Unknown'
    
    resources['cpu_count'] = psutil.cpu_count(logical=False)
    resources['cpu_count_logical'] = psutil.cpu_count(logical=True)
    resources['cpu_freq'] = psutil.cpu_freq()
    
    # RAM 정보
    memory = psutil.virtual_memory()
    resources['ram_total_gb'] = memory.total / (1024 ** 3)
    resources['ram_available_gb'] = memory.available / (1024 ** 3)
    resources['ram_used_gb'] = memory.used / (1024 ** 3)
    resources['ram_percent'] = memory.percent
    
    # GPU 정보 (nvidia-smi 시도)
    resources['gpus'] = []
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        gpu_lines = result.stdout.strip().split('\n')
        for line in gpu_lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                resources['gpus'].append({
                    'type': 'NVIDIA',
                    'name': parts[0],
                    'vram_total_gb': float(parts[1]) / 1024,
                    'vram_used_gb': float(parts[2]) / 1024,
                    'vram_free_gb': float(parts[3]) / 1024,
                })
    except (subprocess.CalledProcessError, FileNotFoundError):
        # NVIDIA GPU가 없으면 lspci로 GPU 정보 확인
        try:
            result = subprocess.run(
                ['lspci', '-v'],
                capture_output=True,
                text=True,
                check=True
            )
            lines = result.stdout.split('\n')
            current_gpu = None
            for line in lines:
                if 'VGA compatible controller' in line or 'Display controller' in line or '3D controller' in line:
                    # GPU 이름 추출
                    parts = line.split(': ', 1)
                    if len(parts) >= 2:
                        current_gpu = {
                            'type': 'Generic',
                            'name': parts[1].strip(),
                            'vram_total_gb': None,
                        }
                        resources['gpus'].append(current_gpu)
                elif current_gpu and 'Memory at' in line and 'prefetchable' in line:
                    # VRAM 크기 추정 (prefetchable memory)
                    if '[size=' in line:
                        size_str = line.split('[size=')[1].split(']')[0]
                        if 'G' in size_str:
                            size_gb = float(size_str.replace('G', '').strip())
                            if current_gpu['vram_total_gb'] is None:
                                current_gpu['vram_total_gb'] = size_gb
                        elif 'M' in size_str:
                            size_mb = float(size_str.replace('M', '').strip())
                            if current_gpu['vram_total_gb'] is None:
                                current_gpu['vram_total_gb'] = size_mb / 1024
                    current_gpu = None
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            resources['gpu_error'] = str(e)
    
    return resources


def print_system_report(resources: Dict[str, Any]) -> None:
    """시스템 리소스 정보를 출력합니다."""
    print("=" * 70)
    print("시스템 리소스 보고서")
    print("=" * 70)
    
    print("\n[CPU 정보]")
    print(f"  CPU 이름: {resources['cpu_name']}")
    print(f"  물리 코어: {resources['cpu_count']}개")
    print(f"  논리 코어: {resources['cpu_count_logical']}개")
    if resources['cpu_freq']:
        print(f"  현재 주파수: {resources['cpu_freq'].current:.2f} MHz")
        print(f"  최대 주파수: {resources['cpu_freq'].max:.2f} MHz")
    
    print("\n[RAM 정보]")
    print(f"  전체 용량: {resources['ram_total_gb']:.2f} GB")
    print(f"  사용 중: {resources['ram_used_gb']:.2f} GB")
    print(f"  사용 가능: {resources['ram_available_gb']:.2f} GB")
    print(f"  사용률: {resources['ram_percent']:.1f}%")
    
    print("\n[GPU 정보]")
    if resources['gpus']:
        for idx, gpu in enumerate(resources['gpus'], 1):
            print(f"  GPU {idx}: {gpu['name']}")
            if gpu.get('type') == 'NVIDIA' and 'vram_total_gb' in gpu:
                print(f"    VRAM 전체: {gpu['vram_total_gb']:.2f} GB")
                print(f"    VRAM 사용 중: {gpu['vram_used_gb']:.2f} GB")
                print(f"    VRAM 사용 가능: {gpu['vram_free_gb']:.2f} GB")
            elif gpu.get('vram_total_gb') is not None:
                print(f"    VRAM 전체 (추정): {gpu['vram_total_gb']:.2f} GB")
            else:
                print(f"    VRAM 정보: 측정 불가")
    else:
        print("  GPU를 찾을 수 없습니다.")
        if 'gpu_error' in resources:
            print(f"  오류: {resources['gpu_error']}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    resources = get_system_resources()
    print_system_report(resources)