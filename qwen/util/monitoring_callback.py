#!/usr/bin/env python3
"""
훈련 중 GPU/CPU 모니터링 Callback
- TrainerCallback에 통합
- 주기적으로 GPU/CPU 상태 로깅
- CSV 파일로 저장
"""

import os
import time
import csv
from datetime import datetime
from typing import Optional
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

# GPU/CPU 모니터 임포트
try:
    from .gpu_mnrtg import GPUMonitor
    from .cpu_mntrg import CPUMonitor
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("⚠ GPU/CPU 모니터를 로드할 수 없습니다. 모니터링이 비활성화됩니다.")


class ResourceMonitorCallback(TrainerCallback):
    """GPU/CPU 리소스 모니터링 Callback"""
    
    def __init__(
        self, 
        output_dir: str,
        log_interval: int = 50,  # 50 step마다 로깅
        detailed_interval: int = 500,  # 500 step마다 상세 로깅
    ):
        """
        Args:
            output_dir: 로그 파일 저장 디렉토리
            log_interval: 기본 로깅 주기 (steps)
            detailed_interval: 상세 로깅 주기 (steps)
        """
        self.output_dir = output_dir
        self.log_interval = log_interval
        self.detailed_interval = detailed_interval
        
        # 로그 파일 경로
        self.resource_log = os.path.join(output_dir, "resource_monitor.log")
        self.resource_csv = os.path.join(output_dir, "resource_stats.csv")
        
        # 모니터 초기화
        if GPU_AVAILABLE:
            self.gpu_monitor = GPUMonitor()
            self.cpu_monitor = CPUMonitor()
        else:
            self.gpu_monitor = None
            self.cpu_monitor = None
        
        # CSV 초기화
        self._init_csv()
        
        # 타이밍
        self.start_time = None
        self.last_log_time = None
    
    def _init_csv(self):
        """CSV 파일 초기화"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        with open(self.resource_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'step', 'epoch',
                # GPU
                'gpu0_mem_used_gb', 'gpu0_mem_total_gb', 'gpu0_mem_percent',
                'gpu0_utilization', 'gpu0_temperature',
                # CPU
                'cpu_percent', 'ram_used_gb', 'ram_total_gb', 'ram_percent',
                # 시간
                'elapsed_seconds', 'steps_per_second'
            ])
    
    def on_train_begin(self, args, state, control, **kwargs):
        """훈련 시작"""
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
        # 초기 상태 로깅
        with open(self.resource_log, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("리소스 모니터링 시작\n")
            f.write(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """로깅 시점에 리소스 모니터링"""
        if not GPU_AVAILABLE or self.gpu_monitor is None:
            return
        
        # log_interval마다 로깅
        if state.global_step % self.log_interval != 0:
            return
        
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # GPU/CPU 스냅샷
        gpu_snapshot = self.gpu_monitor.get_snapshot(0)
        cpu_snapshot = self.cpu_monitor.get_snapshot()
        
        # CSV 기록
        with open(self.resource_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            
            steps_per_sec = state.global_step / elapsed if elapsed > 0 else 0
            
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                state.global_step,
                round(state.epoch, 2) if state.epoch else 0,
                # GPU
                round(gpu_snapshot.memory_allocated_gb, 2) if gpu_snapshot else 0,
                round(gpu_snapshot.memory_total_gb, 2) if gpu_snapshot else 0,
                round(gpu_snapshot.memory_percent, 1) if gpu_snapshot else 0,
                round(gpu_snapshot.utilization_percent, 1) if gpu_snapshot else 0,
                round(gpu_snapshot.temperature, 0) if gpu_snapshot and gpu_snapshot.temperature else 0,
                # CPU
                round(cpu_snapshot.cpu_percent, 1),
                round(cpu_snapshot.ram_used_gb, 1),
                round(cpu_snapshot.ram_total_gb, 1),
                round(cpu_snapshot.ram_percent, 1),
                # 시간
                round(elapsed, 1),
                round(steps_per_sec, 3)
            ])
        
        # detailed_interval마다 상세 로깅
        if state.global_step % self.detailed_interval == 0:
            self._detailed_log(state.global_step, gpu_snapshot, cpu_snapshot)
    
    def _detailed_log(self, step, gpu_snapshot, cpu_snapshot):
        """상세 리소스 로깅"""
        with open(self.resource_log, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"[Step {step}] 리소스 상세 정보\n")
            f.write(f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 80 + "\n")
            
            # GPU
            if gpu_snapshot:
                f.write(f"GPU ({gpu_snapshot.name}):\n")
                f.write(f"  메모리: {gpu_snapshot.memory_allocated_gb:.2f}GB / "
                       f"{gpu_snapshot.memory_total_gb:.2f}GB ({gpu_snapshot.memory_percent:.1f}%)\n")
                f.write(f"  예약 메모리: {gpu_snapshot.memory_reserved_gb:.2f}GB\n")
                f.write(f"  사용률: {gpu_snapshot.utilization_percent:.1f}%\n")
                
                if gpu_snapshot.temperature:
                    f.write(f"  온도: {gpu_snapshot.temperature:.0f}°C\n")
                
                if gpu_snapshot.power_usage and gpu_snapshot.power_limit:
                    f.write(f"  전력: {gpu_snapshot.power_usage:.0f}W / "
                           f"{gpu_snapshot.power_limit:.0f}W\n")
                
                # 메모리 상세
                mem_summary = self.gpu_monitor.get_memory_summary(0)
                if mem_summary:
                    f.write(f"  메모리 단편화: {mem_summary['fragmentation']:.1f}%\n")
                    f.write(f"  활성 메모리: {mem_summary['active_gb']:.2f}GB\n")
            
            f.write("\n")
            
            # CPU
            if cpu_snapshot:
                f.write(f"CPU:\n")
                f.write(f"  사용률: {cpu_snapshot.cpu_percent:.1f}% "
                       f"({cpu_snapshot.cpu_count} cores @ {cpu_snapshot.cpu_freq_current:.0f} MHz)\n")
                f.write(f"  RAM: {cpu_snapshot.ram_used_gb:.1f}GB / "
                       f"{cpu_snapshot.ram_total_gb:.1f}GB ({cpu_snapshot.ram_percent:.1f}%)\n")
                
                if cpu_snapshot.swap_total_gb > 0:
                    f.write(f"  Swap: {cpu_snapshot.swap_used_gb:.1f}GB / "
                           f"{cpu_snapshot.swap_total_gb:.1f}GB\n")
                
                f.write(f"  프로세스 CPU: {cpu_snapshot.process_cpu_percent:.1f}%\n")
                f.write(f"  프로세스 RAM: {cpu_snapshot.process_ram_gb:.2f}GB\n")
                f.write(f"  스레드: {cpu_snapshot.process_threads}\n")
            
            f.write("=" * 80 + "\n")
    
    def on_train_end(self, args, state, control, **kwargs):
        """훈련 종료"""
        if not GPU_AVAILABLE:
            return
        
        total_time = time.time() - self.start_time
        
        with open(self.resource_log, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("리소스 모니터링 종료\n")
            f.write(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"총 소요 시간: {total_time/3600:.2f}시간\n")
            f.write("=" * 80 + "\n")
        
        # 요약 통계
        if self.gpu_monitor and self.cpu_monitor:
            gpu_summary = self.gpu_monitor.get_summary(0)
            cpu_summary = self.cpu_monitor.get_summary()
            
            with open(self.resource_log, 'a') as f:
                f.write("\n" + "=" * 80 + "\n")
                f.write("리소스 사용 요약\n")
                f.write("-" * 80 + "\n")
                
                if gpu_summary:
                    f.write(f"GPU 메모리 (평균/최대/최소):\n")
                    f.write(f"  {gpu_summary['memory_avg']:.1f}% / "
                           f"{gpu_summary['memory_max']:.1f}% / "
                           f"{gpu_summary['memory_min']:.1f}%\n")
                    f.write(f"GPU 사용률 (평균/최대/최소):\n")
                    f.write(f"  {gpu_summary['utilization_avg']:.1f}% / "
                           f"{gpu_summary['utilization_max']:.1f}% / "
                           f"{gpu_summary['utilization_min']:.1f}%\n")
                
                if cpu_summary:
                    f.write(f"\nCPU 사용률 (평균/최대/최소):\n")
                    f.write(f"  {cpu_summary['cpu_avg']:.1f}% / "
                           f"{cpu_summary['cpu_max']:.1f}% / "
                           f"{cpu_summary['cpu_min']:.1f}%\n")
                    f.write(f"RAM 사용률 (평균/최대/최소):\n")
                    f.write(f"  {cpu_summary['ram_avg']:.1f}% / "
                           f"{cpu_summary['ram_max']:.1f}% / "
                           f"{cpu_summary['ram_min']:.1f}%\n")
                
                f.write("=" * 80 + "\n")


if __name__ == "__main__":
    # 테스트
    import sys
    sys.path.append('/home/work/tesseract/qwen')
    
    from util.gpu_mnrtg import GPUMonitor
    from util.cpu_mntrg import CPUMonitor
    
    print("GPU/CPU 모니터 테스트")
    
    gpu_monitor = GPUMonitor()
    cpu_monitor = CPUMonitor()
    
    if gpu_monitor.available:
        gpu_snapshot = gpu_monitor.get_snapshot(0)
        print(f"\nGPU: {gpu_snapshot.name}")
        print(f"  메모리: {gpu_snapshot.memory_allocated_gb:.2f}GB / "
              f"{gpu_snapshot.memory_total_gb:.2f}GB ({gpu_snapshot.memory_percent:.1f}%)")
        print(f"  사용률: {gpu_snapshot.utilization_percent:.1f}%")
    
    cpu_snapshot = cpu_monitor.get_snapshot()
    print(f"\nCPU: {cpu_snapshot.cpu_percent:.1f}%")
    print(f"RAM: {cpu_snapshot.ram_used_gb:.1f}GB / "
          f"{cpu_snapshot.ram_total_gb:.1f}GB ({cpu_snapshot.ram_percent:.1f}%)")

