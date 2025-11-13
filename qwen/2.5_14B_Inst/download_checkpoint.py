#!/usr/bin/env python3
"""
HuggingFace Hub에서 checkpoint 다운로드
"""

import os
import json
from huggingface_hub import HfApi, snapshot_download, hf_hub_download

# 설정
REPO_ID = "MyeongHo0621/Qwen2.5-14B-Korean"
HF_TOKEN = "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"
OUTPUT_DIR = "/home/work/tes/ft_llm/qwen/2.5_14B_Inst/output"

def main():
    """Hub에서 checkpoint 다운로드"""
    api = HfApi(token=HF_TOKEN)
    
    print("="*80)
    print("HuggingFace Hub에서 checkpoint 확인 중...")
    print("="*80)
    
    # 모든 파일 목록 가져오기
    try:
        files = api.list_repo_files(REPO_ID, repo_type="model")
        print(f"\n총 {len(files)}개 파일 발견")
        
        # checkpoint 디렉토리 찾기
        checkpoint_dirs = set()
        for f in files:
            if "checkpoint-" in f:
                parts = f.split("/")
                if len(parts) > 1:
                    checkpoint_dirs.add(parts[0])
        
        checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else 0)
        
        if checkpoint_dirs:
            print(f"\n발견된 checkpoint: {checkpoint_dirs}")
            latest_checkpoint = checkpoint_dirs[-1]
            print(f"\n가장 최근 checkpoint: {latest_checkpoint}")
            
            # checkpoint 다운로드
            print(f"\n다운로드 시작: {latest_checkpoint}")
            print(f"출력 경로: {OUTPUT_DIR}")
            
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            # 특정 checkpoint만 다운로드
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="model",
                local_dir=OUTPUT_DIR,
                local_dir_use_symlinks=False,
                token=HF_TOKEN,
                allow_patterns=f"{latest_checkpoint}/*"
            )
            
            print(f"\n[ COMPLETE ] {latest_checkpoint} 다운로드 완료!")
            print(f"경로: {OUTPUT_DIR}/{latest_checkpoint}")
        else:
            print("\n[ WARNING ] checkpoint 디렉토리를 찾을 수 없습니다!")
            print("[ INFO ] trainer_state.json 확인 중...")
            
            # trainer_state.json 확인
            try:
                trainer_state_path = hf_hub_download(
                    repo_id=REPO_ID,
                    filename="trainer_state.json",
                    repo_type="model",
                    token=HF_TOKEN,
                    cache_dir="/tmp"
                )
                with open(trainer_state_path, 'r') as f:
                    state = json.load(f)
                
                global_step = state.get('global_step', 0)
                epoch = state.get('epoch', 0)
                print(f"\n[ INFO ] 최종 학습 상태:")
                print(f"  Step: {global_step}")
                print(f"  Epoch: {epoch:.4f}")
                
                if global_step > 0:
                    print(f"\n[ WARNING ] Hub에 checkpoint 디렉토리가 없습니다.")
                    print(f"[ INFO ] Step {global_step}까지 학습했지만 checkpoint가 업로드되지 않았습니다.")
                    print(f"[ INFO ] Hub의 모든 파일을 다운로드합니다...")
                    
                    # 최종 모델 다운로드 (모든 파일)
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    
                    # Hub의 모든 파일 다운로드
                    print(f"\n[ INFO ] 전체 모델 파일 다운로드 중...")
                    snapshot_download(
                        repo_id=REPO_ID,
                        repo_type="model",
                        local_dir=OUTPUT_DIR,
                        token=HF_TOKEN,
                        ignore_patterns=["*.md", ".git*"]  # README와 git 파일 제외
                    )
                    
                    print(f"\n[ COMPLETE ] 모든 파일 다운로드 완료!")
                    print(f"[ WARNING ] checkpoint 디렉토리가 없어 optimizer state는 없습니다.")
                    print(f"[ INFO ] adapter 가중치는 로드되어 학습을 계속할 수 있습니다.")
                else:
                    print("\n[ INFO ] 학습 기록이 없습니다. 처음부터 시작하세요.")
            except Exception as e:
                print(f"\n[ ERROR ] trainer_state.json 확인 실패: {e}")
        
    except Exception as e:
        print(f"\n[ ERROR ] 다운로드 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

