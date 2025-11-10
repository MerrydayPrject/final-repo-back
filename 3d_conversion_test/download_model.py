"""
MiDaS Depth Estimation 모델 다운로드 스크립트
"""

import torch
from pathlib import Path

def download_midas_model():
    """MiDaS 모델 다운로드"""
    print("=" * 50)
    print("MiDaS Depth Estimation 모델 다운로드")
    print("=" * 50)
    
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    print(f"\n모델 저장 경로: {models_dir}")
    
    try:
        print("\n1. MiDaS 모델 다운로드 중...")
        # MiDaS 모델 로드 (자동으로 다운로드됨)
        model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        print("✓ MiDaS DPT_Large 모델 다운로드 완료!")
        
        # 작은 모델도 다운로드
        print("\n2. MiDaS Small 모델 다운로드 중...")
        model_small = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        print("✓ MiDaS Small 모델 다운로드 완료!")
        
        print("\n" + "=" * 50)
        print("모든 모델 다운로드 완료!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n✗ 다운로드 실패: {e}")
        print("\n모델은 첫 실행 시 자동으로 다운로드됩니다.")
        return False

if __name__ == "__main__":
    download_midas_model()

