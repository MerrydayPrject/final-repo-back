"""
MediaPipe Pose Landmarker 모델 다운로드 스크립트
"""
import urllib.request
import os
from pathlib import Path

def download_model():
    """MediaPipe Pose Landmarker 모델 다운로드"""
    model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    
    # 모델 저장 경로 (models/body_analysis/)
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models" / "body_analysis"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "pose_landmarker_lite.task"
    
    if model_path.exists():
        print(f"✅ 모델 파일이 이미 존재합니다: {model_path}")
        print("다운로드를 건너뜁니다.")
        return
    
    try:
        print(f"모델 다운로드 중...")
        print(f"URL: {model_url}")
        print(f"저장 경로: {model_path}")
        
        urllib.request.urlretrieve(model_url, str(model_path))
        
        # 파일 크기 확인
        file_size = model_path.stat().st_size / (1024 * 1024)  # MB
        print(f"✅ 모델 다운로드 완료!")
        print(f"   파일 크기: {file_size:.2f} MB")
        print(f"   저장 위치: {model_path}")
        
    except Exception as e:
        print(f"❌ 모델 다운로드 실패: {e}")
        print(f"수동 다운로드: {model_url}")
        print(f"저장 위치: {model_path}")
        raise

if __name__ == "__main__":
    download_model()






