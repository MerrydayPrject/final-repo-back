"""
INSwapper 모델 다운로드 스크립트
다양한 소스에서 INSwapper 모델을 다운로드 시도
"""
import urllib.request
import os
from pathlib import Path
import sys

def download_inswapper():
    """INSwapper 모델 다운로드"""
    # 모델 저장 경로
    model_root = Path.home() / '.insightface' / 'models'
    model_root.mkdir(parents=True, exist_ok=True)
    inswapper_path = model_root / 'inswapper_128.onnx'
    
    if inswapper_path.exists():
        file_size = inswapper_path.stat().st_size / (1024 * 1024)  # MB
        print(f"✅ 모델 파일이 이미 존재합니다: {inswapper_path}")
        print(f"   파일 크기: {file_size:.2f} MB")
        print("다운로드를 건너뜁니다.")
        return True
    
    # 다운로드 시도할 URL 목록 (여러 소스)
    download_urls = [
        # 옵션 1: haofanwang 저장소 (가능하면)
        "https://github.com/haofanwang/inswapper/raw/main/checkpoints/inswapper_128.onnx",
        # 옵션 2: 다른 미러 (실제 작동하는지 확인 필요)
        # "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx",
    ]
    
    print("INSwapper 모델 다운로드 시도 중...")
    print(f"저장 경로: {inswapper_path}\n")
    
    for i, url in enumerate(download_urls, 1):
        try:
            print(f"[{i}/{len(download_urls)}] 다운로드 시도: {url}")
            print("다운로드 중... (시간이 걸릴 수 있습니다)")
            
            urllib.request.urlretrieve(url, str(inswapper_path))
            
            # 파일 크기 확인
            if inswapper_path.exists():
                file_size = inswapper_path.stat().st_size / (1024 * 1024)  # MB
                print(f"✅ 모델 다운로드 완료!")
                print(f"   파일 크기: {file_size:.2f} MB")
                print(f"   저장 위치: {inswapper_path}")
                return True
        except Exception as e:
            print(f"❌ 다운로드 실패: {e}")
            if inswapper_path.exists():
                inswapper_path.unlink()  # 실패한 파일 삭제
            continue
    
    print("\n⚠️  모든 다운로드 소스가 실패했습니다.")
    print("\n수동 다운로드 방법:")
    print("1. 다음 저장소에서 모델 파일을 찾아 다운로드:")
    print("   - https://github.com/haofanwang/inswapper")
    print("   - checkpoints 폴더에서 inswapper_128.onnx 파일 찾기")
    print("\n2. 다운로드한 파일을 다음 위치에 저장:")
    print(f"   {inswapper_path}")
    print("\n3. 또는 다른 소스에서 inswapper_128.onnx 파일을 찾아 다운로드")
    
    return False

if __name__ == "__main__":
    success = download_inswapper()
    sys.exit(0 if success else 1)

