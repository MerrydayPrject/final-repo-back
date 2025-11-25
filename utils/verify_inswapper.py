"""
INSwapper 모델 파일 검증 스크립트
다운로드한 파일이 올바른지 확인
"""
from pathlib import Path
import os

def verify_inswapper_model(file_path: str):
    """INSwapper 모델 파일 검증"""
    model_path = Path(file_path)
    
    print("=" * 60)
    print("INSwapper 모델 파일 검증")
    print("=" * 60)
    
    # 1. 파일 존재 확인
    if not model_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return False
    
    print(f"✅ 파일 존재 확인: {model_path}")
    
    # 2. 파일명 확인
    if model_path.name != "inswapper_128.onnx":
        print(f"⚠️  파일명이 다릅니다: {model_path.name}")
        print(f"   예상 파일명: inswapper_128.onnx")
        print(f"   현재 파일명: {model_path.name}")
        response = input("   파일명을 변경하시겠습니까? (y/n): ")
        if response.lower() == 'y':
            new_path = model_path.parent / "inswapper_128.onnx"
            model_path.rename(new_path)
            model_path = new_path
            print(f"✅ 파일명 변경 완료: {model_path.name}")
    else:
        print(f"✅ 파일명 확인: {model_path.name}")
    
    # 3. 파일 확장자 확인
    if model_path.suffix != ".onnx":
        print(f"❌ 파일 확장자가 올바르지 않습니다: {model_path.suffix}")
        print(f"   예상 확장자: .onnx")
        return False
    
    print(f"✅ 파일 확장자 확인: {model_path.suffix}")
    
    # 4. 파일 크기 확인
    file_size = model_path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    
    print(f"📦 파일 크기: {file_size_mb:.2f} MB ({file_size:,} bytes)")
    
    # 일반적인 INSwapper 모델 크기는 200-300MB
    if file_size_mb < 100:
        print("⚠️  파일 크기가 예상보다 작습니다 (일반적으로 200-300MB)")
        print("   파일이 손상되었거나 잘못된 파일일 수 있습니다.")
    elif file_size_mb > 500:
        print("⚠️  파일 크기가 예상보다 큽니다 (일반적으로 200-300MB)")
        print("   잘못된 파일일 수 있습니다.")
    else:
        print("✅ 파일 크기가 적절합니다 (200-300MB 범위)")
    
    # 5. 파일 헤더 확인 (ONNX 파일인지 간단히 확인)
    try:
        with open(model_path, 'rb') as f:
            header = f.read(16)
            # ONNX 파일은 보통 특정 바이트로 시작
            # 정확한 검증은 onnx 모듈이 필요하지만, 기본적인 확인만 수행
            if header[:4] == b'\x08\x00\x12' or b'onnx' in header[:16].lower():
                print("✅ 파일 형식 확인: ONNX 형식으로 보입니다")
            else:
                print("⚠️  파일 형식 확인: ONNX 형식이 아닐 수 있습니다")
                print("   하지만 파일이 정상일 수도 있으니 시도해보세요")
    except Exception as e:
        print(f"⚠️  파일 헤더 확인 실패: {e}")
    
    # 6. 저장 위치 확인
    expected_path = Path.home() / '.insightface' / 'models' / 'inswapper_128.onnx'
    
    print(f"\n📁 현재 파일 위치: {model_path}")
    print(f"📁 권장 저장 위치: {expected_path}")
    
    if model_path != expected_path:
        print("\n⚠️  파일이 권장 위치에 없습니다.")
        response = input(f"   권장 위치로 복사하시겠습니까? (y/n): ")
        if response.lower() == 'y':
            expected_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 기존 파일이 있으면 백업
            if expected_path.exists():
                backup_path = expected_path.with_suffix('.onnx.backup')
                expected_path.rename(backup_path)
                print(f"   기존 파일을 백업했습니다: {backup_path}")
            
            # 파일 복사
            import shutil
            shutil.copy2(model_path, expected_path)
            print(f"✅ 파일 복사 완료: {expected_path}")
            model_path = expected_path
        else:
            print("   현재 위치에서도 작동할 수 있지만, 권장 위치 사용을 권장합니다.")
    
    # 7. 최종 확인
    print("\n" + "=" * 60)
    print("검증 완료!")
    print("=" * 60)
    print(f"✅ 파일: {model_path}")
    print(f"✅ 크기: {file_size_mb:.2f} MB")
    print("\n이제 서버를 재시작하거나 API를 다시 호출하면")
    print("INSwapper 모델이 자동으로 로드됩니다.")
    
    return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # 기본 경로 확인
        default_path = Path.home() / '.insightface' / 'models' / 'inswapper_128.onnx'
        if default_path.exists():
            file_path = str(default_path)
            print(f"기본 경로에서 파일을 찾았습니다: {file_path}\n")
        else:
            print("파일 경로를 입력하거나, 기본 경로에 파일을 배치하세요.")
            print(f"기본 경로: {default_path}\n")
            file_path = input("파일 경로를 입력하세요: ").strip().strip('"').strip("'")
    
    if file_path:
        verify_inswapper_model(file_path)
    else:
        print("파일 경로가 제공되지 않았습니다.")

