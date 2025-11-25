"""
새로운 체형 보정 시스템 테스트 스크립트

사용법:
    python test_new_body_adjustment.py <image_path>

예시:
    python test_new_body_adjustment.py test_images/person.jpg
"""
import sys
import asyncio
from PIL import Image
import os

# 서비스 import
from services.body_adjustment_service import adjust_body_shape_api


async def test_body_adjustment(image_path: str):
    """체형 보정 테스트"""
    
    if not os.path.exists(image_path):
        print(f"❌ 이미지 파일을 찾을 수 없습니다: {image_path}")
        return
    
    print("=" * 60)
    print("🔬 새로운 체형 보정 시스템 테스트")
    print("=" * 60)
    
    # 이미지 로드
    print(f"\n📷 이미지 로드: {image_path}")
    image = Image.open(image_path).convert("RGB")
    print(f"   크기: {image.size}")
    
    # 테스트 케이스들
    test_cases = [
        {
            "name": "허리만 슬림 (0.8)",
            "params": {"waist_factor": 0.8},
            "output": "output_waist_slim.png"
        },
        {
            "name": "어깨만 넓게 (1.2)",
            "params": {"shoulder_factor": 1.2},
            "output": "output_shoulder_wide.png"
        },
        {
            "name": "엉덩이만 슬림 (0.85)",
            "params": {"hip_factor": 0.85},
            "output": "output_hip_slim.png"
        },
        {
            "name": "전체 슬림 (허리0.8, 엉덩이0.9)",
            "params": {"waist_factor": 0.8, "hip_factor": 0.9},
            "output": "output_overall_slim.png"
        },
    ]
    
    print("\n" + "=" * 60)
    print("🧪 테스트 케이스 실행")
    print("=" * 60)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] {case['name']}")
        print(f"   파라미터: {case['params']}")
        
        try:
            # 체형 조정 실행
            result = await adjust_body_shape_api(
                image=image.copy(),
                **case['params']
            )
            
            # 결과 저장
            output_path = case['output']
            result.save(output_path)
            print(f"   ✅ 저장 완료: {output_path}")
            
        except Exception as e:
            print(f"   ❌ 실패: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✨ 테스트 완료!")
    print("=" * 60)
    print("\n📊 결과 확인:")
    print("   - output_*.png 파일들을 열어서 확인하세요")
    print("   - 조정된 부위만 변형되고 나머지는 원본과 동일해야 합니다")
    print("   - 배경이 왜곡되지 않았는지 확인하세요")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python test_new_body_adjustment.py <image_path>")
        print("예시: python test_new_body_adjustment.py test_images/person.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    asyncio.run(test_body_adjustment(image_path))
