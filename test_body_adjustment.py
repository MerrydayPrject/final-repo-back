"""
체형 조정 API 테스트 스크립트
"""
import requests
from PIL import Image
import io

# API 엔드포인트
url = "http://127.0.0.1:8000/body-adjustment/adjust"

# 테스트 이미지 경로 (실제 파일로 교체하세요)
image_path = "test_person.jpg"

# 파라미터
# slim_factor: 0.7 (매우 날씬) ~ 1.3 (넓게)
#   - 0.8: 약간 날씬
#   - 0.9: 살짝 날씬  
#   - 1.0: 원본
#   - 1.1: 약간 넓게
data = {
    "slim_factor": 0.85,  # 날씬하게
    "use_advanced": False  # True면 상하 다르게 적용
}

try:
    # 이미지 파일 열기
    with open(image_path, "rb") as f:
        files = {"image": f}
        
        # API 호출
        print(f"체형 조정 중... (factor: {data['slim_factor']})")
        response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            # 결과 저장
            output_path = "adjusted_output.png"
            with open(output_path, "wb") as out:
                out.write(response.content)
            
            print(f"✅ 성공! 결과: {output_path}")
            
            # 이미지 크기 확인
            result_img = Image.open(io.BytesIO(response.content))
            print(f"결과 이미지 크기: {result_img.size}")
        else:
            print(f"❌ 실패: {response.status_code}")
            print(response.text)
            
except FileNotFoundError:
    print(f"❌ 이미지 파일을 찾을 수 없습니다: {image_path}")
    print("테스트용 이미지를 준비하고 경로를 수정하세요.")
except Exception as e:
    print(f"❌ 에러: {e}")
