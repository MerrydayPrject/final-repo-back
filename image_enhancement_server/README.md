# 이미지 보정 서버

배경 분위기 변경 및 이미지 보정 기능을 제공하는 독립 서버입니다.

## 기능

- **형태 조작**: 어깨, 허리, 엉덩이 등 신체 부위 조정
- **스타일 변경**: 우아하게, 모던하게, 캐주얼하게 등
- **배경 분위기 변경**: 배경 블러, 해변, 정원, 교회, 스튜디오 등
- **분위기 변경**: 로맨틱하게, 밝게, 드라마틱하게 등
- **조명/색감**: 조명 부드럽게, 따뜻한 조명, 색감 선명하게
- **얼굴 보정**: 주름 제거, 피부톤 개선

## 설치

```bash
pip install -r requirements.txt
```

## 실행

### 배치 파일 사용
```bash
start_enhancement_server.bat
```

### 직접 실행
```bash
python enhancement_server.py 8003
```

## 접속 주소

- **서버**: http://localhost:8003
- **API 문서**: http://localhost:8003/docs
- **테스트 페이지**: http://localhost:8003/

## API 사용법

### POST /api/enhance-image

**요청:**
- `file`: 이미지 파일 (multipart/form-data)
- `instruction`: 보정 요청사항 (한국어 가능)
- `num_inference_steps`: 추론 단계 (기본값: 20)
- `image_guidance_scale`: 이미지 가이던스 (기본값: 1.5)
- `use_gfpgan`: GFPGAN 사용 여부 (기본값: true)
- `gfpgan_weight`: GFPGAN 보정 강도 (기본값: 0.5)

**응답:**
```json
{
    "success": true,
    "result_image": "data:image/png;base64,...",
    "prompt_used": "make shoulders narrower, make background blur",
    "message": "이미지 보정이 완료되었습니다."
}
```

## 사용 예시

### cURL
```bash
curl -X POST "http://localhost:8003/api/enhance-image" \
  -F "file=@test.jpg" \
  -F "instruction=어깨 좁게, 배경 블러, 로맨틱한 분위기"
```

### Python
```python
import requests

files = {'file': open('test.jpg', 'rb')}
data = {
    'instruction': '어깨 좁게, 배경 블러, 로맨틱한 분위기',
    'use_gfpgan': True,
    'gfpgan_weight': 0.5
}

response = requests.post('http://localhost:8003/api/enhance-image', files=files, data=data)
result = response.json()
```

## 지원하는 요청사항

### 형태 조작
- 어깨 좁게, 어깨 넓게
- 허리 얇게, 허리 두껍게
- 엉덩이 작게, 엉덩이 크게

### 스타일 변경
- 우아하게, 모던하게
- 캐주얼하게, 빈티지

### 배경 분위기 변경
- 배경 블러
- 배경을 해변/정원/교회/스튜디오/공원으로

### 분위기 변경
- 로맨틱하게, 밝게
- 드라마틱하게, 따뜻하게
- 고급스럽게

### 조명/색감
- 조명 부드럽게
- 따뜻한 조명, 밝은 조명
- 색감 선명하게

### 얼굴 보정
- 주름 제거
- 피부톤 밝게, 피부톤 하얗게
- 피부 매끄럽게

## 모델 정보

- **InstructPix2Pix**: 형태/스타일/배경 분위기 변경
- **GFPGAN**: 얼굴 보정 (주름 제거, 피부톤 개선)

## 주의사항

- GPU 메모리: 최소 8GB VRAM 권장
- 처리 시간: 약 10-20초 (GPU 기준)
- 모델 다운로드: 첫 실행 시 자동 다운로드 (약 2.5GB + GFPGAN 모델)




