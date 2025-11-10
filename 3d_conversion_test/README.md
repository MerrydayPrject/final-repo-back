# 3D 이미지 변환 테스트 모듈

드레스 이미지를 3D로 변환하는 기능을 테스트하는 독립 모듈입니다.

## 📁 폴더 구조

```
3d_conversion_test/
├── models/              # 3D 변환 모델 파일
├── templates/           # HTML 템플릿
├── static/             # CSS, JS 파일
├── 3d_conversion.py    # 메인 FastAPI 앱
├── download_model.py   # 모델 다운로드 스크립트
├── requirements_test.txt  # 의존성 패키지
├── start_test_server_8003.bat  # 서버 실행 스크립트
└── README.md          # 이 파일
```

## 🚀 빠른 시작

### 1. 의존성 설치

```bash
pip install -r requirements_test.txt
```

### 2. 모델 다운로드 (필요시)

```bash
python download_model.py
```

### 3. 서버 실행

```bash
# Windows
start_test_server_8003.bat

# 또는 직접 실행
python 3d_conversion.py
```

서버는 http://localhost:8003 에서 실행됩니다.

## 🎯 기능

### 2D → 3D 변환 옵션

1. **Depth Map 생성**
   - MiDaS 또는 DPT 모델 사용
   - 이미지의 깊이 정보 추출
   - 입체감 있는 이미지 생성

2. **3D 회전 뷰어**
   - Three.js 기반
   - 드레스를 회전하며 볼 수 있는 기능
   - 인터랙티브 3D 뷰

3. **Normal Map 생성**
   - 표면 질감과 법선 정보 추출
   - 3D 렌더링 품질 향상

## 📊 API 엔드포인트

### GET /
테스트 페이지

### POST /api/convert-to-3d
이미지를 3D로 변환

**Request:**
- `image`: 업로드할 이미지 파일
- `mode`: 변환 모드 (depth, normal, view)

**Response:**
```json
{
  "success": true,
  "depth_map": "base64_image_data",
  "normal_map": "base64_image_data",
  "processing_time": 1.23
}
```

## 🔧 사용된 기술

- **FastAPI**: 웹 프레임워크
- **PyTorch**: 딥러닝 모델
- **MiDaS/DPT**: Depth Estimation
- **Three.js**: 3D 렌더링 (프론트엔드)
- **OpenCV**: 이미지 처리

## 📝 개발 노트

- 독립적인 모듈로 개발하여 main 브랜치와 충돌 최소화
- 포트 8003 사용 (body_analysis_test는 8002)
- merge 시 다른 기능에 영향 없음

## 🎨 TODO

- [ ] MiDaS 모델 통합
- [ ] Three.js 3D 뷰어 구현
- [ ] Normal Map 생성 기능
- [ ] 성능 최적화
- [ ] UI/UX 개선

## 🐛 알려진 이슈

- 대용량 이미지 처리 시 메모리 사용량 높음
- GPU 가속 권장

## 📧 문의

프로젝트 관련 문의사항은 이슈로 등록해주세요.

