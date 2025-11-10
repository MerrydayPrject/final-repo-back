# 체형 분석 기능 테스트 폴더

이 폴더는 체형 분석 기능을 독립적으로 테스트하기 위한 파일들을 포함합니다.

## 📁 파일 구조

```
body_analysis_test/
├── body_analysis.py          # 체형 분석 서비스 클래스
├── test_body_analysis.py     # 독립 테스트용 FastAPI 서버
├── README.md                 # 이 파일
├── requirements_test.txt     # 테스트용 패키지 목록
├── start_test_server_8002.bat # 서버 실행 스크립트
└── models/                   # MediaPipe 모델 파일 (자동 다운로드)
    └── pose_landmarker_lite.task
```

## 🚀 사용 방법

### 1. 패키지 설치

```bash
pip install -r requirements_test.txt
```

### 2. 모델 파일 다운로드

첫 실행 시 MediaPipe Pose Landmarker 모델이 자동으로 다운로드됩니다.

### 3. 테스트 서버 실행

#### Windows (배치 파일)
```bash
start_test_server_8002.bat
```

#### 직접 실행
```bash
python test_body_analysis.py
```

또는 포트 번호 지정:
```bash
python test_body_analysis.py 8002
```

### 4. 서버 접속

- **메인 페이지**: http://localhost:8002
- **API 문서 (Swagger)**: http://localhost:8002/docs
- **헬스 체크**: http://localhost:8002/health

## 📡 API 엔드포인트

### POST /api/analyze-body

전신 이미지를 분석하여 체형 정보를 반환합니다.

**요청:**
- Content-Type: `multipart/form-data`
- Body: `file` (이미지 파일)

**응답:**
```json
{
  "success": true,
  "body_analysis": {
    "body_type": "A라인",
    "measurements": {
      "shoulder_width": 0.45,
      "hip_width": 0.52,
      "shoulder_hip_ratio": 0.87,
      "arm_length": 0.38,
      "leg_length": 0.55,
      "estimated_height": 165
    },
    "body_type_category": {
      "type": "A라인",
      "confidence": 0.85,
      "description": "..."
    }
  },
  "pose_landmarks": {
    "total_landmarks": 33,
    "detected_landmarks": [...]
  },
  "gemini_analysis": {
    "body_type": "A라인",
    "analysis": "...",
    "recommended_styles": [...],
    "avoid_styles": [...]
  },
  "message": "체형 분석이 완료되었습니다."
}
```

## 🔧 환경 변수

`.env` 파일에 다음 설정이 필요합니다 (선택사항):

```env
GEMINI_API_KEY=your_gemini_api_key
```

Gemini API 키가 없어도 랜드마크 기반 체형 분석은 가능합니다.

## ⚠️ 주의사항

- 이 서버는 **테스트 전용**입니다.
- 메인 백엔드(8000 포트)와는 **완전히 분리**되어 있습니다.
- 메인 프로젝트에 영향 없이 독립적으로 실행됩니다.

## 🐛 문제 해결

### 모델 다운로드 실패
- 인터넷 연결을 확인하세요.
- 수동 다운로드: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
- `models/` 폴더에 저장하세요.

### 포트 충돌
8002번 포트가 사용 중인 경우:
```bash
python test_body_analysis.py 8003
```




