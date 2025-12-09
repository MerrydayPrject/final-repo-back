# MarryDay Backend API

> 맞춤형 웨딩드레스 가상 피팅 플랫폼 백엔드 서버

MarryDay는 사용자가 전신 또는 얼굴 이미지를 업로드하면 AI가 체형·취향에 맞는 웨딩드레스를 자동 매칭·추천하여 가상 피팅 이미지를 생성하는 서비스입니다.

## 📋 목차

- [주요 기능](#주요-기능)
- [기술 스택](#기술-스택)
- [프로젝트 구조](#프로젝트-구조)
- [설치 및 실행](#설치-및-실행)
- [환경 변수 설정](#환경-변수-설정)
- [데이터베이스 설정](#데이터베이스-설정)
- [주요 API 엔드포인트](#주요-api-엔드포인트)
- [개발 가이드](#개발-가이드)
- [라이선스](#라이선스)

## 🚀 주요 기능

### 이미지 처리
- **세그멘테이션 및 배경 제거**: SegFormer B2 모델을 사용한 의상 분할 및 배경 제거
- **이미지 합성**: gemini-3-pro-image-preview을 활용한 고품질 이미지 합성

### 가상 피팅 (Virtual Try-On)
- **통합 트라이온 파이프라인**: 가상 피팅 파이프라인 제공
- **배경 합성**: 배경 이미지를 포함한 자연스러운 합성 결과 생성

### 체형 분석 및 추천
- **체형 분석**: MediaPipe 및 RTMPose를 활용한 포즈 추정 및 체형 지표 산출
- **드레스 추천**: 체형과 스타일 선호도를 기반으로 한 드레스 자동 추천

### 드레스 관리
- **드레스 카탈로그**: 드레스 이미지 및 메타데이터 관리
- **배치 처리**: 여러 드레스 이미지의 일괄 처리 기능

## 🛠 기술 스택

### 백엔드 프레임워크
- **FastAPI** 0.104.1: 고성능 비동기 웹 프레임워크
- **Uvicorn** 0.24.0: ASGI 서버
- **Python** 3.9+

### AI 모델 및 서비스
- **SegFormer B2**: 의상 분할 및 배경 제거 (HuggingFace Inference API 사용)
- **gemini-3-pro-image-preview**: 이미지 합성
- **OpenAI GPT-4o**: 드레스 분석 후 판별
- **MediaPipe**: 포즈 추정 및 얼굴 감지
- **InsightFace**: 얼굴 분석 (HuggingFace Inference Endpoint 사용)

### 데이터베이스 및 스토리지
- **MySQL 5.7+** / **MariaDB 10.2+**: 메인 데이터베이스
- **AWS S3**: 이미지 및 파일 스토리지
- **Supabase**: 추가 데이터베이스 서비스

### 이미지 처리
- **Pillow** 10.0.0+: 이미지 처리
- **OpenCV** 4.8.0+: 이미지 필터 및 색상 조화
- **NumPy** 1.24.0+: 수치 연산

### 기타
- **SQLAlchemy** 2.0.23+: ORM
- **PyMySQL** 1.1.0+: MySQL 드라이버
- **Boto3** 1.34.0+: AWS SDK
- **Jinja2** 3.1.2: 템플릿 엔진

## 📁 프로젝트 구조

```
final-repo-back/
├── main.py                 # FastAPI 메인 애플리케이션
├── requirements.txt        # Python 의존성 목록
├── pyproject.toml         # 프로젝트 설정
├── models_config.json     # 모델 설정 파일
├── category_rules.json     # 카테고리 규칙
│
├── config/                # 설정 파일
│   ├── settings.py        # 환경 변수 관리
│   ├── database.py        # 데이터베이스 설정
│   ├── cors.py           # CORS 설정
│   ├── auth_middleware.py # 인증 미들웨어
│   └── prompts.py        # 프롬프트 템플릿
│
├── core/                  # 핵심 기능 모듈
│   ├── gemini_client.py  # Gemini API 클라이언트
│   ├── llm_clients.py    # LLM 클라이언트 통합
│   ├── s3_client.py      # AWS S3 클라이언트
│   ├── supabase_client.py # Supabase 클라이언트
│   └── segformer_garment_parser.py # SegFormer 의상 파싱
│
├── routers/               # API 라우터
│   ├── info.py           # 정보/상태 엔드포인트
│   ├── web.py            # 웹 페이지 라우터
│   ├── segmentation.py   # 세그멘테이션 기능
│   ├── composition.py    # 이미지 합성 기능
│   ├── tryon_router.py   # 트라이온 기능
│   ├── fitting_router.py # 피팅 기능
│   ├── body_analysis.py  # 체형 분석
│   ├── body_generation.py # 체형 생성
│   ├── dress_management.py # 드레스 관리
│   ├── image_processing.py # 이미지 처리
│   ├── prompt.py         # 프롬프트 생성
│   ├── custom_v4v5_router.py # V5/V5 파이프라인
│   ├── nukki_v2_router.py # 누끼 V2
│   ├── admin.py         # 관리자 기능
│   ├── auth.py          # 인증 기능
│   └── ...
│
├── services/              # 비즈니스 로직
│   ├── tryon_service.py  # 트라이온 서비스
│   ├── fitting_service.py # 피팅 서비스
│   ├── body_analysis_service.py # 체형 분석 서비스
│   ├── dress_service.py  # 드레스 서비스
│   ├── image_service.py  # 이미지 서비스
│   └── ...
│
├── schemas/               # Pydantic 스키마
│   ├── tryon_schema.py   # 트라이온 스키마
│   ├── fitting_schema.py # 피팅 스키마
│   ├── segmentation.py  # 세그멘테이션 스키마
│   └── ...
│
├── prompts/               # 프롬프트 템플릿
│   ├── v5/               # V5 프롬프트
│   └── ...
│
├── templates/             # HTML 템플릿
├── static/                # 정적 파일 (CSS, JS)
├── models/                # 모델 파일 저장소
├── docs/                  # 문서
│   └── all_in_one.md     # 통합 문서
└── utils/                 # 유틸리티 스크립트
```

## 🚀 설치 및 실행

### 사전 요구사항

- Python 3.9 이상
- MySQL 5.7+ 또는 MariaDB 10.2+
- AWS S3 계정 (이미지 스토리지용)
- API 키:
  - Google Gemini API 키
  - OpenAI API 키 (선택)

### 설치 단계

1. **저장소 클론**
   ```bash
   git clone <repository-url>
   cd final-repo-back
   ```

2. **가상환경 생성 및 활성화**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **의존성 설치**
   ```bash
   pip install -r requirements.txt
   ```

4. **환경 변수 설정**
   
   프로젝트 루트에 `.env` 파일을 생성하고 필요한 환경 변수를 설정합니다.
   자세한 내용은 [환경 변수 설정](#환경-변수-설정) 섹션을 참조하세요.

5. **데이터베이스 설정**
   
   MySQL 데이터베이스를 생성하고 연결 정보를 `.env` 파일에 설정합니다.
   자세한 내용은 [데이터베이스 설정](#데이터베이스-설정) 섹션을 참조하세요.

6. **서버 실행**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

   또는 개발 모드로 실행:
   ```bash
   python main.py
   ```

7. **API 문서 확인**
   
   서버 실행 후 다음 URL에서 API 문서를 확인할 수 있습니다:
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

## ⚙️ 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 다음 환경 변수를 설정하세요:

```env
# MySQL 데이터베이스 설정
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=devuser
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=marryday

# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key
GEMINI_3_API_KEY=your_gemini_3_api_key  # 여러 키는 쉼표로 구분
GEMINI_3_FLASH_MODEL=gemini-3-pro-image-preview

# OpenAI API (선택)
OPENAI_API_KEY=your_openai_api_key
GPT4O_MODEL_NAME=gpt-4o
GPT4O_V2_MODEL_NAME=gpt-4o-2024-08-06

# AWS S3 설정
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_S3_BUCKET_NAME=your_bucket_name
AWS_REGION=ap-northeast-2

# 로그 저장용 S3 (선택)
LOGS_AWS_ACCESS_KEY_ID=your_logs_aws_access_key
LOGS_AWS_SECRET_ACCESS_KEY=your_logs_aws_secret_key
LOGS_AWS_S3_BUCKET_NAME=your_logs_bucket_name
LOGS_AWS_REGION=ap-northeast-2

# Supabase 설정 (선택)
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key

# MediaPipe Spaces API
MEDIAPIPE_SPACE_URL=https://jjunyuongv-mediapipe-pose-api.hf.space

# InsightFace Inference Endpoint (선택)
INSIGHTFACE_ENDPOINT_URL=your_insightface_endpoint_url
INSIGHTFACE_API_KEY=your_insightface_api_key
```

> ⚠️ **주의**: `.env` 파일은 Git에 커밋하지 마세요. `.gitignore`에 이미 포함되어 있습니다.

## 🗄 데이터베이스 설정

### 데이터베이스 생성

MySQL 또는 MariaDB에서 다음 SQL을 실행하여 데이터베이스를 생성하세요:

```sql
CREATE DATABASE IF NOT EXISTS marryday 
    CHARACTER SET utf8mb4 
    COLLATE utf8mb4_unicode_ci;

CREATE USER IF NOT EXISTS 'devuser'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON marryday.* TO 'devuser'@'localhost';
FLUSH PRIVILEGES;
```

### 자동 생성되는 테이블

서버 실행 시 다음 테이블들이 자동으로 생성됩니다:

- **`dresses`**: 드레스 이미지 메타 정보 저장
- **`result_logs`**: 합성 로그 저장 (모델, 프롬프트, 이미지 경로, 성공 여부, 처리 시간 등)
- **`body_type_definitions`**: 체형별 정의 및 드레스 추천 정보 (초기 데이터 10개 자동 삽입)
- **`body_logs`**: 체형 분석 로그 저장

### 데이터베이스 연결 테스트

데이터베이스 연결을 테스트하려면 다음 명령을 실행하세요:

```bash
python utils/check_db.py
```

## 📡 주요 API 엔드포인트

### 이미지 처리

- `POST /api/remove-background`: 배경 제거
- `POST /api/upscale`: 이미지 업스케일
- `POST /api/compose-dress`: 드레스 합성
- `POST /api/compose-enhanced`: 향상된 드레스 합성

### 체형 분석

- `POST /api/pose-estimation`: 포즈 추정
- `POST /api/body-analysis`: 체형 분석
- `POST /api/body-generation`: 체형 생성

### 드레스 관리

- `GET /api/dresses`: 드레스 목록 조회
- `POST /api/dresses`: 드레스 등록
- `GET /api/dresses/{dress_id}`: 드레스 상세 조회
- `PUT /api/dresses/{dress_id}`: 드레스 정보 수정
- `DELETE /api/dresses/{dress_id}`: 드레스 삭제

### 커스텀 파이프라인

- `POST /api/custom/v4/compose`: 커스텀 V4 파이프라인
- `POST /api/custom/v5/compose`: 커스텀 V5 파이프라인
- `POST /api/custom/v4v5/compare`: V4/V5 비교

### 기타

- `GET /api/info`: 서버 정보 및 상태
- `GET /api/models`: 사용 가능한 모델 목록
- `GET /docs`: Swagger UI 문서
- `GET /redoc`: ReDoc 문서

> 📚 전체 API 문서는 서버 실행 후 `http://localhost:8000/docs`에서 확인할 수 있습니다.

## 📖 개발 가이드

### 프로젝트 구조 이해

이 프로젝트는 **라우터 기반 모듈화 구조**로 설계되어 있습니다:

- **`main.py`**: FastAPI 앱 초기화 및 라우터 등록만 담당 (약 85줄)
- **`routers/`**: 각 기능별로 분리된 라우터 모듈
- **`services/`**: 비즈니스 로직 처리
- **`core/`**: 핵심 기능 모듈 (API 클라이언트, 파서 등)
- **`schemas/`**: Pydantic 스키마 정의

### 새로운 기능 추가하기

1. **라우터 생성**: `routers/` 디렉토리에 새 라우터 파일 생성
2. **서비스 로직**: `services/` 디렉토리에 비즈니스 로직 구현
3. **스키마 정의**: `schemas/` 디렉토리에 요청/응답 스키마 정의
4. **라우터 등록**: `main.py`에서 새 라우터를 `app.include_router()`로 등록

### 모델 로딩

애플리케이션 시작 시 `core/model_loader.py`의 `load_models()` 함수가 호출되어 필요한 모델들을 로드합니다.

### 참고 문서

- **통합 문서**: `docs/all_in_one.md` - 프로젝트의 상세한 기술 문서
- **프로젝트 요구사항**: `marryday.prd` - 비즈니스 요구사항 및 기능 명세
- **API 가이드**: 프론트엔드 저장소의 `BACKEND_API_GUIDE.md` 참조

### 유틸리티 스크립트

- `utils/check_db.py`: 데이터베이스 연결 테스트
- `utils/download_model.py`: 모델 다운로드
- `utils/view_results.py`: 결과 조회

## 📝 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

---

## 🤝 기여

버그 리포트, 기능 제안, Pull Request를 환영합니다!

## 📧 문의

프로젝트 관련 문의사항이 있으시면 이슈를 생성해주세요.

---

**Made with ❤️ for MarryDay**

