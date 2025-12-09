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

## 🔄 프론트엔드 API 엔드포인트 처리 파이프라인

프론트엔드에서 호출하는 주요 API 엔드포인트의 백엔드 처리 파이프라인을 상세히 설명합니다.

### 1. POST /tryon/compare (일반 피팅)

**프론트엔드 호출**: `autoMatchImageV5V5()`  
**라우터**: `tryon_router.py::compare_v4v5()`

#### 백엔드 처리 파이프라인

```
1. 요청 수신
   ├─> FormData 파싱
   │   ├─> person_image: UploadFile
   │   ├─> garment_image: UploadFile
   │   ├─> background_image: UploadFile
   │   ├─> dress_id: Optional[int] (Form)
   │   └─> profile_front: Optional[str] (Form, JSON 문자열)
   │
   └─> X-Trace-Id 헤더 추출 또는 UUID 생성

2. 이미지 검증 및 변환
   ├─> 이미지 바이트 읽기
   ├─> PIL Image로 변환 (RGB)
   └─> 빈 파일 검증

3. V4V5 비교 서비스 호출
   └─> tryon_compare_service.py::run_v4v5_compare()
       ├─> V4V5Orchestrator 생성
       └─> 병렬 실행 (asyncio.gather)
           ├─> V5Adapter.run() (인스턴스 1)
           │   └─> tryon_service.py::generate_unified_tryon_v5()
           │       ├─> Gemini 프롬프트 생성
           │       ├─> Gemini 이미지 합성 요청
           │       └─> Base64 인코딩
           │
           └─> V5Adapter.run() (인스턴스 2)
               └─> tryon_service.py::generate_unified_tryon_v5()
                   ├─> Gemini 프롬프트 생성
                   ├─> Gemini 이미지 합성 요청
                   └─> Base64 인코딩

4. 성공 처리 (v5_result 성공 시)
   ├─> synthesis_stats_service.py::increment_synthesis_count()
   │   └─> MySQL: INSERT INTO daily_synthesis_count
   │       └─> ON DUPLICATE KEY UPDATE count = count + 1
   │
   ├─> dress_fitting_log_service.py::log_dress_fitting() (dress_id가 있는 경우)
   │   └─> MySQL: INSERT INTO dress_fitting_logs (dress_id)
   │
   └─> profile_service.py::save_tryon_profile()
       └─> MySQL: INSERT INTO tryon_profile_summary
           ├─> trace_id, endpoint, front_profile_json
           ├─> server_total_ms, gemini_call_ms
           └─> status, error_stage

5. 응답 생성
   └─> JSONResponse
       ├─> v4_result: { success, prompt, result_image, message, llm }
       ├─> v5_result: { success, prompt, result_image, message, llm }
       ├─> total_time: float
       └─> X-Trace-Id 헤더 포함
```

#### 주요 특징
- **병렬 처리**: V5 파이프라인을 2번 병렬 실행하여 결과 비교
- **프로파일링**: 프론트엔드/백엔드 시간 측정 및 저장
- **자동 로깅**: 합성 카운트 및 드레스 피팅 로그 자동 기록

---

### 2. POST /tryon/compare/custom (커스텀 피팅)

**프론트엔드 호출**: `customV5V5MatchImage()`  
**라우터**: `custom_v4v5_router.py::compare_v4v5_custom()`

#### 백엔드 처리 파이프라인

```
1. 요청 수신
   ├─> FormData 파싱
   │   ├─> person_image: UploadFile
   │   ├─> garment_image: UploadFile
   │   ├─> background_image: UploadFile
   │   └─> profile_front: Optional[str] (Form, JSON 문자열)
   │
   └─> X-Trace-Id 헤더 추출 또는 UUID 생성

2. 이미지 검증 및 변환
   ├─> 이미지 바이트 읽기
   ├─> PIL Image로 변환 (RGB)
   └─> 빈 파일 검증

3. V4V5 커스텀 비교 서비스 호출
   └─> custom_v4v5_compare_service.py::run_v4v5_custom_compare()
       ├─> V4V5CustomOrchestrator 생성
       └─> 병렬 실행 (asyncio.gather)
           ├─> CustomV5Adapter.run() (인스턴스 1)
           │   └─> custom_v5_service.py::generate_unified_tryon_custom_v5()
           │       ├─> 의상 누끼 처리
           │       │   └─> segformer_garment_parser.py::parse_garment_image_v4()
           │       │       ├─> HuggingFace Inference API 호출
           │       │       │   └─> SegFormer B2 모델 실행
           │       │       ├─> 세그멘테이션 마스크 생성
           │       │       └─> 배경 제거된 의상 이미지 반환 (RGBA)
           │       ├─> 이미지 전처리
           │       │   ├─> 의상 이미지 리사이징 (1024px)
           │       │   └─> 의상 이미지를 인물 이미지 크기로 조정
           │       ├─> Gemini 프롬프트 생성
           │       │   └─> prompts/v5/unified_prompt.txt 템플릿 사용
           │       ├─> Gemini 3 Flash API 호출
           │       │   └─> 이미지 합성 요청
           │       └─> Base64 인코딩
           │
           └─> CustomV5Adapter.run() (인스턴스 2)
               └─> custom_v5_service.py::generate_unified_tryon_custom_v5()
                   ├─> 의상 누끼 처리 (SegFormer)
                   ├─> 이미지 전처리
                   ├─> Gemini 프롬프트 생성
                   ├─> Gemini 3 Flash API 호출
                   └─> Base64 인코딩

4. 성공 처리 (v5_result 성공 시)
   └─> synthesis_stats_service.py::increment_synthesis_count()
       └─> MySQL: INSERT INTO daily_synthesis_count

5. 응답 생성
   └─> JSONResponse
       ├─> v4_result: { success, prompt, result_image, message, llm }
       ├─> v5_result: { success, prompt, result_image, message, llm }
       ├─> total_time: float
       └─> X-Trace-Id 헤더 포함
```

#### 주요 특징
- **의상 누끼 처리**: SegFormer를 통한 자동 배경 제거
- **병렬 처리**: CustomV5 파이프라인을 2번 병렬 실행
- **로깅 비활성화**: S3 업로드 및 커스텀 피팅 로그 저장 비활성화 (프론트엔드 요청 시)

---

### 3. POST /api/validate-person (이미지 유효성 검사)

**프론트엔드 호출**: `validatePerson()`  
**라우터**: `body_analysis.py::validate_person()`

#### 백엔드 처리 파이프라인

```
1. 요청 수신
   └─> FormData 파싱
       └─> file: UploadFile

2. 이미지 처리
   ├─> 이미지 바이트 읽기
   ├─> PIL Image로 변환 (RGB)
   └─> EXIF orientation 자동 적용 (ImageOps.exif_transpose)

3. 이미지 타입 감지
   └─> face_swap_service.py::detect_image_type()
       ├─> MediaPipe 포즈 랜드마크 추출
       ├─> 랜드마크 기반 전신/상체 판별
       └─> 신뢰도 계산

4. 응답 생성
   └─> JSONResponse
       ├─> success: boolean
       ├─> is_person: boolean
       ├─> image_type: "full_body" | "upper_body" | "face" | "unknown"
       ├─> confidence: float
       └─> message: string
```

#### 주요 특징
- **EXIF 처리**: 스마트폰 사진의 방향 자동 보정
- **타입 감지**: 전신/상체/얼굴 사진 자동 판별
- **신뢰도 제공**: 판별 결과의 신뢰도 점수 포함

---

### 4. POST /api/analyze-body (체형 분석)

**프론트엔드 호출**: `analyzeBody()`  
**라우터**: `body_analysis.py::analyze_body()`

#### 백엔드 처리 파이프라인

```
1. 요청 수신
   └─> FormData 파싱
       ├─> file: UploadFile
       ├─> height: Optional[int] (Form)
       └─> weight: Optional[int] (Form)

2. 이미지 처리
   ├─> 이미지 바이트 읽기
   ├─> PIL Image로 변환 (RGB)
   └─> EXIF orientation 자동 적용

3. 포즈 랜드마크 추출
   └─> body_analysis_service.py::analyze_body()
       └─> pose_landmark_service.py::extract_landmarks()
           └─> MediaPipe API 호출
               └─> 33개 랜드마크 포인트 반환

4. 이미지 분류
   └─> image_classifier_service.py::classify_image()
       └─> 전신/상체 이미지 분류

5. 체형 지표 계산
   └─> body_service.py::determine_body_features()
       ├─> 키 추정 (랜드마크 기반)
       ├─> 몸무게 추정
       ├─> BMI 계산
       └─> 체형 특징 분석

6. Gemini AI 분석
   ├─> body_service.py::analyze_body_with_gemini()
   │   └─> Gemini API: 상세 체형 분석 프롬프트
   │       └─> 체형 특징, 비율, 스타일 분석
   │
   └─> body_service.py::classify_body_line_with_gemini()
       └─> Gemini API: 체형 라인 분류 프롬프트
           └─> A라인, H라인, X라인 등 분류

7. 결과 저장
   └─> body_analysis_database.py::save_body_analysis_result()
       └─> MySQL: INSERT INTO body_logs
           ├─> 키, 몸무게, BMI
           ├─> 체형 특징, 체형 라인
           └─> 분석 결과 JSON

8. 응답 생성
   └─> JSONResponse
       ├─> success: boolean
       ├─> body_features: { height, weight, bmi, features }
       ├─> body_line: string (A라인, H라인 등)
       ├─> recommendations: [카테고리 목록]
       └─> analysis: { gemini_analysis, body_line_classification }
```

#### 주요 특징
- **정확한 포즈 추정**: MediaPipe 33개 랜드마크 포인트
- **AI 기반 분석**: Gemini를 통한 상세 체형 분석
- **드레스 추천**: 체형 라인에 맞는 카테고리 자동 추천

---

### 5. POST /api/apply-image-filters (이미지 필터 적용)

**프론트엔드 호출**: `applyImageFilter()`  
**라우터**: `image_processing.py::apply_image_filters()`

#### 백엔드 처리 파이프라인

```
1. 요청 수신
   └─> FormData 파싱
       ├─> file: UploadFile
       └─> filter_preset: str (Form)

2. 이미지 처리
   ├─> 이미지 바이트 읽기
   ├─> PIL Image로 변환 (RGB)
   └─> NumPy 배열로 변환

3. 필터 적용
   └─> image_filter_service.py::apply_filter_preset()
       ├─> filter_preset에 따른 필터 선택
       │   ├─> "grayscale": 흑백 변환
       │   ├─> "vintage": 빈티지 효과
       │   ├─> "warm": 따뜻한 톤
       │   ├─> "cool": 차가운 톤
       │   └─> "high_contrast": 고대비
       │
       └─> OpenCV를 통한 이미지 필터링
           ├─> 색상 공간 변환 (RGB → HSV/LAB)
           ├─> 채널별 조정
           └─> 색상 공간 변환 (HSV/LAB → RGB)

4. Base64 인코딩
   └─> PIL Image → Base64 문자열

5. 응답 생성
   └─> JSONResponse
       ├─> success: boolean
       ├─> result_image: string (base64)
       ├─> filter_preset: string
       └─> message: string
```

#### 주요 특징
- **다양한 필터**: 5가지 필터 프리셋 제공
- **OpenCV 기반**: 고성능 이미지 처리
- **실시간 적용**: 결과 이미지에 즉시 필터 적용

---

### 6. GET /api/admin/dresses (드레스 목록 조회)

**프론트엔드 호출**: `getDresses()`  
**라우터**: `dress_management.py::get_dresses()`

#### 백엔드 처리 파이프라인

```
1. 요청 수신
   └─> Query 파라미터
       ├─> page: int (기본값: 1)
       └─> limit: int (기본값: 20, 최대: 10000)

2. 데이터베이스 조회
   └─> MySQL 쿼리 실행
       ├─> 전체 건수 조회
       │   └─> SELECT COUNT(*) FROM dresses
       │
       └─> 페이징된 데이터 조회
           └─> SELECT 
               d.idx as id,
               d.file_name as image_name,
               d.style,
               d.url,
               COALESCE(COUNT(l.id), 0) as fitting_count
               FROM dresses d
               LEFT JOIN dress_fitting_logs l ON d.idx = l.dress_id
               GROUP BY d.idx, d.file_name, d.style, d.url
               ORDER BY d.idx DESC
               LIMIT %s OFFSET %s

3. 응답 생성
   └─> JSONResponse
       ├─> success: boolean
       ├─> data: [드레스 목록]
       │   └─> 각 드레스: { id, image_name, style, url, fitting_count }
       ├─> pagination: { page, limit, total, total_pages }
       └─> message: string
```

#### 주요 특징
- **피팅 카운트 포함**: 각 드레스의 피팅 횟수 자동 계산
- **페이징 지원**: 대량 데이터 처리 최적화
- **정렬**: 드레스 ID 내림차순 (최신순)

---

### 7. POST /api/reviews (리뷰 제출)

**프론트엔드 호출**: `submitReview()`  
**라우터**: `review.py::create_review()`

#### 백엔드 처리 파이프라인

```
1. 요청 수신
   └─> JSON Body 파싱
       ├─> category: str ("general" | "custom" | "analysis")
       ├─> rating: int (1-5)
       └─> content: Optional[str]

2. 유효성 검사
   ├─> 카테고리 검증
   └─> 별점 범위 검증

3. 데이터베이스 저장
   └─> MySQL: INSERT INTO reviews
       ├─> rating
       ├─> content (NULL 허용)
       └─> category

4. 응답 생성
   └─> JSONResponse
       ├─> success: boolean
       ├─> review_id: int
       └─> message: string
```

#### 주요 특징
- **카테고리별 리뷰**: 기능별 리뷰 분리 저장
- **선택적 내용**: 리뷰 내용은 선택사항

---

### 8. POST /visitor/visit (방문자 카운팅)

**프론트엔드 호출**: `countVisitor()`  
**라우터**: `visitor_router.py::increment_visitor()`

#### 백엔드 처리 파이프라인

```
1. 요청 수신
   └─> 빈 Body (POST 요청만으로 카운팅)

2. 날짜 계산
   └─> 한국 시간대(KST) 기준 오늘 날짜 계산
       └─> datetime.now(ZoneInfo("Asia/Seoul")).date()

3. 데이터베이스 업데이트
   └─> MySQL: UPSERT
       └─> INSERT INTO daily_visitors (visit_date, count) 
           VALUES (%s, 1)
           ON DUPLICATE KEY UPDATE count = count + 1

4. 응답 생성
   └─> JSONResponse
       ├─> date: string (YYYY-MM-DD)
       └─> count: int (오늘 방문자 수)
```

#### 주요 특징
- **일별 카운팅**: 날짜별 방문자 수 집계
- **UPSERT**: 중복 방문 시 카운트 증가
- **한국 시간 기준**: KST(UTC+9) 기준 날짜 계산

---

### 9. POST /api/dress/check (드레스 검증)

**프론트엔드 호출**: `checkDress()`  
**라우터**: `dress_management.py::check_single_dress()`

#### 백엔드 처리 파이프라인

```
1. 요청 수신
   └─> FormData 파싱
       ├─> file: UploadFile
       ├─> model: str (Form, "gpt-4o-mini" | "gpt-4o")
       └─> mode: str (Form, "fast" | "accurate")

2. 이미지 검증
   ├─> 파일 크기 검증 (5MB 제한)
   ├─> 이미지 형식 검증
   └─> PIL Image로 변환

3. 이미지 해시 생성
   └─> MD5 해시 계산 (중복 검사용)

4. 드레스 판별
   └─> dress_check_service.py::check_dress()
       ├─> OpenAI GPT-4o API 호출
       │   ├─> Vision API를 통한 이미지 분석
       │   └─> 드레스 여부 판별 프롬프트
       │
       └─> 응답 파싱
           ├─> dress: boolean
           ├─> confidence: float
           └─> category: string

5. 로그 저장 (선택)
   └─> MySQL: INSERT INTO dress_check_logs
       ├─> image_hash
       ├─> dress (boolean)
       ├─> confidence
       └─> model, mode

6. 응답 생성
   └─> JSONResponse
       ├─> success: boolean
       ├─> result: { dress, confidence, category }
       └─> message: string
```

#### 주요 특징
- **AI 기반 판별**: GPT-4o Vision API 사용
- **모드 선택**: fast/accurate 모드 지원
- **중복 검사**: 이미지 해시 기반 중복 방지

---

### 10. GET /api/proxy-image (이미지 프록시)

**프론트엔드 호출**: `urlToFile()` 내부 사용  
**라우터**: `proxy.py::proxy_image_by_url()`

#### 백엔드 처리 파이프라인

```
1. 요청 수신
   └─> Query 파라미터
       └─> url: str (S3 이미지 URL)

2. URL 파싱
   └─> urlparse(url)
       └─> 경로에서 파일명 추출

3. S3 이미지 다운로드
   └─> s3_client.py::get_s3_image()
       ├─> AWS S3 클라이언트 생성
       ├─> S3 버킷에서 이미지 다운로드
       └─> 이미지 바이트 반환

4. 응답 생성
   └─> Response
       ├─> content: bytes (이미지 데이터)
       └─> media_type: "image/png"
```

#### 주요 특징
- **CORS 해결**: S3 직접 접근 시 CORS 문제 해결
- **프록시 역할**: 백엔드를 통한 이미지 전달
- **캐싱 가능**: 백엔드에서 캐싱 정책 적용 가능

---

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

