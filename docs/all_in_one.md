# 프로젝트 통합 문서 (`final-repo-back`)

> 기존 개별 Markdown 문서의 핵심 내용을 한 파일로 정리한 자료입니다.

- [1. 개요](#1-개요)
- [2. 프로젝트 구조](#2-프로젝트-구조)
- [3. 환경 구성](#3-환경-구성)
- [4. 데이터베이스 설정](#4-데이터베이스-설정)
- [5. 실행 가이드](#5-실행-가이드)
- [6. API 엔드포인트 전체 목록](#6-api-엔드포인트-전체-목록)
- [7. 라우터 모듈 상세 설명](#7-라우터-모듈-상세-설명)
- [8. 누끼 · 합성 프로세스](#8-누끼--합성-프로세스)
- [9. 이미지 보정 서버](#9-이미지-보정-서버)
- [10. 이미지 보정 모델 & 활용 가이드](#10-이미지-보정-모델--활용-가이드)
- [11. 체형 분석](#11-체형-분석)
- [12. 모델 추천 & 레퍼런스](#12-모델-추천--레퍼런스)
- [13. 3D 연동 시나리오](#13-3d-연동-시나리오)
- [14. 작업 기록 및 향후 계획](#14-작업-기록-및-향후-계획)
- [부록. 참고 자료](#부록-참고-자료)

---

## 1. 개요

- 웨딩드레스 누끼 및 가상 피팅을 제공하는 FastAPI 기반 백엔드 프로젝트.
- 핵심 기능
  - SegFormer 기반 드레스 세그멘테이션과 배경 제거.
  - Google Gemini 2.5 Flash 및 GPT-4o를 활용한 의상 합성 프로세스.
  - InstructPix2Pix·Real-ESRGAN·ControlNet 등으로 화질 보정 및 스타일 조정.
  - MediaPipe Pose Landmarker & RTMPose로 체형 분석 및 포즈 기반 합성 지원.

---

## 2. 프로젝트 구조

### 2.1 아키텍처 개요

프로젝트는 **라우터 기반 모듈화 구조**로 설계되어 있습니다. 이전에는 모든 코드가 `main_original.py` (약 6,109줄)에 집중되어 있었으나, 현재는 기능별로 분리된 라우터 모듈로 구성되어 있습니다.

### 2.2 main.py 모듈화 구조

`main.py`는 **76줄**의 간결한 메인 파일로, 다음과 같은 역할만 수행합니다:

1. **FastAPI 앱 초기화**
   - 앱 메타데이터 설정 (제목, 설명, 버전 등)
   - CORS 미들웨어 설정
   - 정적 파일 및 템플릿 디렉토리 마운트

2. **라우터 등록**
   - 12개의 라우터 모듈을 `app.include_router()`로 등록
   - 각 라우터는 독립적인 기능 영역을 담당

3. **Startup 이벤트**
   - 애플리케이션 시작 시 모델 로딩 (`load_models()`)

**main.py 구조 예시:**
```python
# FastAPI 앱 초기화
app = FastAPI(...)

# CORS 설정
app.add_middleware(CORSMiddleware, ...)

# 라우터 등록
from routers import info, web, segmentation, composition, ...
app.include_router(info.router)
app.include_router(web.router)
# ... 기타 라우터들

# Startup 이벤트
@app.on_event("startup")
async def startup_event():
    await load_models()
```

### 2.3 라우터 모듈 구조

모든 라우터는 `routers/` 디렉토리에 있으며, 각 모듈은 다음과 같은 구조를 가집니다:

```
routers/
├── __init__.py
├── info.py              # 정보/상태 엔드포인트
├── web.py               # 웹 페이지 라우터
├── segmentation.py      # 세그멘테이션 기능
├── composition.py       # 이미지 합성 기능
├── prompt.py            # 프롬프트 생성
├── image_processing.py # 이미지 처리 (업스케일, 색상 보정 등)
├── body_analysis.py     # 체형 분석
├── dress_management.py  # 드레스 관리
├── admin.py            # 관리자 기능
├── models.py           # 모델 관리
├── conversion_3d.py    # 3D 변환
└── proxy.py            # 이미지 프록시
```

### 2.4 모듈화의 장점

1. **유지보수성**: 특정 기능 수정 시 해당 라우터 파일만 수정
2. **가독성**: main.py가 간결해져 전체 구조 파악이 쉬움
3. **협업**: 여러 개발자가 동시에 다른 라우터 작업 가능
4. **확장성**: 새 기능 추가 시 새 라우터 파일만 추가
5. **테스트**: 각 라우터를 독립적으로 테스트 가능

---

## 3. 환경 구성

- **Python/패키지**
  - `pip install -r requirements.txt`
  - GPU 사용 시 PyTorch를 CUDA 버전에 맞춰 설치 (예: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`).
  - 이미지 보정 서버 전용 의존성은 `image_enhancement_server/requirements.txt` 참고.

- **환경 변수 (.env)**
  ```env
  # MySQL
  MYSQL_HOST=localhost
  MYSQL_PORT=3306
  MYSQL_USER=devuser
  MYSQL_PASSWORD=your_password
  MYSQL_DATABASE=marryday

  # Gemini
  GEMINI_API_KEY=your_gemini_api_key
  OPENAI_API_KEY=your_openai_api_key

  # AWS S3 (드레스 업로드)
  AWS_ACCESS_KEY_ID=...
  AWS_SECRET_ACCESS_KEY=...
  AWS_S3_BUCKET_NAME=...
  AWS_REGION=ap-northeast-2

  # AWS S3 (로그 저장, 선택)
  LOGS_AWS_ACCESS_KEY_ID=...
  LOGS_AWS_SECRET_ACCESS_KEY=...
  LOGS_AWS_S3_BUCKET_NAME=...
  LOGS_AWS_REGION=ap-northeast-2

  # Meshy API (3D 변환)
  MESHY_API_KEY=...
  ```
  - `.env`는 Git에 커밋하지 말 것.

---

## 4. 데이터베이스 설정

- 요구 DB: **MySQL 5.7+** 또는 **MariaDB 10.2+**
- 생성 예시
  ```sql
  CREATE DATABASE IF NOT EXISTS marryday 
      CHARACTER SET utf8mb4 
      COLLATE utf8mb4_unicode_ci;

  CREATE USER IF NOT EXISTS 'devuser'@'localhost' IDENTIFIED BY 'your_password';
  GRANT ALL PRIVILEGES ON marryday.* TO 'devuser'@'localhost';
  FLUSH PRIVILEGES;
  ```
- 서버 실행 시 자동 생성되는 테이블:
  - `composition_logs`: 합성 로그 저장 (모델, 프롬프트, 이미지 경로, 성공 여부, 처리 시간 등)
  - `dresses`: 드레스 이미지 메타 정보 저장
  - `body_analysis_logs`: 체형 분석 로그 저장

---

## 5. 실행 가이드

### 5.1 주요 서버

| 서버 | 포트 | 설명 | 실행 파일 |
|------|------|------|-----------|
| 메인 백엔드 | 8000 | 세그멘테이션 · 합성 API | `main.py` |
| 이미지 보정 | 8003 | InstructPix2Pix + GFPGAN 기반 보정 | `image_enhancement_server/enhancement_server.py` |
| 체형 분석 테스트 | 8002 | MediaPipe 기반 체형 측정 | `body_analysis_test/test_body_analysis.py` |

### 5.2 실행 명령 예시

```powershell
# 메인 백엔드
cd C:\Users\301\Dev\Project\final-repo-back
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 이미지 보정 서버
cd C:\Users\301\Dev\Project\final-repo-back\image_enhancement_server
python enhancement_server.py 8003

# 체형 분석 서버
cd C:\Users\301\Dev\Project\final-repo-back\body_analysis_test
python test_body_analysis.py 8002
```

- Windows 배치 스크립트
  - `start-backend.bat`: 백엔드 서버 실행
  - `start_all_servers.bat`: 세 개의 서버를 각각 새 콘솔에서 실행
  - `start_all_servers.ps1`: PowerShell 버전

- 접속 경로
  - 메인 API 문서: `http://localhost:8000/docs`
  - 이미지 보정 서버 문서: `http://localhost:8003/docs`

---

## 6. API 엔드포인트 전체 목록

### 6.1 정보 조회 (info.py)

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| GET | `/health` | 서버 상태 및 SegFormer 로딩 여부 확인 |
| GET | `/test` | 간단한 테스트 엔드포인트 |
| GET | `/labels` | SegFormer B2가 지원하는 18개 레이블 반환 |

### 6.2 웹 인터페이스 (web.py)

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| GET | `/` | 메인 웹 인터페이스 (테스트 페이지 선택) |
| GET | `/nukki` | 웨딩드레스 누끼 서비스 페이지 |
| GET | `/body-analysis` | 체형 분석 웹 페이지 |
| GET | `/gemini-test` | Gemini 이미지 합성 테스트 페이지 |
| GET | `/3d-conversion` | 3D 이미지 변환 페이지 |
| GET | `/model-comparison` | 모델 비교 테스트 페이지 |
| GET | `/admin` | 관리자 페이지 |
| GET | `/admin/dress-insert` | 드레스 이미지 삽입 관리자 페이지 |
| GET | `/admin/dress-manage` | 드레스 관리자 페이지 |
| GET | `/favicon.ico` | 파비콘 제공 |

### 6.3 세그멘테이션 (segmentation.py)

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| POST | `/api/segment` | 드레스 레이블(7)만 누끼 처리 |
| POST | `/api/segment-custom` | 선택 레이블(콤마 구분)만 누끼 처리 |
| POST | `/api/segment-b0` | SegFormer B0 모델로 세그멘테이션 |
| POST | `/api/remove-background` | 배경 제거 후 인물만 추출 |
| POST | `/api/analyze` | 전체 이미지 분석 및 레이블 비율 반환 |

### 6.4 이미지 합성 (composition.py)

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| POST | `/api/compose-dress` | Gemini 2.5 Flash로 드레스 합성 |
| POST | `/api/gpt4o-gemini/compose` | GPT-4o 프롬프트 + Gemini 합성 |
| POST | `/api/compose-enhanced` | 고품질 파이프라인 (SegFormer B2, RTMPose, HR-VITON, Real-ESRGAN, Color Harmonization) |
| POST | `/api/hr-viton-compose` | HR-VITON 기반 가상 피팅 |

### 6.5 프롬프트 생성 (prompt.py)

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| POST | `/api/gemini/generate-prompt` | Gemini로 커스텀 합성 프롬프트 생성 |
| POST | `/api/gpt4o-gemini/generate-prompt` | GPT-4o로 커스텀 합성 프롬프트 생성 |
| POST | `/api/prompt/generate-short` | GPT-4o-V2로 x.ai 최적화 short prompt 생성 (≤1024자) |
| POST | `/api/xai/generate-prompt` | x.ai grok 모델로 이미지 기반 프롬프트 생성 |

### 6.5.1 통합 트라이온 (tryon_router.py)

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| POST | `/api/tryon/unified` | X.AI 프롬프트 생성 + Gemini 2.5 Flash 이미지 합성 통합 파이프라인 (배경 합성 지원) |

### 6.6 이미지 처리 (image_processing.py)

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| POST | `/api/upscale` | Real-ESRGAN을 활용한 업스케일링 (x2/x4) |
| POST | `/api/color-harmonize` | 색상 보정 (Color Harmonization) |
| POST | `/api/generate-shoes` | 구두 생성 (Gemini 또는 SDXL) |
| POST | `/api/generate-image-xai` | x.ai API를 사용한 텍스트 to 이미지 생성 |
| POST | `/api/tps-warp` | TPS Warp 변환 |
| POST | `/api/pose-estimation` | RTMPose-s로 포즈 키포인트 추론 |

### 6.7 체형 분석 (body_analysis.py)

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| POST | `/api/analyze-body` | 체형 분석 (키, 몸무게 입력) |
| GET | `/api/admin/body-logs` | 체형 분석 로그 조회 (페이징) |
| GET | `/api/admin/body-logs/{log_id}` | 체형 분석 로그 상세 조회 |

### 6.8 드레스 관리 (dress_management.py)

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| GET | `/api/admin/dresses` | 드레스 목록 조회 (페이징) |
| POST | `/api/admin/dresses` | 드레스 추가 (S3 URL 또는 이미지명 입력) |
| POST | `/api/admin/dresses/upload` | 여러 드레스 이미지 업로드 및 S3 저장 |
| DELETE | `/api/admin/dresses/{dress_id}` | 드레스 삭제 (S3 및 DB) |
| GET | `/api/admin/dresses/export` | 드레스 목록 내보내기 (JSON/CSV) |
| POST | `/api/admin/dresses/import` | 드레스 목록 가져오기 (JSON/CSV) |

### 6.9 관리자 기능 (admin.py)

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| GET | `/api/admin/stats` | 관리자 통계 정보 조회 |
| GET | `/api/admin/logs` | 합성 로그 조회 (페이징, 모델 필터) |
| GET | `/api/admin/logs/{log_id}` | 합성 로그 상세 조회 |
| GET | `/api/admin/category-rules` | 카테고리 규칙 조회 |
| POST | `/api/admin/category-rules` | 카테고리 규칙 추가 |
| DELETE | `/api/admin/category-rules` | 카테고리 규칙 삭제 |

### 6.10 모델 관리 (models.py)

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| GET | `/api/models` | 사용 가능한 AI 모델 목록 조회 |
| POST | `/api/models` | 새로운 모델 추가 |

### 6.11 3D 변환 (conversion_3d.py)

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| POST | `/api/convert-to-3d` | Meshy.ai를 사용하여 이미지를 3D 모델로 변환 |
| GET | `/api/check-3d-status/{task_id}` | 3D 변환 작업 상태 확인 |
| POST | `/api/save-3d-model/{task_id}` | 3D 모델을 서버에 저장 |
| GET | `/api/proxy-3d-model` | 3D 모델 프록시 (다운로드) |
| OPTIONS | `/api/proxy-3d-model` | CORS 프리플라이트 |

### 6.12 이미지 프록시 (proxy.py)

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| GET | `/api/proxy-image` | URL로 이미지 프록시 |
| GET | `/api/images/{file_name:path}` | S3 이미지 프록시 |
| GET | `/api/admin/s3-image-proxy` | 관리자용 S3 이미지 프록시 |

---

## 7. 라우터 모듈 상세 설명

### 7.1 info.py - 정보/상태 엔드포인트

**역할**: 서버 상태 확인 및 기본 정보 제공

**주요 기능**:
- `/health`: 모델 로딩 상태 확인
- `/test`: 서버 응답 테스트
- `/labels`: SegFormer 레이블 목록 제공

### 7.2 web.py - 웹 인터페이스

**역할**: HTML 페이지 제공

**주요 기능**:
- 메인 페이지, 각종 테스트 페이지, 관리자 페이지 등 웹 UI 제공
- Jinja2 템플릿을 사용한 동적 HTML 생성

### 7.3 segmentation.py - 세그멘테이션

**역할**: 이미지 세그멘테이션 및 배경 제거

**주요 기능**:
- SegFormer B2/B0 모델을 사용한 의상 세그멘테이션
- 드레스 누끼 처리
- 커스텀 레이블 선택 세그멘테이션
- 배경 제거
- 이미지 분석 (레이블 비율 계산)

**사용 모델**:
- `mattmdjaga/segformer_b2_clothes`: 기본 세그멘테이션
- `matei-dorian/segformer-b0-finetuned-human-parsing`: 경량 모델
- `yolo12138/segformer-b2-human-parse-24`: Human Parsing 특화

### 7.4 composition.py - 이미지 합성

**역할**: 사람과 드레스 이미지 합성

**주요 기능**:
- Gemini 2.5 Flash 기반 이미지 합성
- GPT-4o 프롬프트 생성 후 Gemini 합성
- 고품질 합성 파이프라인 (여러 모델 통합)
- HR-VITON 기반 가상 피팅

**합성 파이프라인** (`/api/compose-enhanced`):
1. SegFormer B2 Human Parsing
2. 드레스 전처리 및 배경 제거
3. RTMPose 포즈 추정
4. HR-VITON 워핑 및 합성
5. Real-ESRGAN 업스케일
6. Color Harmonization

### 7.5 prompt.py - 프롬프트 생성

**역할**: AI 모델을 위한 프롬프트 생성

**주요 기능**:
- Gemini를 사용한 이미지 분석 및 프롬프트 생성
- GPT-4o를 사용한 고품질 프롬프트 생성
- GPT-4o-V2를 사용한 x.ai 최적화 short prompt 생성 (≤1024자)
- x.ai grok 모델을 사용한 이미지 분석 기반 프롬프트 생성
- 사람 이미지와 드레스 이미지를 분석하여 맞춤 프롬프트 생성

### 7.5.1 tryon_router.py - 통합 트라이온 파이프라인

**역할**: X.AI 프롬프트 생성과 Gemini 2.5 Flash 이미지 합성을 통합한 단일 파이프라인

**주요 기능**:
- 통합 트라이온 엔드포인트 (`/api/tryon/unified`)
- X.AI 프롬프트 생성과 Gemini 이미지 합성을 순차적으로 실행
- 단일 API 호출로 프롬프트 생성부터 최종 합성 이미지까지 완료

**통합 파이프라인 V1** (`/api/tryon/unified`):
1. 이미지 전처리 (person, dress, background)
2. X.AI grok-2-vision-1212 모델로 프롬프트 생성
3. 생성된 프롬프트와 이미지들(인물, 드레스, 배경)로 Gemini 2.5 Flash 이미지 합성
4. 결과 이미지 base64 인코딩 및 S3 업로드
5. 테스트 로그 저장

**통합 파이프라인 V2** (`/api/compose_xai_gemini_v2`):
V2는 SegFormer B2 Garment Parsing을 추가하여 더 정확한 의상 추출을 수행합니다.
1. 의상 이미지 전처리
2. **SegFormer B2 Human Parsing** → garment_only 이미지 추출
   - **HuggingFace Inference API 사용** (`yolo12138/segformer-b2-human-parse-24` 모델)
   - API 엔드포인트: `https://router.huggingface.co/hf-inference/models/yolo12138/segformer-b2-human-parse-24`
   - Human Parsing을 통해 의상 영역 자동 추출
   - garment_mask.png, garment_only.png 생성
   - API 호출만으로 동작 (로컬 모델 다운로드 불필요)
   - 인터넷 연결 필요
3. X.AI grok-2-vision-1212 모델로 프롬프트 생성 (person_image, garment_only_image 사용)
4. 생성된 프롬프트와 이미지들(인물, garment_only, 배경)로 Gemini 2.5 Flash 이미지 합성
   - V2는 배경 이미지를 포함하여 합성 (person_image, garment_image, background_image 사용)
5. 결과 이미지 base64 인코딩 및 S3 업로드
6. 테스트 로그 저장

**V1 vs V2 비교**:
- **V1**: 배경 이미지 필요, 원본 드레스 이미지를 XAI에 전달
- **V2**: 배경 이미지 필요, SegFormer B2 Parsing으로 추출한 garment_only 이미지를 XAI에 전달
- **V2 장점**: SegFormer B2 Human Parsing을 통해 의상만 정확히 추출하여 더 나은 프롬프트 생성 및 합성 품질 기대

**환경 변수 설정** (필수):
V2 기능을 사용하려면 `.env` 파일에 다음 환경 변수를 추가해야 합니다:
- `HUGGINGFACE_API_KEY`: HuggingFace Inference API 토큰 (필수)
  - https://huggingface.co/settings/tokens 에서 API 키 발급
  - "Make calls to Inference Providers" 권한 필요

**선택적 환경 변수**:
- `HUGGINGFACE_API_BASE_URL`: API 베이스 URL (기본값: "https://router.huggingface.co/hf-inference/models")
- `SEGFORMER_API_TIMEOUT`: API 요청 타임아웃 (초, 기본값: 60)

**통합 파이프라인 V2.5** (`/fit/v2.5/compose`):
V2.5는 인물 전처리 파이프라인을 추가하여 더욱 정교한 합성 결과를 생성합니다.
1. 의상 이미지 전처리
2. **SegFormer B2 Garment Parsing** → garment_only 이미지 추출
3. **(옵션) 인물 전처리 파이프라인 (1~5단계)**:
   - **Step 1**: SegFormer B2 Human Parsing으로 인물 이미지 파싱
     - face_mask: 레이블 {11(face), 18(skin), 2(hair)}
     - cloth_mask: 레이블 {4(upper), 5(skirt), 6(pants), 7(dress), 8(belt), 16(bag), 17(scarf)}
     - body_mask: 레이블 {12(left-leg), 13(right-leg), 14(left-arm), 15(right-arm)}
   - **Step 2**: face_patch 추출 (face_mask + hair_mask 영역)
   - **Step 3**: base_img 생성 (cloth_mask 영역을 neutral_color(128,128,128)로 덮기)
   - **Step 4**: inpaint_mask 생성 (body_mask - face_mask)
   - **Step 5**: face_patch, base_img, inpaint_mask를 base64(PNG)로 변환
4. X.AI grok-2-vision-1212 모델로 프롬프트 생성 (person_image, garment_only_image 사용)
5. 생성된 프롬프트와 이미지들(base_img 또는 person_image, garment_only, 배경)로 Gemini 2.5 Flash 이미지 합성
6. **(옵션) face_patch 합성 및 경계 블렌딩**: Gemini 생성 이미지에 face_patch를 합성하고 경계 블렌딩 수행
7. 결과 이미지 base64 인코딩 및 S3 업로드
8. 테스트 로그 저장

**V1 vs V2 vs V2.5 비교**:
- **V1**: 배경 이미지 필요, 원본 드레스 이미지를 XAI에 전달
- **V2**: 배경 이미지 필요, SegFormer B2 Parsing으로 추출한 garment_only 이미지를 XAI에 전달
- **V2.5**: 배경 이미지 필요, 인물 전처리 파이프라인 추가로 base_img 사용 및 face_patch 합성
  - 인물 전처리로 의상 영역을 중립색으로 덮어 더 정확한 합성
  - face_patch 합성으로 얼굴 보존 품질 향상
  - 경계 블렌딩으로 자연스러운 합성 결과

**인물 전처리 전용 엔드포인트** (`/fit/v2.5/preprocess-person`):
- 인물 이미지만 업로드하여 face_mask, face_patch, base_img, inpaint_mask를 추출
- 디버깅 및 테스트용으로 사용 가능

**API 사용 예시**:
```bash
curl -X POST "http://localhost:8000/api/compose_xai_gemini_v2" \
  -F "person_image=@person.jpg" \
  -F "garment_image=@dress.jpg" \
  -F "background_image=@background.jpg"

# 응답:
# {
#   "success": true,
#   "prompt": "생성된 프롬프트...",
#   "result_image": "data:image/png;base64,...",
#   "message": "통합 트라이온 파이프라인 V2가 성공적으로 완료되었습니다.",
#   "llm": "segformer-b2-parsing+grok-2-vision-1212+gemini-2.5-flash-image"
# }
```

**입력**:
- **V1** (`/api/tryon/unified`):
  - `person_image`: 사람 이미지 파일 (필수)
  - `dress_image`: 드레스 이미지 파일 (필수)
  - `background_image`: 배경 이미지 파일 (필수)
- **V2** (`/api/compose_xai_gemini_v2`):
  - `person_image`: 사람 이미지 파일 (필수)
  - `garment_image`: 의상 이미지 파일 (필수)
  - `background_image`: 배경 이미지 파일 (필수)

**배경 합성 기능**:
- 배경 이미지를 인물 이미지 크기에 맞춰 자동 조정
- Gemini API에 배경 이미지를 함께 전달하여 한 번의 호출로 배경까지 합성
- 배경 화질을 유지하면서 인물을 자연스럽게 배경에 통합
- 조명, 그림자, 색상, 원근감을 배경에 맞춰 자동 조정

**출력**:
- `success`: 성공 여부
- `prompt`: 생성된 프롬프트
- `result_image`: 합성된 이미지 (base64 data URL)
- `message`: 응답 메시지
- `llm`: 사용된 LLM 정보 (예: "grok-2-vision-1212+gemini-2.5-flash-image")

**Short Prompt 파이프라인** (`/api/prompt/generate-short`):
- GPT-4o-V2 모델을 사용하여 x.ai 이미지 생성을 위한 최적화된 short prompt 생성
- 프롬프트는 최대 1024자로 제한되며, 단일 연속 문단 형식으로 생성
- 얼굴, 몸, 포즈, 배경을 정확히 유지하면서 원본 의상을 완전히 제거하고 드레스만 적용
- 노출된 신체 부위에 자연스러운 피부 생성 및 드레스에 맞는 신발 추가

**x.ai 프롬프트 생성 파이프라인** (`/api/xai/generate-prompt`):
- x.ai grok-2-vision-1212 모델을 사용하여 이미지 분석 기반 프롬프트 생성
- Image 1(사람)과 Image 2(드레스)를 분석하여 상세한 프롬프트 생성
- 드레스의 실루엣, 패브릭, 색상, 넥라인, 디테일 등을 자동 분석하여 프롬프트에 포함
- 사람의 얼굴, 체형, 포즈, 조명, 배경을 정확히 유지하면서 드레스만 교체하는 프롬프트 생성
- 드레스에 어울리는 신발 자동 추가

### 7.6 image_processing.py - 이미지 처리

**역할**: 다양한 이미지 후처리 기능

**주요 기능**:
- Real-ESRGAN 업스케일링 (x2/x4)
- Color Harmonization (색상 보정)
- 구두 생성 (Gemini 또는 SDXL)
- TPS Warp 변환
- RTMPose 포즈 추정

**사용 모델**:
- Real-ESRGAN: 해상도 향상
- SDXL Pipeline: 구두 생성
- RTMPose-s: 포즈 추정

### 7.7 body_analysis.py - 체형 분석

**역할**: 사용자 체형 분석 및 추천

**주요 기능**:
- MediaPipe 기반 포즈 랜드마크 추출
- 체형 측정 (어깨, 허리, 엉덩이 등)
- 체형 분류 (A/H/X 라인 등)
- Gemini 기반 스타일 추천
- 분석 로그 저장 및 조회

### 7.8 dress_management.py - 드레스 관리

**역할**: 드레스 데이터베이스 및 S3 관리

**주요 기능**:
- 드레스 목록 조회 (페이징)
- 드레스 추가 (S3 URL 또는 이미지명)
- 드레스 이미지 일괄 업로드
- 드레스 삭제 (S3 및 DB)
- 드레스 목록 내보내기/가져오기 (JSON/CSV)

### 7.9 admin.py - 관리자 기능

**역할**: 관리자 대시보드 및 로그 관리

**주요 기능**:
- 통계 정보 조회 (전체/성공/실패 건수 등)
- 합성 로그 조회 및 상세 조회
- 카테고리 규칙 관리 (스타일 자동 감지 규칙)

### 7.10 models.py - 모델 관리

**역할**: AI 모델 설정 관리

**주요 기능**:
- `models_config.json` 파일 기반 모델 목록 관리
- 모델 추가 (JSON 형식)
- 모델 목록 조회

### 7.11 conversion_3d.py - 3D 변환

**역할**: 2D 이미지를 3D 모델로 변환

**주요 기능**:
- Meshy.ai API 연동
- 3D 변환 작업 시작 및 상태 확인
- 완성된 3D 모델 다운로드 및 저장
- 3D 모델 프록시 제공

### 7.12 proxy.py - 이미지 프록시

**역할**: 외부 이미지 프록시 및 CORS 처리

**주요 기능**:
- URL 기반 이미지 프록시
- S3 이미지 프록시
- CORS 문제 해결

---

## 8. 누끼 · 합성 프로세스

### 8.1 누끼(세그멘테이션) 흐름

1. 업로드 이미지 → PIL 변환 → 원본 크기 저장
2. `SegformerImageProcessor` 전처리
3. `mattmdjaga/segformer_b2_clothes` 또는 `matei-dorian/segformer-b0-finetuned-human-parsing` 추론
4. Bilinear 업샘플링 및 argmax로 레이블 결정
5. 대상 레이블 마스크 생성
6. RGBA 이미지 (알파 채널=마스크) 반환

### 8.2 합성 파이프라인 (`/api/compose-enhanced`)

1. SegFormer B2 Human Parsing → 배경 제거
2. 드레스 이미지 배경 제거 및 정렬
3. RTMPose로 포즈 키포인트 추출 (허리 좌표 등)
4. 다시 SegFormer로 의상 영역 마스크 추출
5. HR-VITON 워핑 & 합성 (필요 시 Fallback)
6. Real-ESRGAN 업스케일
7. Color Harmonization으로 색상·조명 보정

시험/로그를 위해 S3 업로드 함수가 포함되어 있으며 실패 시에도 Fallback 처리가 정의되어 있습니다.

---

## 9. 이미지 보정 서버

- **핵심 기능**: InstructPix2Pix로 이미지 편집, GFPGAN/Real-ESRGAN으로 얼굴·해상도 보정, ControlNet+SDXL Inpaint로 정밀 편집.
- **API**
  - `POST /api/enhance-image`: 이미지와 보정 지시(한국어 가능)를 받아 결과 이미지 반환.
  - 파라미터: `instruction`, `num_inference_steps`, `image_guidance_scale`, `use_gfpgan`, `gfpgan_weight`.
- **실행**
  ```bash
  python enhancement_server.py 8003
  ```
- **지원 요청 예시**: 형태(어깨/허리), 스타일(우아하게, 캐주얼), 배경(해변/정원/스튜디오), 분위기(로맨틱, 밝게), 조명, 얼굴 보정 등.
- **성능 주의사항**: GPU VRAM 8GB 이상 권장, 첫 실행 시 모델 다운로드(약 2.5GB + GFPGAN).

---

## 10. 이미지 보정 모델 & 활용 가이드

### 10.1 InstructPix2Pix 가이드

- 설치: `pip install diffusers transformers accelerate torch torchvision pillow`
- 기본 사용 예제 제공 (prompt: `"make shoulders narrower and more natural"` 등).
- 한국어 → 영어 프롬프트 매핑 예시 및 FastAPI 연동 코드 예시 포함.
- 텍스트 지시에 따라 형태/스타일/배경/분위기/조명/품질을 조정할 수 있음.

### 10.2 모델 라인업

| 카테고리 | 모델 | 용도 |
|----------|------|------|
| 텍스트 기반 편집 | InstructPix2Pix, MagicBrush, IP-Adapter, ControlNet | 자연어 지시 편집, 이미지+텍스트 편집, 포즈 제어 |
| 인체 형태 조작 | DensePose, SMPL, MediaPipe+ControlNet | 3D 형태 추정 및 포즈 제어 |
| 품질 향상 | Real-ESRGAN, GFPGAN, CodeFormer | 업스케일, 얼굴 복원 |
| Stable Diffusion 기반 | SDXL Base, SD 2.1, SD Inpainting | 고해상도 생성, 인페인팅 |

**추천 워크플로**
1. InstructPix2Pix로 1차 수정
2. Real-ESRGAN 업스케일
3. GFPGAN/CodeFormer로 얼굴 추가 보정

### 10.3 CONTROLNET & 대체 모델

- `diffusers/controlnet-openpose-sdxl-1.0` + `diffusers/stable-diffusion-xl-1.0-inpaint` 조합.
- IP-Adapter + SDXL, ControlNet + Inpaint 등 하이브리드 구성 제안.
- Replicate 등의 API 기반 대안도 문서화 되어 있음 (엔터프라이즈 확장 시 참고).

### 10.4 이미지 리터칭 체크리스트

- 요청 분석 → 프롬프트 변환 → 마스크 생성 → 편집 → 품질 보정 → 검수.
- 오류 및 공통 실수, 고객 피드백 대응 가이드 포함.

---

## 11. 체형 분석

- MediaPipe `pose_landmarker_lite.task`를 사용해 33개 포즈 랜드마크 추출.
- 측정 항목
  - 어깨/엉덩이 폭, 팔/다리 길이, 어깨-엉덩이 비율 등.
  - 체형 분류 (A/H/X 라인 등) 및 Gemini 기반 스타일 추천.
- 테스트 서버(`body_analysis_test/test_body_analysis.py`)는 이미지 업로드 후 실시간 분석 UI 제공.
- Gemini 2.5 Flash (Image)로 상세 분석·코칭 멘트 생성 가능.

---

## 12. 모델 추천 & 레퍼런스

- **SegFormer 계열**
  - `mattmdjaga/segformer_b2_clothes`: 기본 누끼
  - `matei-dorian/segformer-b0-finetuned-human-parsing`: 경량 모델
  - `yolo12138/segformer-b2-human-parse-24`: human parsing 특화

- **포즈/체형**
  - `RTMPose-s`: 고속 포즈 인식
  - MediaPipe Pose Landmarker Lite/Full/Heavy 비교 제공 (정확도 vs 속도)

- **합성 & 보정**
  - HR-VITON: 드레스 워핑/합성, 추가 세팅 필요
  - Real-ESRGAN, GFPGAN, CodeFormer: 화질/얼굴 복원
  - Stable Diffusion XL, ControlNet, IP-Adapter: 고품질 편집

- **배경 변경 가이드**
  - 누끼 결과를 기반으로 배경 템플릿 합성
  - 색감/노출 맞춤 팁, 스튜디오·정원·비치 등 테마별 프롬프트 예시

---

## 13. 3D 연동 시나리오

- `3d_conversion_test` 문서에 3D 모델 변환 및 Meshy 활용 단계가 정리되어 있음.
- 핵심 포인트
  - 입력 준비 → Meshy 업로드 → 파라미터 조정 → 모델 다운로드 → 후처리.
  - 3D 프린팅·AR/VR용 확장 가능성 언급.

---

## 14. 작업 기록 및 향후 계획

- **모듈화 작업**: `main_original.py` (6,109줄)를 12개의 라우터 모듈로 분리
- **모델 비교 페이지**: 모델 추가/관리 기능 구현
- **배경 합성 기능 추가** (2025년 11월):
  - `/api/tryon/unified` 엔드포인트에 배경 이미지 파라미터 추가
  - XAI + Gemini 2.5 Flash V1 모델에 배경 합성 기능 통합
  - model-comparison 페이지에 배경 이미지 입력 필드 추가
  - 배경 화질 유지 및 자연스러운 인물 통합을 위한 프롬프트 개선
  - 향후 확장: 샘플 배경 이미지 선택 기능 (프론트엔드)
- 향후 개선 아이디어
  - 모델 수정/삭제 UI
  - ControlNet/SDXL 기반 정밀 합성 자동화
  - Gemini 결과 품질 향상 및 안전 필터 대응
  - 각 라우터별 단위 테스트 추가
  - 샘플 배경 이미지 관리 기능 (관리자 페이지)

---

## 부록. 참고 자료

- SegFormer Paper: [https://arxiv.org/abs/2105.15203](https://arxiv.org/abs/2105.15203)
- Hugging Face Transformers 문서
- FastAPI 공식 문서
- MediaPipe Pose Landmarker 문서
- Real-ESRGAN, GFPGAN, CodeFormer GitHub
- Google Gemini API 문서
- Meshy.ai API 문서

> 추가 세부 정보가 필요하면 Git 히스토리 또는 관련 Python/스크립트 파일을 참고하세요.
