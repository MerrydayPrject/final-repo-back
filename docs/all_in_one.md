# 프로젝트 통합 문서 (`final-repo-back`)

> 기존 개별 Markdown 문서의 핵심 내용을 한 파일로 정리한 자료입니다. 원본 문서는 모두 통합되어 삭제되었습니다.

- [1. 개요](#1-개요)
- [2. 환경 구성](#2-환경-구성)
- [3. 데이터베이스 설정](#3-데이터베이스-설정)
- [4. 실행 가이드](#4-실행-가이드)
- [5. API 요약](#5-api-요약)
- [6. 누끼 · 합성 프로세스](#6-누끼--합성-프로세스)
- [7. 이미지 보정 서버](#7-이미지-보정-서버)
- [8. 이미지 보정 모델 & 활용 가이드](#8-이미지-보정-모델--활용-가이드)
- [9. 체형 분석](#9-체형-분석)
- [10. 모델 추천 & 레퍼런스](#10-모델-추천--레퍼런스)
- [11. 3D 연동 시나리오](#11-3d-연동-시나리오)
- [12. 작업 기록 및 향후 계획](#12-작업-기록-및-향후-계획)
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

## 2. 환경 구성

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
  ```
  - `.env`는 Git에 커밋하지 말 것.

---

## 3. 데이터베이스 설정

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
- 서버 실행 시 `composition_logs`, `dress_info` 테이블을 자동 생성.
- 수동 생성이 필요한 경우 기존 SQL 스크립트 참고:
  - `composition_logs`: 합성 로그 저장 (모델, 프롬프트, 이미지 경로, 성공 여부, 처리 시간 등).
  - `dress_info`: 드레스 이미지 메타 정보 저장.

---

## 4. 실행 가이드

### 4.1 주요 서버

| 서버 | 포트 | 설명 | 실행 파일 |
|------|------|------|-----------|
| 메인 백엔드 | 8000 | 세그멘테이션 · 합성 API | `main.py` |
| 이미지 보정 | 8003 | InstructPix2Pix + GFPGAN 기반 보정 | `image_enhancement_server/enhancement_server.py` |
| 체형 분석 테스트 | 8002 | MediaPipe 기반 체형 측정 | `body_analysis_test/test_body_analysis.py` |

### 4.2 실행 명령 예시 (PowerShell)

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

## 5. API 요약

### 5.1 정보 조회

- `GET /health`: 서버 상태 및 SegFormer 로딩 여부 확인
- `GET /labels`: SegFormer B2가 지원하는 18개 레이블 반환

### 5.2 세그멘테이션

- `POST /api/segment`: 드레스 레이블(7)만 누끼 처리
- `POST /api/segment-custom`: 선택 레이블(콤마 구분)만 누끼 처리
- `POST /api/remove-background`: 배경 제거 후 인물만 추출

### 5.3 합성 & 프롬프트

- `POST /api/compose-dress`: Gemini 2.5 Flash로 드레스 합성. 이미지 2장과 옵션 프롬프트를 입력.
- `POST /api/gpt4o-gemini/generate-prompt`: GPT-4o로 커스텀 합성 프롬프트 생성.
- `POST /api/gpt4o-gemini/compose`: GPT-4o가 만든 프롬프트를 Gemini 합성에 사용.
- `/gemini-test`: 합성 테스트용 웹 페이지.

### 5.4 기타

- `POST /api/pose-estimation`: RTMPose-s로 133개 키포인트 추론.
- `POST /api/upscale`: Real-ESRGAN을 활용한 업스케일링 (x2/x4).
- `POST /api/compose-enhanced`: 고품질 파이프라인 (SegFormer B2, 드레스 전처리, RTMPose, HR-VITON, Real-ESRGAN, Color Harmonization).
- `POST /api/analyze`: 전체 이미지 분석 및 레이블 비율 반환.

---

## 6. 누끼 · 합성 프로세스

### 6.1 누끼(세그멘테이션) 흐름

1. 업로드 이미지 → PIL 변환 → 원본 크기 저장
2. `SegformerImageProcessor` 전처리
3. `mattmdjaga/segformer_b2_clothes` 또는 `matei-dorian/segformer-b0-finetuned-human-parsing` 추론
4. Bilinear 업샘플링 및 argmax로 레이블 결정
5. 대상 레이블 마스크 생성
6. RGBA 이미지 (알파 채널=마스크) 반환

### 6.2 합성 파이프라인 (`/api/compose-enhanced`)

1. SegFormer B2 Human Parsing → 배경 제거
2. 드레스 이미지 배경 제거 및 정렬
3. RTMPose로 포즈 키포인트 추출 (허리 좌표 등)
4. 다시 SegFormer로 의상 영역 마스크 추출
5. HR-VITON 워핑 & 합성 (필요 시 Fallback)
6. Real-ESRGAN 업스케일
7. Color Harmonization으로 색상·조명 보정

시험/로그를 위해 S3 업로드 함수가 포함되어 있으며 실패 시에도 Fallback 처리가 정의되어 있습니다.

---

## 7. 이미지 보정 서버

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

## 8. 이미지 보정 모델 & 활용 가이드

### 8.1 InstructPix2Pix 가이드

- 설치: `pip install diffusers transformers accelerate torch torchvision pillow`
- 기본 사용 예제 제공 (prompt: `"make shoulders narrower and more natural"` 등).
- 한국어 → 영어 프롬프트 매핑 예시 및 FastAPI 연동 코드 예시 포함.
- 텍스트 지시에 따라 형태/스타일/배경/분위기/조명/품질을 조정할 수 있음.

### 8.2 모델 라인업 (IMAGE_ENHANCEMENT_MODELS.md 요약)

| 카테고리 | 모델 | 용도 |
|----------|------|------|
| 텍스트 기반 편집 | InstructPix2Pix, MagicBrush, IP-Adapter, ControlNet | 자연어 지시 편집, 이미지+텍스트 편집, 포즈 제어 |
| 인체 형태 조작 | DensePose, SMPL, MediaPipe+ControlNet | 3D 형태 추정 및 포즈 제어 |
| 품질 향상 | Real-ESRGAN, GFPGAN, CodeFormer | 업스케일, 얼굴 복원 |
| Stable Diffusion 기반 | SDXL Base, SD 2.1, SD Inpainting | 고해상도 생성, 인페인팅 |

추천 워크플로
1. InstructPix2Pix로 1차 수정
2. Real-ESRGAN 업스케일
3. GFPGAN/CodeFormer로 얼굴 추가 보정

### 8.3 CONTROLNET & 대체 모델

- `diffusers/controlnet-openpose-sdxl-1.0` + `diffusers/stable-diffusion-xl-1.0-inpaint` 조합.
- IP-Adapter + SDXL, ControlNet + Inpaint 등 하이브리드 구성 제안.
- Replicate 등의 API 기반 대안도 문서화 되어 있음 (엔터프라이즈 확장 시 참고).

### 8.4 이미지 리터칭 체크리스트

- 요청 분석 → 프롬프트 변환 → 마스크 생성 → 편집 → 품질 보정 → 검수.
- 오류 및 공통 실수, 고객 피드백 대응 가이드 포함.

---

## 9. 체형 분석

- MediaPipe `pose_landmarker_lite.task`를 사용해 33개 포즈 랜드마크 추출.
- 측정 항목
  - 어깨/엉덩이 폭, 팔/다리 길이, 어깨-엉덩이 비율 등.
  - 체형 분류 (A/H/X 라인 등) 및 Gemini 기반 스타일 추천.
- 테스트 서버(`body_analysis_test/test_body_analysis.py`)는 이미지 업로드 후 실시간 분석 UI 제공.
- Gemini 2.5 Flash (Image)로 상세 분석·코칭 멘트 생성 가능.

---

## 10. 모델 추천 & 레퍼런스

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

## 11. 3D 연동 시나리오

- `3d_conversion_test` 문서에 3D 모델 변환 및 Meshy 활용 단계가 정리되어 있음.
- 핵심 포인트
  - 입력 준비 → Meshy 업로드 → 파라미터 조정 → 모델 다운로드 → 후처리.
  - 3D 프린팅·AR/VR용 확장 가능성 언급.

---

## 12. 작업 기록 및 향후 계획

- **`plan.md`**: 모델 비교 페이지 개선 (버튼 UI 축소, 모델 추가 모달, `POST /api/models` 등) 작업 내역.
- **`JY.md`**: 파이프라인 세부 점검 로그, 실패 시 Fallback 전략, HR-VITON·SegFormer 단계별 유의 사항과 테스트 결과.
- 향후 개선 아이디어
  - 모델 수정/삭제 UI
  - ControlNet/SDXL 기반 정밀 합성 자동화
  - Gemini 결과 품질 향상 및 안전 필터 대응

---

## 부록. 참고 자료

- SegFormer Paper: [https://arxiv.org/abs/2105.15203](https://arxiv.org/abs/2105.15203)
- Hugging Face Transformers 문서
- FastAPI 공식 문서
- MediaPipe Pose Landmarker 문서
- Real-ESRGAN, GFPGAN, CodeFormer GitHub
- Google Gemini API 문서

> 추가 세부 정보가 필요하면 Git 히스토리 또는 관련 Python/스크립트 파일을 참고하세요.

# 프로젝트 통합 문서 (`final-repo-back`)

> 기존 개별 Markdown 문서의 핵심 내용을 한 파일로 정리한 자료입니다. 원본 문서는 모두 통합되어 삭제되었습니다.

- [1. 개요](#1-개요)
- [2. 환경 구성](#2-환경-구성)
- [3. 데이터베이스 설정](#3-데이터베이스-설정)
- [4. 실행 가이드](#4-실행-가이드)
- [5. API 요약](#5-api-요약)
- [6. 누끼 · 합성 프로세스](#6-누끼--합성-프로세스)
- [7. 이미지 보정 서버](#7-이미지-보정-서버)
- [8. 이미지 보정 모델 & 활용 가이드](#8-이미지-보정-모델--활용-가이드)
- [9. 체형 분석](#9-체형-분석)
- [10. 모델 추천 & 레퍼런스](#10-모델-추천--레퍼런스)
- [11. 3D 연동 시나리오](#11-3d-연동-시나리오)
- [12. 작업 기록 및 향후 계획](#12-작업-기록-및-향후-계획)
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

## 2. 환경 구성

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
  ```
  - `.env`는 Git에 커밋하지 말 것.

---

## 3. 데이터베이스 설정

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
- 서버 실행 시 `composition_logs`, `dress_info` 테이블을 자동 생성.
- 수동 생성이 필요한 경우 기존 SQL 스크립트 참고:
  - `composition_logs`: 합성 로그 저장 (모델, 프롬프트, 이미지 경로, 성공 여부, 처리 시간 등).
  - `dress_info`: 드레스 이미지 메타 정보 저장.

---

## 4. 실행 가이드

### 4.1 주요 서버

| 서버 | 포트 | 설명 | 실행 파일 |
|------|------|------|-----------|
| 메인 백엔드 | 8000 | 세그멘테이션 · 합성 API | `main.py` |
| 이미지 보정 | 8003 | InstructPix2Pix + GFPGAN 기반 보정 | `image_enhancement_server/enhancement_server.py` |
| 체형 분석 테스트 | 8002 | MediaPipe 기반 체형 측정 | `body_analysis_test/test_body_analysis.py` |

### 4.2 실행 명령 예시 (PowerShell)

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

## 5. API 요약

### 5.1 정보 조회

- `GET /health`: 서버 상태 및 SegFormer 로딩 여부 확인
- `GET /labels`: SegFormer B2가 지원하는 18개 레이블 반환

### 5.2 세그멘테이션

- `POST /api/segment`: 드레스 레이블(7)만 누끼 처리
- `POST /api/segment-custom`: 선택 레이블(콤마 구분)만 누끼 처리
- `POST /api/remove-background`: 배경 제거 후 인물만 추출

### 5.3 합성 & 프롬프트

- `POST /api/compose-dress`: Gemini 2.5 Flash로 드레스 합성. 이미지 2장과 옵션 프롬프트를 입력.
- `POST /api/gpt4o-gemini/generate-prompt`: GPT-4o로 커스텀 합성 프롬프트 생성.
- `POST /api/gpt4o-gemini/compose`: GPT-4o가 만든 프롬프트를 Gemini 합성에 사용.
- `/gemini-test`: 합성 테스트용 웹 페이지.

### 5.4 기타

- `POST /api/pose-estimation`: RTMPose-s로 133개 키포인트 추론.
- `POST /api/upscale`: Real-ESRGAN을 활용한 업스케일링 (x2/x4).
- `POST /api/compose-enhanced`: 고품질 파이프라인 (SegFormer B2, 드레스 전처리, RTMPose, HR-VITON, Real-ESRGAN, Color Harmonization).
- `POST /api/analyze`: 전체 이미지 분석 및 레이블 비율 반환.

---

## 6. 누끼 · 합성 프로세스

### 6.1 누끼(세그멘테이션) 흐름

1. 업로드 이미지 → PIL 변환 → 원본 크기 저장
2. `SegformerImageProcessor` 전처리
3. `mattmdjaga/segformer_b2_clothes` 또는 `matei-dorian/segformer-b0-finetuned-human-parsing` 추론
4. Bilinear 업샘플링 및 argmax로 레이블 결정
5. 대상 레이블 마스크 생성
6. RGBA 이미지 (알파 채널=마스크) 반환

### 6.2 합성 파이프라인 (`/api/compose-enhanced`)

1. SegFormer B2 Human Parsing → 배경 제거
2. 드레스 이미지 배경 제거 및 정렬
3. RTMPose로 포즈 키포인트 추출 (허리 좌표 등)
4. 다시 SegFormer로 의상 영역 마스크 추출
5. HR-VITON 워핑 & 합성 (필요 시 Fallback)
6. Real-ESRGAN 업스케일
7. Color Harmonization으로 색상·조명 보정

시험/로그를 위해 S3 업로드 함수가 포함되어 있으며 실패 시에도 Fallback 처리가 정의되어 있습니다.

---

## 7. 이미지 보정 서버

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

## 8. 이미지 보정 모델 & 활용 가이드

### 8.1 InstructPix2Pix 가이드

- 설치: `pip install diffusers transformers accelerate torch torchvision pillow`
- 기본 사용 예제 제공 (prompt: `"make shoulders narrower and more natural"` 등).
- 한국어 → 영어 프롬프트 매핑 예시 및 FastAPI 연동 코드 예시 포함.
- 텍스트 지시에 따라 형태/스타일/배경/분위기/조명/품질을 조정할 수 있음.

### 8.2 모델 라인업 (IMAGE_ENHANCEMENT_MODELS.md 요약)

| 카테고리 | 모델 | 용도 |
|----------|------|------|
| 텍스트 기반 편집 | InstructPix2Pix, MagicBrush, IP-Adapter, ControlNet | 자연어 지시 편집, 이미지+텍스트 편집, 포즈 제어 |
| 인체 형태 조작 | DensePose, SMPL, MediaPipe+ControlNet | 3D 형태 추정 및 포즈 제어 |
| 품질 향상 | Real-ESRGAN, GFPGAN, CodeFormer | 업스케일, 얼굴 복원 |
| Stable Diffusion 기반 | SDXL Base, SD 2.1, SD Inpainting | 고해상도 생성, 인페인팅 |

추천 워크플로
1. InstructPix2Pix로 1차 수정
2. Real-ESRGAN 업스케일
3. GFPGAN/CodeFormer로 얼굴 추가 보정

### 8.3 CONTROLNET & 대체 모델

- `diffusers/controlnet-openpose-sdxl-1.0` + `diffusers/stable-diffusion-xl-1.0-inpaint` 조합.
- IP-Adapter + SDXL, ControlNet + Inpaint 등 하이브리드 구성 제안.
- Replicate 등의 API 기반 대안도 문서화 되어 있음 (엔터프라이즈 확장 시 참고).

### 8.4 이미지 리터칭 체크리스트

- 요청 분석 → 프롬프트 변환 → 마스크 생성 → 편집 → 품질 보정 → 검수.
- 오류 및 공통 실수, 고객 피드백 대응 가이드 포함.

---

## 9. 체형 분석

- MediaPipe `pose_landmarker_lite.task`를 사용해 33개 포즈 랜드마크 추출.
- 측정 항목
  - 어깨/엉덩이 폭, 팔/다리 길이, 어깨-엉덩이 비율 등.
  - 체형 분류 (A/H/X 라인 등) 및 Gemini 기반 스타일 추천.
- 테스트 서버(`body_analysis_test/test_body_analysis.py`)는 이미지 업로드 후 실시간 분석 UI 제공.
- Gemini 2.5 Flash (Image)로 상세 분석·코칭 멘트 생성 가능.

---

## 10. 모델 추천 & 레퍼런스

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

## 11. 3D 연동 시나리오

- `3d_conversion_test` 문서에 3D 모델 변환 및 Meshy 활용 단계가 정리되어 있음.
- 핵심 포인트
  - 입력 준비 → Meshy 업로드 → 파라미터 조정 → 모델 다운로드 → 후처리.
  - 3D 프린팅·AR/VR용 확장 가능성 언급.

---

## 12. 작업 기록 및 향후 계획

- **`plan.md`**: 모델 비교 페이지 개선 (버튼 UI 축소, 모델 추가 모달, `POST /api/models` 등) 작업 내역.
- **`JY.md`**: 파이프라인 세부 점검 로그, 실패 시 Fallback 전략, HR-VITON·SegFormer 단계별 유의 사항과 테스트 결과.
- 향후 개선 아이디어
  - 모델 수정/삭제 UI
  - ControlNet/SDXL 기반 정밀 합성 자동화
  - Gemini 결과 품질 향상 및 안전 필터 대응

---

## 부록. 참고 자료

- SegFormer Paper: [https://arxiv.org/abs/2105.15203](https://arxiv.org/abs/2105.15203)
- Hugging Face Transformers 문서
- FastAPI 공식 문서
- MediaPipe Pose Landmarker 문서
- Real-ESRGAN, GFPGAN, CodeFormer GitHub
- Google Gemini API 문서

> 추가 세부 정보가 필요하면 Git 히스토리 또는 관련 Python/스크립트 파일을 참고하세요.


