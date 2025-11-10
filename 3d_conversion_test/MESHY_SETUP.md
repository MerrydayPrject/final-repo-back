# Meshy.ai 3D 변환 설정 가이드

[Meshy.ai](https://www.meshy.ai)를 사용하여 실제 3D 모델을 생성합니다.

## 🔑 API 키 발급

1. **회원가입**
   - https://www.meshy.ai 접속
   - 계정 생성 (Google, Discord 등으로 간편 가입)

2. **API 키 발급**
   - 로그인 후 Dashboard → API Keys 메뉴
   - "Create New Key" 클릭
   - API 키 복사 (나중에 다시 볼 수 없으니 안전하게 저장)

3. **무료 크레딧**
   - 가입 시 무료 크레딧 제공
   - 이미지 → 3D 변환: 크레딧당 약 20-50개 생성 가능

## ⚙️ 설정 방법

### 1. `.env` 파일에 API 키 추가

`final-repo-back/.env` 파일을 열고 다음 줄을 추가하세요:

```bash
# Meshy.ai API Key
MESHY_API_KEY=msy_your_api_key_here
```

### 2. .env 파일 예시

```bash
# 데이터베이스
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=devuser
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=marryday

# AWS S3
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=ap-northeast-2
S3_BUCKET_NAME=your-bucket

# Gemini API (선택사항)
GEMINI_API_KEY=your_gemini_key

# Meshy.ai API (3D 변환)
MESHY_API_KEY=msy_your_api_key_here
```

## 🚀 사용 방법

### 1. 서버 실행

```bash
cd final-repo-back
uvicorn main:app --reload
```

### 2. 웹 브라우저에서 접속

```
http://localhost:8000/3d-conversion
```

### 3. 이미지 업로드 및 3D 변환

1. 드레스 이미지 업로드
2. "3D 모델 생성 시작" 클릭
3. 2-5분 대기 (자동으로 5초마다 상태 확인)
4. 완료되면 결과 화면에 표시:
   - 원본 이미지 & 3D 모델 썸네일
   - GLB, FBX, USDZ, OBJ 다운로드 링크
   - 💾 "서버에 저장" 버튼 클릭 시 서버에 저장

### 4. 서버 저장 기능

완료된 3D 모델을 서버에 저장하려면:
- 결과 화면에서 **"💾 서버에 저장"** 버튼 클릭
- 모든 포맷(GLB, FBX, USDZ, OBJ) + 썸네일이 저장됨
- 저장 위치: `final-repo-back/3d_models/{task_id}/`
- 저장된 파일 목록이 화면에 표시됨

## 📊 API 사용량 확인

- Dashboard → Usage 메뉴에서 확인
- 크레딧 소진 시 유료 플랜 구매 필요

## 🎨 지원 기능

### Image to 3D (API v1)
- **API 엔드포인트**: `POST /openapi/v1/image-to-3d`
- **입력**: PNG, JPG, JPEG 이미지 (base64 data URI 형식)
- **출력**: GLB, FBX, USDZ, OBJ 등
- **텍스처**: PBR 텍스처 자동 생성 (`enable_pbr: true`)
- **처리 시간**: 2-5분
- **AI 모델**: 
  - `meshy-4` (안정적)
  - `meshy-5` 
  - `latest` (Meshy 6 Preview - 기본값)

### 3D 모델 특징
- ✅ **고품질 메쉬**: 자동 리메시 및 리토폴로지
- ✅ **PBR 텍스처**: Base Color, Normal, Roughness, Metallic
- ✅ **다양한 포맷**: 주요 3D 소프트웨어 호환
- ✅ **AR 지원**: USDZ 포맷으로 Apple AR 지원
- ✅ **폴리곤 제어**: `target_polycount` (100~300,000)

### API 파라미터
```json
{
  "image_url": "data:image/png;base64,...",  // 필수
  "ai_model": "meshy-4",                      // 선택
  "enable_pbr": true,                         // PBR 맵 생성
  "should_remesh": true,                      // 리메시 활성화
  "should_texture": true,                     // 텍스처 생성
  "topology": "triangle",                     // quad or triangle
  "target_polycount": 30000                   // 목표 폴리곤 수
}
```

### API 응답 (성공)
```json
{
  "result": "018a210d-8ba4-705c-b111-1f1776f7f578"  // task_id
}
```

### 상태 확인 (GET)
- **엔드포인트**: `GET /openapi/v1/image-to-3d/{task_id}`
- **상태 값**: `PENDING`, `IN_PROGRESS`, `SUCCEEDED`, `FAILED`, `CANCELED`
- **진행률**: 0-100

---

## 🔌 백엔드 API 엔드포인트

### 1. 3D 변환 시작
```http
POST /api/convert-to-3d
Content-Type: multipart/form-data

- image: 이미지 파일

Response:
{
  "success": true,
  "task_id": "019a6c69-f886-7134-9a97-25edda821f1a",
  "message": "3D 모델 생성 작업이 시작되었습니다."
}
```

### 2. 상태 확인
```http
GET /api/check-3d-status/{task_id}?save_to_server=false

Response:
{
  "success": true,
  "status": "SUCCEEDED",
  "progress": 100,
  "model_urls": {
    "glb": "https://...",
    "fbx": "https://...",
    "usdz": "https://...",
    "obj": "https://..."
  },
  "thumbnail_url": "https://...",
  "message": "상태: SUCCEEDED"
}
```

### 3. 서버에 저장
```http
POST /api/save-3d-model/{task_id}

Response:
{
  "success": true,
  "task_id": "019a6c69-...",
  "saved_files": {
    "glb": "3d_models/019a6c69-.../model.glb",
    "fbx": "3d_models/019a6c69-.../model.fbx",
    "usdz": "3d_models/019a6c69-.../model.usdz",
    "obj": "3d_models/019a6c69-.../model.obj",
    "thumbnail": "3d_models/019a6c69-.../thumbnail.png"
  },
  "message": "5개 파일이 서버에 저장되었습니다."
}
```

## 🔧 문제 해결

### API 키 오류
```
MESHY_API_KEY가 설정되지 않았습니다
```
→ `.env` 파일에 API 키를 추가했는지 확인

### 크레딧 부족
```
API 오류: 402
```
→ Dashboard에서 크레딧 충전

### 타임아웃
```
작업이 너무 오래 걸립니다
```
→ 정상입니다. 3D 생성은 2-5분 소요됩니다.

## 📚 참고 자료

- **공식 웹사이트**: https://www.meshy.ai
- **공식 문서**: https://docs.meshy.ai
- **Image to 3D API 문서**: https://docs.meshy.ai/en/api/image-to-3d
- **API 인증**: https://docs.meshy.ai/en/api/authentication
- **가격 정책**: https://docs.meshy.ai/en/api/pricing
- **커뮤니티 Discord**: Discord에서 다른 사용자와 교류

## 💰 가격 정책 (2024년 기준)

- **무료 플랜**: 가입 시 크레딧 제공
- **스타터**: $49/월
- **프로**: $99/월
- **엔터프라이즈**: 별도 문의

자세한 가격은 https://www.meshy.ai/pricing 참고

## 🎯 활용 팁

1. **이미지 품질**: 깔끔하고 명확한 이미지 사용
2. **배경 제거**: 누끼딴 이미지가 더 좋은 결과
3. **조명**: 균일한 조명의 이미지 권장
4. **각도**: 정면/측면 이미지가 가장 효과적

## 🆘 지원

문제가 있으면 Meshy.ai 지원팀에 문의하세요:
- 이메일: support@meshy.ai
- Discord: https://discord.gg/meshy

