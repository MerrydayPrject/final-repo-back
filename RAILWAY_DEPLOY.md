# Railway 배포 가이드

## 배포 준비 완료 사항

✅ `Procfile` 생성 완료
- Railway가 자동으로 시작 명령어를 감지합니다
- 포트는 Railway가 제공하는 `$PORT` 환경변수를 사용합니다

✅ `requirements.txt` 루트 경로에 존재
- Railway가 자동으로 Python 프로젝트를 감지하고 의존성을 설치합니다

## Railway 배포 설정

### 1. 프로젝트 생성 및 GitHub 연결

1. Railway 대시보드에서 "New Project" 클릭
2. "Deploy from GitHub repo" 선택
3. 저장소 선택 및 연결
4. **Root Directory**를 `final-repo-back`로 설정 (필요한 경우)

### 2. 환경변수 설정

Railway 대시보드에서 프로젝트 설정 → Variables 탭에서 다음 환경변수들을 추가하세요:

#### 필수 환경변수

##### 데이터베이스 (MySQL)
```
MYSQL_HOST=your_mysql_host
MYSQL_PORT=3306
MYSQL_USER=your_mysql_user
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=marryday
```

##### Gemini API (필수)
```
GEMINI_API_KEY=your_gemini_api_key
```

##### AWS S3 (드레스 이미지 업로드용)
```
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_S3_BUCKET_NAME=your_s3_bucket_name
AWS_REGION=ap-northeast-2
```

#### 선택적 환경변수

##### Gemini 3 API (CustomV4 기능 사용 시)
```
GEMINI_3_API_KEY=your_gemini_3_api_key
```

##### x.ai API (프롬프트 생성용)
```
XAI_API_KEY=your_xai_api_key
XAI_API_BASE_URL=https://api.x.ai/v1
XAI_IMAGE_MODEL=grok-2-image
XAI_PROMPT_MODEL=grok-2-vision-1212
```

##### HuggingFace API (SegFormer 사용 시)
```
HUGGINGFACE_API_KEY=your_huggingface_api_key
HUGGINGFACE_API_BASE_URL=https://api-inference.huggingface.co/models
SEGFORMER_API_TIMEOUT=60
```

##### InsightFace API (얼굴 교체 기능)
```
INSIGHTFACE_ENDPOINT_URL=your_insightface_endpoint_url
INSIGHTFACE_API_KEY=your_insightface_api_key
```

##### MediaPipe Spaces API (포즈 인식)
```
MEDIAPIPE_SPACE_URL=https://jjunyuongv-mediapipe-pose-api.hf.space
```

##### OpenAI API (프롬프트 생성용, 선택적)
```
OPENAI_API_KEY=your_openai_api_key
```

##### 로그 저장용 AWS S3 (선택적)
```
LOGS_AWS_ACCESS_KEY_ID=your_logs_aws_access_key
LOGS_AWS_SECRET_ACCESS_KEY=your_logs_aws_secret_key
LOGS_AWS_S3_BUCKET_NAME=your_logs_s3_bucket_name
LOGS_AWS_REGION=ap-northeast-2
```

##### 모델 설정 (선택적, 기본값 사용 가능)
```
GPT4O_MODEL_NAME=gpt-4o
GPT4O_V2_MODEL_NAME=gpt-4o-2024-08-06
GEMINI_FLASH_MODEL=gemini-2.5-flash-image
GEMINI_3_FLASH_MODEL=gemini-3-pro-image-preview
GEMINI_PROMPT_MODEL=gemini-2.0-flash-exp
```

### 3. 데이터베이스 설정

Railway에서 MySQL 서비스를 추가하거나, 외부 MySQL 데이터베이스를 사용할 수 있습니다.

**Railway MySQL 서비스 사용 시:**
- Railway 대시보드에서 "New" → "Database" → "Add MySQL" 선택
- Railway가 자동으로 환경변수에 연결 정보를 추가합니다
- 생성된 `MYSQLHOST`, `MYSQLDATABASE`, `MYSQLUSER`, `MYSQLPASSWORD`, `MYSQLPORT`를 프로젝트 환경변수에 매핑:
  ```
  MYSQL_HOST=${{MySQL.MYSQLHOST}}
  MYSQL_PORT=${{MySQL.MYSQLPORT}}
  MYSQL_USER=${{MySQL.MYSQLUSER}}
  MYSQL_PASSWORD=${{MySQL.MYSQLPASSWORD}}
  MYSQL_DATABASE=${{MySQL.MYSQLDATABASE}}
  ```

### 4. 배포 확인

배포가 완료되면:
1. Railway 대시보드에서 생성된 URL 확인
2. `https://your-app.railway.app/health` 엔드포인트로 헬스체크
3. `https://your-app.railway.app/docs`로 API 문서 확인

## 주의사항

- `--reload` 플래그는 production 환경에서 사용하지 않습니다 (Procfile에 포함되지 않음)
- 포트는 Railway가 자동으로 할당하므로 `$PORT` 환경변수를 사용해야 합니다
- 데이터베이스 마이그레이션은 서버 시작 시 자동으로 실행됩니다
- 정적 파일과 템플릿 디렉토리는 런타임에 자동 생성됩니다

## 문제 해결

### 서버가 시작되지 않는 경우
- Railway 로그를 확인하여 오류 메시지 확인
- 환경변수가 올바르게 설정되었는지 확인
- 데이터베이스 연결 정보 확인

### 데이터베이스 연결 실패
- MySQL 호스트가 외부 접속을 허용하는지 확인
- 방화벽 설정 확인
- 환경변수 값이 올바른지 확인

