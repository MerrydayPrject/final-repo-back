# 환경변수 설정 가이드

프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 다음 환경변수들을 설정하세요.

## MySQL 데이터베이스 설정

```env
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=devuser
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=marryday
```

- `MYSQL_HOST`: MySQL 서버 호스트 (기본값: localhost)
- `MYSQL_PORT`: MySQL 서버 포트 (기본값: 3306)
- `MYSQL_USER`: MySQL 사용자명
- `MYSQL_PASSWORD`: MySQL 비밀번호
- `MYSQL_DATABASE`: 사용할 데이터베이스명 (기본값: marryday)

## Gemini API 키 (이미지 합성용)

```env
GEMINI_API_KEY=your_gemini_api_key
```

- Gemini API를 사용한 이미지 합성 기능에 필요합니다.
- Google AI Studio에서 발급받을 수 있습니다.

## AWS S3 설정 (드레스 이미지 업로드용)

```env
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_S3_BUCKET_NAME=your_bucket_name
AWS_REGION=ap-northeast-2
```

- `AWS_ACCESS_KEY_ID`: AWS 액세스 키 ID
- `AWS_SECRET_ACCESS_KEY`: AWS 시크릿 액세스 키
- `AWS_S3_BUCKET_NAME`: S3 버킷 이름
- `AWS_REGION`: AWS 리전 (기본값: ap-northeast-2)

## AWS S3 설정 (모델 테스트 로그 이미지 저장용)

```env
LOGS_AWS_ACCESS_KEY_ID=your_logs_aws_access_key_id
LOGS_AWS_SECRET_ACCESS_KEY=your_logs_aws_secret_access_key
LOGS_AWS_S3_BUCKET_NAME=your_logs_bucket_name
LOGS_AWS_REGION=ap-northeast-2
```

- `LOGS_AWS_ACCESS_KEY_ID`: 로그용 AWS 액세스 키 ID (별도 계정)
- `LOGS_AWS_SECRET_ACCESS_KEY`: 로그용 AWS 시크릿 액세스 키 (별도 계정)
- `LOGS_AWS_S3_BUCKET_NAME`: 로그 이미지 저장용 S3 버킷 이름
- `LOGS_AWS_REGION`: AWS 리전 (기본값: ap-northeast-2)
- 모델 비교 테스트에서 생성된 이미지(person, dress, result)만 저장하는 별도 버킷입니다.

## 전체 예시

`.env` 파일 전체 내용 예시:

```env
# MySQL 데이터베이스 설정
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=devuser
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=marryday

# Gemini API 키 (이미지 합성용)
GEMINI_API_KEY=your_gemini_api_key

# AWS S3 설정 (드레스 이미지 업로드용)
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_S3_BUCKET_NAME=your_bucket_name
AWS_REGION=ap-northeast-2

# AWS S3 설정 (모델 테스트 로그 이미지 저장용 - 별도 계정)
LOGS_AWS_ACCESS_KEY_ID=your_logs_aws_access_key_id
LOGS_AWS_SECRET_ACCESS_KEY=your_logs_aws_secret_access_key
LOGS_AWS_S3_BUCKET_NAME=your_logs_bucket_name
LOGS_AWS_REGION=ap-northeast-2
```

## 주의사항

1. `.env` 파일은 절대 Git에 커밋하지 마세요. (`.gitignore`에 포함되어 있습니다)
2. 실제 값으로 `your_*` 부분을 모두 교체하세요
3. 프로덕션 환경에서는 환경변수를 안전하게 관리하세요
