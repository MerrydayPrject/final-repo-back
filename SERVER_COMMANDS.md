# 서버별 실행 명령어 가이드

이 문서는 프로젝트의 각 서버를 실행하는 방법을 설명합니다.

## 📋 서버 목록

| 서버 | 포트 | 설명 | 파일 경로 |
|------|------|------|-----------|
| 메인 백엔드 서버 | 8000 | 의류 세그멘테이션, 드레스 관리, Gemini API | `main.py` |
| 이미지 보정 서버 | 8003 | 배경 분위기 변경, 이미지 보정 (InstructPix2Pix + GFPGAN) | `image_enhancement_server/enhancement_server.py` |
| 체형 분석 서버 | 8002 | MediaPipe 기반 체형 분석 테스트 | `body_analysis_test/test_body_analysis.py` |

---

## 🚀 서버 실행 방법

### PowerShell 명령어 (빠른 참조)

```powershell
# 1. 메인 백엔드 서버 (포트 8000)
cd c:\proj\final-repo-back; uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 2. 이미지 보정 서버 (포트 8003)
cd c:\proj\final-repo-back\image_enhancement_server; python enhancement_server.py 8003

# 3. 체형 분석 서버 (포트 8002)
cd c:\proj\final-repo-back\body_analysis_test; python test_body_analysis.py 8002
```

---

### 1. 메인 백엔드 서버 (포트 8000)

#### PowerShell
```powershell
cd c:\proj\final-repo-back
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### Windows (배치 파일 사용)
```batch
start-backend.bat
```

#### Windows (직접 실행)
```batch
cd c:\proj\final-repo-back
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### Linux/Mac
```bash
cd /path/to/final-repo-back
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### 접속 주소
- 서버: http://localhost:8000
- API 문서: http://localhost:8000/docs
- 관리자 페이지: http://localhost:8000/admin

---

### 2. 이미지 보정 서버 (포트 8003)

#### PowerShell
```powershell
cd c:\proj\final-repo-back\image_enhancement_server
python enhancement_server.py 8003
```

#### Windows (배치 파일 사용)
```batch
cd c:\proj\final-repo-back\image_enhancement_server
start_enhancement_server.bat
```

#### Windows (직접 실행)
```batch
cd c:\proj\final-repo-back\image_enhancement_server
python enhancement_server.py 8003
```

#### Linux/Mac
```bash
cd /path/to/final-repo-back/image_enhancement_server
python enhancement_server.py 8003
```

#### 포트 변경 (예: 8004)
```batch
python enhancement_server.py 8004
```

#### 접속 주소
- 서버: http://localhost:8003
- API 문서: http://localhost:8003/docs
- 테스트 페이지: http://localhost:8003/

---

### 3. 체형 분석 서버 (포트 8002)

#### PowerShell
```powershell
cd c:\proj\final-repo-back\body_analysis_test
python test_body_analysis.py 8002
```

#### Windows (배치 파일 사용)
```batch
cd c:\proj\final-repo-back\body_analysis_test
start_test_server_8002.bat
```

#### Windows (직접 실행)
```batch
cd c:\proj\final-repo-back\body_analysis_test
python test_body_analysis.py 8002
```

#### Linux/Mac
```bash
cd /path/to/final-repo-back/body_analysis_test
python test_body_analysis.py 8002
```

#### 포트 변경 (예: 8001)
```batch
python test_body_analysis.py 8001
```

#### 접속 주소
- 서버: http://localhost:8002
- API 문서: http://localhost:8002/docs
- 테스트 페이지: http://localhost:8002/

---

## 🔧 모든 서버 동시 실행

### PowerShell (별도 창에서 실행)

각 서버를 별도의 PowerShell 창에서 실행:

```powershell
# 창 1: 메인 백엔드 서버
cd c:\proj\final-repo-back
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 창 2: 이미지 보정 서버
cd c:\proj\final-repo-back\image_enhancement_server
python enhancement_server.py 8003

# 창 3: 체형 분석 서버
cd c:\proj\final-repo-back\body_analysis_test
python test_body_analysis.py 8002
```

### PowerShell (백그라운드 작업으로 실행)

```powershell
# 메인 백엔드 서버 (백그라운드)
cd c:\proj\final-repo-back
Start-Job -ScriptBlock { cd c:\proj\final-repo-back; uvicorn main:app --host 0.0.0.0 --port 8000 --reload }

# 이미지 보정 서버 (백그라운드)
Start-Job -ScriptBlock { cd c:\proj\final-repo-back\image_enhancement_server; python enhancement_server.py 8003 }

# 체형 분석 서버 (백그라운드)
Start-Job -ScriptBlock { cd c:\proj\final-repo-back\body_analysis_test; python test_body_analysis.py 8002 }

# 작업 상태 확인
Get-Job

# 작업 중지
Get-Job | Stop-Job
```

### Windows (배치 파일로 한 번에 실행)
`start_all_servers.bat` 파일 생성:

```batch
@echo off
echo 모든 서버 시작 중...
start "메인 백엔드 서버" cmd /k "cd /d %~dp0 && uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
timeout /t 2 /nobreak >nul
start "이미지 보정 서버" cmd /k "cd /d %~dp0\image_enhancement_server && python enhancement_server.py 8003"
timeout /t 2 /nobreak >nul
start "체형 분석 서버" cmd /k "cd /d %~dp0\body_analysis_test && python test_body_analysis.py 8002"
echo 모든 서버가 시작되었습니다.
pause
```

### Linux/Mac (터미널 탭)
```bash
# 탭 1: 메인 백엔드 서버
cd /path/to/final-repo-back
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 탭 2: 이미지 보정 서버
cd /path/to/final-repo-back/image_enhancement_server
python enhancement_server.py 8003

# 탭 3: 체형 분석 서버
cd /path/to/final-repo-back/body_analysis_test
python test_body_analysis.py 8002
```

---

## 📝 사전 요구사항

### 1. Python 환경 설정
```bash
# 가상환경 생성 (권장)
python -m venv venv

# 가상환경 활성화
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 각 서버별 추가 의존성

#### 이미지 보정 서버
```bash
cd image_enhancement_server
pip install -r requirements.txt
```

#### 체형 분석 서버
```bash
cd body_analysis_test
pip install -r requirements_test.txt
```

### 3. 환경 변수 설정
`.env` 파일이 프로젝트 루트에 있어야 합니다:
```
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=your_user
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=your_database
GEMINI_API_KEY=your_gemini_api_key
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
```

### 4. 모델 파일 준비

#### GFPGAN 모델 (이미지 보정 서버)
- 자동 다운로드되거나 수동으로 `image_enhancement_server/models/GFPGANv1.4.pth`에 배치

#### MediaPipe 모델 (체형 분석 서버)
- 자동 다운로드되거나 `body_analysis_test/models/pose_landmarker_lite.task`에 배치

---

## 🧪 테스트 스크립트 실행

### 배경 분위기 변경 + 보정 테스트
```batch
cd c:\proj\final-repo-back
python test_background_remove_enhance.py test.jpg -i "어깨 좁게, 배경 블러, 로맨틱한 분위기, 주름 제거"
```

### 이미지 보정 테스트
```batch
cd c:\proj\final-repo-back
python test_image_enhancement.py test.jpg "어깨 좁게, 주름 제거, 피부톤 밝게"
```

---

## 🔍 서버 상태 확인

### 포트 사용 확인

#### PowerShell
```powershell
# 특정 포트 확인
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
Get-NetTCPConnection -LocalPort 8002 -ErrorAction SilentlyContinue
Get-NetTCPConnection -LocalPort 8003 -ErrorAction SilentlyContinue

# 모든 서버 포트 한 번에 확인
8000, 8002, 8003 | ForEach-Object { Get-NetTCPConnection -LocalPort $_ -ErrorAction SilentlyContinue | Select-Object LocalPort, State, OwningProcess }
```

#### Windows (CMD)
```batch
netstat -ano | findstr :8000
netstat -ano | findstr :8002
netstat -ano | findstr :8003
```

### 서버 로그 확인
각 서버는 콘솔에 실시간 로그를 출력합니다. 오류 발생 시 콘솔 출력을 확인하세요.

---

## ⚠️ 문제 해결

### 포트가 이미 사용 중인 경우
다른 포트로 실행:
```batch
# 예: 8000 대신 8001 사용
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### 모델 로딩 실패
- 모델 파일 경로 확인
- GPU 메모리 부족 시 CPU 모드로 전환 (자동 처리됨)
- 필요한 모델 파일 다운로드 확인

### 의존성 오류
```bash
pip install --upgrade -r requirements.txt
```

---

## 📚 추가 정보

- API 문서는 각 서버의 `/docs` 엔드포인트에서 확인 가능
- 테스트 페이지는 각 서버의 루트 경로(`/`)에서 확인 가능
- CORS 설정은 각 서버의 설정 파일에서 확인 가능

