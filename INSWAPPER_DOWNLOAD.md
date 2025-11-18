# INSwapper 모델 수동 다운로드 가이드

## 현재 상태
- ✅ InsightFace 모델 (`buffalo_l`) - 자동 다운로드 완료
- ❌ INSwapper 모델 (`inswapper_128.onnx`) - 수동 다운로드 필요

## 다운로드 방법

### Step 1: 모델 파일 다운로드

**옵션 1: Hugging Face (추천)**
1. https://huggingface.co/deepinsight/inswapper 접속
2. `inswapper_128.onnx` 파일 다운로드

**옵션 2: GitHub Releases**
1. https://github.com/deepinsight/insightface/releases 접속
2. 최신 릴리즈에서 `inswapper_128.onnx` 파일 찾기

**옵션 3: 직접 검색**
- Google에서 "inswapper_128.onnx download" 검색

### Step 2: 저장 위치

**Windows 경로:**
```
C:\Users\301\.insightface\models\inswapper_128.onnx
```

**확인 방법:**
- 파일 탐색기에서 위 경로로 이동
- 또는 PowerShell에서:
  ```powershell
  $env:USERPROFILE\.insightface\models\
  ```

### Step 3: 디렉토리 생성 (없으면)

PowerShell에서 실행:
```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.insightface\models"
```

### Step 4: 파일 저장

다운로드한 `inswapper_128.onnx` 파일을 위 경로에 저장

## 파일 크기
- 약 200-300MB

## 확인 방법

서버를 재시작하거나 API를 다시 호출하면:
- `✅ InsightFace + INSwapper 초기화 완료` 메시지가 나와야 합니다
- 또는 에러 없이 페이스스왑이 작동해야 합니다

## 빠른 다운로드 링크 (참고)

- Hugging Face: https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx
- (직접 다운로드 링크가 작동하지 않을 수 있으니 위 사이트에서 다운로드 권장)

