# INSwapper 모델 다운로드 대안 가이드

## 현재 상황
- Hugging Face 저장소 비활성화
- GitHub 공식 링크 404 에러
- 자동 다운로드 실패

## 대안 다운로드 방법

### 방법 1: haofanwang 저장소 (추천)

1. **저장소 접속**: https://github.com/haofanwang/inswapper
2. **checkpoints 폴더**에서 `inswapper_128.onnx` 파일 찾기
3. 파일 다운로드
4. 저장 위치: `C:\Users\301\.insightface\models\inswapper_128.onnx`

### 방법 2: 자동 다운로드 스크립트 실행

프로젝트에 `download_inswapper.py` 스크립트를 추가했습니다:

```bash
python download_inswapper.py
```

이 스크립트는 여러 소스에서 자동으로 다운로드를 시도합니다.

### 방법 3: 다른 소스 검색

Google에서 다음 키워드로 검색:
- "inswapper_128.onnx download"
- "inswapper model download alternative"
- "face swap onnx model download"

### 방법 4: InsightFace 공식 웹사이트

- https://www.insightface.ai/ 접속
- 상업적 사용을 위한 라이선스 구매 시 모델 제공 가능

## 저장 위치

**Windows:**
```
C:\Users\301\.insightface\models\inswapper_128.onnx
```

**확인 방법:**
```powershell
Test-Path "$env:USERPROFILE\.insightface\models\inswapper_128.onnx"
```

## 파일 크기
- 약 200-300MB

## 참고사항

- 모델 파일을 다운로드한 후 서버를 재시작하거나 API를 다시 호출하면 자동으로 인식됩니다
- 파일이 올바른 위치에 있으면 코드가 자동으로 로드합니다

