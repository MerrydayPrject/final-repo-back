# 페이스스왑 기능 설정 가이드

## 1단계: 패키지 설치

```bash
pip install -r requirements.txt
```

또는 페이스스왑 관련 패키지만 설치:
```bash
pip install insightface>=0.7.3 onnxruntime>=1.16.0
```

## 2단계: InsightFace 모델 자동 다운로드

서버를 처음 실행하면 InsightFace가 자동으로 모델을 다운로드합니다:
- `buffalo_l` 모델 (얼굴 감지용)
- 다운로드 위치: `~/.insightface/models/` (Windows: `C:\Users\사용자명\.insightface\models\`)

## 3단계: INSwapper 모델 다운로드

INSwapper 모델은 자동 다운로드가 안 될 수 있으므로 수동으로 다운로드해야 할 수 있습니다.

### 방법 1: 자동 다운로드 시도
서버 실행 시 자동으로 다운로드를 시도합니다. 실패하면 방법 2를 사용하세요.

### 방법 2: 수동 다운로드

1. **모델 다운로드 링크**:
   - GitHub: https://github.com/deepinsight/insightface/releases
   - 또는 Hugging Face: https://huggingface.co/models?search=inswapper

2. **다운로드할 파일**: `inswapper_128.onnx`

3. **저장 위치**:
   - Windows: `C:\Users\사용자명\.insightface\models\inswapper_128.onnx`
   - Linux/Mac: `~/.insightface/models/inswapper_128.onnx`

4. **디렉토리 생성** (없으면):
   ```bash
   # Windows PowerShell
   New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.insightface\models"
   
   # Linux/Mac
   mkdir -p ~/.insightface/models
   ```

## 4단계: 템플릿 이미지 확인

템플릿 이미지가 제대로 들어갔는지 확인:
- 경로: `templates/face_swap_templates/`
- 현재 4개 이미지 확인됨 ✅

## 5단계: 서버 실행 및 테스트

### 서버 실행
```bash
cd final-repo-back
python main.py
```

또는 uvicorn으로:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 테스트 방법

1. **웹 브라우저에서 테스트**:
   - http://localhost:8000/body-generation 접속
   - 얼굴 사진 업로드 후 "페이스스왑 실행하기" 클릭

2. **API로 테스트**:
   ```bash
   curl -X POST "http://localhost:8000/api/body-generation" \
     -F "file=@your_face_image.jpg"
   ```

## 6단계: 문제 해결

### InsightFace 초기화 실패
- 패키지가 제대로 설치되었는지 확인: `pip list | grep insightface`
- 모델 다운로드 경로 확인: `~/.insightface/models/`

### INSwapper 모델을 찾을 수 없음
- 모델 파일이 올바른 위치에 있는지 확인
- 파일명이 정확히 `inswapper_128.onnx`인지 확인

### 템플릿 이미지를 찾을 수 없음
- `templates/face_swap_templates/` 디렉토리 확인
- 이미지 파일 확장자 확인 (.jpg, .png 등)

### 얼굴을 찾을 수 없음
- 업로드한 이미지에 얼굴이 명확하게 보이는지 확인
- 정면 또는 약간 측면 얼굴 권장

## 체크리스트

- [ ] `pip install -r requirements.txt` 실행 완료
- [ ] InsightFace 모델 자동 다운로드 확인 (서버 실행 시)
- [ ] INSwapper 모델 다운로드 완료 (`inswapper_128.onnx`)
- [ ] 템플릿 이미지 4개 확인 완료
- [ ] 서버 실행 성공
- [ ] 웹 페이지 접속 확인
- [ ] 페이스스왑 테스트 성공

## 참고사항

- 첫 실행 시 모델 다운로드로 인해 시간이 걸릴 수 있습니다
- CUDA가 있으면 GPU로 실행되며, 없으면 CPU로 실행됩니다
- 3050 6GB 환경에서도 충분히 동작합니다

