# InsightFace + INSwapper 모델 다운로드 가이드

## 다운로드 필요한 모델: 2개

### 1. InsightFace 얼굴 감지 모델 (buffalo_l) - 자동 다운로드 ✅
- **자동 다운로드**: 서버 실행 시 자동으로 다운로드됨
- **다운로드 위치**: `~/.insightface/models/` (Windows: `C:\Users\사용자명\.insightface\models\`)
- **파일명**: `buffalo_l.zip` (압축 해제됨)
- **용도**: 얼굴 감지 및 분석

**→ 별도 다운로드 불필요! 서버 실행하면 자동으로 다운로드됩니다.**

---

### 2. INSwapper 페이스스왑 모델 (inswapper_128.onnx) - 수동 다운로드 필요 ⚠️
- **수동 다운로드 필요**: 자동 다운로드가 실패할 수 있음
- **다운로드 위치**: `~/.insightface/models/` (Windows: `C:\Users\사용자명\.insightface\models\`)
- **파일명**: `inswapper_128.onnx`
- **용도**: 실제 페이스스왑 수행

**→ 수동으로 다운로드해야 합니다!**

---

## 다운로드 방법

### 방법 1: 자동 다운로드 시도 (먼저 시도)

1. 서버 실행:
   ```bash
   python main.py
   ```

2. 서버 로그 확인:
   - `✅ InsightFace + INSwapper 초기화 완료` 메시지가 나오면 성공!
   - `⚠️ INSwapper 모델을 찾을 수 없습니다` 메시지가 나오면 방법 2 사용

### 방법 2: 수동 다운로드 (자동 다운로드 실패 시)

#### Step 1: 다운로드 링크 찾기

**옵션 A: GitHub Releases**
1. https://github.com/deepinsight/insightface/releases 접속
2. 최신 릴리즈에서 `inswapper_128.onnx` 파일 찾기
3. 또는 검색: "inswapper_128.onnx download"

**옵션 B: Hugging Face**
1. https://huggingface.co/models?search=inswapper 접속
2. `inswapper_128.onnx` 파일 다운로드

**옵션 C: 직접 검색**
- Google에서 "inswapper_128.onnx download" 검색

#### Step 2: 저장 위치 확인

**Windows:**
```
C:\Users\사용자명\.insightface\models\inswapper_128.onnx
```

**Linux/Mac:**
```
~/.insightface/models/inswapper_128.onnx
```

#### Step 3: 디렉토리 생성 (없으면)

**Windows PowerShell:**
```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.insightface\models"
```

**Linux/Mac:**
```bash
mkdir -p ~/.insightface/models
```

#### Step 4: 파일 저장

다운로드한 `inswapper_128.onnx` 파일을 위 경로에 저장

---

## 체크리스트

### 자동 다운로드 (서버 실행 시)
- [ ] 서버 실행
- [ ] `buffalo_l` 모델 자동 다운로드 확인
- [ ] `inswapper_128.onnx` 자동 다운로드 확인 (성공하면 끝!)

### 수동 다운로드 (자동 실패 시)
- [ ] `inswapper_128.onnx` 파일 다운로드
- [ ] `~/.insightface/models/` 디렉토리 생성
- [ ] 파일을 올바른 위치에 저장
- [ ] 서버 재시작하여 확인

---

## 확인 방법

서버 실행 후 로그에서 다음 메시지 확인:

✅ **성공:**
```
✅ InsightFace + INSwapper 초기화 완료
```

❌ **실패:**
```
⚠️ INSwapper 모델을 찾을 수 없습니다
⚠️ INSwapper 모델 로드 실패
```

---

## 파일 크기 참고

- `buffalo_l` 모델: 약 200-300MB (압축 해제 후)
- `inswapper_128.onnx`: 약 200-300MB

---

## 요약

1. **buffalo_l 모델**: 자동 다운로드 (서버 실행 시)
2. **inswapper_128.onnx**: 수동 다운로드 필요할 수 있음

**먼저 서버를 실행해서 자동 다운로드가 되는지 확인하고, 안 되면 수동으로 다운로드하세요!**

