# 체형 분석 모델 정확한 이름

## 현재 프로젝트에서 사용 중인 모델

### 1. 포즈 랜드마크 분석 모델

#### 모델명 (정확한 이름)
**MediaPipe Pose Landmarker Lite**

#### 상세 정보
- **공식 모델명**: `pose_landmarker_lite`
- **모델 파일명**: `pose_landmarker_lite.task`
- **버전**: `float16` (반정밀도)
- **제조사**: Google MediaPipe
- **모델 타입**: Pose Landmarker (Lite 버전)

#### 다운로드 URL
```
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

#### 저장 경로
```
final-repo-back/body_analysis_test/models/pose_landmarker_lite.task
```

#### Python 패키지
- **패키지명**: `mediapipe`
- **최소 버전**: `mediapipe>=0.10.0`
- **클래스**: `mediapipe.tasks.python.vision.PoseLandmarker`

#### 기능
- 33개 포즈 랜드마크 추출
- 얼굴, 어깨, 팔, 손, 허리, 다리 등 신체 부위 감지
- 실시간 처리 가능

---

### 2. AI 상세 분석 모델

#### 모델명 (정확한 이름)
**Google Gemini 2.5 Flash (Image)**

#### 상세 정보
- **공식 모델명**: `gemini-2.5-flash-image`
- **제조사**: Google
- **모델 타입**: 멀티모달 LLM (이미지 + 텍스트)
- **용도**: 체형 상세 분석 및 드레스 스타일 추천

#### API 사용
- **Python 패키지**: `google-genai` 또는 `google-generativeai`
- **최소 버전**: `google-genai>=0.2.0`
- **API 엔드포인트**: Google Gemini API

#### 기능
- 이미지와 텍스트를 동시에 이해
- 체형 특징 분석
- 드레스 스타일 추천
- 자연어 설명 생성

---

## 모델 사용 흐름

```
1. MediaPipe Pose Landmarker Lite
   ↓
   포즈 랜드마크 추출 (33개 포인트)
   ↓
   체형 측정값 계산 (어깨/엉덩이 비율 등)
   ↓
   체형 타입 분류 (A라인, H라인, X라인 등)
   ↓
2. Google Gemini 2.5 Flash Image
   ↓
   이미지 + 측정값 + 체형 타입 입력
   ↓
   상세 분석 및 스타일 추천 생성
```

---

## 모델 비교

### MediaPipe Pose Landmarker 버전별

| 버전 | 모델명 | 정확도 | 속도 | 용도 |
|------|--------|--------|------|------|
| **Lite** | `pose_landmarker_lite` | 중간 | 빠름 | **현재 사용 중** |
| Full | `pose_landmarker_full` | 높음 | 중간 | 더 정확한 분석 필요 시 |
| Heavy | `pose_landmarker_heavy` | 매우 높음 | 느림 | 최고 정확도 필요 시 |

### Gemini 모델 버전별

| 모델명 | 용도 | 특징 |
|--------|------|------|
| `gemini-2.5-flash-image` | **현재 사용 중** | 빠르고 이미지 이해 우수 |
| `gemini-pro-vision` | 대안 | 더 정확하지만 느림 |
| `gemini-1.5-pro` | 대안 | 최고 성능, 비용 높음 |

---

## 모델 다운로드 정보

### MediaPipe Pose Landmarker Lite
- **파일 크기**: 약 2-3MB
- **다운로드 방식**: 자동 (첫 실행 시)
- **수동 다운로드**: 위 URL에서 직접 다운로드 가능

### Google Gemini
- **다운로드 불필요**: API로 사용
- **API 키 필요**: GEMINI_API_KEY 환경 변수 설정
- **비용**: 사용량 기반 (무료 티어 제공)

---

## 참고 링크

### MediaPipe
- **공식 문서**: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
- **모델 허브**: https://storage.googleapis.com/mediapipe-models/pose_landmarker/
- **GitHub**: https://github.com/google/mediapipe

### Google Gemini
- **공식 문서**: https://ai.google.dev/docs
- **모델 목록**: https://ai.google.dev/models/gemini
- **API 문서**: https://ai.google.dev/api

---

## 요약

### 현재 사용 중인 모델 (정확한 이름)

1. **포즈 랜드마크 분석**
   - 모델명: `MediaPipe Pose Landmarker Lite`
   - 파일명: `pose_landmarker_lite.task`
   - 버전: `float16`

2. **AI 상세 분석**
   - 모델명: `Google Gemini 2.5 Flash Image`
   - API 모델명: `gemini-2.5-flash-image`




