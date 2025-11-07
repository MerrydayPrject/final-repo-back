# 의상 합성 모델 추가 가이드 (JY.md)

## 개요
의상 합성 기능에 7개의 새로운 AI 모델을 추가하여 고급 이미지 처리 파이프라인을 구축합니다.

## 모델 목록 및 설치 가이드

### 1. SegFormer B0 - 세그멘테이션
- **용도**: 배경 제거 및 옷 영역 인식
- **모델**: `matei-dorian/segformer-b0-finetuned-human-parsing`
- **라이브러리**: transformers (이미 설치됨)
- **API 엔드포인트**: `/api/segment-b0`
- **설치 방법**:
  ```bash
  # transformers는 이미 requirements.txt에 포함됨
  pip install transformers==4.35.2
  ```
- **사용 방법**:
  ```python
  from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
  
  processor = SegformerImageProcessor.from_pretrained("matei-dorian/segformer-b0-finetuned-human-parsing")
  model = AutoModelForSemanticSegmentation.from_pretrained("matei-dorian/segformer-b0-finetuned-human-parsing")
  ```

### 2. RTMPose-s - 포즈/관절 키포인트 인식
- **용도**: 인체 포즈 및 관절 위치 추출
- **라이브러리**: mmpose, mmcv, mmengine
- **API 엔드포인트**: `/api/pose-estimation`
- **설치 방법**:
  ```bash
  pip install mmpose>=0.31.0
  pip install mmcv>=2.0.0
  pip install mmengine>=0.10.0
  ```
- **모델 다운로드**:
  - RTMPose-s 체크포인트는 자동으로 다운로드됩니다
  - 또는 수동 다운로드: https://github.com/open-mmlab/mmpose/tree/main/configs/rtmpose
- **사용 방법**:
  ```python
  from mmpose.apis import init_model, inference_top_down_pose_model
  
  config_file = 'configs/rtmpose/rtmpose-s_8xb256-420e_coco-256x192.py'
  checkpoint_file = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth'
  model = init_model(config_file, checkpoint_file, device='cuda:0')
  ```

### 3. HR-VITON - 가상 피팅
- **용도**: 옷 교체/워핑/합성
- **라이브러리**: torch, torchvision, opencv-python (이미 설치됨)
- **API 엔드포인트**: `/api/hr-viton-compose`
- **설치 방법**:
  ```bash
  # HR-VITON 저장소 클론 필요
  git clone https://github.com/sangyun884/HR-VITON.git
  cd HR-VITON
  pip install -r requirements.txt
  ```
- **모델 다운로드**:
  - HR-VITON 모델 가중치를 `models/` 디렉토리에 저장
  - 체크포인트 다운로드: HR-VITON GitHub 저장소 참조
- **주의사항**: 
  - HR-VITON은 별도 저장소이므로 로컬 경로로 모듈 import 필요
  - 또는 HR-VITON의 핵심 로직만 재구현

### 4. SDXL-LoRA / Gemini 2.5 Image - 구두 생성
- **용도**: 구두 이미지 생성
- **라이브러리**: diffusers (SDXL-LoRA) 또는 google-genai (이미 설치됨)
- **API 엔드포인트**: `/api/generate-shoes`
- **설치 방법**:
  ```bash
  # SDXL-LoRA 사용 시
  pip install diffusers>=0.21.0
  pip install accelerate>=0.20.0
  
  # 또는 Gemini 2.5 Image 사용 (이미 설치됨)
  # google-genai>=0.2.0
  ```
- **사용 방법**:
  ```python
  # SDXL-LoRA
  from diffusers import StableDiffusionXLPipeline
  
  # Gemini 2.5 Image
  from google import genai
  ```

### 5. TPS Warp - 구두 워핑
- **용도**: 구두 이미지 워핑 및 착용 삽입
- **라이브러리**: scipy, opencv-python
- **API 엔드포인트**: `/api/tps-warp`
- **설치 방법**:
  ```bash
  pip install scipy>=1.11.0
  pip install opencv-python>=4.8.0
  ```
- **사용 방법**:
  ```python
  import cv2
  from scipy.spatial import distance
  # TPS 변환 구현 필요
  ```

### 6. Real-ESRGAN - 해상도 향상
- **용도**: 이미지 해상도 및 질감 향상
- **라이브러리**: basicsr, realesrgan
- **API 엔드포인트**: `/api/upscale`
- **설치 방법**:
  ```bash
  pip install basicsr>=1.4.0
  pip install realesrgan>=0.3.0
  ```
- **모델 다운로드**:
  - Real-ESRGAN 모델 가중치 자동 다운로드
  - 또는 수동: https://github.com/xinntao/Real-ESRGAN/releases
- **사용 방법**:
  ```python
  from realesrgan import RealESRGANer
  
  model = RealESRGANer(scale=4, model_path='weights/RealESRGAN_x4plus.pth')
  ```

### 7. Color Harmonization - 색상 보정
- **용도**: 조명 및 색상 보정
- **라이브러리**: opencv-python, numpy (이미 설치됨)
- **API 엔드포인트**: `/api/color-harmonize`
- **설치 방법**:
  ```bash
  pip install opencv-python>=4.8.0
  # numpy는 이미 설치됨
  ```
- **사용 방법**:
  ```python
  import cv2
  import numpy as np
  # Color Harmonization 알고리즘 구현
  ```

## uv 패키지 관리

### uv 설치
```bash
pip install uv
```

### 가상 환경 생성
```bash
cd final-repo-back
uv venv
```

### 가상 환경 활성화
- Windows:
  ```bash
  .venv\Scripts\activate
  ```
- macOS/Linux:
  ```bash
  source .venv/bin/activate
  ```

### 의존성 설치
```bash
# pyproject.toml 기반 설치
uv pip install -e .

# 또는 requirements.txt 사용
uv pip install -r requirements.txt
```

## 버전 충돌 방지 전략

### 호환성 확인된 버전
- **torch>=2.2.0**: 대부분의 모델과 호환
- **transformers==4.35.2**: SegFormer B0 호환
- **numpy>=1.24.0**: 모든 모델 호환
- **opencv-python>=4.8.0**: 최신 기능 지원

### 주의사항
1. **mmpose와 mmcv**: 특정 버전 조합 필요
   - mmpose 0.31.0+ 와 mmcv 2.0.0+ 호환
2. **diffusers**: torch 2.2.0+ 권장
3. **Real-ESRGAN**: basicsr 1.4.0+ 필요
4. **HR-VITON**: 별도 저장소이므로 로컬 설치 필요

### 의존성 충돌 해결
```bash
# 특정 버전 고정
pip install package==version

# 또는 uv 사용
uv pip install package==version
```

## 모델 로딩 전략

### Lazy Loading
메모리 사용량 고려하여 모델을 필요할 때만 로드:

```python
# 전역 변수로 모델 저장
models_cache = {}

def load_model(model_name):
    if model_name not in models_cache:
        # 모델 로드
        models_cache[model_name] = load_model_impl(model_name)
    return models_cache[model_name]
```

### GPU/CPU 자동 선택
```python
import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
```

## 테스트 방법

### 1. 개별 모델 테스트
```bash
# API 엔드포인트 직접 테스트
curl -X POST "http://localhost:8000/api/segment-b0" \
  -F "file=@test_image.jpg"
```

### 2. model-comparison.html에서 테스트
1. 서버 실행: `uvicorn main:app --reload`
2. 브라우저에서 `http://localhost:8000/model-comparison` 접속
3. 각 모델 버튼 클릭하여 테스트

### 3. 통합 파이프라인 테스트
모든 모델을 순차적으로 실행하여 최종 결과 확인

## 기존 파이프라인 (구정 전)

### 의상합성 고품화 파이프라인 (`/api/compose-enhanced`)

7개 모델이 순차적으로 실행되어 고품질 의상 합성 이미지를 생성합니다.

```
┌─────────────────────────────────────────────────────────────┐
│ 입력: 인물 이미지 + 의상 이미지                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 1: SegFormer B0 - 세그멘테이션                         │
│ - 인물 이미지: 얼굴, 상체, 하체 마스크 생성                  │
│ - 의상 이미지: 상체/하체 의상 영역 추출                      │
│ 출력: person_mask, person_face_mask, person_upper_mask,      │
│       dress_lower_mask                                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: RTMPose-s - 포즈/키포인트 인식                      │
│ - 17개 키포인트 추출 (COCO 포맷)                            │
│ - 허리 위치 계산 (골반 키포인트 11, 12 평균)                │
│ - 발 위치 추출 (발목 키포인트 15, 16)                       │
│ 출력: waist_y, left_foot_pos, right_foot_pos                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: HR-VITON - 정교한 가상 피팅                         │
│ - 마스크 기반 의상 영역 추출                                 │
│ - 키포인트 기반 상체/하체 구분                               │
│ - 상체/얼굴 보존, 하체만 의상으로 교체                      │
│ - Gaussian Blur로 경계 부드럽게 블렌딩                      │
│ 출력: 합성된 이미지 (current_image)                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4 & 5: 구두 생성 및 합성 (선택적)                      │
│ - Gemini 2.5 Image API로 구두 이미지 생성                     │
│ - RTMPose-s 키포인트 기반 발 위치에 구두 합성                │
│ - 마스크 기반 배경 제거 및 자연스러운 합성                   │
│ - 키포인트 없으면 하단 중앙에 배치                           │
│ 출력: 구두가 합성된 이미지                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 6: Real-ESRGAN - 해상도 향상                          │
│ - Real-ESRGAN 모델로 2배 업스케일                            │
│ - 실패 시 Lanczos 리사이즈로 대체                            │
│ 출력: 해상도 향상된 이미지 (2배 크기)                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 7: Color Harmonization - 색상 보정                     │
│ - LAB 색공간으로 변환                                        │
│ - CLAHE (Contrast Limited Adaptive Histogram Equalization)  │
│   적용으로 명도 보정                                         │
│ - 자연스러운 색상 조정                                       │
│ 출력: 최종 고품화 이미지                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 최종 결과: 고품질 의상 합성 이미지                           │
│ - base64 인코딩하여 JSON 응답                               │
│ - S3에 로그 저장                                             │
│ - pipeline_steps로 각 단계 상태 반환                        │
└─────────────────────────────────────────────────────────────┘
```

### 각 단계별 상세 설명

#### Step 1: SegFormer B0
- **입력**: 인물 이미지, 의상 이미지
- **처리**:
  - 인물 이미지에서 얼굴(레이블 11), 상체(레이블 4), 머리(레이블 2) 영역 추출
  - 의상 이미지에서 하체 의상(레이블 5: Skirt, 7: Dress) 추출
  - 배경 및 불필요한 영역 제거
- **출력**: 각 영역별 마스크 (numpy array, uint8)

#### Step 2: RTMPose-s
- **입력**: 인물 이미지
- **처리**:
  - COCO 포맷 17개 키포인트 감지
  - 골반 키포인트(11, 12)로 허리 Y 좌표 계산
  - 발목 키포인트(15, 16)로 발 위치 추출
- **출력**: `waist_y` (int), `left_foot_pos`, `right_foot_pos` (tuple)

#### Step 3: HR-VITON
- **입력**: 인물 이미지, 의상 이미지, 마스크, 키포인트
- **처리**:
  1. 의상 이미지를 인물 이미지 크기로 리사이즈
  2. 마스크 리사이즈 및 3채널 확장
  3. 키포인트 기반 상체/하체 구분:
     - `waist_y`가 있으면: 허리 위치 + 20px부터 하체로 간주
     - `waist_y`가 없으면: 이미지 하단 60%를 하체로 간주
  4. 상체/얼굴 보존 마스크 생성
  5. 하체 마스크와 의상 마스크 결합
  6. 의상 영역만 추출하여 합성
  7. Gaussian Blur로 경계 부드럽게 블렌딩
  8. 보존 영역은 원본 유지
- **Fallback**: 마스크가 없으면 하단 50% 직접 교체
- **출력**: 합성된 이미지 (PIL Image)

#### Step 4 & 5: 구두 생성 및 합성 (선택적)
- **조건**: `generate_shoes=true` 및 `shoes_prompt` 제공 시 실행
- **입력**: 합성된 이미지, 발 위치 키포인트
- **처리**:
  1. Gemini 2.5 Image API로 구두 이미지 생성
  2. 발 위치 키포인트가 있으면:
     - 발 위치에 맞춰 구두 크기 조정 (이미지 크기의 1/8)
     - 발목 위에 배치
     - 마스크 기반 배경 제거 및 합성
  3. 키포인트가 없으면:
     - 하단 중앙에 배치
     - 마스크 기반 합성
- **출력**: 구두가 합성된 이미지

#### Step 6: Real-ESRGAN
- **입력**: 합성된 이미지
- **처리**:
  - Real-ESRGAN 모델로 2배 업스케일 (메모리 고려)
  - 모델이 없으면 Lanczos 리사이즈로 대체
- **출력**: 해상도 향상된 이미지 (2배 크기)

#### Step 7: Color Harmonization
- **입력**: 해상도 향상된 이미지
- **처리**:
  1. RGB → BGR → LAB 색공간 변환
  2. L 채널(명도)에 CLAHE 적용
  3. LAB → BGR → RGB 변환
- **출력**: 최종 보정된 이미지

### 에러 처리 및 Fallback

각 단계는 독립적으로 실행되며, 실패 시 다음 단계로 진행:

- **SegFormer 실패**: 마스크 없이 기본 합성 진행
- **RTMPose 실패**: 키포인트 없이 이미지 하단 60% 교체
- **HR-VITON 실패**: 원본 이미지 사용
- **구두 생성 실패**: 구두 없이 진행
- **Real-ESRGAN 실패**: Lanczos 리사이즈로 대체
- **Color Harmonization 실패**: 보정 없이 진행

### API 응답 형식

```json
{
  "success": true,
  "person_image": "data:image/png;base64,...",
  "dress_image": "data:image/png;base64,...",
  "result_image": "data:image/png;base64,...",
  "pipeline_steps": [
    {"step": "SegFormer B0", "status": "success", "message": "..."},
    {"step": "RTMPose-s", "status": "success", "message": "..."},
    ...
  ],
  "run_time": 12.34,
  "message": "의상합성 고품화 파이프라인 완료 (5/7 단계 성공)"
}
```

## 새로운 파이프라인 (구정 후)

### 의상합성 개선 파이프라인

7단계 파이프라인이 순차적으로 실행되어 고품질 의상 합성 이미지를 생성합니다.

```
┌─────────────────────────────────────────────────────────────┐
│ 입력: 인물 이미지 + 드레스 이미지                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 1: RMBG - 인물 배경 제거                               │
│ - BRIA RMBG-1.4 모델 사용                                    │
│ - 512×768 정규화 후 입력                                     │
│ - 배경 마스크 추출 → OpenCV bitwise AND 적용                 │
│ 출력: person_rgba.png (배경이 투명한 RGBA 이미지)            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Dress Preprocessing - 드레스 배경 제거 + 정렬       │
│ - 배경 제거 (RMBG 또는 remove.bg API)                       │
│ - 세로 크기 기준 리사이즈 (높이 768px 맞춤)                  │
│ - 중심 정렬 (드레스 중심선 = 사람 중심선)                    │
│ - 목선 또는 어깨선 기준으로 위쪽 여백 맞추기                  │
│ 출력: dress_ready.png (배경 제거된 정렬 드레스 이미지)       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: SegFormer B0 - 상체 마스크 생성                     │
│ - human parsing 수행 → 각 영역 id 확인                       │
│ - 'upper body, arm, neck, head' 클래스만 1로 mask 생성       │
│ - 하단은 0 (나중에 드레스로 덮을 부분)                       │
│ Fallback: 세로 45% 이하를 0으로 자르고 위 부분만 1로 처리   │
│ 출력: mask_upper.png (상체 마스크)                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: HR-VITON - 드레스 중심 워핑 및 합성                 │
│ - 두 이미지를 중앙 정렬                                       │
│ - mask_upper 기준으로 상체는 보존, 하체 영역에 드레스 오버레이│
│ - HR-VITON 모델의 "cloth-warp" 모듈 실행                     │
│   → TPS 변형 아닌 기본 grid-warp 모드로 드레스 하단 fit     │
│ - 합성 후 α-블렌딩 (0.7~0.9 비율)                           │
│ Fallback: 모델 출력 없으면 단순 addWeighted 블렌딩           │
│ 출력: viton_result.png (합성 이미지)                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 5: RTMPose - 포즈 보정 (Optional)                      │
│ - 인물이 기본 정면이 아닌 경우(팔을 벌리거나 측면)에 사용     │
│ - keypoints 추출 (허리, 어깨, 무릎 좌표 등)                  │
│ - HR-VITON warp 시 anchor point 보정 용도                    │
│ - 특히 허리 Y 좌표 기준으로 dress 위치 미세 이동             │
│ 생략 가능: 정면 서있는 사진이면 RTMPose pass 해도 무방       │
│ 출력: keypoints (허리, 어깨, 무릎 좌표 등)                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 6: Real-ESRGAN - 질감/해상도 업스케일                  │
│ - realesrgan-x4plus 또는 x4plus-anime 모델 사용              │
│ - 512×768 → 1024×1536 으로 업스케일                         │
│ - 저장 후 픽셀 샤프닝 적용                                    │
│ Fallback: GPU 메모리 부족 → OpenCV resize 대체              │
│ 출력: upscaled.png (업스케일 이미지)                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 7: Color Harmonization - 색상/조명 보정                │
│ - HSV 또는 LAB 색공간으로 변환                               │
│ - 인물 영역 평균 밝기 → 드레스 영역 밝기로 보정              │
│ - 채도·명도 조정 후 inverse transform                        │
│ - blend ratio = 0.3(인물) + 0.7(드레스)                     │
│ Fallback: color match 실패 → 감마 보정 (1.1× + beta 5)      │
│ 출력: final_result.png (최종 결과)                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 최종 결과: 고품질 의상 합성 이미지                           │
│ - base64 인코딩하여 JSON 응답                               │
│ - pipeline_steps로 각 단계 상태 반환                        │
└─────────────────────────────────────────────────────────────┘
```

### 각 단계별 상세 설명

#### Step 1: RMBG — 인물 배경 제거

**목적**: 드레스 위에 얹힐 "깨끗한 인물 실루엣" 만들기

**해야 할 일**:
- **입력**: person image (JPG 또는 PNG)
- **출력**: 배경이 투명한 RGBA 이미지 (person_rgba.png)
- **모델**: BRIA RMBG-1.4 (Hugging Face 또는 onnx export 버전)

**작동 절차**:
1. 512×768 정규화 후 입력
2. 배경 마스크 추출 → OpenCV bitwise AND 적용

**검증 포인트**:
- RGBA 채널이 제대로 생성되었는지 확인
- 배경이 완전히 제거되었는지 확인

#### Step 2: Dress Preprocessing — 드레스 배경 제거 + 정렬

**목적**: 드레스 이미지를 인체 스케일에 맞게 보정해 "착용 가능 형태"로 만들기

**해야 할 일**:
- **입력**: 드레스 이미지(마네킹 또는 제품 사진)
- **출력**: 배경 제거된 정렬 드레스 이미지 (dress_ready.png)

**처리 단계**:
1. 배경 제거 (RMBG 또는 remove.bg API 사용 가능)
2. 세로 크기 기준 리사이즈 (예: 높이 768 px 맞춤)
3. 중심 정렬 (드레스 중심선 = 사람 중심선)
4. 목선 또는 어깨선 기준으로 위쪽 여백 맞추기

**검증 포인트**:
- 너무 크거나 작지 않나 → 사람 전체 높이의 80~90% 범위 권장
- 상단 목 좌표 ≈ 사람 어깨선 Y 좌표

#### Step 3: SegFormer B0 — 상체 마스크 생성

**목적**: "인체 상체 부분만 보이게 하고 하체는 드레스로 덮기" 위한 마스크 제작

**해야 할 일**:
- **입력**: person_rgba.png
- **출력**: 상체 마스크 (mask_upper.png)

**처리 단계**:
1. SegFormer B0 모델로 human parsing 수행 → 각 영역 id 확인
2. 'upper body, arm, neck, head' 클래스만 1로 mask 만듦
3. 하단은 0 (나중에 드레스로 덮을 부분)

**Fallback**:
- SegFormer 결과 없으면: 세로 45% 이하를 0으로 자르고 위 부분만 1로 처리

**검증 포인트**:
- mask 픽셀 합계 > 전체의 0.2 이상인지 확인 (너무 작으면 비정상)

#### Step 4: HR-VITON — 드레스 중심 워핑 및 합성

**목적**: 드레스 형상을 인체 위치에 맞춰 자연스럽게 워핑 + 합성

**해야 할 일**:
- **입력**: person_rgba, dress_ready, mask_upper
- **출력**: 합성 이미지 (viton_result.png)

**처리 단계**:
1. 두 이미지를 중앙 정렬
2. mask_upper 기준으로 상체는 보존, 하체 영역에 드레스 오버레이
3. HR-VITON 모델의 "cloth-warp" 모듈을 실행
   → TPS 변형 아닌 기본 grid-warp 모드로 드레스 하단 fit
4. 합성 후 α-블렌딩 (0.7~0.9 비율)

**Fallback**:
- 모델 출력 없으면 단순 addWeighted 블렌딩

**검증 포인트**:
- 허리선 위치 맞는지 (대략 높이의 0.45~0.5)
- 드레스 경계가 인체 실루엣과 겹치지 않는지

#### Step 5: RTMPose — 자세 보정 (Optional)

**언제 필요한가**: 인물이 기본 정면이 아닌 경우(팔을 벌리거나 측면)

**해야 할 일**:
- **입력**: person_rgba
- **출력**: keypoints (허리, 어깨, 무릎 좌표 등)

**활용**:
- HR-VITON warp 시 anchor point 보정 용도
- 특히 허리 Y 좌표 기준으로 dress 위치 미세 이동

**생략 가능**: 정면 서있는 사진이면 RTMPose pass 해도 무방

#### Step 6: Real-ESRGAN — 질감 및 해상도 업스케일

**목적**: 드레스 질감 (레이스·자수 등) 복원, 모서리 선명화

**해야 할 일**:
- **입력**: viton_result.png
- **출력**: 업스케일 이미지 (upscaled.png)
- **모델**: realesrgan-x4plus 또는 x4plus-anime (피부톤 자연형)

**절차**:
1. 512×768 → 1024×1536 으로 업스케일
2. 저장 후 픽셀 샤프닝 적용

**Fallback**:
- GPU 메모리 부족 → OpenCV resize 대체

**디버그**: 전후 크기 비교, 처리 시간 로그

#### Step 7: Color Harmonization — 색상/조명 보정

**목적**: 인물과 드레스의 톤을 자연스럽게 맞추기

**해야 할 일**:
- **입력**: upscaled.png
- **출력**: 최종 결과 (final_result.png)

**방법**:
1. HSV 또는 LAB 색공간으로 변환
2. 인물 영역 평균 밝기 → 드레스 영역 밝기로 보정
3. 채도·명도 조정 후 inverse transform
4. blend ratio = 0.3(인물) + 0.7(드레스)

**Fallback**:
- color match 실패 → 감마 보정 (1.1× + beta 5)

**검증 포인트**:
- 피부색 붉게 변하지 않는지
- 드레스 하단 명암 자연스러운지

### 결과 검증 체크리스트

| 검증 항목 | 기준 |
|----------|------|
| 이미지 크기 통일 | 모든 단계 512×768 (업스케일 전까지) |
| 투명도 정보 | RGBA 유지 |
| 허리선 정렬 | 인물 중앙 Y≈드레스 상단 Y |
| 색상 조화 | 인물과 드레스 조명 차 < 20 (LAB ΔE) |

### 에러 처리 및 Fallback

각 단계는 독립적으로 실행되며, 실패 시 다음 단계로 진행:

- **RMBG 실패**: 원본 이미지 사용 (배경 제거 없이 진행)
- **Dress Preprocessing 실패**: 원본 드레스 이미지 사용
- **SegFormer 실패**: 세로 45% 이하를 0으로 자르고 위 부분만 1로 처리
- **HR-VITON 실패**: 단순 addWeighted 블렌딩으로 대체
- **RTMPose 실패**: 키포인트 없이 진행 (생략 가능 단계)
- **Real-ESRGAN 실패**: OpenCV resize로 대체
- **Color Harmonization 실패**: 감마 보정 (1.1× + beta 5)으로 대체

### API 응답 형식

```json
{
  "success": true,
  "person_image": "data:image/png;base64,...",
  "dress_image": "data:image/png;base64,...",
  "result_image": "data:image/png;base64,...",
  "pipeline_steps": [
    {"step": "RMBG", "status": "success", "message": "인물 배경 제거 완료"},
    {"step": "Dress Preprocessing", "status": "success", "message": "드레스 정렬 완료"},
    {"step": "SegFormer B0", "status": "success", "message": "상체 마스크 생성 완료"},
    {"step": "HR-VITON", "status": "success", "message": "드레스 워핑 및 합성 완료"},
    {"step": "RTMPose", "status": "skipped", "message": "정면 사진이므로 생략"},
    {"step": "Real-ESRGAN", "status": "success", "message": "업스케일 완료"},
    {"step": "Color Harmonization", "status": "success", "message": "색상 보정 완료"}
  ],
  "run_time": 15.67,
  "message": "의상합성 개선 파이프라인 완료 (6/7 단계 성공)"
}
```

## 문제 해결

### 모델 로딩 실패
- 체크포인트 파일 경로 확인
- GPU 메모리 부족 시 CPU 사용
- 모델 다운로드 재시도

### 의존성 충돌
- 가상 환경 사용 권장
- uv 패키지로 버전 고정
- requirements.txt 또는 pyproject.toml 확인

### 메모리 부족
- 모델 lazy loading 구현
- 배치 크기 감소
- GPU 메모리 정리 (`torch.cuda.empty_cache()`)

## 참고 자료

- SegFormer B0: https://huggingface.co/matei-dorian/segformer-b0-finetuned-human-parsing
- RTMPose-s: https://github.com/open-mmlab/mmpose
- HR-VITON: https://github.com/sangyun884/HR-VITON
- Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN
- SDXL-LoRA: https://huggingface.co/docs/diffusers

## 업데이트 기록

- 2024-01-XX: 초기 모델 추가 계획 작성
- 7개 모델 API 엔드포인트 구현 완료

