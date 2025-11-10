# 이미지 보정 서버 - 모델 및 프롬프트 정리

## 📦 현재 사용 중인 모델

### 1. Stable Diffusion Inpainting
- **용도**: 신체 보정 (어깨, 허리, 엉덩이)
- **모델 ID**: `runwayml/stable-diffusion-inpainting` (우선순위)
- **대체 모델**: `diffusers/stable-diffusion-inpainting`
- **로딩 옵션**: 
  - `variant="fp16"` (GPU 메모리 최적화)
  - `use_safetensors=True` (보안 강화)
- **최적화**: `enable_attention_slicing()` (GPU 메모리 절약)

### 2. GFPGAN
- **용도**: 얼굴 보정 (주름 제거, 피부톤 개선)
- **모델 파일**: `models/GFPGANv1.4.pth`
- **다운로드 URL**: https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
- **설정**:
  - `upscale=1` (크기 유지)
  - `arch='clean'`
  - `channel_multiplier=2`
  - `bg_upsampler=None`

### 3. MediaPipe BodyAnalysisService
- **용도**: 포즈 랜드마크 추출 (마스크 생성용)
- **모델 파일**: `body_analysis_test/models/pose_landmarker_lite.task`
- **출력**: 33개 랜드마크 포인트

---

## 🗑️ 제거된 모델 (사용 안 함)

### 1. ControlNet OpenPose
- **이유**: 사용하지 않아 제거 (로딩 시간 단축)
- **이전 용도**: 포즈 기반 이미지 생성 (현재 미사용)

### 2. OpenPose Detector (controlnet-aux)
- **이유**: 사용하지 않아 제거
- **이전 용도**: 포즈 이미지 생성 (현재 미사용)

---

## 📝 프롬프트 시스템

### 1. `translate_instruction()` 함수
**위치**: `enhancement_server.py` (351-421줄)

**기능**: 한국어 요청을 영어 프롬프트로 변환

**지원 기능**:

#### 신체 보정
- **어깨**:
  - "넓게/크게" → `"make shoulders wider"`
  - "좁게/작게" → `"make shoulders narrower"`
  
- **허리**:
  - "얇게/작게" → `"make waist thinner"`
  - "넓게" → `"make waist wider"`
  
- **엉덩이**:
  - "작게" → `"make hips smaller"`
  - "넓게/크게" → `"make hips larger"`

#### 배경 변경
- "교회" → `"church background"`
- "해변" → `"beach background"`
- "바다" → `"ocean background"`
- "정원" → `"garden background"`
- "공원" → `"park background"`
- "스튜디오" → `"studio background"`
- "카페" → `"cafe background"`
- "호텔" → `"hotel background"`
- "웨딩홀" → `"wedding hall background"`
- "블러/흐릿" → `"blurred background"`
- "흰색/하얀" → `"white background"`

#### 스타일 변경
- "우아" → `"elegant style"`
- "모던" → `"modern style"`
- "캐주얼" → `"casual style"`
- "로맨틱" → `"romantic style"`

#### 분위기
- "밝" → `"bright"`
- "어둡" → `"dark"`

---

### 2. `parse_body_edit_params()` 함수
**위치**: `enhancement_server.py` (326-348줄)

**기능**: 사용자 요청에서 보정 강도 파라미터 추출

**기본값**:
- `strength = 0.7` (변경 강도)
- `steps = 34` (추론 스텝)
- `mask_scale = 1.2` (마스크 크기 배율)
- `iterations = 2` (반복 횟수)

**강한 키워드** ("많이", "확", "대폭", "강하게", "크게"):
- `strength = 0.85` (최대 0.90)
- `steps = 40` (최대 40)
- `mask_scale = 1.5` (최대 1.6)
- `iterations = 3`

**약한 키워드** ("살짝", "조금", "약하게", "미세하게"):
- `strength = 0.55` (최소 0.45)
- `steps = 26` (최소 24)
- `mask_scale = 1.0` (최소 0.9)
- `iterations = 1`

---

### 3. 신체 부위별 프롬프트 생성 로직

#### 어깨 보정 (539-582줄)
```python
if "넓게/넓게/크게":
    additional_prompt = ", increase shoulder width by 25 percent, make shoulders noticeably wider"
else:  # 좁게
    additional_prompt = ", reduce shoulder width by 25 percent, make shoulders noticeably narrower"

최종 프롬프트 = translate_instruction(instruction) + additional_prompt + 
               ", keep face completely unchanged, keep original image style, natural, realistic, high quality, detailed, preserve face"
```

#### 허리 보정 (584-626줄)
```python
if "얇게/얇게/작게":
    additional_prompt = ", emphasize a slimmer waistline, reduce waist circumference by 20 percent"
else:  # 넓게
    additional_prompt = ", increase waist width by 20 percent, make waist noticeably wider"

최종 프롬프트 = translate_instruction(instruction) + additional_prompt + 
               ", keep face completely unchanged, keep original image style, natural, realistic, high quality, detailed, preserve face"
```

#### 엉덩이 보정 (628-670줄)
```python
if "작게":
    additional_prompt = ", reduce hip width by 20 percent, make hips noticeably smaller"
else:  # 크게
    additional_prompt = ", increase hip width by 20 percent, make hips noticeably larger"

최종 프롬프트 = translate_instruction(instruction) + additional_prompt + 
               ", keep face completely unchanged, keep original image style, natural, realistic, high quality, detailed, preserve face"
```

---

## 🔧 Inpainting 파라미터

### 사용되는 파라미터
```python
inpaint_pipe(
    prompt=prompt_inpaint,           # 프롬프트
    image=current_image,              # 원본 이미지
    mask_image=mask,                 # 마스크 (흰색=편집 영역)
    num_inference_steps=body_steps,  # 추론 스텝 (24-40)
    strength=body_strength           # 변경 강도 (0.45-0.90)
)
```

### 파라미터 범위
- **strength**: 0.45 ~ 0.90 (기본 0.7)
- **num_inference_steps**: 24 ~ 40 (기본 34)
- **iterations**: 1 ~ 3 (기본 2)

---

## 🎯 얼굴 보호 메커니즘

### 1. 얼굴 마스크 생성
- **함수**: `create_face_protection_mask()` (156-203줄)
- **영역**: 코 기준으로 얼굴 폭 35%, 높이 40% (타원형)
- **용도**: 얼굴 영역을 마스크에서 제외하여 편집 방지

### 2. 얼굴 원본 복원
- **방법**: Inpainting 전 원본 얼굴 픽셀 추출 → Inpainting 후 복원
- **구현**: `face_mask_bool` (얼굴 영역 True)로 픽셀 단위 복원
- **위치**: 각 신체 보정 후 (563-579줄, 608-623줄, 652-667줄)

---

## 📊 처리 흐름

1. **이미지 로드** → 리사이징 (최대 768px)
2. **MediaPipe 랜드마크 추출** → 마스크 생성용
3. **얼굴 영역 추출** → 원본 보존
4. **신체 보정 (Inpainting)**:
   - 마스크 생성 (신체 부위 + 얼굴 제외)
   - 프롬프트 생성 (한국어 → 영어)
   - Inpainting 실행 (반복 가능)
   - 얼굴 원본 복원
5. **얼굴 보정 (GFPGAN)** (선택사항)
6. **원본 크기로 복원**

---

## 🐛 수정된 버그

### 1. 프롬프트 방향 오류 (수정됨)
- **문제**: "어깨 넓게" 요청 시 "narrower" 프롬프트 생성
- **해결**: 넓게/좁게 키워드에 따라 올바른 프롬프트 생성 (542-545줄)

### 2. 사용하지 않는 모델 로딩 (제거됨)
- **문제**: ControlNet, OpenPose Detector 로딩으로 인한 지연
- **해결**: 사용하지 않는 모델 제거 (로딩 시간 단축)

### 3. 얼굴 보호 마스크 오작동 (수정됨)
- **문제**: 얼굴 영역이 편집되지 않음
- **해결**: 픽셀 단위 복원 방식으로 변경 (얼굴만 원본 유지)

---

## 📌 주요 변경 이력

1. **InstructPix2Pix → Inpainting 전환**
   - 이유: 원본 이미지 보존 필요
   - 결과: 더 정확한 영역별 편집 가능

2. **ControlNet 제거**
   - 이유: 사용하지 않음, 로딩 시간 단축
   - 결과: 모델 로딩 시간 감소

3. **프롬프트 동적 생성**
   - 추가: `parse_body_edit_params()` 함수
   - 결과: 사용자 요청에 따른 강도 조절 가능

4. **얼굴 보호 강화**
   - 방법: 픽셀 단위 복원
   - 결과: 얼굴 완전 보존

---

## 💡 사용 예시

### 어깨 넓게 (강하게)
```
요청: "어깨 확 넓게"
→ strength: 0.85, steps: 40, iterations: 3
→ 프롬프트: "make shoulders wider, increase shoulder width by 25 percent, make shoulders noticeably wider, keep face completely unchanged..."
```

### 허리 얇게 (살짝)
```
요청: "허리 살짝 얇게"
→ strength: 0.55, steps: 26, iterations: 1
→ 프롬프트: "make waist thinner, emphasize a slimmer waistline, reduce waist circumference by 20 percent, keep face completely unchanged..."
```

---

## 📚 참고

- **Stable Diffusion Inpainting**: https://huggingface.co/runwayml/stable-diffusion-inpainting
- **GFPGAN**: https://github.com/TencentARC/GFPGAN
- **MediaPipe**: https://developers.google.com/mediapipe


