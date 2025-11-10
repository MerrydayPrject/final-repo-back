# InstructPix2Pix 대체 모델 - 하이브리드 구성

## 추천 하이브리드 조합 (2개 모델)

### 조합 1: ControlNet OpenPose + Stable Diffusion Inpainting ⭐⭐⭐⭐⭐
**가장 정밀한 형태 조작**

#### 모델 1: ControlNet OpenPose
- **용도**: MediaPipe 포즈 정보를 활용한 정밀한 형태 조작
- **Hugging Face**: https://huggingface.co/lllyasviel/sd-controlnet-openpose
- **특징**: 
  - 현재 프로젝트에서 이미 MediaPipe 사용 중 → 포즈 정보 활용 가능
  - 어깨, 허리, 엉덩이 등 부위별 정밀 조작
  - 원본 이미지 구조 완벽 유지

#### 모델 2: Stable Diffusion Inpainting
- **용도**: 특정 영역(어깨, 허리 등)만 선택적으로 편집
- **Hugging Face**: https://huggingface.co/runwayml/stable-diffusion-inpainting
- **특징**:
  - 마스크 영역만 수정 (나머지는 원본 유지)
  - 불필요한 얼굴 생성 문제 해결
  - 자연스러운 편집

#### 구현 흐름:
```
1. MediaPipe로 포즈 랜드마크 추출 (이미 구현됨)
2. 포즈 정보 조작 (어깨 좁게 → 어깨 랜드마크 좁게)
3. ControlNet OpenPose로 포즈 제어 이미지 생성
4. Stable Diffusion Inpainting으로 어깨 영역만 마스크하여 편집
5. 원본과 합성
```

---

### 조합 2: IP-Adapter + Stable Diffusion XL ⭐⭐⭐⭐
**스타일 유지하면서 편집**

#### 모델 1: IP-Adapter
- **용도**: 원본 이미지 스타일을 유지하면서 텍스트 지시 반영
- **Hugging Face**: https://huggingface.co/h94/IP-Adapter
- **특징**:
  - 원본 이미지의 스타일, 색감, 분위기 완벽 유지
  - 텍스트 프롬프트로 편집 지시
  - InstructPix2Pix보다 안정적

#### 모델 2: Stable Diffusion XL
- **용도**: 고품질 이미지 생성/편집
- **Hugging Face**: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
- **특징**:
  - 고해상도 이미지 처리
  - 더 자연스러운 결과
  - IP-Adapter와 완벽 호환

#### 구현 흐름:
```
1. IP-Adapter로 원본 이미지 스타일 임베딩 추출
2. 텍스트 프롬프트 변환 ("어깨 좁게" → "make shoulders narrower")
3. Stable Diffusion XL + IP-Adapter로 스타일 유지하면서 편집
4. 결과 이미지 반환
```

---

## 구현 코드 예시

### 조합 1: ControlNet + Inpainting

```python
from diffusers import (
    ControlNetModel, 
    StableDiffusionControlNetPipeline,
    StableDiffusionInpaintPipeline
)
from controlnet_aux import OpenposeDetector
import torch
from PIL import Image
import numpy as np
import cv2

# 1. ControlNet OpenPose 로드
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose",
    torch_dtype=torch.float16
)

controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# 2. Inpainting 파이프라인 로드
inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
).to("cuda")

# 3. MediaPipe 포즈 추출 (이미 구현됨)
# body_analysis_service.extract_landmarks(image)

# 4. 포즈 조작 (어깨 좁게)
def adjust_pose_landmarks(landmarks, instruction):
    """포즈 랜드마크 조작"""
    if "어깨" in instruction and "좁" in instruction:
        # 어깨 랜드마크 좁게 조정
        landmarks[11][0] -= 0.05  # 왼쪽 어깨
        landmarks[12][0] += 0.05  # 오른쪽 어깨
    return landmarks

# 5. ControlNet으로 포즈 제어 이미지 생성
def edit_with_controlnet(image, landmarks, prompt):
    # 포즈 이미지 생성
    pose_image = draw_pose(landmarks)
    
    # ControlNet으로 편집
    result = controlnet_pipe(
        prompt=prompt,
        image=pose_image,
        num_inference_steps=20,
        controlnet_conditioning_scale=1.0
    ).images[0]
    
    return result

# 6. Inpainting으로 특정 영역만 편집
def edit_with_inpainting(image, mask, prompt):
    """어깨 영역만 마스크하여 편집"""
    result = inpaint_pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=20,
        strength=0.8
    ).images[0]
    
    return result
```

### 조합 2: IP-Adapter + SDXL

```python
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from diffusers.utils import load_image
import torch
from PIL import Image

# 1. IP-Adapter 로드
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter

adapter = T2IAdapter.from_pretrained(
    "TencentARC/t2i-adapter-canny-sdxl-1.0",
    torch_dtype=torch.float16
)

# 2. Stable Diffusion XL 로드
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

# 3. IP-Adapter 파이프라인
adapter_pipe = StableDiffusionXLAdapterPipeline(
    vae=pipe.vae,
    text_encoder=pipe.text_encoder,
    tokenizer=pipe.tokenizer,
    unet=pipe.unet,
    scheduler=pipe.scheduler,
    adapter=adapter
).to("cuda")

# 4. 이미지 편집
def edit_with_ipadapter(image, prompt):
    """IP-Adapter로 스타일 유지하면서 편집"""
    result = adapter_pipe(
        prompt=prompt,
        image=image,
        num_inference_steps=20,
        guidance_scale=7.5,
        adapter_conditioning_scale=0.8
    ).images[0]
    
    return result
```

---

## 비교표

| 조합 | 정밀도 | 구현 난이도 | 속도 | 품질 | 추천도 |
|------|--------|------------|------|------|--------|
| **ControlNet + Inpainting** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **IP-Adapter + SDXL** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| InstructPix2Pix (현재) | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |

---

## 최종 추천

### **조합 1: ControlNet OpenPose + Stable Diffusion Inpainting**

**이유:**
1. ✅ 현재 프로젝트에서 이미 MediaPipe 사용 중 → 포즈 정보 활용 가능
2. ✅ 정밀한 형태 조작 가능 (어깨, 허리, 엉덩이 등)
3. ✅ Inpainting으로 특정 영역만 편집 → 불필요한 얼굴 생성 문제 해결
4. ✅ 원본 이미지 구조 완벽 유지
5. ✅ 안정적인 결과

**구현 우선순위:**
1. ControlNet OpenPose 파이프라인 구축
2. MediaPipe 포즈 정보 → ControlNet 포즈 이미지 변환
3. 포즈 조작 로직 구현
4. Inpainting으로 특정 영역만 편집
5. 원본과 합성

---

## 설치 명령어

```bash
# ControlNet + Inpainting
pip install diffusers transformers accelerate controlnet-aux

# IP-Adapter + SDXL
pip install diffusers transformers accelerate
```

---

## 참고 링크

- **ControlNet**: https://github.com/lllyasviel/ControlNet
- **Stable Diffusion Inpainting**: https://huggingface.co/runwayml/stable-diffusion-inpainting
- **IP-Adapter**: https://github.com/tencent-ailab/IP-Adapter
- **Stable Diffusion XL**: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0


