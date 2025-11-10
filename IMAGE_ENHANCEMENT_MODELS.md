# 이미지 보정 모델 리스트

사용자가 텍스트로 요청한 내용(예: "어깨가 너무 넓게 나왔어, 좁게 수정해줘")에 맞춰 이미지를 보정하는 기능을 위한 모델 목록입니다.

## 1. 텍스트 지시 기반 이미지 편집 모델

### 1.1 InstructPix2Pix
**용도**: 텍스트 지시에 따라 이미지 편집
- **Hugging Face**: https://huggingface.co/timbrooks/instruct-pix2pix
- **GitHub**: https://github.com/timothybrooks/instruct-pix2pix
- **특징**: 
  - "make shoulders narrower", "make waist thinner" 같은 자연어 명령 이해
  - 원본 이미지 구조 유지하면서 편집
  - Stable Diffusion 기반
- **사용 예시**: "어깨를 좁게 해줘", "허리를 더 얇게 해줘"

### 1.2 MagicBrush
**용도**: 텍스트 지시 기반 인페인팅/아웃페인팅
- **GitHub**: https://github.com/OSU-NLP-Group/MagicBrush
- **특징**: 정확한 텍스트 지시에 따른 이미지 편집
- **사용 예시**: 특정 부위만 수정

### 1.3 IP-Adapter
**용도**: 이미지 + 텍스트 프롬프트로 편집
- **Hugging Face**: https://huggingface.co/h94/IP-Adapter
- **GitHub**: https://github.com/tencent-ailab/IP-Adapter
- **특징**: 
  - 이미지 스타일 유지하면서 텍스트 지시 반영
  - ControlNet과 함께 사용 가능

### 1.4 ControlNet
**용도**: 제어 가능한 이미지 생성/편집
- **Hugging Face**: https://huggingface.co/lllyasviel/ControlNet-v1-1
- **GitHub**: https://github.com/lllyasviel/ControlNet
- **특징**:
  - OpenPose, Canny, Depth 등 다양한 제어 방식
  - 인체 포즈/형태 제어 가능
  - **주요 ControlNet 모델들**:
    - OpenPose: https://huggingface.co/lllyasviel/sd-controlnet-openpose
    - Canny: https://huggingface.co/lllyasviel/sd-controlnet-canny
    - Depth: https://huggingface.co/lllyasviel/sd-controlnet-depth

## 2. 인체 형태 조작 모델

### 2.1 DensePose
**용도**: 인체 3D 형태 추정 및 조작
- **GitHub**: https://github.com/facebookresearch/DensePose
- **Hugging Face**: https://huggingface.co/papers/1802.00434
- **특징**: 인체의 3D 형태를 정확히 파악하여 형태 조작 가능

### 2.2 SMPL/Body Shape Manipulation
**용도**: 인체 형태 파라미터 조작
- **SMPL**: https://smpl.is.tue.mpg.de/
- **특징**: 어깨, 허리, 엉덩이 등 부위별 형태 조작 가능

### 2.3 MediaPipe Pose + ControlNet
**용도**: 현재 프로젝트에서 이미 사용 중인 MediaPipe와 ControlNet 조합
- **MediaPipe**: 이미 사용 중 (body_analysis_test)
- **ControlNet OpenPose**: 포즈 정보를 활용한 이미지 편집

## 3. 이미지 품질 향상 모델

### 3.1 Real-ESRGAN
**용도**: 이미지 업스케일링 및 품질 향상
- **GitHub**: https://github.com/xinntao/Real-ESRGAN
- **Hugging Face**: https://huggingface.co/spaces/akhaliq/Real-ESRGAN
- **특징**: 
  - 2x, 4x 업스케일링
  - 블러 제거, 노이즈 제거
  - 합성 후 최종 품질 향상에 사용

### 3.2 GFPGAN
**용도**: 얼굴 복원 및 향상
- **GitHub**: https://github.com/TencentARC/GFPGAN
- **Hugging Face**: https://huggingface.co/spaces/Xintao/GFPGAN
- **특징**: 얼굴 부분만 고품질로 복원

### 3.3 CodeFormer
**용도**: 얼굴 복원 및 향상
- **GitHub**: https://github.com/sczhou/CodeFormer
- **Hugging Face**: https://huggingface.co/spaces/sczhou/CodeFormer
- **특징**: GFPGAN보다 더 자연스러운 얼굴 복원

## 4. Stable Diffusion 기반 편집 모델

### 4.1 Stable Diffusion XL
**용도**: 고품질 이미지 생성/편집
- **Hugging Face**: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
- **특징**: 고해상도 이미지 생성 가능

### 4.2 Stable Diffusion 2.1
**용도**: 기본 이미지 생성/편집
- **Hugging Face**: https://huggingface.co/stabilityai/stable-diffusion-2-1
- **특징**: 안정적인 이미지 생성

### 4.3 Stable Diffusion Inpainting
**용도**: 특정 영역만 편집
- **Hugging Face**: https://huggingface.co/runwayml/stable-diffusion-inpainting
- **특징**: 마스크 영역만 선택적으로 수정

## 5. 추천 구현 방식

### 방식 1: InstructPix2Pix (가장 간단)
```
사용자 요청 → 텍스트 프롬프트 변환 → InstructPix2Pix → 결과 이미지
예: "어깨가 너무 넓게 나왔어" → "make shoulders narrower" → 편집
```

### 방식 2: ControlNet + Stable Diffusion (더 정밀)
```
원본 이미지 → MediaPipe Pose 추출 → 포즈 조작 → ControlNet OpenPose → Stable Diffusion → 결과
```

### 방식 3: 하이브리드 (추천)
```
1. InstructPix2Pix로 전체적인 수정
2. Real-ESRGAN으로 품질 향상
3. GFPGAN/CodeFormer로 얼굴 부분만 추가 보정
```

## 6. Python 패키지 및 라이브러리

### 필수 패키지
```python
# 이미지 처리
pip install pillow opencv-python numpy

# Stable Diffusion
pip install diffusers transformers accelerate

# ControlNet
pip install controlnet-aux

# Real-ESRGAN
pip install basicsr facexlib gfpgan realesrgan

# 기타
pip install torch torchvision
```

## 7. 모델 다운로드 및 사용 예시

### InstructPix2Pix 사용 예시
```python
from diffusers import StableDiffusionInstructPix2PixPipeline
import torch
from PIL import Image

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix",
    torch_dtype=torch.float16
).to("cuda")

image = Image.open("composed_image.png")
prompt = "make shoulders narrower and more natural"
result = pipe(prompt, image=image, num_inference_steps=20).images[0]
```

### ControlNet OpenPose 사용 예시
```python
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from controlnet_aux import OpenposeDetector
import torch

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose",
    torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# MediaPipe로 포즈 추출 → 조작 → ControlNet에 입력
```

### Real-ESRGAN 사용 예시
```python
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

model = RRDBNet(num_in_feat=3, num_out_feat=3, num_feat=64, 
                num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(scale=4, model_path='models/RealESRGAN_x4plus.pth', 
                        model=model, tile=0, tile_pad=10, pre_pad=0)
output, _ = upsampler.enhance(image, outscale=4)
```

## 8. 모델 선택 가이드

| 요구사항 | 추천 모델 | 난이도 |
|---------|---------|--------|
| 빠른 구현, 간단한 수정 | InstructPix2Pix | ⭐⭐ |
| 정밀한 형태 조작 | ControlNet + OpenPose | ⭐⭐⭐⭐ |
| 얼굴 품질 향상 | GFPGAN/CodeFormer | ⭐⭐⭐ |
| 전체 품질 향상 | Real-ESRGAN | ⭐⭐⭐ |
| 복합 편집 | 하이브리드 방식 | ⭐⭐⭐⭐⭐ |

## 9. 참고 자료

- **Diffusers 공식 문서**: https://huggingface.co/docs/diffusers
- **ControlNet 공식 문서**: https://github.com/lllyasviel/ControlNet
- **Stable Diffusion 공식**: https://stability.ai/
- **Real-ESRGAN 문서**: https://github.com/xinntao/Real-ESRGAN

## 10. 구현 우선순위

1. **1단계**: InstructPix2Pix로 기본 편집 기능 구현
2. **2단계**: Real-ESRGAN으로 품질 향상 추가
3. **3단계**: ControlNet으로 더 정밀한 조작 추가
4. **4단계**: GFPGAN으로 얼굴 보정 추가




