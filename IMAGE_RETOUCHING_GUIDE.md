# 이미지 보정(Retouching) 가이드

## 문제점: InstructPix2Pix의 한계

InstructPix2Pix는 **전반적인 편집**에는 좋지만, **세밀한 보정**에는 제한적입니다:
- ❌ 주름 제거: 제한적
- ❌ 피부톤 보정: 제한적
- ❌ 얼굴 디테일 보정: 제한적
- ✅ 스타일 변경: 잘 작동
- ✅ 형태 조작: 잘 작동
- ✅ 배경 변경: 잘 작동

## 해결책: 모델 조합 사용

### 추천 조합: 3단계 파이프라인

```
1단계: InstructPix2Pix (형태/스타일 변경)
    ↓
2단계: GFPGAN/CodeFormer (얼굴 보정, 피부톤 개선)
    ↓
3단계: Real-ESRGAN (전체 품질 향상)
```

## 1. 얼굴 보정 모델 (주름 제거, 피부톤 개선)

### GFPGAN ⭐⭐⭐ (추천)
- **용도**: 얼굴 보정, 주름 제거, 피부톤 개선, 피부 매끄럽게
- **Hugging Face**: https://huggingface.co/spaces/Xintao/GFPGAN
- **GitHub**: https://github.com/TencentARC/GFPGAN
- **특징**:
  - 얼굴 영역 자동 감지
  - 주름, 잡티 제거
  - 피부톤 밝게/개선
  - 자연스러운 보정

### CodeFormer ⭐⭐⭐⭐ (더 자연스러움)
- **용도**: GFPGAN보다 더 자연스러운 얼굴 보정
- **GitHub**: https://github.com/sczhou/CodeFormer
- **Hugging Face**: https://huggingface.co/spaces/sczhou/CodeFormer
- **특징**:
  - 더 자연스러운 결과
  - GFPGAN보다 약간 느림

### 설치 및 사용
```bash
pip install gfpgan realesrgan basicsr facexlib
```

```python
from gfpgan import GFPGANer
import cv2

# 모델 로드
restorer = GFPGANer(
    model_path='models/GFPGANv1.4.pth',
    upscale=1,  # 업스케일 비율 (1 = 원본 크기)
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
)

# 이미지 보정
input_img = cv2.imread('input.jpg', cv2.IMREAD_COLOR)
_, _, restored_img = restorer.enhance(
    input_img,
    has_aligned=False,
    only_center_face=False,
    paste_back=True,
    weight=0.5  # 보정 강도 (0.0-1.0, 낮을수록 자연스러움)
)

cv2.imwrite('output.jpg', restored_img)
```

## 2. 전체 품질 향상 모델

### Real-ESRGAN
- **용도**: 전체 이미지 품질 향상, 선명도 개선
- **GitHub**: https://github.com/xinntao/Real-ESRGAN
- **특징**: 보정 후 최종 품질 향상에 사용

## 3. 통합 파이프라인 구현

### 3단계 보정 파이프라인
```python
from PIL import Image
import cv2
import numpy as np
from diffusers import StableDiffusionInstructPix2PixPipeline
from gfpgan import GFPGANer
import torch

class ImageRetouchingPipeline:
    def __init__(self):
        # 1단계: InstructPix2Pix (형태/스타일 변경)
        self.instruct_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            torch_dtype=torch.float16
        ).to("cuda")
        
        # 2단계: GFPGAN (얼굴 보정)
        self.gfpgan = GFPGANer(
            model_path='models/GFPGANv1.4.pth',
            upscale=1,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None
        )
        
        # 3단계: Real-ESRGAN (품질 향상) - 필요시 추가
    
    def enhance_image(self, image, instruction):
        """
        이미지 보정 파이프라인
        
        Args:
            image: PIL Image
            instruction: 사용자 요청 (예: "어깨 좁게, 주름 제거, 피부톤 밝게")
        
        Returns:
            PIL Image (보정된 이미지)
        """
        # PIL → OpenCV 변환
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 1단계: 형태/스타일 변경 (InstructPix2Pix)
        if "어깨" in instruction or "허리" in instruction or "스타일" in instruction:
            prompt = self.translate_instruction(instruction)
            result = self.instruct_pipe(prompt, image=image, num_inference_steps=20).images[0]
            img_cv = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        
        # 2단계: 얼굴 보정 (GFPGAN)
        if "주름" in instruction or "피부" in instruction or "톤" in instruction:
            _, _, img_cv = self.gfpgan.enhance(
                img_cv,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=0.5  # 보정 강도
            )
        
        # OpenCV → PIL 변환
        result_image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        
        return result_image
    
    def translate_instruction(self, instruction):
        """한국어 요청을 영어 프롬프트로 변환"""
        # 기본 변환 로직
        if "어깨" in instruction:
            return "make shoulders narrower and more natural"
        # ... (기존 변환 로직)
        return "make it more natural and realistic"
```

## 4. FastAPI 통합 예시

```python
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
import cv2
import numpy as np
from gfpgan import GFPGANer
from diffusers import StableDiffusionInstructPix2PixPipeline
import torch

# 전역 변수
instruct_pipe = None
gfpgan_restorer = None

def load_models():
    """모든 모델 로드"""
    global instruct_pipe, gfpgan_restorer
    
    # InstructPix2Pix
    if instruct_pipe is None:
        instruct_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            torch_dtype=torch.float16
        ).to("cuda")
    
    # GFPGAN
    if gfpgan_restorer is None:
        gfpgan_restorer = GFPGANer(
            model_path='models/GFPGANv1.4.pth',
            upscale=1,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None
        )

@app.post("/api/enhance-image")
async def enhance_image(
    file: UploadFile = File(...),
    instruction: str = Form("")
):
    """
    통합 이미지 보정 API
    
    지원 기능:
    - 형태 조작: "어깨 좁게", "허리 얇게"
    - 스타일 변경: "우아하게", "모던하게"
    - 얼굴 보정: "주름 제거", "피부톤 밝게", "피부 매끄럽게"
    - 분위기: "로맨틱하게", "밝게"
    - 배경: "배경 블러"
    """
    try:
        # 모델 로드
        load_models()
        
        # 이미지 읽기
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 1단계: 형태/스타일 변경 (InstructPix2Pix)
        needs_shape_edit = any(keyword in instruction for keyword in 
                              ["어깨", "허리", "엉덩이", "스타일", "배경", "분위기"])
        
        if needs_shape_edit:
            prompt = translate_instruction(instruction)
            result = instruct_pipe(prompt, image=image, num_inference_steps=20).images[0]
            img_cv = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        
        # 2단계: 얼굴 보정 (GFPGAN)
        needs_face_retouch = any(keyword in instruction for keyword in 
                                ["주름", "피부", "톤", "얼굴", "보정"])
        
        if needs_face_retouch:
            # 보정 강도 결정
            weight = 0.5  # 기본값
            if "밝게" in instruction or "하얗게" in instruction:
                weight = 0.7  # 더 강하게
            elif "자연스럽게" in instruction:
                weight = 0.3  # 더 자연스럽게
            
            _, _, img_cv = gfpgan_restorer.enhance(
                img_cv,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=weight
            )
        
        # 결과 변환
        result_image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        
        # Base64로 변환
        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return JSONResponse({
            "success": True,
            "result_image": f"data:image/png;base64,{img_base64}",
            "message": "이미지 보정이 완료되었습니다."
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"이미지 보정 중 오류 발생: {str(e)}"
        }, status_code=500)
```

## 5. 사용자 요청 예시 및 처리

### 예시 1: "어깨 좁게, 주름 제거, 피부톤 밝게"
```python
# 1단계: InstructPix2Pix - "make shoulders narrower"
# 2단계: GFPGAN - 주름 제거 + 피부톤 밝게
```

### 예시 2: "우아한 스타일로, 주름 없애주고, 배경 블러"
```python
# 1단계: InstructPix2Pix - "make it more elegant, make background blur"
# 2단계: GFPGAN - 주름 제거
```

### 예시 3: "허리 얇게, 피부톤 하얗게, 자연스럽게"
```python
# 1단계: InstructPix2Pix - "make waist thinner"
# 2단계: GFPGAN - 피부톤 밝게 (weight=0.3, 자연스럽게)
```

## 6. 모델 다운로드

### GFPGAN 모델 다운로드
```python
# 자동 다운로드 (첫 실행 시)
from gfpgan import GFPGANer

# 모델 자동 다운로드
restorer = GFPGANer(
    model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
    upscale=1,
    arch='clean'
)
```

또는 수동 다운로드:
```bash
# GFPGAN 모델 다운로드
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P models/
```

## 7. requirements.txt 추가

```
# 기존 패키지
diffusers>=0.21.0
transformers>=4.35.0
accelerate>=0.24.0

# 얼굴 보정
gfpgan>=1.3.8
realesrgan>=0.3.0
basicsr>=1.4.2
facexlib>=0.3.0

# 이미지 처리
opencv-python>=4.8.0
pillow>=10.0.0
numpy>=1.24.0
```

## 8. 최종 추천 파이프라인

### 단계별 처리
```
사용자 요청 분석
    ↓
[형태/스타일 변경 필요?]
    ↓ YES
InstructPix2Pix 실행
    ↓
[얼굴 보정 필요?]
    ↓ YES
GFPGAN 실행 (주름 제거, 피부톤 개선)
    ↓
[품질 향상 필요?]
    ↓ YES (선택적)
Real-ESRGAN 실행
    ↓
최종 결과 반환
```

## 9. 주의사항

### GFPGAN 사용 시
- **얼굴 감지**: 얼굴이 작거나 가려진 경우 보정이 제한적
- **보정 강도**: weight 파라미터로 조절 (0.0-1.0)
  - 낮음 (0.3): 자연스럽지만 보정 약함
  - 높음 (0.7): 강하게 보정하지만 부자연스러울 수 있음
- **처리 시간**: 약 2-5초 (GPU 기준)

### 모델 조합 시
- **순서 중요**: InstructPix2Pix → GFPGAN 순서 권장
- **메모리**: 두 모델 모두 로드 시 약 10-12GB VRAM 필요
- **처리 시간**: 총 10-20초 (GPU 기준)

## 10. 대안: 더 정밀한 보정이 필요한 경우

### Stable Diffusion Inpainting
- 특정 영역만 선택적으로 보정
- 마스크 영역 지정 가능

### ControlNet + Inpainting
- 더 정밀한 제어 가능
- 포즈/구조 유지하면서 특정 영역만 수정

## 결론

**InstructPix2Pix만으로는 부족**합니다. 
**GFPGAN을 추가**하여 주름 제거, 피부톤 개선 등 세밀한 보정을 수행해야 합니다.

**추천 구현**:
1. InstructPix2Pix (형태/스타일)
2. GFPGAN (얼굴 보정) ← **필수 추가**
3. Real-ESRGAN (품질 향상, 선택적)




