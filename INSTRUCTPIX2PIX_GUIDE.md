# InstructPix2Pix 사용 가이드

## 1. 설치

### 필수 패키지 설치
```bash
pip install diffusers transformers accelerate torch torchvision pillow
```

### requirements.txt에 추가
```
diffusers>=0.21.0
transformers>=4.35.0
accelerate>=0.24.0
torch>=2.0.0
torchvision>=0.15.0
pillow>=10.0.0
```

### ⚠️ 중요: 학습 불필요!
- **사전 학습된 모델 사용**: 학습(fine-tuning) 없이 바로 사용 가능
- **자동 다운로드**: `from_pretrained()` 호출 시 자동으로 모델 다운로드 (약 2.5GB)
- **첫 실행 시**: 인터넷 연결 필요 (모델 다운로드)
- **이후 사용**: 다운로드된 모델 캐시 사용 (재다운로드 불필요)

## 2. 기본 사용법

### 기본 코드
```python
from diffusers import StableDiffusionInstructPix2PixPipeline
import torch
from PIL import Image

# 모델 로드
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix",
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
)

# GPU 사용 시
if torch.cuda.is_available():
    pipe = pipe.to("cuda")
else:
    pipe = pipe.to("cpu")

# 이미지 편집
image = Image.open("composed_image.png").convert("RGB")
prompt = "make shoulders narrower and more natural"
result = pipe(prompt, image=image, num_inference_steps=20, image_guidance_scale=1.5).images[0]
result.save("edited_image.png")
```

## 3. InstructPix2Pix로 가능한 기능들

### ✅ 가능한 편집 기능
InstructPix2Pix는 단순한 형태 조작뿐만 아니라 **스타일, 분위기, 배경 변경**도 가능합니다!

#### 1. 형태 조작
- "make shoulders narrower" - 어깨 좁게
- "make waist thinner" - 허리 얇게
- "make the person taller" - 키 크게

#### 2. 스타일 변경
- "change to casual style" - 캐주얼 스타일로
- "make it more elegant" - 더 우아하게
- "change to vintage style" - 빈티지 스타일로
- "make it modern" - 모던하게

#### 3. 분위기 변경
- "make the mood more romantic" - 더 로맨틱하게
- "change to bright and cheerful atmosphere" - 밝고 경쾌한 분위기로
- "make it more dramatic" - 더 드라마틱하게
- "change to warm and cozy feeling" - 따뜻하고 아늑한 느낌으로

#### 4. 배경 변경
- "change background to beach" - 배경을 해변으로
- "change background to garden" - 배경을 정원으로
- "make background blur" - 배경 블러 처리
- "remove background and add studio background" - 배경 제거 후 스튜디오 배경

#### 5. 조명/색감 변경
- "make lighting more soft" - 조명을 더 부드럽게
- "change to warm lighting" - 따뜻한 조명으로
- "make colors more vibrant" - 색감을 더 선명하게
- "change to black and white" - 흑백으로

#### 6. 전체적인 품질 향상
- "make it more realistic" - 더 현실적으로
- "improve image quality" - 이미지 품질 향상
- "make it more professional" - 더 전문적으로

### 실제 사용 예시
```python
# 형태 조작
result = pipe("make shoulders narrower", image=image).images[0]

# 스타일 변경
result = pipe("change to elegant wedding dress style", image=image).images[0]

# 분위기 변경
result = pipe("make the mood more romantic with soft lighting", image=image).images[0]

# 배경 변경
result = pipe("change background to beautiful garden with flowers", image=image).images[0]

# 복합 요청
result = pipe("make shoulders narrower, change to elegant style, and make background blur", image=image).images[0]
```

## 4. 한국어 요청을 영어 프롬프트로 변환

### 변환 함수 (확장 버전)
```python
def translate_korean_to_prompt(korean_text):
    """한국어 요청을 영어 프롬프트로 변환"""
    
    # 형태 조작 매핑
    shape_mappings = {
        "어깨가 너무 넓게": "make shoulders narrower",
        "어깨를 좁게": "make shoulders narrower",
        "허리를 얇게": "make waist thinner",
        "엉덩이를 작게": "make hips smaller",
    }
    
    # 스타일 변경 매핑
    style_mappings = {
        "우아하게": "make it more elegant",
        "캐주얼하게": "change to casual style",
        "모던하게": "make it more modern",
        "빈티지": "change to vintage style",
        "클래식": "change to classic style",
    }
    
    # 분위기 변경 매핑
    mood_mappings = {
        "로맨틱하게": "make the mood more romantic",
        "밝게": "make it bright and cheerful",
        "드라마틱하게": "make it more dramatic",
        "따뜻하게": "change to warm and cozy feeling",
    }
    
    # 배경 변경 매핑
    background_mappings = {
        "배경을 해변으로": "change background to beach",
        "배경을 정원으로": "change background to garden",
        "배경 블러": "make background blur",
        "배경 제거": "remove background",
    }
    
    # 모든 매핑 검사
    prompt_parts = []
    
    for kor, eng in {**shape_mappings, **style_mappings, **mood_mappings, **background_mappings}.items():
        if kor in korean_text:
            prompt_parts.append(eng)
    
    if prompt_parts:
        return ", ".join(prompt_parts) + " and make it more natural"
    else:
        # 기본 변환
        return f"adjust {korean_text} to make it more natural and realistic"
```

## 4. FastAPI 엔드포인트 구현 예시

### main.py에 추가
```python
from diffusers import StableDiffusionInstructPix2PixPipeline
import torch
from PIL import Image
import io
import base64

# 전역 변수로 모델 저장
instruct_pix2pix_pipe = None

def load_instruct_pix2pix_model():
    """InstructPix2Pix 모델 로드"""
    global instruct_pix2pix_pipe
    if instruct_pix2pix_pipe is None:
        print("InstructPix2Pix 모델 로딩 중...")
        instruct_pix2pix_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        instruct_pix2pix_pipe = instruct_pix2pix_pipe.to(device)
        print(f"InstructPix2Pix 모델 로드 완료: {device}")
    return instruct_pix2pix_pipe

def translate_korean_request(korean_text):
    """한국어 요청을 영어 프롬프트로 변환 (확장 버전)"""
    
    # 형태 조작
    shape_mappings = {
        "어깨가 너무 넓게": "make shoulders narrower",
        "어깨를 좁게": "make shoulders narrower",
        "허리를 얇게": "make waist thinner",
        "엉덩이를 작게": "make hips smaller",
        "팔을 짧게": "make arms shorter",
        "다리를 길게": "make legs longer",
    }
    
    # 스타일 변경
    style_mappings = {
        "우아하게": "make it more elegant",
        "캐주얼하게": "change to casual style",
        "모던하게": "make it more modern",
        "빈티지": "change to vintage style",
    }
    
    # 분위기 변경
    mood_mappings = {
        "로맨틱하게": "make the mood more romantic",
        "밝게": "make it bright and cheerful",
        "드라마틱하게": "make it more dramatic",
    }
    
    # 배경 변경
    background_mappings = {
        "배경을 해변으로": "change background to beach",
        "배경을 정원으로": "change background to garden",
        "배경 블러": "make background blur",
    }
    
    # 모든 매핑 병합
    all_mappings = {**shape_mappings, **style_mappings, **mood_mappings, **background_mappings}
    
    # 매칭된 프롬프트 수집
    prompt_parts = []
    for kor, eng in all_mappings.items():
        if kor in korean_text:
            prompt_parts.append(eng)
    
    if prompt_parts:
        return ", ".join(prompt_parts) + " and make it more natural"
    else:
        return f"adjust {korean_text} to make it more natural and realistic"

@app.post("/api/enhance-image")
async def enhance_image(
    file: UploadFile = File(...),
    instruction: str = Form(""),  # 사용자 요청 텍스트
    num_inference_steps: int = Form(20),
    image_guidance_scale: float = Form(1.5)
):
    """
    사용자 요청에 따라 이미지 보정
    
    - file: 합성된 이미지
    - instruction: 사용자 요청 (예: "어깨가 너무 넓게 나왔어, 좁게 수정해줘")
    - num_inference_steps: 추론 단계 (20-50, 높을수록 품질 좋지만 느림)
    - image_guidance_scale: 이미지 가이던스 (1.0-2.0, 높을수록 원본 유지)
    """
    try:
        # 이미지 읽기
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # 한국어 요청을 영어 프롬프트로 변환
        if instruction:
            prompt = translate_korean_request(instruction)
        else:
            prompt = "make the image more natural and realistic"
        
        # 모델 로드
        pipe = load_instruct_pix2pix_model()
        
        # 이미지 편집
        result_image = pipe(
            prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            image_guidance_scale=image_guidance_scale
        ).images[0]
        
        # Base64로 변환
        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return JSONResponse({
            "success": True,
            "result_image": f"data:image/png;base64,{img_base64}",
            "prompt_used": prompt,
            "message": "이미지 보정이 완료되었습니다."
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"이미지 보정 중 오류 발생: {str(e)}"
        }, status_code=500)
```

## 5. 사용 예시

### 프론트엔드에서 호출
```javascript
// api.js에 추가
export const enhanceImage = async (imageFile, instruction) => {
    try {
        const formData = new FormData()
        formData.append('file', imageFile)
        formData.append('instruction', instruction)
        formData.append('num_inference_steps', 20)
        formData.append('image_guidance_scale', 1.5)
        
        const response = await api.post('/api/enhance-image', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        })
        
        return response.data
    } catch (error) {
        console.error('이미지 보정 오류:', error)
        throw error
    }
}

// 사용 예시
const result = await enhanceImage(
    imageFile, 
    "어깨가 너무 넓게 나왔어, 좁게 수정해줘"
)
```

## 6. 파라미터 설명

### num_inference_steps
- **범위**: 10-50 (기본값: 20)
- **설명**: 추론 단계 수
- **효과**: 높을수록 품질이 좋아지지만 처리 시간이 길어짐
- **추천**: 20-30

### image_guidance_scale
- **범위**: 1.0-2.0 (기본값: 1.5)
- **설명**: 원본 이미지 유지 정도
- **효과**: 
  - 낮음 (1.0-1.2): 더 많이 변경, 창의적
  - 높음 (1.5-2.0): 원본 유지, 안정적
- **추천**: 1.5

## 7. 주의사항

### GPU 메모리
- **최소 요구사항**: 8GB VRAM
- **권장**: 16GB+ VRAM
- **CPU 사용 시**: 매우 느림 (권장하지 않음)

### 처리 시간
- **GPU**: 약 5-10초 (20 steps 기준)
- **CPU**: 약 1-3분 (20 steps 기준)

### 모델 크기
- **다운로드 크기**: 약 2.5GB
- **메모리 사용량**: 약 4-6GB

## 8. 최적화 팁

### 배치 처리
```python
# 여러 이미지를 한 번에 처리 (GPU 메모리 허용 시)
results = pipe(prompt, image=[img1, img2, img3], num_inference_steps=20)
```

### 모델 캐싱
```python
# 전역 변수로 모델을 한 번만 로드하고 재사용
# 위의 load_instruct_pix2pix_model() 함수 참조
```

### 저해상도 처리 후 업스케일
```python
# 1. 저해상도로 빠르게 편집
small_image = image.resize((512, 512))
result = pipe(prompt, image=small_image, num_inference_steps=10)

# 2. Real-ESRGAN으로 업스케일
# (별도 가이드 참조)
```

## 9. 에러 처리

### 일반적인 에러
```python
try:
    result = pipe(prompt, image=image)
except torch.cuda.OutOfMemoryError:
    # GPU 메모리 부족
    # 이미지 크기 줄이기 또는 CPU 사용
    image = image.resize((512, 512))
    result = pipe(prompt, image=image)
except Exception as e:
    print(f"오류 발생: {e}")
```

## 10. 프로젝트 통합 예시

### main.py에 통합
```python
# 앱 시작 시 모델 로드
@app.on_event("startup")
async def startup_event():
    # 기존 모델들 로드
    load_model()
    init_database()
    
    # InstructPix2Pix 모델 로드 (선택적)
    try:
        load_instruct_pix2pix_model()
        print("✅ InstructPix2Pix 모델 로드 완료")
    except Exception as e:
        print(f"⚠️ InstructPix2Pix 모델 로드 실패: {e}")
```

### requirements.txt에 추가
```
diffusers>=0.21.0
transformers>=4.35.0
accelerate>=0.24.0
```

## 11. 테스트 코드

```python
# test_instruct_pix2pix.py
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline
import torch

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix",
    torch_dtype=torch.float16
).to("cuda")

image = Image.open("test_image.png").convert("RGB")

# 다양한 테스트
test_prompts = [
    "make shoulders narrower",  # 형태 조작
    "change to elegant wedding dress style",  # 스타일 변경
    "make the mood more romantic with soft lighting",  # 분위기 변경
    "change background to beautiful garden",  # 배경 변경
    "make it more realistic and professional",  # 품질 향상
]

for i, prompt in enumerate(test_prompts):
    result = pipe(prompt, image=image, num_inference_steps=20).images[0]
    result.save(f"result_{i+1}.png")
    print(f"테스트 {i+1} 완료: {prompt}")

print("모든 테스트 완료!")
```

## 12. 스타일/분위기 변경에 대한 제한사항

### ⚠️ 주의사항
- **형태 조작**: 비교적 정확하게 작동
- **스타일 변경**: 일반적으로 잘 작동하지만 원하는 스타일과 차이가 있을 수 있음
- **배경 변경**: 간단한 배경은 잘 작동, 복잡한 배경은 제한적
- **분위기 변경**: 조명/색감 변경은 잘 작동, 전체적인 분위기는 제한적

### 💡 더 정확한 스타일 변경을 원한다면
- **ControlNet**: 포즈/구조를 유지하면서 스타일 변경
- **IP-Adapter**: 이미지 스타일을 참조하여 변경
- **Stable Diffusion Inpainting**: 특정 영역만 선택적으로 변경

## 13. 실제 사용 시나리오

### 시나리오 1: 형태 + 스타일 동시 변경
```python
prompt = "make shoulders narrower and change to elegant wedding dress style"
result = pipe(prompt, image=image, num_inference_steps=25, image_guidance_scale=1.5).images[0]
```

### 시나리오 2: 분위기 + 배경 변경
```python
prompt = "make the mood more romantic with soft lighting and change background to garden"
result = pipe(prompt, image=image, num_inference_steps=25).images[0]
```

### 시나리오 3: 복합 요청
```python
prompt = "make waist thinner, change to modern style, make background blur, and improve image quality"
result = pipe(prompt, image=image, num_inference_steps=30, image_guidance_scale=1.5).images[0]
```

