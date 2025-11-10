# API 키 기반 이미지 보정 모델 옵션

현재는 로컬 모델(InstructPix2Pix, GFPGAN)을 사용하고 있지만, API 키를 사용하는 클라우드 기반 보정 모델들도 있습니다.

## 1. Google Gemini Vision API (이미 사용 중 ✅)

### 현재 사용 상황
- **용도**: 이미지 합성 (`/api/compose-dress`)
- **모델**: `gemini-2.5-flash-image`
- **API 키**: `GEMINI_API_KEY`

### 보정 기능으로 확장 가능
```python
# Gemini로 이미지 보정 예시
from google import genai
import base64
from PIL import Image
import io

def enhance_with_gemini(image: Image.Image, instruction: str):
    """Gemini API로 이미지 보정"""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    # 이미지를 base64로 변환
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    # 프롬프트 생성
    prompt = f"""
    Please edit this image according to the following instruction: {instruction}
    
    Requirements:
    - Keep the person's face unchanged
    - Make adjustments naturally
    - Preserve image quality
    """
    
    # Gemini API 호출
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[
            types.Part.from_bytes(
                data=base64.b64decode(img_base64),
                mime_type="image/png"
            ),
            prompt
        ]
    )
    
    # 응답에서 이미지 추출 (Gemini는 텍스트 응답만 제공하므로 제한적)
    return response.text
```

**장점**:
- 이미 API 키가 설정되어 있음
- 무료 티어 제공
- 빠른 응답 속도

**단점**:
- 이미지 편집보다는 이미지 이해/분석에 특화
- 직접적인 이미지 편집 결과를 반환하지 않음 (텍스트 응답)

---

## 2. OpenAI DALL-E 3 (이미지 편집)

### 특징
- **API**: OpenAI API
- **모델**: `dall-e-3`
- **용도**: 이미지 생성 및 편집
- **비용**: 유료 (이미지당 $0.04~$0.12)

### 사용 예시
```python
from openai import OpenAI

def enhance_with_dalle(image: Image.Image, instruction: str):
    """DALL-E로 이미지 편집"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # DALL-E는 이미지 편집보다는 생성에 특화
    # 이미지 편집은 제한적
    response = client.images.edit(
        image=image,
        prompt=instruction,
        n=1,
        size="1024x1024"
    )
    
    return response.data[0].url
```

**장점**:
- 고품질 이미지 생성
- 자연어 이해 우수

**단점**:
- 이미지 편집 기능 제한적 (생성에 특화)
- 비용 발생
- 기존 이미지 편집보다는 새 이미지 생성

---

## 3. Stability AI API

### 특징
- **API**: Stability AI API
- **모델**: Stable Diffusion XL, Stable Diffusion 3
- **용도**: 이미지 생성 및 편집
- **비용**: 유료 (크레딧 기반)

### 사용 예시
```python
import stability_sdk.client

def enhance_with_stability(image: Image.Image, instruction: str):
    """Stability AI로 이미지 편집"""
    stability_api = stability_sdk.client.StabilityInference(
        key=os.getenv("STABILITY_API_KEY"),
        engine="stable-diffusion-xl-1024-v1-0"
    )
    
    # 이미지 편집 (inpainting/outpainting)
    answers = stability_api.edit(
        init_image=image,
        prompt=instruction,
        strength=0.7  # 편집 강도
    )
    
    return answers[0].image
```

**장점**:
- 고품질 결과
- 다양한 편집 옵션

**단점**:
- 비용 발생
- API 복잡도 높음

---

## 4. Replicate API (다양한 모델)

### 특징
- **API**: Replicate API
- **모델**: 다양한 오픈소스 모델 제공
- **용도**: InstructPix2Pix, GFPGAN 등 클라우드 버전
- **비용**: 사용량 기반

### 사용 예시
```python
import replicate

def enhance_with_replicate(image: Image.Image, instruction: str):
    """Replicate API로 InstructPix2Pix 사용"""
    
    # 이미지를 base64로 변환
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    output = replicate.run(
        "timbrooks/instruct-pix2pix:latest",
        input={
            "image": f"data:image/png;base64,{img_base64}",
            "prompt": instruction,
            "num_inference_steps": 20,
            "image_guidance_scale": 1.5
        }
    )
    
    return output
```

**장점**:
- 현재 사용 중인 모델의 클라우드 버전
- 로컬 GPU 불필요
- 다양한 모델 선택 가능

**단점**:
- 비용 발생 (사용량 기반)
- 네트워크 지연
- API 키 필요

---

## 5. Hugging Face Inference API

### 특징
- **API**: Hugging Face Inference API
- **모델**: 다양한 Hugging Face 모델
- **용도**: InstructPix2Pix 등
- **비용**: 무료 티어 + 유료

### 사용 예시
```python
import requests

def enhance_with_hf(image: Image.Image, instruction: str):
    """Hugging Face Inference API 사용"""
    
    API_URL = "https://api-inference.huggingface.co/models/timbrooks/instruct-pix2pix"
    headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}
    
    # 이미지 변환
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    
    response = requests.post(
        API_URL,
        headers=headers,
        json={
            "inputs": {
                "image": base64.b64encode(buffered.getvalue()).decode(),
                "prompt": instruction
            }
        }
    )
    
    return response.json()
```

**장점**:
- 무료 티어 제공
- 다양한 모델 접근
- 현재 사용 중인 모델과 동일

**단점**:
- 무료 티어는 큐 대기 시간 있음
- 유료는 비용 발생

---

## 추천 조합

### 옵션 1: Gemini API 확장 (추천 ⭐)
- **이유**: 이미 API 키가 있고, 무료 티어 제공
- **용도**: 이미지 분석 + 프롬프트 생성 보조
- **구현**: Gemini로 요청 분석 → 로컬 모델로 실제 편집

### 옵션 2: Replicate API (로컬 GPU 없을 때)
- **이유**: 현재 사용 중인 모델과 동일
- **용도**: 로컬 모델 대체
- **구현**: Replicate로 InstructPix2Pix + GFPGAN 실행

### 옵션 3: 하이브리드 방식
- **로컬 모델**: 기본 편집 (빠르고 무료)
- **API 모델**: 복잡한 요청 또는 로컬 실패 시 fallback

---

## 구현 예시: Gemini 보조 기능 추가

```python
# enhancement_server.py에 추가

async def analyze_request_with_gemini(instruction: str, image: Image.Image):
    """Gemini로 요청 분석 및 프롬프트 개선"""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None
        
        client = genai.Client(api_key=api_key)
        
        # 이미지를 base64로 변환
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        prompt = f"""
        Analyze this image and the user's request: "{instruction}"
        
        Please provide:
        1. What specific edits are needed?
        2. What parts of the image should be preserved?
        3. An improved English prompt for image editing
        
        Format your response as JSON:
        {{
            "edits": ["edit1", "edit2"],
            "preserve": ["face", "background"],
            "improved_prompt": "English prompt here"
        }}
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[
                types.Part.from_bytes(
                    data=base64.b64decode(img_base64),
                    mime_type="image/png"
                ),
                prompt
            ]
        )
        
        # JSON 파싱
        import json
        result = json.loads(response.text)
        return result.get("improved_prompt", None)
        
    except Exception as e:
        print(f"Gemini 분석 실패: {e}")
        return None
```

---

## 결론

**현재 상황**: 로컬 모델(InstructPix2Pix + GFPGAN)이 잘 작동 중

**추가 고려사항**:
1. **Gemini API**: 이미 있으니 프롬프트 개선 보조로 활용
2. **Replicate API**: 로컬 GPU 문제 시 대체 옵션
3. **하이브리드**: 로컬 기본 + API fallback

**권장사항**: 
- 현재 로컬 모델 유지
- 필요시 Gemini로 프롬프트 개선 보조 기능 추가
- 로컬 GPU 문제 시 Replicate API 고려



