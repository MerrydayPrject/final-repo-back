# 배경 변경 가이드

## 질문: 배경 이미지를 제공하면 그대로 바꿔줄 수 있나요?

### 답변: InstructPix2Pix만으로는 제한적입니다

**InstructPix2Pix의 배경 변경 방식:**
- ✅ **텍스트로 배경 변경**: 가능
  - 예: "change background to beach"
  - 예: "change background to garden"
  - 예: "make background blur"
- ❌ **특정 배경 이미지로 교체**: 직접적으로는 불가능
  - InstructPix2Pix는 텍스트 프롬프트만 받음
  - 배경 이미지를 직접 입력으로 받지 않음

## 해결 방법

### 방법 1: 텍스트로 배경 변경 (InstructPix2Pix)
```python
# 텍스트로 배경 변경
prompt = "change background to beautiful beach with ocean"
result = pipe(prompt, image=image).images[0]
```

**장점**: 간단하고 빠름  
**단점**: 원하는 정확한 배경 이미지와 다를 수 있음

### 방법 2: 배경 제거 후 합성 (추천)
```python
# 1. 배경 제거 (기존 세그멘테이션 사용)
person_image = remove_background(original_image)

# 2. 새 배경 이미지와 합성
result = composite_images(person_image, background_image)
```

**장점**: 정확한 배경 이미지 사용 가능  
**단점**: 배경 제거 품질에 의존

### 방법 3: Stable Diffusion Inpainting + ControlNet
```python
# 1. 배경 영역 마스크 생성
mask = create_background_mask(image)

# 2. 새 배경 이미지를 참조하여 Inpainting
result = inpainting_pipe(
    prompt="beautiful background",
    image=image,
    mask_image=mask,
    control_image=background_reference_image  # 참조 이미지
).images[0]
```

**장점**: 배경 이미지를 참조하여 변경 가능  
**단점**: 구현이 복잡함

### 방법 4: IP-Adapter 사용
```python
# 배경 이미지를 스타일 참조로 사용
result = ip_adapter_pipe(
    prompt="change background",
    image=image,
    ip_adapter_image=background_image  # 참조 배경 이미지
).images[0]
```

**장점**: 배경 이미지 스타일을 참조하여 변경  
**단점**: 정확히 동일한 배경은 아님

## 추천 구현 방식

### 현재 프로젝트에 맞는 방법: 배경 제거 + 합성

이미 배경 제거 기능(SegFormer)이 있으므로:

```python
# 1. 배경 제거 (기존 API 사용)
person_segmented = await remove_background(original_image)

# 2. 사용자가 제공한 배경 이미지와 합성
result = composite_person_background(person_segmented, user_background_image)

# 3. 필요시 InstructPix2Pix로 추가 보정
if user_instruction:
    result = instruct_pix2pix(result, user_instruction)
```

## API 설계 예시

### 옵션 1: 텍스트로만 배경 변경
```python
@app.post("/api/enhance-image")
async def enhance_image(
    file: UploadFile = File(...),
    instruction: str = Form("")  # "배경을 해변으로"
):
    # InstructPix2Pix로 텍스트 기반 배경 변경
    prompt = translate_instruction(instruction)
    result = pipe(prompt, image=image).images[0]
```

### 옵션 2: 배경 이미지 제공 가능
```python
@app.post("/api/enhance-image")
async def enhance_image(
    file: UploadFile = File(...),
    instruction: str = Form(""),
    background_image: UploadFile = File(None)  # 배경 이미지 (선택적)
):
    if background_image:
        # 배경 이미지 제공 시: 제거 후 합성
        person = await remove_background(file)
        result = composite(person, background_image)
    else:
        # 텍스트만 제공 시: InstructPix2Pix 사용
        prompt = translate_instruction(instruction)
        result = pipe(prompt, image=image).images[0]
```

## 결론

**현재 상황:**
- ✅ **텍스트로 배경 변경**: InstructPix2Pix로 가능
- ❌ **배경 이미지로 직접 교체**: InstructPix2Pix만으로는 불가능

**추천:**
1. **텍스트 요청**: InstructPix2Pix 사용
2. **배경 이미지 제공**: 기존 배경 제거 기능 + 합성 사용

**구현 우선순위:**
1. 먼저 텍스트 기반 배경 변경 구현 (InstructPix2Pix)
2. 필요시 배경 이미지 합성 기능 추가 (기존 세그멘테이션 활용)




