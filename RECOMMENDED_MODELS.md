# 이미지 보정 추천 모델 및 링크

## 텍스트 지시 기반 편집 (추천)

### 1. InstructPix2Pix ⭐⭐⭐
- **Hugging Face**: https://huggingface.co/timbrooks/instruct-pix2pix
- **GitHub**: https://github.com/timothybrooks/instruct-pix2pix
- **용도**: "어깨를 좁게 해줘" 같은 자연어 명령으로 이미지 편집

### 2. MagicBrush
- **GitHub**: https://github.com/OSU-NLP-Group/MagicBrush
- **용도**: 정밀한 텍스트 지시 기반 인페인팅

### 3. IP-Adapter
- **Hugging Face**: https://huggingface.co/h94/IP-Adapter
- **GitHub**: https://github.com/tencent-ailab/IP-Adapter
- **용도**: 이미지 스타일 유지하면서 텍스트 지시 반영

## 정밀한 형태 조작

### 4. ControlNet
- **Hugging Face**: https://huggingface.co/lllyasviel/ControlNet-v1-1
- **GitHub**: https://github.com/lllyasviel/ControlNet
- **OpenPose**: https://huggingface.co/lllyasviel/sd-controlnet-openpose
- **용도**: 포즈/형태를 정밀하게 제어하여 이미지 편집

### 5. DensePose
- **GitHub**: https://github.com/facebookresearch/DensePose
- **용도**: 인체 3D 형태 추정 및 조작

## 품질 향상

### 6. Real-ESRGAN
- **GitHub**: https://github.com/xinntao/Real-ESRGAN
- **Hugging Face**: https://huggingface.co/spaces/akhaliq/Real-ESRGAN
- **용도**: 이미지 업스케일링 및 블러 제거

### 7. GFPGAN
- **GitHub**: https://github.com/TencentARC/GFPGAN
- **Hugging Face**: https://huggingface.co/spaces/Xintao/GFPGAN
- **용도**: 얼굴 복원 및 향상

### 8. CodeFormer
- **GitHub**: https://github.com/sczhou/CodeFormer
- **Hugging Face**: https://huggingface.co/spaces/sczhou/CodeFormer
- **용도**: 얼굴 복원 (GFPGAN보다 자연스러움)

## Stable Diffusion 기반

### 9. Stable Diffusion Inpainting
- **Hugging Face**: https://huggingface.co/runwayml/stable-diffusion-inpainting
- **용도**: 특정 영역만 선택적으로 수정

### 10. Stable Diffusion XL
- **Hugging Face**: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
- **용도**: 고해상도 이미지 생성/편집

## 최종 추천 순위

1. **InstructPix2Pix** - 가장 간단하고 빠른 구현
2. **Real-ESRGAN** - 품질 향상 필수
3. **ControlNet + OpenPose** - 정밀한 형태 조작 필요 시
4. **GFPGAN/CodeFormer** - 얼굴 보정 필요 시




