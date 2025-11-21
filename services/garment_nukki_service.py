"""의상 누끼(배경 제거) 서비스"""
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from typing import Optional

from core.model_loader import get_processor, get_model


def remove_garment_background(garment_img: Image.Image) -> Image.Image:
    """
    의상 이미지에서 배경을 제거하고 드레스만 추출
    
    기존 `/api/segment` 엔드포인트 로직을 참고하여 구현되었습니다.
    SegFormer 모델을 사용하여 드레스(레이블 7)만 추출하고 배경을 제거합니다.
    
    Args:
        garment_img: 원본 의상 이미지 (PIL Image, RGB)
    
    Returns:
        배경 제거된 의상 이미지 (PIL Image, RGBA)
        배경이 투명한 알파 채널을 가진 이미지입니다.
    
    Raises:
        RuntimeError: 모델이 로드되지 않았거나 처리 중 오류가 발생한 경우
    """
    processor = get_processor()
    model = get_model()
    
    if processor is None or model is None:
        raise RuntimeError("SegFormer 모델이 로드되지 않았습니다. 서버를 재시작해주세요.")
    
    # 이미지를 RGB로 변환
    if garment_img.mode != 'RGB':
        garment_img = garment_img.convert('RGB')
    
    original_size = garment_img.size
    
    # 모델 추론
    inputs = processor(images=garment_img, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.cpu()
    
    # 업샘플링
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=original_size[::-1],  # (height, width)
        mode="bilinear",
        align_corners=False,
    )
    
    # 세그멘테이션 마스크 생성
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
    
    # 드레스 마스크 생성 (레이블 7: Dress)
    dress_mask = (pred_seg == 7).astype(np.uint8) * 255
    
    # 원본 이미지를 numpy 배열로 변환
    image_array = np.array(garment_img)
    
    # 누끼 이미지 생성 (RGBA)
    result_image = np.zeros((image_array.shape[0], image_array.shape[1], 4), dtype=np.uint8)
    result_image[:, :, :3] = image_array  # RGB 채널
    result_image[:, :, 3] = dress_mask    # 알파 채널
    
    # PIL 이미지로 변환
    result_pil = Image.fromarray(result_image, mode='RGBA')
    
    return result_pil

