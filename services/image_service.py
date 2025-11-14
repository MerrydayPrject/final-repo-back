"""이미지 전처리 서비스"""
import numpy as np
from PIL import Image


def preprocess_dress_image(dress_img: Image.Image, target_size: int = 1024) -> Image.Image:
    """
    드레스 이미지를 전처리하여 배경 정보를 제거하고 중앙 정렬합니다.
    
    Args:
        dress_img: 원본 드레스 이미지 (PIL Image)
        target_size: 출력 이미지 크기 (정사각형)
    
    Returns:
        전처리된 드레스 이미지 (흰색 배경에 중앙 정렬)
    """
    # RGB로 변환 (투명도 채널이 있을 경우를 대비)
    if dress_img.mode == 'RGBA':
        # 알파 채널을 사용하여 드레스 영역 감지
        alpha = dress_img.split()[3]
        bbox = alpha.getbbox()  # 투명하지 않은 영역의 경계 상자
        
        if bbox:
            # 드레스 영역만 크롭
            dress_cropped = dress_img.crop(bbox)
        else:
            dress_cropped = dress_img
    else:
        dress_cropped = dress_img
    
    # 드레스 이미지 크기 조정 (비율 유지) - 더 크게 표시
    dress_cropped.thumbnail((target_size * 0.95, target_size * 0.95), Image.Resampling.LANCZOS)
    
    # 흰색 배경 생성
    white_bg = Image.new('RGB', (target_size, target_size), (255, 255, 255))
    
    # 드레스를 중앙에 배치
    dress_rgb = dress_cropped.convert('RGB')
    offset_x = (target_size - dress_rgb.width) // 2
    offset_y = (target_size - dress_rgb.height) // 2
    
    # RGBA 모드인 경우 알파 채널을 마스크로 사용
    if dress_cropped.mode == 'RGBA':
        white_bg.paste(dress_rgb, (offset_x, offset_y), dress_cropped.split()[3])
    else:
        white_bg.paste(dress_rgb, (offset_x, offset_y))
    
    return white_bg


def _create_rtmpose_fallback_mask(height, width, waist_y=None):
    """
    RTMPose 키포인트 기반 Fallback 마스크 생성
    
    Args:
        height: 이미지 높이
        width: 이미지 너비
        waist_y: 허리 Y 좌표 (None이면 기본값 사용)
    
    Returns:
        상체 마스크 (numpy array, uint8)
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if waist_y is not None and waist_y > 0:
        # 허리 위치 + 20px부터 하체로 간주
        cutoff_y = min(waist_y + 20, height)
    else:
        # 이미지 하단 60%를 하체로 간주 (상단 40%가 상체)
        cutoff_y = int(height * 0.4)
    
    mask[:cutoff_y, :] = 255
    return mask

