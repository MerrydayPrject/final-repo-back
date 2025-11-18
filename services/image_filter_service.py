"""이미지 필터 및 스티커 서비스"""
import io
from typing import Optional, List, Dict
from PIL import Image, ImageEnhance, ImageOps, ImageDraw, ImageFilter


# 필터 프리셋 정의
FILTER_PRESETS = {
    "grayscale": {
        "type": "grayscale"
    },
    "vintage": {
        "brightness": 1.1,
        "contrast": 0.8,
        "saturation": 0.7,
        "color_temperature": "warm"
    },
    "warm": {
        "brightness": 1.15,
        "contrast": 1.1,
        "saturation": 1.05,
        "color_temperature": "warm"
    },
    "cool": {
        "brightness": 0.95,
        "contrast": 1.15,
        "saturation": 1.1,
        "color_temperature": "cool"
    },
    "high_contrast": {
        "brightness": 1.05,
        "contrast": 1.4,
        "saturation": 1.15,
        "color_temperature": "neutral"
    }
}

# 프레임 프리셋 정의
FRAME_PRESETS = {
    "none": {
        "frame_type": "none"
    },
    "black": {
        "frame_type": "solid",
        "frame_color": "#000000",
        "frame_width": 15
    },
    "white": {
        "frame_type": "solid",
        "frame_color": "#FFFFFF",
        "frame_width": 15
    },
    "gold": {
        "frame_type": "solid",
        "frame_color": "#FFD700",
        "frame_width": 20
    },
    "red": {
        "frame_type": "solid",
        "frame_color": "#FF0000",
        "frame_width": 15
    },
    "blue": {
        "frame_type": "solid",
        "frame_color": "#0066FF",
        "frame_width": 15
    }
}


def apply_filter_preset(image: Image.Image, preset_name: str) -> Image.Image:
    """
    프리셋 필터 적용
    
    Args:
        image: 원본 이미지 (PIL Image)
        preset_name: 필터 프리셋 이름 ("none", "grayscale", "vintage", "warm", "cool", "high_contrast")
    
    Returns:
        필터가 적용된 이미지
    """
    if preset_name == "none" or preset_name not in FILTER_PRESETS:
        return image.copy()
    
    result = image.copy()
    preset = FILTER_PRESETS[preset_name]
    
    # 흑백 필터
    if preset.get("type") == "grayscale":
        if result.mode == "L":
            # 이미 흑백이면 RGB로 변환
            result = result.convert("RGB")
        else:
            result = ImageOps.grayscale(result)
            # RGB로 변환하여 반환 (다른 처리와 호환성 유지)
            result = result.convert("RGB")
        return result
    
    # 색감 조정 필터
    # 밝기 조정
    if "brightness" in preset:
        enhancer = ImageEnhance.Brightness(result)
        result = enhancer.enhance(preset["brightness"])
    
    # 대비 조정
    if "contrast" in preset:
        enhancer = ImageEnhance.Contrast(result)
        result = enhancer.enhance(preset["contrast"])
    
    # 채도 조정
    if "saturation" in preset:
        enhancer = ImageEnhance.Color(result)
        result = enhancer.enhance(preset["saturation"])
    
    # 색온도 조정 (RGB 채널 조정으로 시뮬레이션)
    color_temp = preset.get("color_temperature", "neutral")
    if color_temp == "warm":
        # 따뜻한 톤: 빨강/노랑 채널 증가
        r, g, b = result.split()
        # 빈티지 필터는 핑크색을 줄이기 위해 빨강 채널 증가량을 줄임
        if preset_name == "vintage":
            r = ImageEnhance.Brightness(r).enhance(1.05)  # 1.1 -> 1.05로 감소
        else:
            r = ImageEnhance.Brightness(r).enhance(1.1)
        result = Image.merge("RGB", (r, g, b))
    elif color_temp == "cool":
        # 차가운 톤: 파랑 채널 증가
        r, g, b = result.split()
        b = ImageEnhance.Brightness(b).enhance(1.1)
        result = Image.merge("RGB", (r, g, b))
    
    return result


def apply_frame_preset(image: Image.Image, preset_name: str) -> Image.Image:
    """
    프레임 프리셋 적용
    
    Args:
        image: 원본 이미지 (PIL Image)
        preset_name: 프레임 프리셋 이름
    
    Returns:
        프레임이 적용된 이미지
    """
    if preset_name == "none" or preset_name not in FRAME_PRESETS:
        return image.copy()
    
    preset = FRAME_PRESETS[preset_name]
    return apply_frame(
        image,
        preset["frame_type"],
        preset.get("frame_color", "#000000"),
        preset.get("frame_width", 10),
        None
    )


def apply_frame(
    image: Image.Image,
    frame_type: str,
    frame_color: str = "#000000",
    frame_width: int = 10,
    frame_image: Optional[Image.Image] = None
) -> Image.Image:
    """
    프레임 추가
    
    Args:
        image: 원본 이미지 (PIL Image)
        frame_type: 프레임 타입 ("none", "solid", "decorative")
        frame_color: 단색 프레임 색상 (hex 코드)
        frame_width: 프레임 두께
        frame_image: 장식 프레임 이미지 (PIL Image)
    
    Returns:
        프레임이 추가된 이미지
    """
    if frame_type == "none":
        return image.copy()
    
    result = image.copy()
    
    if frame_type == "solid":
        # 단색 프레임
        width, height = result.size
        draw = ImageDraw.Draw(result)
        
        # hex 색상을 RGB로 변환
        hex_color = frame_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # 테두리 그리기
        for i in range(frame_width):
            draw.rectangle(
                [i, i, width - 1 - i, height - 1 - i],
                outline=rgb,
                width=1
            )
    
    elif frame_type == "decorative" and frame_image:
        # 장식 프레임
        frame = frame_image.copy()
        frame = frame.convert("RGBA")
        
        # 프레임을 원본 이미지 크기에 맞춰 리사이즈
        frame = frame.resize(result.size, Image.Resampling.LANCZOS)
        
        # 프레임과 이미지 합성
        if result.mode != "RGBA":
            result = result.convert("RGBA")
        
        result = Image.alpha_composite(result, frame)
        result = result.convert("RGB")
    
    return result


def apply_sticker(
    base_image: Image.Image,
    sticker_image: Image.Image,
    x: int = -1,
    y: int = -1,
    width: Optional[int] = None,
    height: Optional[int] = None,
    opacity: float = 1.0,
    rotation: float = 0.0
) -> Image.Image:
    """
    스티커 적용
    
    Args:
        base_image: 기본 이미지 (PIL Image)
        sticker_image: 스티커 이미지 (PIL Image)
        x: X 좌표 (-1이면 중앙)
        y: Y 좌표 (-1이면 중앙)
        width: 스티커 너비 (None이면 원본 크기)
        height: 스티커 높이 (None이면 원본 크기)
        opacity: 투명도 (0.0 ~ 1.0)
        rotation: 회전 각도 (0 ~ 360)
    
    Returns:
        스티커가 적용된 이미지
    """
    result = base_image.copy()
    sticker = sticker_image.copy()
    
    # 스티커를 RGBA로 변환
    if sticker.mode != "RGBA":
        sticker = sticker.convert("RGBA")
    
    # 크기 조정
    if width or height:
        original_size = sticker.size
        if width and height:
            new_size = (width, height)
        elif width:
            ratio = width / original_size[0]
            new_size = (width, int(original_size[1] * ratio))
        else:
            ratio = height / original_size[1]
            new_size = (int(original_size[0] * ratio), height)
        
        sticker = sticker.resize(new_size, Image.Resampling.LANCZOS)
    
    # 회전
    if rotation != 0:
        sticker = sticker.rotate(rotation, expand=True)
    
    # 투명도 적용
    if opacity < 1.0:
        alpha = sticker.split()[3]
        alpha = alpha.point(lambda p: int(p * opacity))
        sticker.putalpha(alpha)
    
    # 위치 계산
    sticker_width, sticker_height = sticker.size
    base_width, base_height = result.size
    
    if x == -1:
        x = (base_width - sticker_width) // 2
    if y == -1:
        y = (base_height - sticker_height) // 2
    
    # 기본 이미지를 RGBA로 변환
    if result.mode != "RGBA":
        result = result.convert("RGBA")
    
    # 스티커 합성
    result.paste(sticker, (x, y), sticker)
    
    return result


def process_image_with_filters_and_stickers(
    image: Image.Image,
    filter_preset: str = "none",
    frame_options: Optional[Dict] = None,
    stickers: Optional[List[Dict]] = None
) -> Image.Image:
    """
    필터와 스티커를 통합 처리하는 함수
    
    Args:
        image: 원본 이미지 (PIL Image)
        filter_preset: 필터 프리셋 이름
        frame_options: 프레임 옵션 딕셔너리
        stickers: 스티커 옵션 리스트
    
    Returns:
        처리된 이미지
    """
    result = image.copy()
    
    # 1. 필터 적용
    result = apply_filter_preset(result, filter_preset)
    
    # 2. 프레임 적용
    if frame_options:
        frame_type = frame_options.get("frame_type", "none")
        frame_color = frame_options.get("frame_color", "#000000")
        frame_width = frame_options.get("frame_width", 10)
        frame_image = frame_options.get("frame_image")
        
        result = apply_frame(
            result,
            frame_type,
            frame_color,
            frame_width,
            frame_image
        )
    
    # 3. 스티커 적용
    if stickers:
        for sticker_data in stickers:
            sticker_image = sticker_data.get("sticker_image")
            if sticker_image:
                result = apply_sticker(
                    result,
                    sticker_image,
                    x=sticker_data.get("x", -1),
                    y=sticker_data.get("y", -1),
                    width=sticker_data.get("width"),
                    height=sticker_data.get("height"),
                    opacity=sticker_data.get("opacity", 1.0),
                    rotation=sticker_data.get("rotation", 0.0)
                )
    
    return result
