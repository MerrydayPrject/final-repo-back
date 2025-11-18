"""이미지 필터 및 스티커 스키마"""
from typing import Optional, List
from pydantic import BaseModel


class FilterPreset:
    """필터 프리셋 타입"""
    NONE = "none"
    GRAYSCALE = "grayscale"
    VINTAGE = "vintage"
    WARM = "warm"
    COOL = "cool"
    HIGH_CONTRAST = "high_contrast"


class FrameOptions(BaseModel):
    """프레임 옵션 스키마"""
    frame_type: str = "none"  # "none", "solid", "decorative"
    frame_color: Optional[str] = "#000000"  # hex 코드
    frame_width: int = 10  # 프레임 두께
    frame_image: Optional[bytes] = None  # 장식 프레임 이미지 (bytes)


class StickerOptions(BaseModel):
    """스티커 옵션 스키마"""
    sticker_image: bytes  # 스티커 이미지 (bytes)
    x: int = -1  # X 좌표, -1이면 중앙
    y: int = -1  # Y 좌표, -1이면 중앙
    width: Optional[int] = None  # 스티커 너비
    height: Optional[int] = None  # 스티커 높이
    opacity: float = 1.0  # 투명도, 0.0 ~ 1.0
    rotation: float = 0.0  # 회전 각도, 0 ~ 360


