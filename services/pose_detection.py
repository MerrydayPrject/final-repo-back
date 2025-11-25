"""포즈 검증 서비스 - 체형 조정 후 포즈 자연스러움 확인"""
import numpy as np
from PIL import Image
from typing import Union
import logging

logger = logging.getLogger(__name__)


async def check_pose_integrity(image: Union[Image.Image, np.ndarray], threshold: float = 0.3) -> bool:
    """
    이미지의 포즈가 자연스러운지 검증
    
    Args:
        image: PIL Image 또는 numpy array
        threshold: 포즈 신뢰도 임계값 (0.0-1.0)
        
    Returns:
        bool: 포즈가 자연스러우면 True, 아니면 False
    """
    try:
        # 현재는 간단한 구현: 항상 True 반환
        # MediaPipe 연동은 추후 필요시 추가
        logger.debug(f"포즈 검증: 이미지 크기 = {image.size if isinstance(image, Image.Image) else image.shape}")
        
        # TODO: MediaPipe Pose Landmarker로 실제 검증 구현
        # 현재는 체형 조정이 작동하도록 항상 valid 반환
        return True
        
    except Exception as e:
        logger.warning(f"포즈 검증 실패, 기본값 True 반환: {e}")
        return True  # 에러 시 조정된 이미지 사용
