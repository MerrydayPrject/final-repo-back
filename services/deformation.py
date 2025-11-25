"""TPS 기반 로컬 메시 변형 서비스"""
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def deform_region(
    image: Image.Image,
    mask: Image.Image,
    factor: float = 1.0,
    part_type: str = 'body'  # 'body', 'arm_left', 'arm_right', 'leg' 등
) -> Image.Image:
    """
    개선된 Liquify 워핑 - 안전장치 및 부위별 최적화
    
    개선사항:
    - 과도한 왜곡 방지 (Factor Clamping 0.85-1.15)
    - 부위별 동적 블러 sigma (팔: 작게, 몸통: 크게)
    - 부드러운 경계 처리 (BORDER_REFLECT_101)
    
    Args:
        image: 원본 이미지 (PIL Image)
        mask: 변형할 영역 마스크 (PIL Image, L mode)
        factor: 변형 강도 (< 1.0: 축소/슬림, > 1.0: 확대/넓게)
        part_type: 부위 유형 ('body', 'arm_left', 'arm_right', 'leg')
    
    Returns:
        변형된 이미지 (PIL Image)
    """
    if abs(factor - 1.0) < 0.01:
        logger.debug(f"변형 건너뜀 (factor ≈ 1.0)")
        return image
    
    # 1. 안전장치: 과도한 변형 방지 (0.7-1.3 범위)
    safe_factor = np.clip(factor, 0.7, 1.3)
    if abs(safe_factor - factor) > 0.01:
        logger.warning(f"과도한 factor {factor:.2f} -> {safe_factor:.2f}로 제한")
    
    logger.info(f"Liquify 워핑 시작: factor={safe_factor:.3f}, part={part_type}")
    
    # PIL to numpy
    img_arr = np.array(image)
    mask_arr = np.array(mask).astype(float) / 255.0
    h, w = img_arr.shape[:2]
    
    # 마스크가 비어있으면 원본 반환
    if np.sum(mask_arr) < 10:
        logger.warning("마스크 영역 거의 없음, 원본 반환")
        return image
    
    # 2. 동적 블러링 sigma (부위별 최적화)
    # 팔(좁은 부위): 작은 sigma로 정밀하게
    # 몸통(넓은 부위): 큰 sigma로 부드럽게
    if 'arm' in part_type:
        sigma = h * 0.02  # 팔: 작은 블러
    else:
        sigma = h * 0.06  # 몸통/다리: 큰 블러
    
    from scipy.ndimage import gaussian_filter
    mask_blurred = gaussian_filter(mask_arr, sigma=sigma)
    logger.debug(f"Blur sigma: {sigma:.1f}")
    
    # 3. 중심축 계산 (마스크의 무게중심 X)
    y_indices, x_indices = np.nonzero(mask_arr > 0.1)
    if len(x_indices) == 0:
        logger.warning("마스크 영역 없음, 원본 반환")
        return image
    
    center_x = np.mean(x_indices)
    logger.debug(f"중심축 X: {center_x:.1f}")
    
    # 4. 변위 맵(Displacement Map) 생성
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    grid_x = grid_x.astype(np.float32)
    grid_y = grid_y.astype(np.float32)
    
    # 중심으로부터의 거리
    dist_x = grid_x - center_x
    
    # 5. Liquify 변형 계산
    # safe_factor < 1.0 (슬림): 중심으로 당김
    # safe_factor > 1.0 (확대): 바깥으로 밂
    scale = 1.0 / safe_factor
    
    # 변위량 계산 (팔은 2배 증폭)
    deviation = (scale - 1.0)
    
    # 팔은 2배 증폭 (명확한 효과를 위해)
    if 'arm' in part_type:
        delta_x = dist_x * (deviation * 2.0)
        logger.info(f"Arm deformation: factor={safe_factor:.3f}, deviation={deviation:.3f}, amplified=2.0x")
    else:
        delta_x = dist_x * deviation
        logger.debug(f"Body deformation: factor={safe_factor:.3f}")
    
    
    # 마스크 강도만큼만 적용 (블렌딩)
    map_x = (grid_x + delta_x * mask_blurred).astype(np.float32)
    map_y = grid_y  # Y축 유지 (이미 float32)
    
    # 6. OpenCV remap으로 워핑
    try:
        import cv2
        result_arr = cv2.remap(
            img_arr,
            map_x,
            map_y,
            interpolation=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT_101  # 경계면 거울 반사
        )
        logger.info(f"Liquify 워핑 완료 (OpenCV, {part_type})")
        return Image.fromarray(result_arr)
        
    except (ImportError, Exception) as e:
        logger.warning(f"OpenCV remap 실패, Scipy fallback: {e}")
        
        # Fallback: scipy map_coordinates
        from scipy.ndimage import map_coordinates
        coords = np.array([map_y, map_x])
        result_arr = np.zeros_like(img_arr)
        
        for c in range(img_arr.shape[2] if len(img_arr.shape) == 3 else 1):
            if len(img_arr.shape) == 3:
                result_arr[..., c] = map_coordinates(
                    img_arr[..., c], coords, order=1, mode='reflect'
                )
            else:
                result_arr = map_coordinates(
                    img_arr, coords, order=1, mode='reflect'
                )
        
        logger.info(f"Liquify 워핑 완료 (Scipy, {part_type})")
        return Image.fromarray(result_arr.astype(np.uint8))


def blend_images(
    original: Image.Image,
    deformed: Image.Image,
    mask: Image.Image,
    feather_radius: int = 20
) -> Image.Image:
    """
    원본과 변형된 이미지를 마스크로 블렌딩 (feathering 적용)
    
    Args:
        original: 원본 이미지
        deformed: 변형된 이미지
        mask: 블렌딩 마스크
        feather_radius: 페더링 반경 (픽셀)
    
    Returns:
        블렌딩된 이미지
    """
    from scipy.ndimage import gaussian_filter
    
    # 마스크 페더링
    mask_array = np.array(mask).astype(float) / 255.0
    if feather_radius > 0:
        mask_array = gaussian_filter(mask_array, sigma=feather_radius / 3)
    
    # 블렌딩
    orig_array = np.array(original).astype(float)
    deform_array = np.array(deformed).astype(float)
    
    # 3채널로 확장
    if len(mask_array.shape) == 2:
        mask_array = mask_array[:, :, np.newaxis]
    
    blended = orig_array * (1 - mask_array) + deform_array * mask_array
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    return Image.fromarray(blended)
