"""체형 보정 서비스 - Segmentation + Liquify Deformation 기반"""
from PIL import Image
from typing import Optional
import logging
import torch

from .segmentation import HumanSegmentation
from .body_parts import BodyPartsSegmentor
from .deformation import deform_region

logger = logging.getLogger(__name__)


class BodyAdjustmentService:
    """
    새로운 체형 보정 서비스
    
    전체 이미지 왜곡 없이 부위별로만 자연스럽게 조정하는 서비스
    
    동작 원리:
    1. SegFormer로 전체 인물 세그멘테이션
    2. 부위별 세그멘테이션 (BBox 기반)
    3. 각 부위별로 Liquify 기반 로컬 메시 변형
    4. 자연스러운 블렌딩
    """
    
    _instance = None
    
    def __new__(cls):
        """싱글톤 패턴으로 모델 중복 로딩 방지"""
        if cls._instance is None:
            cls._instance = super(BodyAdjustmentService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, device: str = None):
        """
        Args:
            device: 'cuda' or 'cpu' (None이면 자동 선택)
        """
        if self._initialized:
            return
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        logger.info(f"BodyAdjustmentService 초기화 중... (device: {device})")
        
        try:
            # 전체 인물 세그멘테이션
            self.segmentation = HumanSegmentation(device=device)
            
            # 부위별 세그멘테이션 (모델 파일 없으면 fallback)
            self.body_parts = BodyPartsSegmentor(
                model_path="models/bodyparts_unet.pth",
                device=device
            )
            
            self._initialized = True
            logger.info("BodyAdjustmentService 초기화 완료")
            
        except Exception as e:
            logger.error(f"BodyAdjustmentService 초기화 실패: {e}")
            raise
    
    async def adjust(
        self,
        image: Image.Image,
        shoulder_factor: float = 1.0,
        waist_factor: float = 1.0,
        hip_factor: float = 1.0,
        chest_factor: float = 1.0,
        arm_factor: float = 1.0,
        leg_factor: float = 1.0,
        neck_factor: float = 1.0
    ) -> Image.Image:
        """
        체형 조정 수행
        
        Args:
            image: 원본 이미지 (PIL Image)
            shoulder_factor: 어깨 조정 (< 1.0: 좁게, > 1.0: 넓게)
            waist_factor: 허리 조정 (< 1.0: 슬림, > 1.0: 넓게)
            hip_factor: 엉덩이 조정 (< 1.0: 슬림, > 1.0: 넓게)
            chest_factor: 가슴 조정 (< 1.0: 작게, > 1.0: 크게)
            arm_factor: 팔 조정 (< 1.0: 얇게, > 1.0: 굵게)
            leg_factor: 다리 조정 (< 1.0: 얇게, > 1.0: 굵게)
            neck_factor: 목 조정 (< 1.0: 얇게, > 1.0: 굵게)
        
        Returns:
            조정된 이미지 (PIL Image)
        """
        logger.info(f"체형 조정 시작: shoulder={shoulder_factor:.2f}, waist={waist_factor:.2f}, "
                   f"hip={hip_factor:.2f}, chest={chest_factor:.2f}, arm={arm_factor:.2f}, "
                   f"leg={leg_factor:.2f}, neck={neck_factor:.2f}")
        
        try:
            # 1. 전체 인물 마스크 추출
            person_mask = self.segmentation.get_person_mask(image)
            
            # 2. 부위별 마스크 추출 (BBox 기반, 팔 좌우 분리)
            parts = self.body_parts.segment(image, person_mask=person_mask)
            
            # 3. 결과 이미지 (순차 적용)
            current_image = image.copy()
            
            # 적용 순서: 중심부(허리/가슴) -> 외곽(어깨/골반) -> 다리 -> 팔 (좌우 개별)
            
            # (1) 가슴/허리/골반 (몸통 라인)
            if abs(chest_factor - 1.0) > 0.01 and 'chest' in parts:
                logger.info(f"가슴 조정 적용: {chest_factor:.2f}")
                current_image = deform_region(current_image, parts['chest'], chest_factor, 'body')
            
            if abs(waist_factor - 1.0) > 0.01 and 'waist' in parts:
                logger.info(f"허리 조정 적용: {waist_factor:.2f}")
                current_image = deform_region(current_image, parts['waist'], waist_factor, 'body')
            
            if abs(hip_factor - 1.0) > 0.01 and 'hip' in parts:
                logger.info(f"엉덩이 조정 적용: {hip_factor:.2f}")
                current_image = deform_region(current_image, parts['hip'], hip_factor, 'body')
            
            # (2) 어깨 (상체 프레임)
            if abs(shoulder_factor - 1.0) > 0.01 and 'shoulder' in parts:
                logger.info(f"어깨 조정 적용: {shoulder_factor:.2f}")
                current_image = deform_region(current_image, parts['shoulder'], shoulder_factor, 'body')
            
            # (3) 다리 (드레스)
            if abs(leg_factor - 1.0) > 0.01 and 'leg' in parts:
                logger.info(f"다리 조정 적용: {leg_factor:.2f}")
                current_image = deform_region(current_image, parts['leg'], leg_factor, 'leg')
            
            # (4) 팔 - 좌/우 개별 적용 (뽀빠이 팔 현상 방지!)
            if abs(arm_factor - 1.0) > 0.01:
                if 'arm_left' in parts:
                    logger.info(f"왼팔 조정 적용: {arm_factor:.2f}")
                    current_image = deform_region(current_image, parts['arm_left'], arm_factor, 'arm_left')
                
                if 'arm_right' in parts:
                    logger.info(f"오른팔 조정 적용: {arm_factor:.2f}")
                    current_image = deform_region(current_image, parts['arm_right'], arm_factor, 'arm_right')
            
            # (5) 목 (선택적)
            if abs(neck_factor - 1.0) > 0.01 and 'neck' in parts:
                logger.info(f"목 조정 적용: {neck_factor:.2f}")
                current_image = deform_region(current_image, parts['neck'], neck_factor, 'body')
            
            logger.info("체형 조정 완료")
            return current_image
            
        except Exception as e:
            logger.error(f"체형 조정 실패: {e}", exc_info=True)
            # 실패 시 원본 반환
            return image


# 전역 서비스 인스턴스
_service_instance: Optional[BodyAdjustmentService] = None


def get_service() -> BodyAdjustmentService:
    """서비스 인스턴스 가져오기 (싱글톤)"""
    global _service_instance
    if _service_instance is None:
        _service_instance = BodyAdjustmentService()
    return _service_instance


async def adjust_body_shape_api(
    image: Image.Image,
    slim_factor: float = 0.9,
    neck_length_factor: float = 1.0,
    neck_thickness_factor: float = 1.0,
    chest_factor: float = 1.0,
    waist_factor: float = 1.0,
    shoulder_factor: float = 1.0,
    hip_factor: float = 1.0,
    leg_factor: float = 1.0,
    arm_factor: float = 1.0,
    **kwargs
) -> Image.Image:
    """
    체형 조정 API (기존 인터페이스 호환)
    
    Args:
        image: PIL Image
        shoulder_factor: 어깨 조정 (0.7-1.3)
        waist_factor: 허리 조정 (0.7-1.3)
        hip_factor: 엉덩이 조정 (0.7-1.3)
        chest_factor: 가슴 조정 (0.7-1.3)
        leg_factor: 다리 조정
        arm_factor: 팔 조정
        neck_length_factor, neck_thickness_factor: 목 조정
        slim_factor: 전체 슬림 조정 (레거시, 무시됨)
    
    Returns:
        조정된 PIL Image
    """
    logger.info(f"체형 조정 API 호출: shoulder={shoulder_factor:.2f}, waist={waist_factor:.2f}, "
               f"hip={hip_factor:.2f}, chest={chest_factor:.2f}")
    
    # 범위 제한 (과도한 변형 방지)
    shoulder_factor = max(0.7, min(1.3, shoulder_factor))
    waist_factor = max(0.7, min(1.3, waist_factor))
    hip_factor = max(0.7, min(1.3, hip_factor))
    chest_factor = max(0.7, min(1.3, chest_factor))
    arm_factor = max(0.7, min(1.3, arm_factor))
    leg_factor = max(0.7, min(1.3, leg_factor))
    neck_factor = max(0.7, min(1.3, neck_thickness_factor))
    
    logger.info(f"범위 제한 후: shoulder={shoulder_factor:.2f}, waist={waist_factor:.2f}, "
               f"hip={hip_factor:.2f}, chest={chest_factor:.2f}")
    
    # 서비스 인스턴스 가져오기
    service = get_service()
    
    # 조정 수행
    result = await service.adjust(
        image=image,
        shoulder_factor=shoulder_factor,
        waist_factor=waist_factor,
        hip_factor=hip_factor,
        chest_factor=chest_factor,
        arm_factor=arm_factor,
        leg_factor=leg_factor,
        neck_factor=neck_factor
    )
    
    return result
