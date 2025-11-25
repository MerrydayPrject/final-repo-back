"""인물 전체 세그멘테이션 서비스 - SegFormer B2 기반"""
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import logging

logger = logging.getLogger(__name__)


class HumanSegmentation:
    """SegFormer B2를 사용한 인물 세그멘테이션"""
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Args:
            device: 'cuda' or 'cpu'
        """
        self.device = device
        logger.info(f"HumanSegmentation 초기화 중... (device: {self.device})")
        
        try:
            # SegFormer B2 모델 로드 (ADE20K 데이터셋으로 사전학습됨)
            self.processor = SegformerImageProcessor.from_pretrained(
                "nvidia/segformer-b2-finetuned-ade-512-512"
            )
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b2-finetuned-ade-512-512"
            ).to(device)
            self.model.eval()
            logger.info("SegFormer B2 모델 로드 완료")
        except Exception as e:
            logger.error(f"SegFormer 모델 로드 실패: {e}")
            raise
    
    @torch.no_grad()
    def get_person_mask(self, image: Image.Image) -> Image.Image:
        """
        이미지에서 인물 영역 마스크 추출
        
        Args:
            image: PIL Image (RGB)
        
        Returns:
            PIL Image: 이진 마스크 (255=인물, 0=배경)
        """
        try:
            # 입력 이미지 전처리
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # 모델 추론
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # 원본 이미지 크기로 업샘플링
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=image.size[::-1],  # (height, width)
                mode="bilinear",
                align_corners=False
            )
            
            # 가장 높은 확률의 클래스 선택
            pred = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
            
            # ADE20K 데이터셋에서 person 클래스는 12번
            # 참고: https://github.com/CSAILVision/ADE20K/blob/main/data/index_ade20k.pkl
            person_mask = (pred == 12).astype(np.uint8) * 255
            
            logger.info(f"인물 마스크 생성 완료: {person_mask.sum() / 255} 픽셀")
            
            return Image.fromarray(person_mask, mode='L')
            
        except Exception as e:
            logger.error(f"인물 마스크 생성 실패: {e}")
            raise
