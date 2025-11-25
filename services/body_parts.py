"""부위별 세그멘테이션 서비스"""
import torch
import numpy as np
from PIL import Image
import logging
import os

logger = logging.getLogger(__name__)


class BodyPartsSegmentor:
    """
    부위별 세그멘테이션 (Fallback: BBox 기반 위치 추정)
    
    개선사항:
    - 팔을 arm_left, arm_right로 분리
    - 드레스 하단 보호 (leg 0.92까지만)
    - 허리 영역 최소화 (배경 보호)
    """
    
    def __init__(self, model_path=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model_path = model_path
        self.use_model = False
        
        if model_path and os.path.exists(model_path):
            try:
                self.model = torch.load(model_path, map_location=device)
                self.model.eval()
                self.use_model = True
                logger.info(f"모델 로드 완료: {model_path}")
            except Exception as e:
                logger.warning(f"모델 로드 실패, Fallback 모드 사용: {e}")
        else:
            logger.info("모델 없음, Fallback 모드 사용")
    
    def segment(self, image: Image.Image, person_mask: Image.Image = None) -> dict:
        """부위별 마스크 추출"""
        if self.use_model:
            return self._segment_with_model(image)
        else:
            return self._segment_fallback(image, person_mask)
    
    @torch.no_grad()
    def _segment_with_model(self, image: Image.Image) -> dict:
        """모델 기반 세그멘테이션 (미구현)"""
        raise NotImplementedError("모델 기반 세그멘테이션은 아직 구현되지 않았습니다.")
    
    def _segment_fallback(self, image: Image.Image, person_mask: Image.Image = None) -> dict:
        """
        Fallback: Bounding Box 기반 부위 추정 (개선된 버전 v2)
        
        개선사항:
        - 팔을 arm_left, arm_right로 분리 (뽀빠이 팔 현상 방지)
        - 드레스 하단 보호 (leg 0.92까지만)
        - 허리 영역 좁게 조정 (배경 보호)
        
        인물의 실제 위치와 크기를 기준으로 비율 계산:
        - 머리/목: 0.13-0.18
        - 어깨: 0.15-0.26
        - 가슴: 0.26-0.40
        - 허리: 0.38-0.53 (폭 20-80%)
        - 골반/엉덩이: 0.53-0.65
        - 다리: 0.65-0.92 (드레스 하단 보호)
        - 팔: 좌우 분리 (0.20-0.60)
        """
        w, h = image.size
        
        # 기본 빈 마스크
        if person_mask is None:
            person_mask = Image.new('L', (w, h), 255)
        
        mask_arr = np.array(person_mask)
        
        # ===== Bounding Box 계산 =====
        rows = np.any(mask_arr > 128, axis=1)
        cols = np.any(mask_arr > 128, axis=0)
        
        if not np.any(rows):
            logger.warning("인물이 감지되지 않음. 전체 이미지 기준으로 fallback")
            y_min, y_max = 0, h
            x_min, x_max = 0, w
        else:
            y_indices = np.where(rows)[0]
            x_indices = np.where(cols)[0]
            y_min, y_max = y_indices[0], y_indices[-1]
            x_min, x_max = x_indices[0], x_indices[-1]
        
        p_height = y_max - y_min  # 인물 실제 키
        p_width = x_max - x_min   # 인물 실제 너비
        
        logger.info(f"인물 BBox: y=[{y_min}, {y_max}], x=[{x_min}, {x_max}], height={p_height}, width={p_width}")
        
        parts = {}
        
        def create_mask(y_start_ratio, y_end_ratio, x_start_ratio=0.0, x_end_ratio=1.0):
            """BBox 기준 비율로 마스크 생성 (X축 범위 추가)"""
            m = np.zeros_like(mask_arr)
            ys = int(y_min + p_height * y_start_ratio)
            ye = int(y_min + p_height * y_end_ratio)
            xs = int(x_min + p_width * x_start_ratio)
            xe = int(x_min + p_width * x_end_ratio)
            
            # 경계 체크
            ys = max(0, min(h, ys))
            ye = max(0, min(h, ye))
            xs = max(0, min(w, xs))
            xe = max(0, min(w, xe))
            
            # 원본 마스크와 교집합
            m[ys:ye, xs:xe] = mask_arr[ys:ye, xs:xe]
            return Image.fromarray(m, mode='L')
        
        # 인체 비례학 기반 비율 (웨딩드레스 고려)
        parts['neck'] = create_mask(0.13, 0.18, 0.3, 0.7)  # 목: 폭 좁게
        parts['shoulder'] = create_mask(0.15, 0.26)
        parts['chest'] = create_mask(0.26, 0.40)
        
        # 허리: 가장 중요한 라인. 폭을 좁게 잡아 배경 왜곡 최소화
        parts['waist'] = create_mask(0.38, 0.53, 0.2, 0.8)
        
        parts['hip'] = create_mask(0.53, 0.65)
        
        # 다리: 드레스 하단 보호 (0.92까지만, 바닥 안 건드림)
        parts['leg'] = create_mask(0.65, 0.92)
        
        # ===== 팔: 좌우 분리 (개선 v3) =====
        # Upper arm까지 포함하도록 범위 확대
        y_arm_s = int(y_min + p_height * 0.15)  # 어깨 시작부터 (upper arm 포함)
        y_arm_e = int(y_min + p_height * 0.70)  # 팔꿈치 아래까지
        
        # 중심 영역을 40%로 축소 (30-70%만 제외)
        center_start = int(x_min + p_width * 0.30)
        center_end = int(x_min + p_width * 0.70)
        
        # 경계 체크
        y_arm_s = max(0, min(h, y_arm_s))
        y_arm_e = max(0, min(h, y_arm_e))
        center_start = max(0, min(w, center_start))
        center_end = max(0, min(w, center_end))
        
        # 왼팔 (이미지상 왼쪽)
        arm_left_mask = np.zeros_like(mask_arr)
        arm_left_mask[y_arm_s:y_arm_e, x_min:center_start] = mask_arr[y_arm_s:y_arm_e, x_min:center_start]
        
        # 오른팔 (이미지상 오른쪽)
        arm_right_mask = np.zeros_like(mask_arr)
        arm_right_mask[y_arm_s:y_arm_e, center_end:x_max] = mask_arr[y_arm_s:y_arm_e, center_end:x_max]
        
        # 마스크가 비어있으면 fallback: 전체 좌우 영역 사용
        left_pixels = np.sum(arm_left_mask > 128)
        right_pixels = np.sum(arm_right_mask > 128)
        
        if left_pixels < 500:  # 너무 작으면 전체 왼쪽 사용
            logger.warning(f"왼팔 마스크가 너무 작음 ({left_pixels}px), 전체 왼쪽 영역 사용")
            arm_left_mask[y_arm_s:y_arm_e, x_min:int(x_min + p_width * 0.5)] = mask_arr[y_arm_s:y_arm_e, x_min:int(x_min + p_width * 0.5)]
            left_pixels = np.sum(arm_left_mask > 128)
        
        if right_pixels < 500:  # 너무 작으면 전체 오른쪽 사용
            logger.warning(f"오른팔 마스크가 너무 작음 ({right_pixels}px), 전체 오른쪽 영역 사용")
            arm_right_mask[y_arm_s:y_arm_e, int(x_min + p_width * 0.5):x_max] = mask_arr[y_arm_s:y_arm_e, int(x_min + p_width * 0.5):x_max]
            right_pixels = np.sum(arm_right_mask > 128)
        
        parts['arm_left'] = Image.fromarray(arm_left_mask, mode='L')
        parts['arm_right'] = Image.fromarray(arm_right_mask, mode='L')
        
        # 통합 팔 마스크
        arm_combined = np.maximum(arm_left_mask, arm_right_mask)
        parts['arm'] = Image.fromarray(arm_combined, mode='L')
        
        logger.info(f"BBox 기반 부위별 마스크 생성 완료: {list(parts.keys())}")
        logger.info(f"팔 마스크 크기: 왼팔={left_pixels}px, 오른팔={right_pixels}px")
        
        return parts
