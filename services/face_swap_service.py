"""
페이스스왑 서비스 모듈
HuggingFace Inference Endpoint를 사용한 얼굴 분석 및 페이스스왑

기능:
1. 사용자 얼굴 이미지에서 얼굴 인식 및 정렬 (API 기반)
2. 템플릿 이미지의 얼굴을 사용자 얼굴로 교체 (로컬 INSwapper 필요)
3. 자연스러운 페이스스왑 결과 생성
"""
import os
import cv2
import numpy as np
from PIL import Image
from typing import Optional, List, Dict
from pathlib import Path
from services.face_analysis_service import FaceAnalysisService
from services.pose_landmark_service import PoseLandmarkService


class FaceSwapService:
    """페이스스왑 서비스"""
    
    def __init__(self, endpoint_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        서비스 초기화
        
        Args:
            endpoint_url: HuggingFace Inference Endpoint URL
            api_key: HuggingFace API 키
        """
        self.face_analysis_service = FaceAnalysisService(endpoint_url=endpoint_url, api_key=api_key)
        self.pose_landmark_service = PoseLandmarkService()
        self.swapper = None
        self.is_initialized = self.face_analysis_service.is_initialized
        
        # INSwapper는 로컬 모델이 필요하므로 경고 메시지 출력
        if not self.is_initialized:
            print("⚠️  얼굴 분석 서비스를 사용할 수 없습니다.")
        else:
            print("✅ 얼굴 분석 서비스 초기화 완료 (페이스스왑 기능은 INSwapper 모델이 필요합니다)")
    
    def is_available(self) -> bool:
        """서비스 사용 가능 여부 확인"""
        # 얼굴 감지는 API로 가능하지만, 페이스스왑은 INSwapper 모델이 필요
        return self.face_analysis_service.is_initialized
    
    def detect_face(self, image: np.ndarray) -> Optional[Dict]:
        """
        이미지에서 얼굴 감지 및 분석
        
        Args:
            image: BGR 형식의 numpy 배열 이미지
            
        Returns:
            감지된 얼굴 정보 딕셔너리 (없으면 None)
        """
        if not self.is_available():
            return None
        
        try:
            # FaceAnalysisService를 통해 얼굴 감지
            face_data = self.face_analysis_service.detect_face(image)
            return face_data
        except Exception as e:
            print(f"얼굴 감지 오류: {e}")
            return None
    
    def swap_face(
        self,
        source_image: Image.Image,
        target_image: Image.Image,
        source_face_index: int = 0,
        target_face_index: int = 0
    ) -> Optional[Image.Image]:
        """
        템플릿 이미지에 사용자 얼굴을 교체
        
        주의: 페이스스왑 기능은 INSwapper 모델이 필요합니다.
        현재는 얼굴 감지만 API로 수행되며, 실제 페이스스왑은 로컬 모델이 필요합니다.
        
        Args:
            source_image: 사용자 얼굴 이미지 (PIL Image)
            target_image: 템플릿 이미지 (PIL Image)
            source_face_index: 소스 이미지에서 사용할 얼굴 인덱스 (기본값: 0)
            target_face_index: 타겟 이미지에서 교체할 얼굴 인덱스 (기본값: 0)
            
        Returns:
            페이스스왑된 이미지 (PIL Image) 또는 None (실패 시)
        """
        if not self.is_available():
            print("⚠️  페이스스왑 서비스를 사용할 수 없습니다.")
            return None
        
        # INSwapper 모델이 없으면 페이스스왑 불가
        if self.swapper is None:
            print("⚠️  페이스스왑 기능은 INSwapper 모델이 필요합니다.")
            print("   현재는 얼굴 감지만 API로 수행됩니다.")
            return None
        
        try:
            # PIL Image를 BGR numpy 배열로 변환
            source_np = np.array(source_image.convert('RGB'))[:, :, ::-1]  # RGB -> BGR
            target_np = np.array(target_image.convert('RGB'))[:, :, ::-1]  # RGB -> BGR
            
            # 소스 이미지에서 얼굴 감지 (API 사용)
            source_faces = self.face_analysis_service.detect_faces(source_np)
            if len(source_faces) == 0:
                print("⚠️  소스 이미지에서 얼굴을 찾을 수 없습니다.")
                return None
            
            if source_face_index >= len(source_faces):
                source_face_index = 0
            
            source_face_data = source_faces[source_face_index]
            
            # 타겟 이미지에서 얼굴 감지 (API 사용)
            target_faces = self.face_analysis_service.detect_faces(target_np)
            if len(target_faces) == 0:
                print("⚠️  타겟 이미지에서 얼굴을 찾을 수 없습니다.")
                return None
            
            if target_face_index >= len(target_faces):
                target_face_index = 0
            
            target_face_data = target_faces[target_face_index]
            
            # INSwapper로 페이스스왑 (로컬 모델 필요)
            # API에서 받은 얼굴 데이터를 INSwapper 형식으로 변환 필요
            # 현재는 INSwapper 모델이 없으므로 원본 이미지 반환
            print("⚠️  페이스스왑 기능은 INSwapper 모델이 필요합니다.")
            return target_image
            
        except Exception as e:
            print(f"페이스스왑 오류: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    
    def detect_image_type(self, image: Image.Image) -> Dict[str, any]:
        """
        이미지 타입 감지 (전신 vs 얼굴/상체)
        
        Args:
            image: 분석할 이미지 (PIL Image)
            
        Returns:
            Dict with keys:
            - type: "full_body" or "upper_body" or "face_only"
            - confidence: 신뢰도 (0.0 ~ 1.0)
            - details: 상세 정보
        """
        try:
            # 이미지 크기 및 비율 확인
            width, height = image.size
            aspect_ratio = height / width if width > 0 else 1.0
            
            # 얼굴 크기 비율 계산
            source_np = np.array(image.convert('RGB'))[:, :, ::-1]  # RGB -> BGR
            faces = self.face_analysis_service.detect_faces(source_np)
            
            face_ratio = 0.0
            if len(faces) > 0:
                face = faces[0]
                # API 응답 형식에 따라 bbox 추출
                if isinstance(face, dict):
                    face_bbox = face.get("bbox", [0, 0, 0, 0])
                else:
                    face_bbox = [0, 0, 0, 0]
                face_area = (face_bbox[2] - face_bbox[0]) * (face_bbox[3] - face_bbox[1])
                image_area = width * height
                face_ratio = face_area / image_area if image_area > 0 else 0.0
            
            # 포즈 랜드마크로 하체 감지 시도 (API 사용)
            has_lower_body = False
            try:
                # PoseLandmarkService를 통해 포즈 감지
                landmarks = self.pose_landmark_service.extract_landmarks(image)
                
                if landmarks:
                    # 하체 랜드마크 확인 (발목: 27, 28, 무릎: 25, 26, 엉덩이: 23, 24)
                    lower_body_ids = [23, 24, 25, 26, 27, 28]
                    visible_lower_body = sum(
                        1 for landmark in landmarks
                        if landmark.get("id") in lower_body_ids and landmark.get("visibility", 0) > 0.5
                    )
                    has_lower_body = visible_lower_body >= 3  # 최소 3개 이상 보이면 하체 있음
            except Exception as e:
                # 포즈 감지 실패 시 다른 방법 사용
                pass
            
            # 판단 로직
            image_type = "upper_body"
            confidence = 0.5
            
            # 1. 하체 랜드마크가 있으면 전신
            if has_lower_body:
                image_type = "full_body"
                confidence = 0.9
            
            # 2. 이미지 비율이 세로로 길면 (전신 가능성)
            elif aspect_ratio > 1.5:
                image_type = "full_body"
                confidence = 0.7
            
            # 3. 얼굴 비율이 크면 (얼굴/상체)
            elif face_ratio > 0.15:  # 얼굴이 이미지의 15% 이상
                image_type = "face_only" if face_ratio > 0.3 else "upper_body"
                confidence = 0.8
            
            # 4. 이미지 비율이 정사각형에 가까우면 (얼굴/상체)
            elif 0.8 < aspect_ratio < 1.2:
                image_type = "upper_body"
                confidence = 0.7
            
            return {
                "type": image_type,
                "confidence": confidence,
                "details": {
                    "aspect_ratio": aspect_ratio,
                    "face_ratio": face_ratio,
                    "has_lower_body": has_lower_body,
                    "image_size": (width, height)
                }
            }
            
        except Exception as e:
            print(f"이미지 타입 감지 오류: {e}")
            return {
                "type": "unknown",
                "confidence": 0.0,
                "details": {"error": str(e)}
            }
    
    def get_template_images(self, template_dir: Optional[Path] = None) -> List[Path]:
        """
        템플릿 이미지 목록 가져오기
        
        Args:
            template_dir: 템플릿 이미지 디렉토리 경로 (None이면 기본 경로 사용)
            
        Returns:
            템플릿 이미지 파일 경로 리스트
        """
        if template_dir is None:
            template_dir = Path(__file__).parent.parent / 'templates' / 'face_swap_templates'
        
        template_dir = Path(template_dir)
        if not template_dir.exists():
            template_dir.mkdir(parents=True, exist_ok=True)
            print(f"⚠️  템플릿 디렉토리가 없어 생성했습니다: {template_dir}")
            return []
        
        # 이미지 파일만 필터링
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        template_files = [
            f for f in template_dir.iterdir()
            if f.suffix.lower() in image_extensions and f.is_file()
        ]
        
        return sorted(template_files)

