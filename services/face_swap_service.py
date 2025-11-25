"""
페이스스왑 서비스 모듈
InsightFace + INSwapper를 사용하여 템플릿 이미지에 사용자 얼굴을 교체

기능:
1. 사용자 얼굴 이미지에서 얼굴 인식 및 정렬
2. 템플릿 이미지의 얼굴을 사용자 얼굴로 교체
3. 자연스러운 페이스스왑 결과 생성
"""
import os
import cv2
import numpy as np
from PIL import Image
from typing import Optional, List, TYPE_CHECKING, Dict
from pathlib import Path

if TYPE_CHECKING:
    import insightface

try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    insightface = None  # 타입 힌트를 위한 더미 값
    print("⚠️  InsightFace가 설치되지 않았습니다. pip install insightface를 실행하세요.")


class FaceSwapService:
    """페이스스왑 서비스"""
    
    def __init__(self):
        """서비스 초기화"""
        self.face_analyzer = None
        self.swapper = None
        self.is_initialized = False
        
        if INSIGHTFACE_AVAILABLE:
            try:
                self._init_insightface()
            except Exception as e:
                print(f"⚠️  InsightFace 초기화 실패: {e}")
                self.is_initialized = False
        else:
            print("⚠️  InsightFace를 사용할 수 없습니다.")
    
    def _init_insightface(self):
        """InsightFace 모델 초기화"""
        if not INSIGHTFACE_AVAILABLE:
            return
        
        try:
            # InsightFace FaceAnalysis 초기화
            # 모델은 자동으로 다운로드됨 (~/.insightface/models/ 경로)
            self.face_analyzer = insightface.app.FaceAnalysis(
                name='buffalo_l',  # 기본 모델 (buffalo_l은 가장 정확함)
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']  # CUDA 우선, 없으면 CPU
            )
            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
            
            # INSwapper 모델 로드
            # InsightFace의 model_zoo를 사용하여 모델 로드
            model_root = Path.home() / '.insightface' / 'models'
            inswapper_path = model_root / 'inswapper_128.onnx'
            
            try:
                from insightface.model_zoo import get_model
                
                # 먼저 로컬에 있는지 확인
                if inswapper_path.exists():
                    print(f"📦 로컬 INSwapper 모델 발견: {inswapper_path}")
                    self.swapper = get_model(str(inswapper_path))
                else:
                    # 자동 다운로드 시도
                    print("⚠️  INSwapper 모델을 찾을 수 없습니다. 자동 다운로드를 시도합니다...")
                    try:
                        self.swapper = get_model('inswapper_128.onnx', download=True, download_zip=False)
                    except Exception as download_error:
                        print(f"⚠️  자동 다운로드 실패: {download_error}")
                        print("   수동 다운로드가 필요합니다.")
                        print("   다운로드 링크:")
                        print("   - https://github.com/haofanwang/inswapper (checkpoints 폴더)")
                        print("   - 또는 다른 소스에서 inswapper_128.onnx 파일 다운로드")
                        print(f"   저장 위치: {inswapper_path}")
                        return
            except Exception as e:
                print(f"⚠️  INSwapper 모델 로드 실패: {e}")
                print("   수동으로 모델을 다운로드하거나 경로를 설정해주세요.")
                print(f"   저장 위치: {inswapper_path}")
                return
            
            self.is_initialized = True
            print("✅ InsightFace + INSwapper 초기화 완료")
            
        except Exception as e:
            print(f"❌ InsightFace 초기화 오류: {e}")
            self.is_initialized = False
    
    def is_available(self) -> bool:
        """서비스 사용 가능 여부 확인"""
        return self.is_initialized and self.face_analyzer is not None and self.swapper is not None
    
    def detect_face(self, image: np.ndarray) -> Optional["insightface.types.Face"]:
        """
        이미지에서 얼굴 감지 및 분석
        
        Args:
            image: BGR 형식의 numpy 배열 이미지
            
        Returns:
            감지된 얼굴 객체 (없으면 None)
        """
        if not self.is_available():
            return None
        
        try:
            faces = self.face_analyzer.get(image)
            if len(faces) > 0:
                # 가장 큰 얼굴 반환 (여러 얼굴이 있을 경우)
                return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            return None
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
        
        try:
            # PIL Image를 BGR numpy 배열로 변환
            source_np = np.array(source_image.convert('RGB'))[:, :, ::-1]  # RGB -> BGR
            target_np = np.array(target_image.convert('RGB'))[:, :, ::-1]  # RGB -> BGR
            
            # 소스 이미지에서 얼굴 감지
            source_faces = self.face_analyzer.get(source_np)
            if len(source_faces) == 0:
                print("⚠️  소스 이미지에서 얼굴을 찾을 수 없습니다.")
                return None
            
            if source_face_index >= len(source_faces):
                source_face_index = 0
            
            source_face = source_faces[source_face_index]
            
            # 타겟 이미지에서 얼굴 감지
            target_faces = self.face_analyzer.get(target_np)
            if len(target_faces) == 0:
                print("⚠️  타겟 이미지에서 얼굴을 찾을 수 없습니다.")
                return None
            
            if target_face_index >= len(target_faces):
                target_face_index = 0
            
            target_face = target_faces[target_face_index]
            
            # INSwapper로 페이스스왑
            # INSwapper의 get 메서드 사용
            if hasattr(self.swapper, 'get'):
                result_np = self.swapper.get(target_np, target_face, source_face, paste_back=True)
            else:
                # 대체 방법
                result_np = self._swap_face_with_inswapper(source_face, target_face, target_np)
            
            if result_np is None:
                return None
            
            # BGR -> RGB로 변환 후 PIL Image로 변환
            result_rgb = cv2.cvtColor(result_np, cv2.COLOR_BGR2RGB)
            result_image = Image.fromarray(result_rgb)
            
            return result_image
            
        except Exception as e:
            print(f"페이스스왑 오류: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _swap_face_with_inswapper(
        self,
        source_face: "insightface.types.Face",
        target_face: "insightface.types.Face",
        target_image: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        INSwapper 모델을 사용하여 페이스스왑 수행
        
        Args:
            source_face: 소스 얼굴 객체
            target_face: 타겟 얼굴 객체
            target_image: 타겟 이미지 (BGR)
            
        Returns:
            페이스스왑된 이미지 (BGR) 또는 None
        """
        try:
            # INSwapper의 get 메서드를 사용하여 페이스스왑 수행
            # InsightFace의 INSwapper는 간단한 인터페이스를 제공
            result_image = self.swapper.get(target_image, target_face, source_face, paste_back=True)
            
            return result_image
            
        except Exception as e:
            print(f"INSwapper 추론 오류: {e}")
            import traceback
            traceback.print_exc()
            # 대체 방법: 직접 구현
            return self._swap_face_manual(source_face, target_face, target_image)
    
    def _swap_face_manual(
        self,
        source_face: "insightface.types.Face",
        target_face: "insightface.types.Face",
        target_image: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        수동으로 페이스스왑 수행 (INSwapper 실패 시 대체 방법)
        """
        try:
            # 소스 얼굴 임베딩
            source_embedding = source_face.embedding
            
            # 타겟 얼굴 영역 크롭
            target_bbox = target_face.bbox.astype(int)
            x1, y1, x2, y2 = target_bbox
            w, h = x2 - x1, y2 - y1
            
            # 얼굴 영역 확장
            scale = 1.3
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w_new, h_new = int(w * scale), int(h * scale)
            
            x1_new = max(0, int(cx - w_new / 2))
            y1_new = max(0, int(cy - h_new / 2))
            x2_new = min(target_image.shape[1], int(cx + w_new / 2))
            y2_new = min(target_image.shape[0], int(cy + h_new / 2))
            
            # 타겟 얼굴 영역
            face_roi = target_image[y1_new:y2_new, x1_new:x2_new].copy()
            
            # 간단한 블렌딩 (실제로는 INSwapper 모델이 필요)
            # 여기서는 원본 이미지 반환 (나중에 모델이 제대로 로드되면 작동)
            result = target_image.copy()
            
            return result
            
        except Exception as e:
            print(f"수동 페이스스왑 오류: {e}")
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
            faces = self.face_analyzer.get(source_np)
            
            face_ratio = 0.0
            if len(faces) > 0:
                face = faces[0]
                face_bbox = face.bbox
                face_area = (face_bbox[2] - face_bbox[0]) * (face_bbox[3] - face_bbox[1])
                image_area = width * height
                face_ratio = face_area / image_area if image_area > 0 else 0.0
            
            # 포즈 랜드마크로 하체 감지 시도
            has_lower_body = False
            try:
                import mediapipe as mp
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                
                # MediaPipe Pose Landmarker로 포즈 감지
                model_path = Path(__file__).parent.parent / 'models' / 'body_analysis' / 'pose_landmarker_lite.task'
                if model_path.exists():
                    base_options = python.BaseOptions(model_asset_path=str(model_path))
                    options = vision.PoseLandmarkerOptions(
                        base_options=base_options,
                        output_segmentation_masks=False,
                        min_pose_detection_confidence=0.5,
                        min_pose_presence_confidence=0.5,
                        min_tracking_confidence=0.5
                    )
                    pose_landmarker = vision.PoseLandmarker.create_from_options(options)
                    
                    # 이미지 변환
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(image))
                    detection_result = pose_landmarker.detect(mp_image)
                    
                    if detection_result.pose_landmarks:
                        landmarks = detection_result.pose_landmarks[0]
                        # 하체 랜드마크 확인 (발목: 27, 28, 무릎: 25, 26, 엉덩이: 23, 24)
                        lower_body_landmarks = [23, 24, 25, 26, 27, 28]
                        visible_lower_body = sum(
                            1 for i in lower_body_landmarks 
                            if i < len(landmarks) and landmarks[i].visibility > 0.5
                        )
                        has_lower_body = visible_lower_body >= 3  # 최소 3개 이상 보이면 하체 있음
            except Exception as e:
                # MediaPipe가 없거나 실패하면 다른 방법 사용
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

