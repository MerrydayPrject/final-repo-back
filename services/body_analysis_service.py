"""
체형 분석 서비스 클래스
MediaPipe Pose Landmarker를 사용한 체형 분석
"""
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
import numpy as np
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import os


class BodyAnalysisService:
    """체형 분석 서비스"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        초기화
        
        Args:
            model_path: MediaPipe 모델 파일 경로 (None이면 자동 다운로드)
        """
        self.model_path = model_path
        self.pose_landmarker = None
        self.is_initialized = False
        
        # MediaPipe 초기화
        self._init_mediapipe()
    
    def _init_mediapipe(self):
        """MediaPipe Pose Landmarker 초기화"""
        try:
            # 모델 파일 경로 설정
            if self.model_path is None:
                # 기본 모델 경로 (자동 다운로드 시 사용)
                model_path = self._get_default_model_path()
            else:
                model_path = Path(self.model_path)
            
            # 모델 파일이 없으면 다운로드
            if not model_path.exists():
                print("모델 파일이 없습니다. 다운로드를 시도합니다...")
                self._download_model(model_path)
            
            # Pose Landmarker 옵션 설정
            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Pose Landmarker 생성
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)
            self.is_initialized = True
            print("체형 분석 서비스 초기화 완료!")
            
        except Exception as e:
            print(f"MediaPipe 초기화 오류: {e}")
            self.is_initialized = False
    
    def _get_default_model_path(self) -> Path:
        """기본 모델 경로 반환"""
        # 프로젝트 루트 기준으로 models/body_analysis/ 경로 사용
        project_root = Path(__file__).parent.parent
        models_dir = project_root / "models" / "body_analysis"
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir / "pose_landmarker_lite.task"
    
    def _download_model(self, model_path: Path):
        """MediaPipe 모델 다운로드"""
        try:
            import urllib.request
            
            model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
            
            print(f"모델 다운로드 중: {model_url}")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(model_url, str(model_path))
            print(f"모델 다운로드 완료: {model_path}")
            
        except Exception as e:
            print(f"모델 다운로드 실패: {e}")
            raise
    
    def extract_landmarks(self, image: Image.Image) -> Optional[List[Dict]]:
        """
        이미지에서 포즈 랜드마크 추출
        
        Args:
            image: PIL Image 객체
            
        Returns:
            랜드마크 좌표 리스트 (33개 포인트) 또는 None
        """
        if not self.is_initialized:
            print("서비스가 초기화되지 않았습니다.")
            return None
        
        try:
            # PIL Image를 numpy array로 변환
            image_array = np.array(image)
            
            # RGB 형식으로 변환 (MediaPipe는 RGB 필요)
            if len(image_array.shape) == 3 and image_array.shape[2] == 4:
                # RGBA -> RGB
                image_array = image_array[:, :, :3]
            
            # MediaPipe 형식으로 변환
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
            
            # 랜드마크 추출
            detection_result = self.pose_landmarker.detect(mp_image)
            
            # 랜드마크가 없으면 None 반환
            if not detection_result.pose_landmarks:
                return None
            
            # 첫 번째 포즈의 랜드마크 사용
            pose_landmarks = detection_result.pose_landmarks[0]
            
            # 랜드마크를 딕셔너리 리스트로 변환
            landmarks = []
            for idx, landmark in enumerate(pose_landmarks):
                landmarks.append({
                    "id": idx,
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility
                })
            
            return landmarks
            
        except Exception as e:
            print(f"랜드마크 추출 오류: {e}")
            return None
    
    def calculate_measurements(self, landmarks: List[Dict]) -> Dict:
        """
        랜드마크로부터 체형 측정값 계산
        
        Args:
            landmarks: 랜드마크 좌표 리스트
            
        Returns:
            측정값 딕셔너리
        """
        if not landmarks or len(landmarks) < 33:
            return {}
        
        # 랜드마크 인덱스 정의
        # 어깨: 11 (왼쪽), 12 (오른쪽)
        # 엉덩이: 23 (왼쪽), 24 (오른쪽)
        # 팔꿈치: 13 (왼쪽), 14 (오른쪽)
        # 손목: 15 (왼쪽), 16 (오른쪽)
        # 무릎: 25 (왼쪽), 26 (오른쪽)
        # 발목: 27 (왼쪽), 28 (오른쪽)
        # 머리: 0 (코)
        # 발: 31 (왼쪽), 32 (오른쪽)
        
        def get_landmark(idx: int) -> Tuple[float, float, float]:
            landmark = landmarks[idx]
            return landmark["x"], landmark["y"], landmark["z"]
        
        def distance(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)
        
        # 어깨 폭
        left_shoulder = get_landmark(11)
        right_shoulder = get_landmark(12)
        shoulder_width = distance(left_shoulder, right_shoulder)
        
        # 엉덩이 폭
        left_hip = get_landmark(23)
        right_hip = get_landmark(24)
        hip_width = distance(left_hip, right_hip)
        
        # 어깨/엉덩이 비율
        shoulder_hip_ratio = shoulder_width / hip_width if hip_width > 0 else 0
        
        # 팔 길이 (어깨 -> 팔꿈치 -> 손목)
        left_elbow = get_landmark(13)
        left_wrist = get_landmark(15)
        left_arm_length = distance(left_shoulder, left_elbow) + distance(left_elbow, left_wrist)
        
        right_elbow = get_landmark(14)
        right_wrist = get_landmark(16)
        right_arm_length = distance(right_shoulder, right_elbow) + distance(right_elbow, right_wrist)
        arm_length = (left_arm_length + right_arm_length) / 2
        
        # 다리 길이 (엉덩이 -> 무릎 -> 발목)
        left_knee = get_landmark(25)
        left_ankle = get_landmark(27)
        left_leg_length = distance(left_hip, left_knee) + distance(left_knee, left_ankle)
        
        right_knee = get_landmark(26)
        right_ankle = get_landmark(28)
        right_leg_length = distance(right_hip, right_knee) + distance(right_knee, right_ankle)
        leg_length = (left_leg_length + right_leg_length) / 2
        
        # 키 추정 (머리 -> 발)
        nose = get_landmark(0)
        left_foot = get_landmark(31)
        right_foot = get_landmark(32)
        foot_center = ((left_foot[0] + right_foot[0]) / 2, 
                      (left_foot[1] + right_foot[1]) / 2, 
                      (left_foot[2] + right_foot[2]) / 2)
        estimated_height = distance(nose, foot_center)
        
        # 허리 폭 (엉덩이와 어깨의 중간점으로 추정)
        waist_width = (shoulder_width + hip_width) / 2
        
        # 상체 길이 (어깨 중심 → 엉덩이 중심)
        shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2,
                          (left_shoulder[1] + right_shoulder[1]) / 2,
                          (left_shoulder[2] + right_shoulder[2]) / 2)
        hip_center = ((left_hip[0] + right_hip[0]) / 2,
                     (left_hip[1] + right_hip[1]) / 2,
                     (left_hip[2] + right_hip[2]) / 2)
        torso_length = distance(shoulder_center, hip_center)
        
        # 하체 길이 (이미 계산된 leg_length 사용)
        lower_body_length = leg_length
        
        # 비율 계산
        waist_shoulder_ratio = waist_width / shoulder_width if shoulder_width > 0 else 1.0
        waist_hip_ratio = waist_width / hip_width if hip_width > 0 else 1.0
        torso_leg_ratio = torso_length / leg_length if leg_length > 0 else 1.0
        arm_leg_ratio = arm_length / leg_length if leg_length > 0 else 1.0
        
        return {
            "shoulder_width": float(shoulder_width),
            "hip_width": float(hip_width),
            "waist_width": float(waist_width),
            "shoulder_hip_ratio": float(shoulder_hip_ratio),
            "waist_shoulder_ratio": float(waist_shoulder_ratio),
            "waist_hip_ratio": float(waist_hip_ratio),
            "arm_length": float(arm_length),
            "leg_length": float(leg_length),
            "torso_length": float(torso_length),
            "lower_body_length": float(lower_body_length),
            "torso_leg_ratio": float(torso_leg_ratio),
            "arm_leg_ratio": float(arm_leg_ratio),
            "estimated_height": float(estimated_height),
            "body_length": float(estimated_height)
        }
    
    def classify_body_type(self, measurements: Dict) -> Dict:
        """
        측정값으로부터 체형 타입 분류
        
        Args:
            measurements: 측정값 딕셔너리
            
        Returns:
            체형 타입 정보
        """
        if not measurements:
            return {
                "type": "unknown",
                "confidence": 0.0,
                "description": "측정값을 계산할 수 없습니다."
            }
        
        shoulder_hip_ratio = measurements.get("shoulder_hip_ratio", 1.0)
        waist_shoulder_ratio = measurements.get("waist_shoulder_ratio", 1.0)
        waist_hip_ratio = measurements.get("waist_hip_ratio", 1.0)
        torso_leg_ratio = measurements.get("torso_leg_ratio", 1.0)
        arm_leg_ratio = measurements.get("arm_leg_ratio", 1.0)
        shoulder_width = measurements.get("shoulder_width", 0)
        hip_width = measurements.get("hip_width", 0)
        waist_width = measurements.get("waist_width", 0)
        
        # 디버깅: 측정값 출력
        print(f"\n[체형 분석 측정값]")
        print(f"  어깨 폭: {shoulder_width:.4f}")
        print(f"  엉덩이 폭: {hip_width:.4f}")
        print(f"  허리 폭: {waist_width:.4f}")
        print(f"  어깨/엉덩이 비율: {shoulder_hip_ratio:.3f}")
        print(f"  허리/어깨 비율: {waist_shoulder_ratio:.3f}")
        print(f"  허리/엉덩이 비율: {waist_hip_ratio:.3f}")
        print(f"  상체/하체 비율: {torso_leg_ratio:.3f}")
        print(f"  팔/다리 비율: {arm_leg_ratio:.3f}")
        
        # 체형 분류 (4가지 기본 체형으로 단순화, 조건을 확연하게 구분)
        # 실제 측정값 분포를 반영하여 조건 조정
        # 측정값: 어깨/엉덩이 비율이 1.3-1.7 범위로 나오는 경우가 많음
        
        # 디버깅: 조건 체크
        print(f"  [조건 체크]")
        print(f"    X라인 (허리/어깨<0.82, 허리/엉덩이<1.30): {waist_shoulder_ratio < 0.82 and waist_hip_ratio < 1.30}")
        print(f"    A라인 (< 1.40): {shoulder_hip_ratio < 1.40}")
        print(f"    H라인 (1.40-1.65, 허리>=0.82): {1.40 <= shoulder_hip_ratio <= 1.65 and waist_shoulder_ratio >= 0.82}")
        print(f"    O라인 (> 1.65): {shoulder_hip_ratio > 1.65}")
        
        # 1. X라인 (모래시계형) - 허리가 매우 얇음
        # 허리/어깨 비율이 낮고 (< 0.82), 허리/엉덩이 비율도 낮은 경우
        if (waist_shoulder_ratio < 0.82 and waist_hip_ratio < 1.30):
            body_type = "X라인"
            confidence = 0.90
            description = "X라인(모래시계형) 체형에 가깝습니다. 어깨와 엉덩이가 비슷하고 허리가 얇은 특징을 보입니다."
        
        # 2. A라인 (역삼각형) - 어깨 < 엉덩이 (상대적으로)
        # 실제 측정값: 1.4 미만인 경우 (엉덩이가 상대적으로 넓음)
        elif shoulder_hip_ratio < 1.40:
            body_type = "A라인"
            confidence = 0.85
            description = "A라인 체형에 가깝습니다. 어깨보다 엉덩이가 넓은 특징을 보입니다."
        
        # 3. H라인 (직선형) - 어깨 ≈ 엉덩이, 허리도 비슷
        # 실제 측정값: 1.4-1.65 범위에서 허리가 그렇게 얇지 않은 경우
        elif (1.40 <= shoulder_hip_ratio <= 1.65 and
              waist_shoulder_ratio >= 0.82):  # 허리가 그렇게 얇지 않음
            body_type = "H라인"
            confidence = 0.85
            description = "H라인 체형에 가깝습니다. 어깨와 엉덩이가 비슷한 직선형 특징을 보입니다."
        
        # 4. O라인 (원형) - 어깨 > 엉덩이 또는 둥근 체형
        # 실제 측정값: 1.65 이상인 경우 (어깨가 상대적으로 넓음)
        elif shoulder_hip_ratio > 1.65:
            body_type = "O라인"
            confidence = 0.80
            description = "O라인 체형에 가깝습니다. 어깨가 넓거나 균형잡힌 둥근 특징을 보입니다."
        
        # 5. 기본 (기타) - 위 조건에 해당하지 않는 경우
        else:
            body_type = "균형형"
            confidence = 0.75
            description = "균형잡힌 체형에 가깝습니다."
        
        print(f"  → 분류 결과: {body_type} (신뢰도: {confidence:.2f})")
        
        return {
            "type": body_type,
            "confidence": confidence,
            "description": description
        }



