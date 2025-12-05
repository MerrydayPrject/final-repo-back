"""
체형 분석 서비스 클래스
HuggingFace Spaces API를 사용한 체형 분석
"""
from PIL import Image
import numpy as np
from typing import Dict, Optional, List, Tuple
from services.pose_landmark_service import PoseLandmarkService


class BodyAnalysisService:
    """체형 분석 서비스"""
    
    def __init__(self, space_url: Optional[str] = None):
        """
        초기화
        
        Args:
            space_url: HuggingFace Spaces URL (None이면 설정에서 가져옴)
        """
        self.pose_landmark_service = PoseLandmarkService(space_url=space_url)
        self.is_initialized = self.pose_landmark_service.is_initialized
        
        if self.is_initialized:
            print("체형 분석 서비스 초기화 완료!")
        else:
            print("⚠️  체형 분석 서비스 초기화 실패")
    
    def _detect_orientation(self, landmarks: List[Dict]) -> Dict[str, float]:
        """
        랜드마크로부터 이미지 방향 감지
        
        Args:
            landmarks: 랜드마크 좌표 리스트
            
        Returns:
            방향 정보 딕셔너리
        """
        if not landmarks or len(landmarks) < 33:
            return {"is_vertical": True, "rotation_angle": 0.0, "needs_rotation": False}
        
        def get_landmark(idx: int) -> Tuple[float, float, float]:
            landmark = landmarks[idx]
            return landmark["x"], landmark["y"], landmark["z"]
        
        # 어깨 중심
        left_shoulder = get_landmark(11)
        right_shoulder = get_landmark(12)
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
        
        # 엉덩이 중심
        left_hip = get_landmark(23)
        right_hip = get_landmark(24)
        hip_center_x = (left_hip[0] + right_hip[0]) / 2
        hip_center_y = (left_hip[1] + right_hip[1]) / 2
        
        # 어깨와 엉덩이 사이의 x, y 차이
        dx = abs(hip_center_x - shoulder_center_x)
        dy = abs(hip_center_y - shoulder_center_y)
        
        # 정면 사진: 세로 방향 (y 차이가 x 차이보다 훨씬 큼)
        # 가로로 누운 사진: 가로 방향 (x 차이가 y 차이보다 큼)
        is_vertical = dy > dx
        
        # 회전 각도 계산 (어깨 중심에서 엉덩이 중심으로의 벡터)
        angle_rad = np.arctan2(hip_center_y - shoulder_center_y, hip_center_x - shoulder_center_x)
        angle_deg = np.degrees(angle_rad)
        
        # 정면 사진이어야 하는데 가로로 누워있는 경우 감지
        # 정면: 어깨가 위, 엉덩이가 아래 (y 차이가 크고, 어깨 y < 엉덩이 y)
        # 가로: 어깨와 엉덩이가 좌우로 (x 차이가 크거나, y 차이가 작음)
        
        # 정면 사진 기준: 어깨 y < 엉덩이 y (어깨가 위에 있음)
        shoulder_above_hip = shoulder_center_y < hip_center_y
        
        # 가로로 누운지 판단: x 차이가 y 차이보다 크거나, 어깨가 엉덩이보다 아래에 있으면
        is_horizontal = dx > dy or not shoulder_above_hip
        
        # 회전이 필요한지 판단
        needs_rotation = is_horizontal
        
        # 회전 각도 결정 (90도 또는 -90도)
        rotation_angle = 0.0
        if needs_rotation:
            # 어깨가 엉덩이보다 오른쪽에 있으면 -90도, 왼쪽에 있으면 90도
            if shoulder_center_x > hip_center_x:
                rotation_angle = -90.0
            else:
                rotation_angle = 90.0
        
        # 어깨와 엉덩이 사이의 실제 거리 계산
        distance = np.sqrt(dx**2 + dy**2)
        
        print(f"[방향 감지] 세로: {is_vertical}, 가로: {is_horizontal}, 회전 필요: {needs_rotation}, 각도: {rotation_angle:.1f}도")
        print(f"  어깨 중심: ({shoulder_center_x:.3f}, {shoulder_center_y:.3f})")
        print(f"  엉덩이 중심: ({hip_center_x:.3f}, {hip_center_y:.3f})")
        print(f"  거리: {distance:.3f}")
        
        return {
            "is_vertical": is_vertical,
            "is_horizontal": is_horizontal,
            "rotation_angle": rotation_angle,
            "needs_rotation": needs_rotation,
            "dx": dx,
            "dy": dy
        }
    
    def _correct_image_orientation(self, image: Image.Image, rotation_angle: float) -> Image.Image:
        """
        이미지 회전 보정
        
        Args:
            image: PIL Image 객체
            rotation_angle: 회전 각도 (도)
            
        Returns:
            보정된 이미지
        """
        if abs(rotation_angle) < 0.1:
            return image
        
        print(f"[이미지 보정] {rotation_angle:.1f}도 회전 적용")
        # PIL Image.rotate는 반시계 방향이 양수
        corrected_image = image.rotate(-rotation_angle, expand=True, fillcolor='white')
        return corrected_image
    
    def extract_landmarks(self, image: Image.Image) -> Optional[List[Dict]]:
        """
        이미지에서 포즈 랜드마크 추출
        
        Args:
            image: PIL Image 객체 (EXIF orientation이 이미 적용된 이미지)
            
        Returns:
            랜드마크 좌표 리스트 (33개 포인트) 또는 None
        """
        if not self.is_initialized:
            print("서비스가 초기화되지 않았습니다.")
            return None
        
        # 랜드마크 추출
        landmarks = self.pose_landmark_service.extract_landmarks(image)
        
        return landmarks
    
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



