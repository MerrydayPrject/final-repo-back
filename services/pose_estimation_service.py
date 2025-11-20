"""Mediapipe를 사용한 포즈 추출 서비스"""
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from core.model_loader import get_mediapipe_pose

async def extract_pose_from_image(image_file):
    """
    이미지 파일에서 포즈를 추출하고 스켈레톤을 그립니다.
    """
    # 모델이 필요할 때 getter 함수를 호출하여 로드
    pose_estimator, drawing_utils = get_mediapipe_pose()

    if not pose_estimator or not drawing_utils:
        raise RuntimeError("포즈 예측 모델을 로드할 수 없습니다.")
    
    image = Image.open(image_file.file)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Pillow 이미지를 OpenCV 형식(NumPy array)으로 변환
    open_cv_image = np.array(image)
    
    # Mediapipe 처리
    results = pose_estimator.process(open_cv_image)

    # 포즈 스켈레톤을 그릴 검은색 배경 이미지 생성
    annotated_image = np.zeros_like(open_cv_image)
    if results.pose_landmarks:
        # 스켈레톤 그리기
        drawing_utils.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_utils.DrawingSpec(color=(255,255,0), thickness=2, circle_radius=2),
            connection_drawing_spec=drawing_utils.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
        )
    
    return Image.fromarray(annotated_image)