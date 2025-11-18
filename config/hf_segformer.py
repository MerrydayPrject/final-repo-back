"""SegFormer B2 Human Parsing 설정"""
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# HuggingFace Inference API 설정
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
SEGFORMER_MODEL_ID = "yolo12138/segformer-b2-human-parse-24"
HUGGINGFACE_API_BASE_URL = os.getenv(
    "HUGGINGFACE_API_BASE_URL",
    "https://router.huggingface.co/hf-inference/models"
)
SEGFORMER_API_URL = f"{HUGGINGFACE_API_BASE_URL}/{SEGFORMER_MODEL_ID}"
API_TIMEOUT = int(os.getenv("SEGFORMER_API_TIMEOUT", "60"))

# 레이블 매핑 상수
FACE_MASK_IDS = [11, 18, 2]  # face, skin, hair
CLOTH_MASK_IDS = [4, 5, 6, 7, 8, 16, 17]  # upper, skirt, pants, dress, belt, bag, scarf
BODY_MASK_IDS = [12, 13, 14, 15]  # left-leg, right-leg, left-arm, right-arm

# 중립 색상 (base_img 생성용)
NEUTRAL_COLOR = (128, 128, 128)

