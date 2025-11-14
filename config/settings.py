"""환경변수 및 설정값 관리"""
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# LLM 모델 설정
GPT4O_MODEL_NAME = os.getenv("GPT4O_MODEL_NAME", "gpt-4o")
GEMINI_FLASH_MODEL = os.getenv("GEMINI_FLASH_MODEL", "gemini-2.5-flash-image")
GEMINI_PROMPT_MODEL = os.getenv("GEMINI_PROMPT_MODEL", "gemini-2.0-flash-exp")

# Meshy API 설정
MESHY_API_KEY = os.getenv("MESHY_API_KEY", "")
MESHY_API_URL = "https://api.meshy.ai"

# AWS S3 설정
AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")

# 레이블 정보
LABELS = {
    0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses",
    4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "Dress",
    8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face",
    12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm",
    16: "Bag", 17: "Scarf"
}

