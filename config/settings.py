"""환경변수 및 설정값 관리"""
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# LLM 모델 설정
GPT4O_MODEL_NAME = os.getenv("GPT4O_MODEL_NAME", "gpt-4o")
GPT4O_V2_MODEL_NAME = os.getenv("GPT4O_V2_MODEL_NAME", "gpt-4o-2024-08-06")
GEMINI_FLASH_MODEL = os.getenv("GEMINI_FLASH_MODEL", "gemini-2.5-flash-image")
GEMINI_3_FLASH_MODEL = os.getenv("GEMINI_3_FLASH_MODEL", "gemini-3-pro-image-preview")
GEMINI_PROMPT_MODEL = os.getenv("GEMINI_PROMPT_MODEL", "gemini-2.0-flash-exp")

# Gemini API 키 설정
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_3_API_KEY = os.getenv("GEMINI_3_API_KEY", "")

# x.ai API 설정
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
XAI_API_BASE_URL = os.getenv("XAI_API_BASE_URL", "https://api.x.ai/v1")
XAI_IMAGE_MODEL = os.getenv("XAI_IMAGE_MODEL", "grok-2-image")
XAI_PROMPT_MODEL = os.getenv("XAI_PROMPT_MODEL", "grok-2-vision-1212")

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

