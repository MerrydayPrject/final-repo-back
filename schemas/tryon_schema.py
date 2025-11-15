"""트라이온 통합 파이프라인 스키마"""
from pydantic import BaseModel
from typing import Optional


class UnifiedTryonResponse(BaseModel):
    """통합 트라이온 응답 스키마"""
    success: bool
    prompt: str
    result_image: str  # base64 인코딩된 이미지
    message: Optional[str] = None
    llm: Optional[str] = None  # 사용된 LLM 정보 (예: "xai-gemini-unified")

