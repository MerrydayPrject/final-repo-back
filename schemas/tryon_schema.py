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


class V4V5CompareResponse(BaseModel):
    """V4V5일반 비교 응답 스키마"""
    success: bool
    v4_result: UnifiedTryonResponse
    v5_result: UnifiedTryonResponse
    total_time: float
    message: Optional[str] = None


class V4V5CustomCompareResponse(BaseModel):
    """V4V5커스텀 비교 응답 스키마"""
    success: bool
    v4_result: UnifiedTryonResponse
    v5_result: UnifiedTryonResponse
    total_time: float
    message: Optional[str] = None