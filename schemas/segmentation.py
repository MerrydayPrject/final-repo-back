"""세그멘테이션 관련 스키마"""
from typing import List
from pydantic import BaseModel, Field


class LabelInfo(BaseModel):
    """레이블 정보 모델"""
    id: int = Field(..., description="레이블 ID")
    name: str = Field(..., description="레이블 이름")
    percentage: float = Field(..., description="이미지 내 해당 레이블이 차지하는 비율 (%)")


class SegmentationResponse(BaseModel):
    """세그멘테이션 응답 모델"""
    success: bool = Field(..., description="처리 성공 여부")
    original_image: str = Field(..., description="원본 이미지 (base64)")
    result_image: str = Field(..., description="결과 이미지 (base64)")
    detected_labels: List[LabelInfo] = Field(..., description="감지된 레이블 목록")
    message: str = Field(..., description="처리 결과 메시지")


class ErrorResponse(BaseModel):
    """에러 응답 모델"""
    success: bool = Field(False, description="처리 성공 여부")
    error: str = Field(..., description="에러 메시지")
    message: str = Field(..., description="사용자 친화적 에러 메시지")

