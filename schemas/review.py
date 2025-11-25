"""리뷰 스키마"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ReviewCreate(BaseModel):
    """리뷰 생성 스키마"""
    rating: int = Field(..., ge=1, le=5, description="별점 (1-5)")
    content: Optional[str] = Field(None, description="리뷰 내용 (선택사항)")
    category: str = Field(..., description="카테고리 ('general', 'custom', 'analysis')")
    
    class Config:
        json_schema_extra = {
            "example": {
                "rating": 5,
                "content": "서비스가 매우 만족스럽습니다!",
                "category": "general"
            }
        }


class ReviewResponse(BaseModel):
    """리뷰 응답 스키마"""
    idx: int
    rating: int
    content: Optional[str]
    category: str
    created_at: datetime
    
    class Config:
        from_attributes = True

