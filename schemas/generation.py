"""이미지 생성 API를 위한 Pydantic 스키마"""
from pydantic import BaseModel
from typing import Optional

class GenerationResponse(BaseModel):
    message: str
    image_url: Optional[str] = None