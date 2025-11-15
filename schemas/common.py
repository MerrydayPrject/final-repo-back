"""공통 스키마"""
from pydantic import BaseModel
from typing import Optional


class ShortPromptResponse(BaseModel):
    """Short prompt 생성 응답 스키마"""
    success: bool
    prompt: str
    llm: str
    message: Optional[str] = None

