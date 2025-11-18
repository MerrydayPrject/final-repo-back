"""Fitting 파이프라인 스키마"""
from pydantic import BaseModel
from typing import Optional


class PersonPreprocessResult(BaseModel):
    """인물 전처리 결과 스키마"""
    face_mask: str  # base64 인코딩된 face_mask 이미지
    face_patch: str  # base64 인코딩된 face_patch 이미지
    base_img: str  # base64 인코딩된 base_img 이미지
    inpaint_mask: str  # base64 인코딩된 inpaint_mask 이미지
    message: Optional[str] = None


class ComposeV25Request(BaseModel):
    """Compose V2.5 요청 스키마 (선택적)"""
    use_person_preprocess: bool = True
    person_preprocess_data: Optional[PersonPreprocessResult] = None

