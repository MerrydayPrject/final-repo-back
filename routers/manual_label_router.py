from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter(prefix="/api/manual-label", tags=["관리자"])

# 수동 라벨용 Pydantic 모델
class ManualLabel(BaseModel):
    image_id: str
    label: str  # "dress" 또는 "not-dress"
    confidence: Optional[float] = None

# 임시 DB (나중에 실제 DB로 연결 가능)
manual_labels_db: List[ManualLabel] = []

@router.post("/", response_model=ManualLabel)
async def save_manual_label(label: ManualLabel):
    """
    수동 라벨 저장
    """
    manual_labels_db.append(label)
    return label

@router.get("/", response_model=List[ManualLabel])
async def get_manual_labels():
    """
    저장된 수동 라벨 조회
    """
    return manual_labels_db
