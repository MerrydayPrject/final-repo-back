import time
from fastapi import APIRouter
# 'get_processor', 'get_model'을 명확한 새 이름으로 변경
from core.model_loader import get_segformer_b2_processor, get_segformer_b2_model
from config.settings import LABELS

router = APIRouter()

@router.get("/health", tags=["정보"])
async def health_check():
    """
    서버 상태 확인
    
    서버와 모델의 로딩 상태를 확인합니다.
    
    Returns:
        dict: 서버 상태 및 모델 로딩 여부
    """
    # 변경된 새 함수 이름으로 호출
    processor = get_segformer_b2_processor()
    model = get_segformer_b2_model()
    
    return {
        "status": "healthy",
        "model_loaded": model is not None and processor is not None,
        # model_loader.py에서 실제 로드하는 모델 이름으로 수정
        "model_name": "yolo12138/segformer-b2-human-parse-24",
        "version": "1.0.0"
    }

@router.get("/test", tags=["테스트"])
async def test_endpoint():
    """
    간단한 테스트 엔드포인트
    
    서버가 정상적으로 응답하는지 확인합니다.
    """
    return {
        "message": "서버가 정상적으로 작동 중입니다!",
        "timestamp": time.time()
    }

@router.get("/labels", tags=["정보"])
async def get_labels():
    """
    사용 가능한 모든 레이블 목록 조회
    
    SegFormer 모델이 감지할 수 있는 의류/신체 부위 레이블 목록을 반환합니다.
    
    Returns:
        dict: 레이블 ID를 키로, 레이블 이름을 값으로 하는 딕셔너리
    """
    return {
        "labels": LABELS,
        "total_labels": len(LABELS),
        "description": "SegFormer B2 모델이 감지할 수 있는 레이블 목록"
    }