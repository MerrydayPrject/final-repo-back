"""정보 엔드포인트 라우터"""
import time
from fastapi import APIRouter
from core.model_loader import processor, model
from config.settings import LABELS
from config.server_session import get_server_session

router = APIRouter()


@router.get("/health", tags=["정보"])
async def health_check():
    """
    서버 상태 확인
    
    서버와 모델의 로딩 상태를 확인합니다.
    서버 재시작 감지를 위한 세션 ID도 포함합니다.
    
    Returns:
        dict: 서버 상태 및 모델 로딩 여부, 서버 세션 ID
    """
    session_info = get_server_session()
    
    return {
        "status": "healthy",
        "model_loaded": model is not None and processor is not None,
        "model_name": "mattmdjaga/segformer_b2_clothes",
        "version": "1.0.0",
        "server_session_id": session_info["server_session_id"],
        "server_start_time": session_info["server_start_time"]
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
    
    SegFormer 모델이 감지할 수 있는 18개 의류/신체 부위 레이블 목록을 반환합니다.
    
    Returns:
        dict: 레이블 ID를 키로, 레이블 이름을 값으로 하는 딕셔너리
    """
    return {
        "labels": LABELS,
        "total_labels": len(LABELS),
        "description": "SegFormer B2 모델이 감지할 수 있는 레이블 목록"
    }

