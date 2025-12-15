"""합성 품질 평가 라우터"""
from fastapi import APIRouter, Query, Path, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Optional

from services.synthesis_quality_service import (
    evaluate_synthesis_quality,
    evaluate_batch_synthesis_quality,
    get_quality_statistics,
    get_unevaluated_results,
    save_manual_evaluation,
    get_manual_evaluation_statistics
)

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.post("/api/synthesis-quality/evaluate/{result_log_idx}", tags=["합성 품질 평가"])
async def evaluate_single_synthesis(
    result_log_idx: int = Path(..., description="result_logs.idx")
):
    """
    단일 합성 결과 이미지를 평가합니다.
    
    Args:
        result_log_idx: result_logs 테이블의 idx
    
    Returns:
        평가 결과
    """
    try:
        result = await evaluate_synthesis_quality(result_log_idx)
        
        if result["success"]:
            return JSONResponse({
                "success": True,
                "result_log_idx": result["result_log_idx"],
                "evaluation_idx": result.get("evaluation_idx"),
                "quality_score": result.get("quality_score"),
                "quality_comment": result.get("quality_comment"),
                "is_success": result.get("is_success"),
                "message": "평가가 완료되었습니다."
            })
        else:
            return JSONResponse({
                "success": False,
                "result_log_idx": result["result_log_idx"],
                "error": result.get("error", "알 수 없는 오류"),
                "message": f"평가 중 오류가 발생했습니다: {result.get('error', '알 수 없는 오류')}"
            }, status_code=500)
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"평가 중 예상치 못한 오류가 발생했습니다: {str(e)}"
        }, status_code=500)


@router.post("/api/synthesis-quality/evaluate-batch", tags=["합성 품질 평가"])
async def evaluate_batch_synthesis(
    model: Optional[str] = Query(None, description="모델 필터 (예: 'xai-gemini-unified-v3')"),
    limit: Optional[int] = Query(None, ge=1, le=1000, description="평가 개수 제한")
):
    """
    배치로 합성 결과 이미지를 평가합니다.
    
    Args:
        model: 모델 필터 (None이면 모든 모델)
        limit: 평가 개수 제한 (None이면 제한 없음, 최대 1000)
    
    Returns:
        배치 평가 결과
    """
    try:
        result = await evaluate_batch_synthesis_quality(model=model, limit=limit)
        
        if result["success"]:
            return JSONResponse({
                "success": True,
                "total": result["total"],
                "evaluated": result["evaluated"],
                "failed": result["failed"],
                "results": result["results"],
                "message": f"배치 평가가 완료되었습니다. 총 {result['total']}개 중 {result['evaluated']}개 성공, {result['failed']}개 실패"
            })
        else:
            return JSONResponse({
                "success": False,
                "error": result.get("error", "알 수 없는 오류"),
                "message": f"배치 평가 중 오류가 발생했습니다: {result.get('error', '알 수 없는 오류')}"
            }, status_code=500)
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"배치 평가 중 예상치 못한 오류가 발생했습니다: {str(e)}"
        }, status_code=500)


@router.get("/api/synthesis-quality/statistics", tags=["합성 품질 평가"])
async def get_synthesis_quality_statistics(
    model: Optional[str] = Query(None, description="모델 필터 (예: 'xai-gemini-unified-v3')"),
    start_date: Optional[str] = Query(None, description="시작 날짜 (YYYY-MM-DD 형식)"),
    end_date: Optional[str] = Query(None, description="종료 날짜 (YYYY-MM-DD 형식)")
):
    """
    성공률 통계를 조회합니다.
    
    Args:
        model: 모델 필터 (None이면 모든 모델)
        start_date: 시작 날짜 (YYYY-MM-DD 형식, None이면 제한 없음)
        end_date: 종료 날짜 (YYYY-MM-DD 형식, None이면 제한 없음)
    
    Returns:
        성공률 통계
    """
    try:
        result = get_quality_statistics(model=model, start_date=start_date, end_date=end_date)
        
        if result["success"]:
            return JSONResponse({
                "success": True,
                "overall": result["overall"],
                "by_model": result["by_model"],
                "message": "통계 조회가 완료되었습니다."
            })
        else:
            return JSONResponse({
                "success": False,
                "error": result.get("error", "알 수 없는 오류"),
                "message": f"통계 조회 중 오류가 발생했습니다: {result.get('error', '알 수 없는 오류')}"
            }, status_code=500)
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"통계 조회 중 예상치 못한 오류가 발생했습니다: {str(e)}"
        }, status_code=500)


@router.get("/synthesis-quality-test", response_class=HTMLResponse, tags=["합성 품질 평가"])
async def synthesis_quality_test_page(request: Request):
    """합성 품질 수동 평가 테스트 페이지"""
    return templates.TemplateResponse("synthesis_quality_test.html", {"request": request})


@router.get("/api/synthesis-quality-test/images", tags=["합성 품질 평가"])
async def get_synthesis_test_images(
    model: Optional[str] = Query(None, description="모델 필터 (예: 'xai-gemini-unified-v3')"),
    limit: Optional[int] = Query(None, ge=1, description="조회할 레코드 수 (None이면 전체 조회)"),
    offset: int = Query(0, ge=0, description="시작 위치"),
    include_evaluated: bool = Query(False, description="이미 평가된 항목도 포함할지 여부")
):
    """
    평가할 이미지 목록 조회 (result_logs에서 result_url 가져오기)
    
    Args:
        model: 모델 필터 (None이면 모든 모델)
        limit: 조회할 레코드 수 (None이면 전체 조회)
        offset: 시작 위치 (기본값: 0)
        include_evaluated: 이미 평가된 항목도 포함할지 여부
    
    Returns:
        이미지 목록 및 통계
    """
    try:
        result = get_unevaluated_results(
            model=model,
            limit=limit,
            offset=offset,
            include_evaluated=include_evaluated
        )
        
        if result["success"]:
            return JSONResponse({
                "success": True,
                "images": result["images"],
                "total": result["total"]
            })
        else:
            return JSONResponse({
                "success": False,
                "error": result.get("error", "알 수 없는 오류"),
                "images": [],
                "total": 0
            }, status_code=500)
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "images": [],
            "total": 0
        }, status_code=500)


@router.post("/api/synthesis-quality-test/submit", tags=["합성 품질 평가"])
async def submit_manual_evaluation(data: dict):
    """
    수동 평가 결과 저장
    
    Request Body:
        {
            "result_log_idx": 1,
            "is_success": true  // true=예, false=아니오
        }
    
    Returns:
        평가 저장 결과
    """
    try:
        result_log_idx = data.get("result_log_idx")
        is_success = data.get("is_success")
        
        if result_log_idx is None:
            return JSONResponse({
                "success": False,
                "error": "result_log_idx는 필수입니다."
            }, status_code=400)
        
        if is_success is None:
            return JSONResponse({
                "success": False,
                "error": "is_success는 필수입니다."
            }, status_code=400)
        
        result = save_manual_evaluation(
            result_log_idx=int(result_log_idx),
            is_success=bool(is_success)
        )
        
        if result["success"]:
            return JSONResponse({
                "success": True,
                "evaluation_idx": result.get("evaluation_idx"),
                "message": "평가가 저장되었습니다."
            })
        else:
            return JSONResponse({
                "success": False,
                "error": result.get("error", "알 수 없는 오류"),
                "message": f"평가 저장 중 오류가 발생했습니다: {result.get('error', '알 수 없는 오류')}"
            }, status_code=500)
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"평가 저장 중 예상치 못한 오류가 발생했습니다: {str(e)}"
        }, status_code=500)


@router.get("/api/synthesis-quality-test/statistics", tags=["합성 품질 평가"])
async def get_manual_evaluation_statistics_api(
    model: Optional[str] = Query(None, description="모델 필터 (예: 'xai-gemini-unified-v3')")
):
    """
    수동 평가 통계 조회 (파이프라인별 성공률)
    
    Args:
        model: 모델 필터 (None이면 모든 모델)
    
    Returns:
        수동 평가 통계
    """
    try:
        result = get_manual_evaluation_statistics(model=model)
        
        if result["success"]:
            return JSONResponse({
                "success": True,
                "overall": result["overall"],
                "by_model": result["by_model"],
                "message": "통계 조회가 완료되었습니다."
            })
        else:
            return JSONResponse({
                "success": False,
                "error": result.get("error", "알 수 없는 오류"),
                "message": f"통계 조회 중 오류가 발생했습니다: {result.get('error', '알 수 없는 오류')}"
            }, status_code=500)
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"통계 조회 중 예상치 못한 오류가 발생했습니다: {str(e)}"
        }, status_code=500)

