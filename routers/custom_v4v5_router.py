"""V4V5커스텀 비교 라우터"""
import io
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from services.custom_v4v5_compare_service import run_v4v5_custom_compare
from schemas.tryon_schema import V4V5CustomCompareResponse, UnifiedTryonResponse

router = APIRouter()


@router.post("/fit/v5v5-custom/compose", tags=["통합 트라이온 V5V5커스텀"], response_model=UnifiedTryonResponse)
async def compose_v5v5_custom(
    person_image: UploadFile = File(..., description="인물 이미지 파일"),
    garment_image: UploadFile = File(..., description="의상 이미지 파일"),
    background_image: UploadFile = File(..., description="배경 이미지 파일"),
):
    """
    V5V5커스텀 통합 트라이온 파이프라인: CustomV5 파이프라인을 두 번 병렬 실행하고 v5_result 반환
    
    - CustomV5-1: 의상 누끼 + V5 프롬프트 + Gemini 3 Flash 직접 처리
    - CustomV5-2: 의상 누끼 + V5 프롬프트 + Gemini 3 Flash 직접 처리
    
    같은 CustomV5 파이프라인을 두 번 병렬로 실행하고 v5_result만 반환합니다.
    /fit/ 경로로 배포 가능하도록 설계되었습니다.
    V4V5일반과 달리 의상 이미지에 누끼(배경 제거) 처리가 적용됩니다.
    
    Returns:
        UnifiedTryonResponse: v5_result를 직접 반환 (단일 결과)
    """
    try:
        # 이미지 읽기
        person_bytes = await person_image.read()
        garment_bytes = await garment_image.read()
        background_bytes = await background_image.read()
        
        if not person_bytes or not garment_bytes or not background_bytes:
            return JSONResponse(
                {
                    "success": False,
                    "prompt": "",
                    "result_image": "",
                    "message": "인물 이미지, 의상 이미지, 배경 이미지를 모두 업로드해주세요.",
                    "llm": None
                },
                status_code=400,
            )
        
        # PIL Image로 변환
        person_img = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        garment_img = Image.open(io.BytesIO(garment_bytes)).convert("RGB")
        background_img = Image.open(io.BytesIO(background_bytes)).convert("RGB")
        
        # V4V5커스텀 비교 실행 (로깅 비활성화 - 프론트엔드 커스텀 피팅용)
        # enable_logging=False: S3 업로드 및 DB 로그 저장 비활성화
        print("[DEBUG 라우터] /fit/v5v5-custom/compose - enable_logging=False로 호출")
        result = await run_v4v5_custom_compare(person_img, garment_img, background_img, enable_logging=False)
        
        # v5_result만 반환
        if result.get("success") and result.get("v5_result"):
            v5_result = result["v5_result"]
            
            # 날짜별 합성 카운트 증가 (v5_result가 성공한 경우에만)
            if v5_result.get("success", False):
                from services.synthesis_stats_service import increment_synthesis_count
                print("[커스텀 피팅] 합성 성공 - 카운팅 시작")
                try:
                    count_success = increment_synthesis_count()
                    if count_success:
                        print("[커스텀 피팅] ✅ 합성 카운트 증가 성공")
                    else:
                        print("[커스텀 피팅] ⚠️ 합성 카운트 증가 실패 (DB 연결 또는 쿼리 오류)")
                except Exception as e:
                    print(f"[커스텀 피팅] ❌ 합성 카운트 증가 예외 발생: {e}")
            
            return JSONResponse({
                "success": v5_result.get("success", False),
                "prompt": v5_result.get("prompt", ""),
                "result_image": v5_result.get("result_image", ""),
                "message": v5_result.get("message") or result.get("message", "V5V5커스텀 파이프라인이 완료되었습니다."),
                "llm": v5_result.get("llm")
            })
        else:
            return JSONResponse(
                {
                    "success": False,
                    "prompt": "",
                    "result_image": "",
                    "message": result.get("message", "V5V5커스텀 파이프라인 처리 중 오류가 발생했습니다."),
                    "llm": None
                },
                status_code=500,
            )
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"V5V5커스텀 통합 트라이온 엔드포인트 오류: {e}")
        print(error_detail)
        
        return JSONResponse(
            {
                "success": False,
                "prompt": "",
                "result_image": "",
                "message": f"V5V5커스텀 통합 트라이온 처리 중 오류가 발생했습니다: {str(e)}",
                "llm": None
            },
            status_code=500,
        )


@router.post("/tryon/compare/custom", tags=["V4V5커스텀"], response_model=V4V5CustomCompareResponse)
async def compare_v4v5_custom(
    person_image: UploadFile = File(..., description="인물 이미지 파일"),
    garment_image: UploadFile = File(..., description="의상 이미지 파일"),
    background_image: UploadFile = File(..., description="배경 이미지 파일"),
):
    """
    V4V5커스텀 비교 엔드포인트: CustomV5 파이프라인을 두 번 병렬 실행하고 두 결과를 반환
    
    - CustomV5-1: 의상 누끼 + V5 프롬프트 + Gemini 3 Flash 직접 처리
    - CustomV5-2: 의상 누끼 + V5 프롬프트 + Gemini 3 Flash 직접 처리
    
    같은 CustomV5 파이프라인을 두 번 병렬로 실행하여 결과를 비교할 수 있습니다.
    V4V5일반과 달리 의상 이미지에 누끼(배경 제거) 처리가 적용됩니다.
    
    Returns:
        V4V5CustomCompareResponse: CustomV5-1과 CustomV5-2 결과를 모두 포함한 비교 응답
    """
    try:
        # 이미지 읽기
        person_bytes = await person_image.read()
        garment_bytes = await garment_image.read()
        background_bytes = await background_image.read()
        
        if not person_bytes or not garment_bytes or not background_bytes:
            return JSONResponse(
                {
                    "success": False,
                    "v4_result": {"success": False, "prompt": "", "result_image": "", "message": "입력 오류", "llm": None},
                    "v5_result": {"success": False, "prompt": "", "result_image": "", "message": "입력 오류", "llm": None},
                    "total_time": 0.0,
                    "message": "인물 이미지, 의상 이미지, 배경 이미지를 모두 업로드해주세요."
                },
                status_code=400,
            )
        
        # PIL Image로 변환
        person_img = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        garment_img = Image.open(io.BytesIO(garment_bytes)).convert("RGB")
        background_img = Image.open(io.BytesIO(background_bytes)).convert("RGB")
        
        # V4V5커스텀 비교 실행
        result = await run_v4v5_custom_compare(person_img, garment_img, background_img)
        
        # 날짜별 합성 카운트 증가 (v5_result가 성공한 경우에만)
        if result.get("success") and result.get("v5_result"):
            v5_result = result["v5_result"]
            if v5_result.get("success", False):
                from services.synthesis_stats_service import increment_synthesis_count
                print("[커스텀 피팅] 합성 성공 - 카운팅 시작")
                try:
                    count_success = increment_synthesis_count()
                    if count_success:
                        print("[커스텀 피팅] ✅ 합성 카운트 증가 성공")
                    else:
                        print("[커스텀 피팅] ⚠️ 합성 카운트 증가 실패 (DB 연결 또는 쿼리 오류)")
                except Exception as e:
                    print(f"[커스텀 피팅] ❌ 합성 카운트 증가 예외 발생: {e}")
        
        if result["success"]:
            return JSONResponse(result)
        else:
            return JSONResponse(result, status_code=500)
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"V4V5커스텀 비교 엔드포인트 오류: {e}")
        print(error_detail)
        
        return JSONResponse(
            {
                "success": False,
                "v4_result": {"success": False, "prompt": "", "result_image": "", "message": str(e), "llm": None},
                "v5_result": {"success": False, "prompt": "", "result_image": "", "message": str(e), "llm": None},
                "total_time": 0.0,
                "message": f"V4V5커스텀 비교 처리 중 오류가 발생했습니다: {str(e)}"
            },
            status_code=500,
        )

