"""V4V5커스텀 비교 라우터"""
import io
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from services.custom_v4v5_compare_service import run_v4v5_custom_compare
from schemas.tryon_schema import V4V5CustomCompareResponse

router = APIRouter()


@router.post("/tryon/compare/custom", tags=["V4V5커스텀"], response_model=V4V5CustomCompareResponse)
async def compare_v4v5_custom(
    person_image: UploadFile = File(..., description="인물 이미지 파일"),
    garment_image: UploadFile = File(..., description="의상 이미지 파일"),
    background_image: UploadFile = File(..., description="배경 이미지 파일"),
):
    """
    V4V5커스텀 비교 엔드포인트: CustomV4/CustomV5 파이프라인을 병렬 실행하고 두 결과를 반환
    
    - CustomV4: 의상 누끼 + X.AI 프롬프트 생성 + Gemini 3 Flash
    - CustomV5: 의상 누끼 + V5 프롬프트 + Gemini 3 Flash 직접 처리
    
    두 파이프라인을 병렬로 실행하여 결과를 비교할 수 있습니다.
    V4V5일반과 달리 의상 이미지에 누끼(배경 제거) 처리가 적용됩니다.
    
    Returns:
        V4V5CustomCompareResponse: CustomV4와 CustomV5 결과를 모두 포함한 비교 응답
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

