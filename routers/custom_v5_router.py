"""CustomV5 통합 트라이온 라우터 - 누끼 + V5 프롬프트 (X.AI 제거)"""
import io
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from services.custom_v5_service import generate_unified_tryon_custom_v5
from schemas.tryon_schema import UnifiedTryonResponse

router = APIRouter()


@router.post("/fit/custom-v5/compose", tags=["통합 트라이온 CustomV5"], response_model=UnifiedTryonResponse)
async def compose_custom_v5_endpoint(
    person_image: UploadFile = File(..., description="인물 이미지 파일"),
    garment_image: UploadFile = File(..., description="의상 이미지 파일"),
    background_image: UploadFile = File(..., description="배경 이미지 파일"),
):
    """
    CustomV5 통합 트라이온 파이프라인: 의상 누끼 + V5 프롬프트 + Gemini 3 Flash 직접 처리
    
    CustomV5는 V5 파이프라인에 누끼 기능을 추가한 버전입니다.
    X.AI 프롬프트 생성 없이 Gemini 3 Flash가 직접 처리합니다.
    
    파이프라인 단계:
    - Stage 0: 의상 이미지 누끼 처리 (배경 제거) - 자동 수행
    - Stage 1: Gemini 3 Flash로 의상 교체 + 배경 합성 통합 처리 (person + garment_nukki + background)
    
    Returns:
        UnifiedTryonResponse: 생성된 프롬프트와 합성 이미지 (base64)
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
        
        # CustomV5 통합 트라이온 서비스 호출
        result = await generate_unified_tryon_custom_v5(person_img, garment_img, background_img)
        
        if result["success"]:
            return JSONResponse(result)
        else:
            status_code = 500 if "error" in result else 400
            return JSONResponse(result, status_code=status_code)
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"CustomV5 통합 트라이온 엔드포인트 오류: {e}")
        print(error_detail)
        
        return JSONResponse(
            {
                "success": False,
                "prompt": "",
                "result_image": "",
                "message": f"CustomV5 통합 트라이온 처리 중 오류가 발생했습니다: {str(e)}",
                "llm": None
            },
            status_code=500,
        )

