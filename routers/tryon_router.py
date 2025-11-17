"""통합 트라이온 라우터"""
import io
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from services.tryon_service import generate_unified_tryon, generate_unified_tryon_v2
from schemas.tryon_schema import UnifiedTryonResponse

router = APIRouter()


@router.post("/api/tryon/unified", tags=["통합 트라이온"], response_model=UnifiedTryonResponse)
async def unified_tryon(
    person_image: UploadFile = File(..., description="사람 이미지 파일"),
    dress_image: UploadFile = File(..., description="드레스 이미지 파일"),
    background_image: UploadFile = File(..., description="배경 이미지 파일"),
):
    """
    통합 트라이온 파이프라인: X.AI 프롬프트 생성 + Gemini 2.5 Flash 이미지 합성 (배경 포함)
    
    이 엔드포인트는 다음 단계를 수행합니다:
    1. X.AI를 사용하여 person_image와 dress_image로부터 프롬프트 생성
    2. 생성된 프롬프트와 이미지들(인물, 드레스, 배경)을 사용하여 Gemini 2.5 Flash로 최종 합성 이미지 생성
    
    Returns:
        UnifiedTryonResponse: 생성된 프롬프트와 합성 이미지 (base64)
    """
    try:
        # 이미지 읽기
        person_bytes = await person_image.read()
        dress_bytes = await dress_image.read()
        background_bytes = await background_image.read()
        
        if not person_bytes or not dress_bytes or not background_bytes:
            return JSONResponse(
                {
                    "success": False,
                    "prompt": "",
                    "result_image": "",
                    "message": "사람 이미지, 드레스 이미지, 배경 이미지를 모두 업로드해주세요.",
                    "llm": None
                },
                status_code=400,
            )
        
        # PIL Image로 변환
        person_img = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        dress_img = Image.open(io.BytesIO(dress_bytes)).convert("RGB")
        background_img = Image.open(io.BytesIO(background_bytes)).convert("RGB")
        
        # 통합 트라이온 서비스 호출
        result = generate_unified_tryon(person_img, dress_img, background_img)
        
        if result["success"]:
            return JSONResponse(result)
        else:
            status_code = 500 if "error" in result else 400
            return JSONResponse(result, status_code=status_code)
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"통합 트라이온 엔드포인트 오류: {e}")
        print(error_detail)
        
        return JSONResponse(
            {
                "success": False,
                "prompt": "",
                "result_image": "",
                "message": f"통합 트라이온 처리 중 오류가 발생했습니다: {str(e)}",
                "llm": None
            },
            status_code=500,
        )


@router.post("/api/compose_xai_gemini_v2", tags=["통합 트라이온 V2"], response_model=UnifiedTryonResponse)
async def compose_xai_gemini_v2(
    person_image: UploadFile = File(..., description="사람 이미지 파일"),
    garment_image: UploadFile = File(..., description="의상 이미지 파일"),
    background_image: UploadFile = File(..., description="배경 이미지 파일"),
):
    """
    통합 트라이온 파이프라인 V2: SegFormer B2 Garment Parsing + X.AI 프롬프트 생성 + Gemini 2.5 Flash 이미지 합성 (배경 포함)
    
    V2는 다음 단계를 수행합니다:
    1. SegFormer B2 Human Parsing을 사용하여 garment_image에서 garment_only 이미지 추출
    2. X.AI를 사용하여 person_image와 garment_only 이미지로부터 프롬프트 생성
    3. 생성된 프롬프트와 이미지들(인물, garment_only, 배경)을 사용하여 Gemini 2.5 Flash로 최종 합성 이미지 생성
    
    V2는 배경 이미지를 포함하여 합성합니다.
    
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
                    "message": "사람 이미지, 의상 이미지, 배경 이미지를 모두 업로드해주세요.",
                    "llm": None
                },
                status_code=400,
            )
        
        # PIL Image로 변환
        person_img = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        garment_img = Image.open(io.BytesIO(garment_bytes)).convert("RGB")
        background_img = Image.open(io.BytesIO(background_bytes)).convert("RGB")
        
        # V2 통합 트라이온 서비스 호출
        result = generate_unified_tryon_v2(person_img, garment_img, background_img)
        
        if result["success"]:
            return JSONResponse(result)
        else:
            status_code = 500 if "error" in result else 400
            return JSONResponse(result, status_code=status_code)
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"통합 트라이온 V2 엔드포인트 오류: {e}")
        print(error_detail)
        
        return JSONResponse(
            {
                "success": False,
                "prompt": "",
                "result_image": "",
                "message": f"통합 트라이온 V2 처리 중 오류가 발생했습니다: {str(e)}",
                "llm": None
            },
            status_code=500,
        )

