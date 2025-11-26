"""CustomV4 통합 트라이온 라우터"""
import io
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from services.custom_v4_service import generate_unified_tryon_custom_v4
from schemas.tryon_schema import UnifiedTryonResponse

router = APIRouter()


@router.post("/fit/custom-v4/compose", tags=["통합 트라이온 CustomV4"], response_model=UnifiedTryonResponse)
async def compose_custom_v4_endpoint(
    person_image: UploadFile = File(..., description="인물 이미지 파일"),
    garment_image: UploadFile = File(..., description="의상 이미지 파일"),
    background_image: UploadFile = File(..., description="배경 이미지 파일"),
):
    """
    CustomV4 통합 트라이온 파이프라인: 의상 누끼 + X.AI 프롬프트 생성 + 2단계 Gemini 3 플로우
    
    CustomV4는 기존 V3 커스텀과 동일한 구조를 가지지만, Gemini 3 Flash 모델을 사용합니다.
    
    파이프라인 단계:
    - Stage 0: 의상 이미지 누끼 처리 (배경 제거) - 자동 수행
    - Stage 1: 누끼 처리된 의상 이미지로 X.AI 프롬프트 생성
    - Stage 2: Gemini 3로 의상 교체만 수행 (person + garment_nukki)
    - Stage 3: Gemini 3로 배경 합성 + 조명 보정 (dressed_person + background)
    
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
        
        # CustomV4 통합 트라이온 서비스 호출
        result = await generate_unified_tryon_custom_v4(person_img, garment_img, background_img)
        
        if result["success"]:
            return JSONResponse(result)
        else:
            status_code = 500 if "error" in result else 400
            return JSONResponse(result, status_code=status_code)
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"CustomV4 통합 트라이온 엔드포인트 오류: {e}")
        print(error_detail)
        
        return JSONResponse(
            {
                "success": False,
                "prompt": "",
                "result_image": "",
                "message": f"CustomV4 통합 트라이온 처리 중 오류가 발생했습니다: {str(e)}",
                "llm": None
            },
            status_code=500,
        )

