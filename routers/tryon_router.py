"""통합 트라이온 라우터"""
import io
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from services.tryon_service import generate_unified_tryon
from services.face_swap_service import FaceSwapService
from schemas.tryon_schema import UnifiedTryonResponse

router = APIRouter()


@router.post("/api/tryon/unified", tags=["통합 트라이온"], response_model=UnifiedTryonResponse)
async def unified_tryon(
    person_image: UploadFile = File(..., description="사람 이미지 파일"),
    dress_image: UploadFile = File(..., description="드레스 이미지 파일"),
):
    """
    통합 트라이온 파이프라인: X.AI 프롬프트 생성 + Gemini 2.5 Flash 이미지 합성
    
    이 엔드포인트는 다음 단계를 수행합니다:
    1. 이미지 타입 감지 (전신 사진인지 확인)
    2. 전신 사진이면 합성 불가 메시지 반환
    3. 상체/얼굴 사진이면 X.AI를 사용하여 프롬프트 생성 후 Gemini 2.5 Flash로 합성
    
    Returns:
        UnifiedTryonResponse: 생성된 프롬프트와 합성 이미지 (base64)
    """
    try:
        # 이미지 읽기
        person_bytes = await person_image.read()
        dress_bytes = await dress_image.read()
        
        if not person_bytes or not dress_bytes:
            return JSONResponse(
                {
                    "success": False,
                    "prompt": "",
                    "result_image": "",
                    "message": "사람 이미지와 드레스 이미지를 모두 업로드해주세요.",
                    "llm": None
                },
                status_code=400,
            )
        
        # PIL Image로 변환
        person_img = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        dress_img = Image.open(io.BytesIO(dress_bytes)).convert("RGB")
        
        # 이미지 타입 감지 (전신 vs 상체/얼굴)
        face_swap_service = FaceSwapService()
        image_type_info = face_swap_service.detect_image_type(person_img)
        image_type = image_type_info.get("type", "unknown")
        confidence = image_type_info.get("confidence", 0.0)
        
        # 전신 사진이면 합성 불가 메시지 반환
        if image_type == "full_body":
            return JSONResponse(
                {
                    "success": False,
                    "prompt": "",
                    "result_image": "",
                    "message": "지금 올려주신 사진은 전신사진입니다. 상체만 나온 사진이나 얼굴만 나온 사진을 업로드해주세요.",
                    "llm": None,
                    "image_type": image_type,
                    "image_type_confidence": round(confidence, 2)
                },
                status_code=400,
            )
        
        # 통합 트라이온 서비스 호출 (상체/얼굴 사진인 경우)
        result = generate_unified_tryon(person_img, dress_img)
        
        # 결과에 이미지 타입 정보 추가
        if isinstance(result, dict):
            result["image_type"] = image_type
            result["image_type_confidence"] = round(confidence, 2)
        
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

