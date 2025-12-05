"""프롬프트 테스트 라우터 - V4/V5 파이프라인 지원"""
import io
from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image

from services.prompt_test_service import (
    get_v4_prompt_files,
    get_v5_prompt_files,
    get_prompt_content,
    get_v5_prompt_content,
    run_single_pipeline,
    run_v5_pipeline,
    run_custom_prompt_pipeline,
    run_v5_custom_prompt_pipeline
)

router = APIRouter()


@router.get("/api/prompts/v4/list", tags=["프롬프트 테스트"])
async def list_v4_prompts():
    """
    V4 프롬프트 파일 목록 반환
    
    Returns:
        {
            "success": bool,
            "stage1": ["stage1_default.txt", "stage1_test.txt"],
            "stage2": ["prompt_stage2_outfit.txt", ...],
            "stage3": ["prompt_stage3_background_lighting.txt", ...]
        }
    """
    try:
        files = get_v4_prompt_files()
        return JSONResponse({
            "success": True,
            "stage1": files["stage1"],
            "stage2": files["stage2"],
            "stage3": files["stage3"]
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": str(e)
        }, status_code=500)


@router.get("/api/prompts/v4/{filename}", tags=["프롬프트 테스트"])
async def get_v4_prompt(filename: str):
    """
    특정 V4 프롬프트 파일 내용 반환
    
    Args:
        filename: 프롬프트 파일명
    
    Returns:
        {
            "success": bool,
            "filename": str,
            "content": str
        }
    """
    try:
        content = get_prompt_content(filename)
        if content is None:
            return JSONResponse({
                "success": False,
                "message": f"파일을 찾을 수 없습니다: {filename}"
            }, status_code=404)
        
        return JSONResponse({
            "success": True,
            "filename": filename,
            "content": content
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": str(e)
        }, status_code=500)


@router.post("/api/prompt-test/run", tags=["프롬프트 테스트"])
async def run_prompt_test_endpoint(
    person_image: UploadFile = File(..., description="인물 이미지"),
    garment_image: UploadFile = File(..., description="의상 이미지"),
    background_image: UploadFile = File(..., description="배경 이미지"),
    stage1_filename: str = Form(..., description="Stage 1 프롬프트 파일명 (Grok 시스템 프롬프트)"),
    stage2_filename: str = Form(..., description="Stage 2 프롬프트 파일명 (의상 교체)"),
    stage3_filename: str = Form(..., description="Stage 3 프롬프트 파일명 (배경/조명)")
):
    """
    선택한 프롬프트로 단일 V4 파이프라인 실행
    
    - Stage 1: Grok 시스템 프롬프트 (의상 분석 프롬프트 생성)
    - Stage 2: Gemini 의상 교체 프롬프트
    - Stage 3: Gemini 배경/조명 합성 프롬프트
    
    Returns:
        {
            "success": bool,
            "xai_prompt": str,
            "result": {
                "success": bool,
                "result_image": str (base64),
                "message": str
            },
            "message": str,
            "run_time": float,
            "prompts_used": {
                "stage1": str,
                "stage2": str,
                "stage3": str
            }
        }
    """
    try:
        # 이미지 읽기
        person_bytes = await person_image.read()
        garment_bytes = await garment_image.read()
        background_bytes = await background_image.read()
        
        if not person_bytes or not garment_bytes or not background_bytes:
            return JSONResponse({
                "success": False,
                "message": "인물, 의상, 배경 이미지를 모두 업로드해주세요."
            }, status_code=400)
        
        # PIL Image로 변환
        person_img = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        garment_img = Image.open(io.BytesIO(garment_bytes)).convert("RGB")
        background_img = Image.open(io.BytesIO(background_bytes)).convert("RGB")
        
        # 파이프라인 실행
        result = await run_single_pipeline(
            person_img,
            garment_img,
            background_img,
            stage1_filename,
            stage2_filename,
            stage3_filename
        )
        
        if result["success"]:
            return JSONResponse(result)
        else:
            return JSONResponse(result, status_code=500)
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"프롬프트 테스트 오류: {e}")
        print(error_detail)
        
        return JSONResponse({
            "success": False,
            "message": f"프롬프트 테스트 중 오류가 발생했습니다: {str(e)}"
        }, status_code=500)


@router.post("/api/prompt-test/run-custom", tags=["프롬프트 테스트"])
async def run_prompt_test_custom_endpoint(
    person_image: UploadFile = File(..., description="인물 이미지"),
    garment_image: UploadFile = File(..., description="의상 이미지"),
    background_image: UploadFile = File(..., description="배경 이미지"),
    stage1_prompt: str = Form(..., description="Stage 1 프롬프트 내용 (Grok 시스템 프롬프트)"),
    stage2_prompt: str = Form(..., description="Stage 2 프롬프트 내용 (의상 교체)"),
    stage3_prompt: str = Form(..., description="Stage 3 프롬프트 내용 (배경/조명)")
):
    """
    사용자가 직접 입력/수정한 프롬프트로 V4 파이프라인 실행
    
    Returns:
        {
            "success": bool,
            "xai_prompt": str,
            "result": {...},
            "message": str,
            "run_time": float
        }
    """
    try:
        # 이미지 읽기
        person_bytes = await person_image.read()
        garment_bytes = await garment_image.read()
        background_bytes = await background_image.read()
        
        if not person_bytes or not garment_bytes or not background_bytes:
            return JSONResponse({
                "success": False,
                "message": "인물, 의상, 배경 이미지를 모두 업로드해주세요."
            }, status_code=400)
        
        # PIL Image로 변환
        person_img = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        garment_img = Image.open(io.BytesIO(garment_bytes)).convert("RGB")
        background_img = Image.open(io.BytesIO(background_bytes)).convert("RGB")
        
        # 커스텀 프롬프트로 파이프라인 실행
        result = await run_custom_prompt_pipeline(
            person_img,
            garment_img,
            background_img,
            stage1_prompt,
            stage2_prompt,
            stage3_prompt
        )
        
        if result["success"]:
            return JSONResponse(result)
        else:
            return JSONResponse(result, status_code=500)
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"커스텀 프롬프트 테스트 오류: {e}")
        print(error_detail)
        
        return JSONResponse({
            "success": False,
            "message": f"커스텀 프롬프트 테스트 중 오류가 발생했습니다: {str(e)}"
        }, status_code=500)


# ============================================================
# V5 파이프라인 엔드포인트
# ============================================================

@router.get("/api/prompts/v5/list", tags=["프롬프트 테스트"])
async def list_v5_prompts():
    """
    V5 프롬프트 파일 목록 반환
    
    Returns:
        {
            "success": bool,
            "files": ["prompt_unified.txt", ...]
        }
    """
    try:
        files = get_v5_prompt_files()
        return JSONResponse({
            "success": True,
            "files": files
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": str(e)
        }, status_code=500)


@router.get("/api/prompts/v5/{filename}", tags=["프롬프트 테스트"])
async def get_v5_prompt(filename: str):
    """
    특정 V5 프롬프트 파일 내용 반환
    
    Args:
        filename: 프롬프트 파일명
    
    Returns:
        {
            "success": bool,
            "filename": str,
            "content": str
        }
    """
    try:
        content = get_v5_prompt_content(filename)
        if content is None:
            return JSONResponse({
                "success": False,
                "message": f"파일을 찾을 수 없습니다: {filename}"
            }, status_code=404)
        
        return JSONResponse({
            "success": True,
            "filename": filename,
            "content": content
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": str(e)
        }, status_code=500)


@router.post("/api/prompt-test/run-v5", tags=["프롬프트 테스트"])
async def run_prompt_test_v5_endpoint(
    person_image: UploadFile = File(..., description="인물 이미지"),
    garment_image: UploadFile = File(..., description="의상 이미지"),
    background_image: UploadFile = File(..., description="배경 이미지"),
    prompt_filename: str = Form(..., description="V5 통합 프롬프트 파일명")
):
    """
    선택한 프롬프트로 V5 파이프라인 실행 (X.AI 제거, Gemini 직접 처리)
    
    Returns:
        {
            "success": bool,
            "result": {
                "success": bool,
                "result_image": str (base64),
                "message": str
            },
            "message": str,
            "run_time": float,
            "prompt_used": str
        }
    """
    try:
        # 이미지 읽기
        person_bytes = await person_image.read()
        garment_bytes = await garment_image.read()
        background_bytes = await background_image.read()
        
        if not person_bytes or not garment_bytes or not background_bytes:
            return JSONResponse({
                "success": False,
                "message": "인물, 의상, 배경 이미지를 모두 업로드해주세요."
            }, status_code=400)
        
        # PIL Image로 변환
        person_img = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        garment_img = Image.open(io.BytesIO(garment_bytes)).convert("RGB")
        background_img = Image.open(io.BytesIO(background_bytes)).convert("RGB")
        
        # V5 파이프라인 실행
        result = await run_v5_pipeline(
            person_img,
            garment_img,
            background_img,
            prompt_filename
        )
        
        if result["success"]:
            return JSONResponse(result)
        else:
            return JSONResponse(result, status_code=500)
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"V5 프롬프트 테스트 오류: {e}")
        print(error_detail)
        
        return JSONResponse({
            "success": False,
            "message": f"V5 프롬프트 테스트 중 오류가 발생했습니다: {str(e)}"
        }, status_code=500)


@router.post("/api/prompt-test/run-v5-custom", tags=["프롬프트 테스트"])
async def run_prompt_test_v5_custom_endpoint(
    person_image: UploadFile = File(..., description="인물 이미지"),
    garment_image: UploadFile = File(..., description="의상 이미지"),
    background_image: UploadFile = File(..., description="배경 이미지"),
    unified_prompt: str = Form(..., description="V5 통합 프롬프트 내용")
):
    """
    사용자가 직접 입력/수정한 프롬프트로 V5 파이프라인 실행
    
    Returns:
        {
            "success": bool,
            "result": {...},
            "message": str,
            "run_time": float
        }
    """
    try:
        # 이미지 읽기
        person_bytes = await person_image.read()
        garment_bytes = await garment_image.read()
        background_bytes = await background_image.read()
        
        if not person_bytes or not garment_bytes or not background_bytes:
            return JSONResponse({
                "success": False,
                "message": "인물, 의상, 배경 이미지를 모두 업로드해주세요."
            }, status_code=400)
        
        # PIL Image로 변환
        person_img = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        garment_img = Image.open(io.BytesIO(garment_bytes)).convert("RGB")
        background_img = Image.open(io.BytesIO(background_bytes)).convert("RGB")
        
        # 커스텀 프롬프트로 V5 파이프라인 실행
        result = await run_v5_custom_prompt_pipeline(
            person_img,
            garment_img,
            background_img,
            unified_prompt
        )
        
        if result["success"]:
            return JSONResponse(result)
        else:
            return JSONResponse(result, status_code=500)
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"V5 커스텀 프롬프트 테스트 오류: {e}")
        print(error_detail)
        
        return JSONResponse({
            "success": False,
            "message": f"V5 커스텀 프롬프트 테스트 중 오류가 발생했습니다: {str(e)}"
        }, status_code=500)
