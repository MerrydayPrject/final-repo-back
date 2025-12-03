"""누끼V2 라우터 - Gemini3 + OpenAI Ghost Mannequin"""
from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import io
from PIL import Image

from services.nukki_v2_service import process_nukki_v2

router = APIRouter(tags=["누끼V2"])
templates = Jinja2Templates(directory="templates")


@router.get("/nukki-v2", response_class=HTMLResponse)
async def nukki_v2_page(request: Request):
    """누끼V2 테스트 페이지"""
    return templates.TemplateResponse("nukki_v2.html", {"request": request})


@router.post("/api/nukki-v2/process")
async def process_nukki_v2_api(
    file: UploadFile = File(..., description="드레스 이미지 파일")
):
    """
    누끼V2 처리 API
    
    Gemini3와 OpenAI DALL-E-3 두 모델을 동시에 실행하여
    Ghost Mannequin 이미지를 생성합니다.
    
    - 입력 이미지와 결과 이미지는 S3에 자동 저장됩니다.
    - 처리 결과는 MySQL에 자동 로깅됩니다.
    - 한 모델이 실패해도 다른 모델 결과는 반환됩니다.
    
    Args:
        file: 드레스 이미지 파일
    
    Returns:
        {
            "success": bool,
            "gemini3": {
                "success": bool,
                "result_image": str (base64 data URL),
                "model": str,
                "run_time": float,
                "message": str,
                "error": str | null,
                "result_url": str (S3 URL)
            },
            "openai": {
                "success": bool,
                "result_image": str (base64 data URL),
                "model": str,
                "run_time": float,
                "message": str,
                "error": str | null,
                "result_url": str (S3 URL)
            },
            "input_url": str (입력 이미지 S3 URL),
            "message": str
        }
    """
    try:
        # 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        print(f"[NukkiV2 Router] 이미지 업로드 완료 - 크기: {image.size}")
        
        # 누끼V2 처리 (S3 저장 + DB 로깅 포함)
        result = await process_nukki_v2(
            dress_img=image,
            save_to_s3=True,
            save_to_db=True
        )
        
        return JSONResponse(result)
        
    except Exception as e:
        print(f"[NukkiV2 Router] 오류: {e}")
        return JSONResponse({
            "success": False,
            "gemini3": {
                "success": False,
                "result_image": None,
                "model": "gemini-3",
                "run_time": 0,
                "message": f"처리 오류: {str(e)}",
                "error": str(e)
            },
            "openai": {
                "success": False,
                "result_image": None,
                "model": "dall-e-3",
                "run_time": 0,
                "message": f"처리 오류: {str(e)}",
                "error": str(e)
            },
            "input_url": None,
            "message": f"처리 중 오류 발생: {str(e)}"
        }, status_code=500)


@router.post("/api/nukki-v2/process-single/{model}")
async def process_nukki_v2_single(
    model: str,
    file: UploadFile = File(..., description="드레스 이미지 파일")
):
    """
    누끼V2 단일 모델 재시도 API
    
    특정 모델만 다시 실행합니다. (재시도용)
    
    Args:
        model: 모델 이름 ("gemini3" 또는 "openai")
        file: 드레스 이미지 파일
    
    Returns:
        해당 모델의 처리 결과
    """
    from services.nukki_v2_service import (
        process_with_gemini3, 
        process_with_openai,
        load_prompt_from_file,
        base64_to_bytes
    )
    from core.s3_client import upload_log_to_s3
    from services.log_service import save_test_log
    from core.openai_image_client import load_prompt_from_file
    
    try:
        # 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        print(f"[NukkiV2 Router] 단일 모델 처리 시작 - 모델: {model}")
        
        prompt = load_prompt_from_file()
        
        if model.lower() == "gemini3":
            result = await process_with_gemini3(image, prompt)
            model_id = "gemini-3"
        elif model.lower() == "openai":
            result = await process_with_openai(image)
            model_id = "dall-e-3"
        else:
            return JSONResponse({
                "success": False,
                "message": f"알 수 없는 모델: {model}. 'gemini3' 또는 'openai'를 사용하세요."
            }, status_code=400)
        
        # 결과 S3 저장 및 DB 로깅
        if result.get("success") and result.get("result_image"):
            try:
                result_bytes = base64_to_bytes(result["result_image"])
                result_url = upload_log_to_s3(
                    file_content=result_bytes,
                    model_id=model_id,
                    image_type="result"
                )
                result["result_url"] = result_url
                
                # DB 저장
                save_test_log(
                    person_url="",
                    result_url=result_url or "",
                    model=f"nukki-v2-{model_id}",
                    prompt=prompt[:500],
                    success=True,
                    run_time=result.get("run_time", 0),
                    dress_url=""
                )
            except Exception as e:
                print(f"[NukkiV2 Router] 결과 저장 실패: {e}")
        
        return JSONResponse(result)
        
    except Exception as e:
        print(f"[NukkiV2 Router] 단일 모델 처리 오류: {e}")
        return JSONResponse({
            "success": False,
            "result_image": None,
            "model": model,
            "run_time": 0,
            "message": f"처리 오류: {str(e)}",
            "error": str(e)
        }, status_code=500)

