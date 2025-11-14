"""3D 변환 라우터"""
import time
import io
from fastapi import APIRouter, File, UploadFile, Query
from fastapi.responses import JSONResponse, Response
from PIL import Image
import requests

from core.meshy_client import create_3d_model_meshy, check_3d_task_status, save_3d_models_to_server

router = APIRouter()


@router.post("/api/convert-to-3d", tags=["3D 변환"])
async def convert_to_3d(
    image: UploadFile = File(..., description="변환할 이미지")
):
    """
    Meshy.ai를 사용하여 이미지를 3D 모델로 변환
    
    작업을 시작하고 task_id를 반환합니다.
    생성 완료까지 2-5분 소요됩니다.
    """
    start_time = time.time()
    
    try:
        # 이미지 읽기
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes))
        
        # RGB 변환 및 리사이즈 (API 최적화)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 이미지 크기 제한 (Meshy.ai 권장사항)
        max_size = 2048
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # 이미지를 바이트로 변환
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Meshy.ai API 호출
        result = create_3d_model_meshy(img_buffer.getvalue())
        
        if result.get("success"):
            return JSONResponse({
                "success": True,
                "task_id": result.get("task_id"),
                "message": "3D 모델 생성 작업이 시작되었습니다. 2-5분 정도 소요됩니다.",
                "processing_time": round(time.time() - start_time, 2)
            })
        else:
            return JSONResponse({
                "success": False,
                "error": result.get("error", "알 수 없는 오류"),
                "message": result.get("message", ""),
                "processing_time": round(time.time() - start_time, 2)
            }, status_code=400)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "processing_time": round(time.time() - start_time, 2)
        }, status_code=500)


@router.get("/api/check-3d-status/{task_id}", tags=["3D 변환"])
async def check_3d_status(task_id: str, save_to_server: bool = Query(False, description="서버에 자동 저장")):
    """
    3D 모델 생성 작업 상태 확인
    
    - task_id: 작업 ID
    - save_to_server: True면 완료 시 서버에 자동 저장
    
    Returns:
        - status: PENDING, IN_PROGRESS, SUCCEEDED, FAILED
        - progress: 0-100
        - model_urls: 완료 시 GLB, FBX 등 모델 다운로드 URL
        - saved_files: 서버에 저장된 파일 경로들 (save_to_server=True일 때)
    """
    try:
        result = check_3d_task_status(task_id)
        
        if result.get("success"):
            # 완료 상태이고 서버 저장 옵션이 활성화된 경우
            if save_to_server and result.get("status") == "SUCCEEDED":
                model_urls = result.get("model_urls", {})
                thumbnail_url = result.get("thumbnail_url")
                
                if model_urls:
                    print(f"[API] 서버에 3D 모델 저장 시작...")
                    saved_files = save_3d_models_to_server(task_id, model_urls, thumbnail_url)
                    result["saved_files"] = saved_files
                    result["saved_to_server"] = True
                    print(f"[API] 서버 저장 완료! 파일 수: {len(saved_files)}")
            
            return JSONResponse(result)
        else:
            return JSONResponse(result, status_code=400)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@router.post("/api/save-3d-model/{task_id}", tags=["3D 변환"])
async def save_3d_model(task_id: str):
    """
    완료된 3D 모델을 서버에 저장
    
    - task_id: 저장할 작업 ID
    
    Returns:
        저장된 파일 경로들
    """
    try:
        # 먼저 상태 확인
        result = check_3d_task_status(task_id)
        
        if not result.get("success"):
            return JSONResponse({
                "success": False,
                "error": "작업 상태 확인 실패"
            }, status_code=400)
        
        if result.get("status") != "SUCCEEDED":
            return JSONResponse({
                "success": False,
                "error": f"작업이 완료되지 않았습니다. 현재 상태: {result.get('status')}"
            }, status_code=400)
        
        # 모델 저장
        model_urls = result.get("model_urls", {})
        thumbnail_url = result.get("thumbnail_url")
        
        if not model_urls:
            return JSONResponse({
                "success": False,
                "error": "다운로드 가능한 모델이 없습니다."
            }, status_code=400)
        
        print(f"[API] 3D 모델 저장 요청 - Task ID: {task_id}")
        saved_files = save_3d_models_to_server(task_id, model_urls, thumbnail_url)
        
        return JSONResponse({
            "success": True,
            "task_id": task_id,
            "saved_files": saved_files,
            "message": f"{len(saved_files)}개 파일이 서버에 저장되었습니다."
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@router.get("/api/proxy-3d-model", tags=["3D 변환"])
async def proxy_3d_model(model_url: str = Query(..., description="다운로드할 3D 모델 URL")):
    """
    CORS 문제를 해결하기 위해 3D 모델 파일을 프록시하여 제공
    
    - model_url: Meshy.ai의 GLB/FBX 파일 URL
    
    Returns:
        3D 모델 파일 (binary)
    """
    try:
        print(f"[API] 3D 모델 프록시 요청: {model_url}")
        
        # Meshy.ai에서 파일 다운로드
        response = requests.get(model_url, timeout=60, stream=True)
        
        if response.status_code == 200:
            # CORS 헤더 추가
            headers = {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Content-Type": response.headers.get("Content-Type", "application/octet-stream"),
                "Content-Disposition": f'attachment; filename="model.glb"'
            }
            
            return Response(
                content=response.content,
                headers=headers,
                media_type=response.headers.get("Content-Type", "application/octet-stream")
            )
        else:
            return JSONResponse({
                "success": False,
                "error": f"파일 다운로드 실패: {response.status_code}"
            }, status_code=response.status_code)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@router.options("/api/proxy-3d-model", tags=["3D 변환"])
async def proxy_3d_model_options():
    """CORS preflight 요청 처리"""
    return Response(
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

