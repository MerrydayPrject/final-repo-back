"""이미지 프록시 라우터"""
import os
import requests
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, Response
from urllib.parse import urlparse, unquote
from core.s3_client import get_s3_image, get_logs_s3_image

router = APIRouter()


@router.get("/api/proxy-image", tags=["이미지 프록시"])
async def proxy_image_by_url(url: str = Query(..., description="S3 이미지 URL")):
    """
    S3 URL로 이미지 프록시 (썸네일용)
    
    프론트엔드에서 S3 이미지를 직접 로드할 때 CORS 문제를 해결하기 위한 프록시
    """
    try:
        # S3 URL에서 파일명 추출
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.strip('/').split('/')
        
        if len(path_parts) >= 2 and path_parts[0] == 'dresses':
            file_name = path_parts[1]
            # URL 디코딩
            file_name = unquote(file_name)
            
            # S3에서 이미지 다운로드
            image_data = get_s3_image(file_name)
            
            if image_data:
                return Response(
                    content=image_data,
                    media_type="image/png"
                )
            else:
                return JSONResponse({
                    "success": False,
                    "error": "Image not found",
                    "message": "이미지를 찾을 수 없습니다."
                }, status_code=404)
        else:
            # 직접 URL로 다운로드 시도
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    return Response(
                        content=response.content,
                        media_type=response.headers.get("Content-Type", "image/png")
                    )
                else:
                    return JSONResponse({
                        "success": False,
                        "error": f"HTTP {response.status_code}",
                        "message": "이미지를 다운로드할 수 없습니다."
                    }, status_code=response.status_code)
            except Exception as e:
                return JSONResponse({
                    "success": False,
                    "error": str(e),
                    "message": f"이미지 다운로드 중 오류: {str(e)}"
                }, status_code=500)
                
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"프록시 처리 중 오류: {str(e)}"
        }, status_code=500)


@router.options("/api/images/{file_name:path}", tags=["이미지 프록시"])
async def proxy_s3_image_options(file_name: str):
    """CORS preflight 요청 처리"""
    return Response(
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )


@router.get("/api/images/{file_name:path}", tags=["이미지 프록시"])
async def proxy_s3_image(file_name: str):
    """
    S3 이미지 프록시
    
    파일명으로 S3에서 이미지를 다운로드하여 반환합니다.
    CORS 문제를 해결하기 위한 프록시입니다.
    """
    try:
        # URL 디코딩
        file_name = unquote(file_name)
        
        # S3에서 이미지 다운로드
        image_data = get_s3_image(file_name)
        
        if image_data:
            # CORS 헤더 추가
            headers = {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Content-Type": "image/png"
            }
            
            return Response(
                content=image_data,
                headers=headers,
                media_type="image/png"
            )
        else:
            return JSONResponse({
                "success": False,
                "error": "Image not found",
                "message": "이미지를 찾을 수 없습니다."
            }, status_code=404)
            
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"이미지 프록시 처리 중 오류: {str(e)}"
        }, status_code=500)


@router.get("/api/admin/s3-image-proxy", tags=["관리자"])
async def get_s3_image_proxy(url: str = Query(..., description="S3 이미지 URL")):
    """
    관리자용 S3 이미지 프록시
    
    S3 URL을 받아서 이미지를 다운로드하여 반환합니다.
    """
    try:
        # URL에서 파일명 추출
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.strip('/').split('/')
        
        if len(path_parts) >= 2:
            folder = path_parts[0]
            file_name = path_parts[1]
            file_name = unquote(file_name)
            
            # dresses 폴더인 경우 기본 S3 클라이언트 사용
            if folder == 'dresses':
                image_data = get_s3_image(file_name)
                if image_data:
                    return Response(
                        content=image_data,
                        media_type="image/png"
                    )
                else:
                    return JSONResponse({
                        "success": False,
                        "error": "Image not found",
                        "message": "이미지를 찾을 수 없습니다."
                    }, status_code=404)
            
            # logs 폴더인 경우 로그용 S3 클라이언트 사용
            elif folder == 'logs':
                image_data = get_logs_s3_image(file_name)
                if image_data:
                    return Response(
                        content=image_data,
                        media_type="image/png"
                    )
                else:
                    return JSONResponse({
                        "success": False,
                        "error": "Image not found",
                        "message": "이미지를 찾을 수 없습니다."
                    }, status_code=404)
        
        # 직접 URL로 다운로드 시도 (fallback)
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return Response(
                    content=response.content,
                    media_type=response.headers.get("Content-Type", "image/png")
                )
            else:
                return JSONResponse({
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "message": "이미지를 다운로드할 수 없습니다."
                }, status_code=response.status_code)
        except Exception as e:
            return JSONResponse({
                "success": False,
                "error": str(e),
                "message": f"이미지 다운로드 중 오류: {str(e)}"
            }, status_code=500)
                
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"프록시 처리 중 오류: {str(e)}"
        }, status_code=500)

