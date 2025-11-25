"""페이스스왑 라우터 (body-generation 엔드포인트)"""
import io
import time
import base64
from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
from typing import Optional

from services.face_swap_service import FaceSwapService

router = APIRouter()


@router.get("/api/body-generation/templates", tags=["페이스스왑"])
async def get_templates():
    """
    사용 가능한 템플릿 이미지 목록 조회
    """
    try:
        service = FaceSwapService()
        template_images = service.get_template_images()
        
        templates = []
        for template_path in template_images:
            templates.append({
                "name": template_path.name,
                "path": str(template_path)
            })
        
        return JSONResponse({
            "success": True,
            "templates": templates,
            "count": len(templates)
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"템플릿 목록 조회 실패: {str(e)}"
        }, status_code=500)


@router.post("/api/body-generation", tags=["페이스스왑"])
async def face_swap(
    file: UploadFile = File(..., description="사용자 얼굴 이미지 파일"),
    template_name: Optional[str] = Form(None, description="템플릿 이미지 이름 (선택사항, 기본값: 첫 번째 템플릿)")
):
    """
    템플릿 이미지에 사용자 얼굴 교체
    
    HuggingFace Inference Endpoint를 사용하여 템플릿 이미지의 얼굴을 사용자 얼굴로 교체합니다.
    주의: 실제 페이스스왑 기능은 INSwapper 모델이 필요합니다.
    
    Args:
        file: 사용자 얼굴 이미지 파일
        template_name: 템플릿 이미지 이름 (선택사항)
    
    Returns:
        JSONResponse: 페이스스왑된 이미지 (base64 인코딩)
    """
    start_time = time.time()
    
    try:
        # 이미지 읽기
        contents = await file.read()
        if not contents:
            return JSONResponse({
                "success": False,
                "error": "Invalid input",
                "message": "이미지 파일이 비어있습니다."
            }, status_code=400)
        
        source_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 페이스스왑 서비스 초기화
        service = FaceSwapService()
        
        # 이미지 타입 감지 (전신 vs 얼굴/상체)
        image_type_info = service.detect_image_type(source_image)
        image_type = image_type_info.get("type", "unknown")
        confidence = image_type_info.get("confidence", 0.0)
        
        # 전신 사진인 경우 합성 불가 메시지 반환
        if image_type == "full_body":
            return JSONResponse({
                "success": False,
                "error": "Full body image detected",
                "message": "지금 올려주신 사진은 전신사진입니다. 상체만 나온 사진이나 얼굴만 나온 사진을 업로드해주세요.",
                "image_type": image_type,
                "image_type_confidence": round(confidence, 2)
            }, status_code=400)
        
        if not service.is_available():
            return JSONResponse({
                "success": False,
                "error": "Service unavailable",
                "message": "페이스스왑 서비스를 사용할 수 없습니다. HuggingFace Inference Endpoint 설정을 확인해주세요."
            }, status_code=500)
        
        # 템플릿 이미지 가져오기
        template_images = service.get_template_images()
        if len(template_images) == 0:
            return JSONResponse({
                "success": False,
                "error": "No templates",
                "message": "템플릿 이미지가 없습니다. templates/face_swap_templates/ 디렉토리에 템플릿 이미지를 추가해주세요."
            }, status_code=500)
        
        # 템플릿 선택
        if template_name:
            template_path = next((t for t in template_images if t.name == template_name), None)
            if template_path is None:
                template_path = template_images[0]
        else:
            template_path = template_images[0]
        
        target_image = Image.open(template_path).convert("RGB")
        
        # 페이스스왑 수행
        result_image = service.swap_face(source_image, target_image)
        
        if result_image is None:
            return JSONResponse({
                "success": False,
                "error": "Face swap failed",
                "message": "페이스스왑에 실패했습니다. 얼굴이 명확하게 보이는 이미지를 사용해주세요."
            }, status_code=500)
        
        # 결과 이미지를 base64로 인코딩
        result_buffered = io.BytesIO()
        result_image.save(result_buffered, format="PNG")
        result_base64 = base64.b64encode(result_buffered.getvalue()).decode()
        
        # 처리 시간 계산
        run_time = time.time() - start_time
        
        return JSONResponse({
            "success": True,
            "result_image": f"data:image/png;base64,{result_base64}",
            "template_name": template_path.name,
            "image_type": image_type,
            "image_type_confidence": round(confidence, 2),
            "run_time": round(run_time, 2),
            "message": f"페이스스왑 완료 (처리 시간: {run_time:.2f}초)"
        })
        
    except Exception as e:
        import traceback
        print(f"페이스스왑 오류: {traceback.format_exc()}")
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"페이스스왑 중 오류 발생: {str(e)}"
        }, status_code=500)

