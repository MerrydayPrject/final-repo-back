"""이미지 생성 및 포즈 추출을 위한 API 라우터"""
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from services import pose_estimation_service, image_generation_service
from schemas.generation import GenerationResponse
from PIL import Image
import io

router = APIRouter(
    prefix="/generation",
    tags=["Image Generation"]
)

@router.post("/extract-pose", response_class=StreamingResponse)
async def extract_pose(image: UploadFile = File(...)):
    """
    이미지에서 포즈를 추출하여 스켈레톤 이미지를 반환합니다.
    """
    try:
        pose_image = await pose_estimation_service.extract_pose_from_image(image)
        
        # 이미지를 바이트 스트림으로 변환하여 반환
        img_byte_arr = io.BytesIO()
        pose_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-with-pose", response_class=StreamingResponse)
async def generate_with_pose(
    prompt: str = Form(...),
    pose_image_file: UploadFile = File(...)
):
    """
    텍스트 프롬프트와 포즈 이미지를 기반으로 이미지를 생성합니다.
    """
    try:
        pose_image = Image.open(pose_image_file.file)
        
        generated_image = await image_generation_service.generate_image_with_pose(prompt, pose_image)
        
        # 이미지를 바이트 스트림으로 변환하여 반환
        img_byte_arr = io.BytesIO()
        generated_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))