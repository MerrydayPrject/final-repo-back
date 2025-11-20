from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import io
import numpy as np
import torch.nn.functional as nn
# routers/segmentation.py
from core.model_loader import get_segformer_b2_processor, get_segformer_b2_model
import base64

# Base64 인코딩을 위한 함수 정의
def image_to_base64(image: Image.Image) -> str:
    """이미지를 base64로 변환하는 함수"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Router 정의
router = APIRouter()

@router.post("/api/segment", tags=["세그멘테이션"])
async def segment_dress(file: UploadFile = File(..., description="세그멘테이션할 이미지 파일")):
    """
    드레스 세그멘테이션 (웨딩드레스 누끼)
    """
    try:
        # 모델 로딩
        processor = get_segformer_b2_processor()  # get_processor()로 가져오기
        model = get_segformer_b2_model()  # get_model()로 가져오기

        # 모델 로드 실패 처리
        if processor is None or model is None:
            return JSONResponse({
                "success": False,
                "error": "Model not loaded",
                "message": "모델이 로드되지 않았습니다. 서버를 재시작해주세요."
            }, status_code=503)

        # 파일 읽기 및 PIL 이미지로 변환
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # 원본 이미지 크기 저장
        original_size = image.size
        original_base64 = image_to_base64(image)

        # 프로세서로 입력 형식 변환
        inputs = processor(images=image, return_tensors="pt")
        
        # 모델 예측
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.cpu()

        # 원본 이미지 크기로 결과 업샘플링
        upsampled_logits = nn.interpolate(
            logits,
            size=original_size[::-1],  # (width, height) -> (height, width)
            mode="bilinear",
            align_corners=False,
        )

        # 드레스 영역 마스크 추출
        pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
        
        # 드레스 클래스를 7로 가정 (클래스 번호는 모델에 따라 다를 수 있음)
        dress_class_id = 7  # 드레스 클래스 ID를 모델에 맞게 설정
        dress_mask = (pred_seg == dress_class_id).astype(np.uint8) * 255

        # 결과 이미지 만들기 (RGBA 형식으로 저장)
        image_array = np.array(image)
        result_image = np.zeros((image_array.shape[0], image_array.shape[1], 4), dtype=np.uint8)
        result_image[:, :, :3] = image_array
        result_image[:, :, 3] = dress_mask

        # 결과를 PIL 이미지로 변환
        result_pil = Image.fromarray(result_image, mode='RGBA')
        result_base64 = image_to_base64(result_pil)

        return JSONResponse({
            "success": True,
            "original_image": f"data:image/png;base64,{original_base64}",
            "result_image": f"data:image/png;base64,{result_base64}",
            "message": "드레스 영역을 성공적으로 감지하고 배경을 제거했습니다."
        })

    except Exception as e:
        # 에러 처리
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"처리 중 오류 발생: {str(e)}"
        }, status_code=500)
