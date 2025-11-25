from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from services.body_adjustment_service import adjust_body_shape_api
from services.text_parser import parse_adjustment_text
from services.pose_detection import check_pose_integrity
from PIL import Image, UnidentifiedImageError
import io
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/body-adjustment",
    tags=["Body Adjustment"]
)


@router.post("/adjust", response_class=StreamingResponse)
async def adjust_body(
    image: UploadFile = File(...),
    slim_factor: float = Form(0.9),  # 기본 슬림/넓게 조정 비율
    use_advanced: bool = Form(False),
    waist_factor: float = Form(None),  # 허리 비율 조정 (기본값은 slim_factor로 설정)
    shoulder_factor: float = Form(None),  # 어깨 비율 조정 (기본값은 slim_factor로 설정)
    hip_factor: float = Form(None),  # 엉덩이 비율 조정 (기본값은 slim_factor로 설정)
    recommend_style: bool = Form(False),  # 스타일 추천
    posture_correction: bool = Form(False),  # 자세 교정
    use_3d_model: bool = Form(False),  # 3D 모델 사용 여부
    live_feedback: bool = Form(False),  # 실시간 피드백
    target_body_part: str = Form('full')  # 특정 부위 조정 ('full', 'waist', 'shoulders', 'hips')
):
    """
    체형을 조정하고 자연스러운 포즈를 유지하는지 확인합니다.
    """
    try:
        # 이미지 파일 처리 (파일 형식 및 크기 체크)
        try:
            input_image = Image.open(image.file)
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="지원되지 않는 이미지 형식입니다.")
        except Exception as e:
            logger.error(f"이미지 처리 실패: {e}")
            raise HTTPException(status_code=500, detail="이미지 처리 중 오류가 발생했습니다.")
        
        # slim_factor에 따른 비율 조정
        # slim_factor가 None일 경우, 기본값을 slim_factor로 설정
        waist_factor = waist_factor if waist_factor is not None else slim_factor
        hip_factor = hip_factor if hip_factor is not None else slim_factor
        shoulder_factor = shoulder_factor if shoulder_factor is not None else slim_factor

        logger.info(f"전체 체형 비율 조정 요청: slim_factor={slim_factor} → waist={waist_factor}, hip={hip_factor}, shoulder={shoulder_factor}")
        
        # 체형 조정
        adjusted_image = await adjust_body_shape_api(
            input_image,
            waist_factor=waist_factor,
            hip_factor=hip_factor,
            shoulder_factor=shoulder_factor,
        )
        
        # 포즈 검증
        pose_valid = await check_pose_integrity(adjusted_image)
        
        if not pose_valid:
            logger.warning("체형 조정 후 포즈가 비정상적입니다. 원본 이미지를 반환합니다.")
            return StreamingResponse(image.file, media_type="image/png")  # 원본 이미지 반환

        # 이미지 바이트로 변환
        img_byte_arr = io.BytesIO()
        adjusted_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type="image/png")
    
    except Exception as e:
        logger.error(f"체형 조정 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"체형 조정 실패: {str(e)}")


@router.post("/text-adjust", response_class=StreamingResponse)
async def adjust_body_by_text(
    image: UploadFile = File(...),
    prompt: str = Form(..., description="조정 명령")
):
    """
    텍스트 프롬프트로 체형을 조정하고, 조정 후 포즈가 자연스러운지 확인합니다.
    """
    try:
        # 이미지 파일 처리 (파일 형식 및 크기 체크)
        try:
            input_image = Image.open(image.file)
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="지원되지 않는 이미지 형식입니다.")
        except Exception as e:
            logger.error(f"이미지 처리 실패: {e}")
            raise HTTPException(status_code=500, detail="이미지 처리 중 오류가 발생했습니다.")

        # 텍스트 프롬프트 파싱
        logger.info(f"텍스트 프롬프트: {prompt}")
        params = parse_adjustment_text(prompt)
        logger.info(f"파싱된 파라미터: {params}")
        
        # 체형 조정
        adjusted_image = await adjust_body_shape_api(
            input_image,
            shoulder_factor=params.get('shoulder_factor', 1.0),
            waist_factor=params.get('waist_factor', 1.0),
            hip_factor=params.get('hip_factor', 1.0),
            leg_factor=params.get('leg_factor', 1.0),
            arm_factor=params.get('arm_factor', 1.0),
            neck_thickness_factor=params.get('neck_thickness_factor', 1.0),
            chest_factor=params.get('chest_factor', 1.0)
        )
        
        # 포즈 검증
        pose_valid = await check_pose_integrity(adjusted_image)
        
        if not pose_valid:
            logger.warning("텍스트 기반 체형 조정 후 포즈가 비정상적입니다. 원본 이미지를 반환합니다.")
            return StreamingResponse(image.file, media_type="image/png")  # 원본 이미지 반환
        
        # 이미지 바이트로 변환
        img_byte_arr = io.BytesIO()
        adjusted_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type="image/png")
    
    except Exception as e:
        logger.error(f"텍스트 기반 체형 조정 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"텍스트 기반 체형 조정 실패: {str(e)}")
