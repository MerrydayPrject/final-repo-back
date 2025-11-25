"""Fitting 라우터"""
import io
from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
from typing import Optional

from services.fitting_service import (
    parse_person_with_b2,
    extract_face_patch,
    generate_base_image,
    generate_inpaint_mask,
    build_preprocessed_person_payload,
    compose_v2_5
)
from services.tryon_service import generate_unified_tryon_v3, generate_unified_tryon_v4
from schemas.fitting_schema import PersonPreprocessResult
from schemas.tryon_schema import UnifiedTryonResponse

router = APIRouter()


@router.post("/fit/v2.5/preprocess-person", tags=["인물 전처리"], response_model=PersonPreprocessResult)
async def preprocess_person(
    person_image: UploadFile = File(..., description="인물 이미지 파일")
):
    """
    인물 전처리 파이프라인 (1~5단계)
    
    SegFormer B2 Human Parsing을 사용하여 인물 이미지를 파싱하고,
    face_mask, cloth_mask, body_mask를 생성한 후,
    face_patch, base_img, inpaint_mask를 추출합니다.
    
    Returns:
        PersonPreprocessResult: {
            "face_mask": str (base64),
            "face_patch": str (base64),
            "base_img": str (base64),
            "inpaint_mask": str (base64),
            "message": str
        }
    """
    try:
        # 이미지 읽기
        person_bytes = await person_image.read()
        
        if not person_bytes:
            return JSONResponse(
                {
                    "face_mask": "",
                    "face_patch": "",
                    "base_img": "",
                    "inpaint_mask": "",
                    "message": "인물 이미지를 업로드해주세요."
                },
                status_code=400,
            )
        
        # PIL Image로 변환
        person_img = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        
        # Step 1: SegFormer B2 Human Parsing
        print("[Preprocess Person] Step 1: SegFormer B2 Human Parsing...")
        parsing_result = parse_person_with_b2(person_img)
        
        if not parsing_result.get("success"):
            error_msg = parsing_result.get("message", "인물 파싱에 실패했습니다.")
            return JSONResponse(
                {
                    "face_mask": "",
                    "face_patch": "",
                    "base_img": "",
                    "inpaint_mask": "",
                    "message": error_msg
                },
                status_code=500,
            )
        
        parsing_mask = parsing_result.get("parsing_mask")
        face_mask_array = parsing_result.get("face_mask")
        
        # Step 2: face_patch 추출
        print("[Preprocess Person] Step 2: face_patch 추출...")
        face_patch = extract_face_patch(person_img, parsing_mask)
        
        # Step 3: base_img 생성
        print("[Preprocess Person] Step 3: base_img 생성...")
        base_img = generate_base_image(person_img, parsing_mask)
        
        # Step 4: inpaint_mask 생성
        print("[Preprocess Person] Step 4: inpaint_mask 생성...")
        inpaint_mask = generate_inpaint_mask(parsing_mask)
        
        # Step 5: face_mask를 base64로 변환 (응답용)
        import base64
        face_mask_img = Image.fromarray(face_mask_array, mode='L')
        face_mask_buffer = io.BytesIO()
        face_mask_img.save(face_mask_buffer, format="PNG")
        face_mask_base64 = base64.b64encode(face_mask_buffer.getvalue()).decode("utf-8")
        
        # payload 생성
        payload = build_preprocessed_person_payload(face_patch, base_img, inpaint_mask)
        
        print("[Preprocess Person] Step 5: 인물 전처리 완료")
        
        return JSONResponse({
            "face_mask": face_mask_base64,
            "face_patch": payload["face_patch"],
            "base_img": payload["base_img"],
            "inpaint_mask": payload["inpaint_mask"],
            "message": "인물 전처리가 성공적으로 완료되었습니다."
        })
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"인물 전처리 엔드포인트 오류: {e}")
        print(error_detail)
        
        return JSONResponse(
            {
                "face_mask": "",
                "face_patch": "",
                "base_img": "",
                "inpaint_mask": "",
                "message": f"인물 전처리 중 오류가 발생했습니다: {str(e)}"
            },
            status_code=500,
        )


@router.post("/fit/v2.5/compose", tags=["통합 트라이온 V2.5"], response_model=UnifiedTryonResponse)
async def compose_v2_5_endpoint(
    person_image: UploadFile = File(..., description="인물 이미지 파일"),
    garment_image: UploadFile = File(..., description="의상 이미지 파일"),
    background_image: UploadFile = File(..., description="배경 이미지 파일"),
    use_person_preprocess: str = Form("true", description="인물 전처리 사용 여부")
):
    """
    통합 트라이온 파이프라인 V2.5: 인물 전처리 + SegFormer B2 Garment Parsing + X.AI 프롬프트 생성 + Gemini 2.5 Flash 이미지 합성
    
    V2.5는 다음 단계를 수행합니다:
    1. SegFormer B2 Garment Parsing을 사용하여 garment_image에서 garment_only 이미지 추출
    2. (옵션) SegFormer B2 Human Parsing을 사용하여 person_image에서 face_mask, cloth_mask, body_mask 생성
    3. (옵션) face_patch, base_img, inpaint_mask 추출
    4. X.AI를 사용하여 person_image와 garment_only 이미지로부터 프롬프트 생성
    5. 생성된 프롬프트와 이미지들(base_img 또는 person_image, garment_only, 배경)을 사용하여 Gemini 2.5 Flash로 최종 합성 이미지 생성
    6. (옵션) Gemini 생성 이미지에 face_patch를 합성하고 경계 블렌딩 수행
    
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
        
        # use_person_preprocess 파라미터 변환
        use_preprocess = use_person_preprocess.lower() == "true"
        
        # V2.5 통합 트라이온 서비스 호출
        result = compose_v2_5(
            person_img,
            garment_img,
            background_img,
            use_person_preprocess=use_preprocess
        )
        
        if result["success"]:
            return JSONResponse(result)
        else:
            status_code = 500 if "error" in result else 400
            return JSONResponse(result, status_code=status_code)
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"통합 트라이온 V2.5 엔드포인트 오류: {e}")
        print(error_detail)
        
        return JSONResponse(
            {
                "success": False,
                "prompt": "",
                "result_image": "",
                "message": f"통합 트라이온 V2.5 처리 중 오류가 발생했습니다: {str(e)}",
                "llm": None
            },
            status_code=500,
        )


@router.post("/fit/v3/compose", tags=["통합 트라이온 V3"], response_model=UnifiedTryonResponse)
async def compose_v3_endpoint(
    person_image: UploadFile = File(..., description="인물 이미지 파일"),
    garment_image: UploadFile = File(..., description="의상 이미지 파일"),
    background_image: UploadFile = File(..., description="배경 이미지 파일"),
):
    """
    통합 트라이온 파이프라인 V3: 2단계 Gemini 플로우
    - Stage 1: X.AI 프롬프트 생성 (V2와 동일)
    - Stage 2: Gemini로 의상 교체만 수행 (person + garment_only)
    - Stage 3: Gemini로 배경 합성 + 조명 보정 (dressed_person + background)
    
    V3는 2단계 Gemini 플로우를 사용하여 더 정확한 의상 교체와 조명 보정을 수행합니다.
    
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
        
        # V3 통합 트라이온 서비스 호출
        result = generate_unified_tryon_v3(person_img, garment_img, background_img)
        
        if result["success"]:
            return JSONResponse(result)
        else:
            status_code = 500 if "error" in result else 400
            return JSONResponse(result, status_code=status_code)
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"통합 트라이온 V3 엔드포인트 오류: {e}")
        print(error_detail)
        
        return JSONResponse(
            {
                "success": False,
                "prompt": "",
                "result_image": "",
                "message": f"통합 트라이온 V3 처리 중 오류가 발생했습니다: {str(e)}",
                "llm": None
            },
            status_code=500,
        )


@router.post("/fit/v4/compose", tags=["통합 트라이온 V4"], response_model=UnifiedTryonResponse)
async def compose_v4_endpoint(
    person_image: UploadFile = File(..., description="인물 이미지 파일"),
    garment_image: UploadFile = File(..., description="의상 이미지 파일"),
    background_image: UploadFile = File(..., description="배경 이미지 파일"),
):
    """
    통합 트라이온 파이프라인 V4: 2단계 Gemini 3 Flash 플로우
    - Stage 1: X.AI 프롬프트 생성 (V3와 동일)
    - Stage 2: Gemini 3 Flash로 의상 교체만 수행 (person + garment_only)
    - Stage 3: Gemini 3 Flash로 배경 합성 + 조명 보정 (dressed_person + background)
    
    V4는 Gemini 3 Flash를 사용하여 더 향상된 의상 교체와 조명 보정을 수행합니다.
    
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
        
        # V4 통합 트라이온 서비스 호출
        result = generate_unified_tryon_v4(person_img, garment_img, background_img)
        
        if result["success"]:
            return JSONResponse(result)
        else:
            status_code = 500 if "error" in result else 400
            return JSONResponse(result, status_code=status_code)
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"통합 트라이온 V4 엔드포인트 오류: {e}")
        print(error_detail)
        
        return JSONResponse(
            {
                "success": False,
                "prompt": "",
                "result_image": "",
                "message": f"통합 트라이온 V4 처리 중 오류가 발생했습니다: {str(e)}",
                "llm": None
            },
            status_code=500,
        )

