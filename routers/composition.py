"""이미지 합성 라우터"""
import os
import time
import io
import base64
import traceback
from typing import Optional, List
from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
from urllib.parse import urlparse
import requests
import boto3
from botocore.exceptions import ClientError
from google import genai

from core.llm_clients import generate_custom_prompt_from_images
from core.s3_client import upload_log_to_s3
# from core.model_loader import _load_segformer_b2_models, _load_rtmpose_model, _load_realesrgan_model  # 주석 처리: torch/transformers 미사용
from services.image_service import preprocess_dress_image
from services.log_service import save_test_log
from services.tryon_service import generate_custom_tryon_v2
from config.settings import GEMINI_FLASH_MODEL
from config.prompts import GEMINI_DEFAULT_COMPOSITION_PROMPT

router = APIRouter()


@router.post("/api/compose-dress", tags=["커스텀 피팅 V2"])
async def compose_dress(
    person_image: UploadFile = File(..., description="전신사진 이미지 파일"),
    dress_image: UploadFile = File(..., description="드레스 이미지 파일")
):
    """
    커스텀 피팅 API: X.AI 프롬프트 생성 + Gemini 2.5 Flash V2 이미지 합성
    
    SegFormer B2 Garment Parsing + X.AI 프롬프트 생성 + Gemini 2.5 Flash V2를 사용하여
    전신사진과 드레스 이미지를 합성합니다. 배경 이미지는 사용하지 않습니다.
    
    Returns:
        JSONResponse: {
            "success": bool,
            "prompt": str,
            "result_image": str (base64),
            "message": str,
            "llm": str
        }
    """
    try:
        # 이미지 읽기
        person_contents = await person_image.read()
        dress_contents = await dress_image.read()
        
        if not person_contents or not dress_contents:
            return JSONResponse({
                "success": False,
                "prompt": "",
                "result_image": "",
                "message": "전신사진과 드레스 이미지를 모두 업로드해주세요.",
                "llm": None
            }, status_code=400)
        
        # PIL Image로 변환
        person_img = Image.open(io.BytesIO(person_contents)).convert("RGB")
        dress_img = Image.open(io.BytesIO(dress_contents)).convert("RGB")
        
        # 커스텀 트라이온 V2 서비스 호출
        result = await generate_custom_tryon_v2(person_img, dress_img)
        
        # 응답 형식 맞추기 (기존 API와 호환)
        if result["success"]:
            return JSONResponse({
                "success": True,
                "result_image": result.get("result_image", ""),
                "message": result.get("message", "이미지 합성이 완료되었습니다."),
                "prompt": result.get("prompt", ""),
                "llm": result.get("llm", "")
            })
        else:
            status_code = 500 if "error" in result else 400
            return JSONResponse({
                "success": False,
                "result_image": "",
                "message": result.get("message", "이미지 합성에 실패했습니다."),
                "prompt": result.get("prompt", ""),
                "llm": result.get("llm", ""),
                "error": result.get("error", "")
            }, status_code=status_code)
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"커스텀 피팅 API 오류: {e}")
        print(error_detail)
        
        return JSONResponse({
            "success": False,
            "result_image": "",
            "message": f"커스텀 피팅 처리 중 오류가 발생했습니다: {str(e)}",
            "prompt": "",
            "llm": None,
            "error": str(e)
        }, status_code=500)


@router.post("/api/gpt4o-gemini/compose", tags=["Gemini 이미지 합성"])
async def compose_gpt4o_gemini(
    person_image: UploadFile = File(..., description="사람 이미지 파일"),
    dress_image: UploadFile = File(..., description="드레스 이미지 파일"),
    prompt: str = Form(..., description="GPT-4o가 생성한 프롬프트"),
    model_name: Optional[str] = Form(None, description="모델명"),
):
    """
    GPT-4o 프롬프트를 사용한 Gemini 이미지 합성
    """
    start_time = time.time()
    model_id = model_name or "gpt4o-gemini"
    used_prompt = (prompt or "").strip()

    if not used_prompt:
        return JSONResponse(
            {
                "success": False,
                "error": "Invalid prompt",
                "message": "프롬프트가 비어 있습니다. GPT-4o로 생성한 프롬프트를 제공해주세요."
            },
            status_code=400,
        )

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return JSONResponse(
            {
                "success": False,
                "error": "API key not found",
                "message": ".env 파일에 GEMINI_API_KEY가 설정되지 않았습니다."
            },
            status_code=500,
        )

    person_bytes = await person_image.read()
    dress_bytes = await dress_image.read()

    if not person_bytes or not dress_bytes:
        return JSONResponse(
            {
                "success": False,
                "error": "Invalid input",
                "message": "사람 이미지와 드레스 이미지를 모두 업로드해주세요."
            },
            status_code=400,
        )

    try:
        person_img = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        dress_img = Image.open(io.BytesIO(dress_bytes)).convert("RGB")
    except Exception as exc:
        return JSONResponse(
            {
                "success": False,
                "error": "Image decoding failed",
                "message": f"업로드한 이미지를 열 수 없습니다: {str(exc)}"
            },
            status_code=400,
        )

    person_buffered = io.BytesIO()
    person_img.save(person_buffered, format="PNG")
    person_base64 = base64.b64encode(person_buffered.getvalue()).decode()

    dress_buffered = io.BytesIO()
    dress_img.save(dress_buffered, format="PNG")
    dress_base64 = base64.b64encode(dress_buffered.getvalue()).decode()

    person_s3_url = upload_log_to_s3(person_buffered.getvalue(), model_id, "person") or ""
    dress_s3_url = upload_log_to_s3(dress_buffered.getvalue(), model_id, "dress") or ""
    result_s3_url = ""

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=GEMINI_FLASH_MODEL,
            contents=[person_img, dress_img, used_prompt]
        )
    except Exception as exc:
        run_time = time.time() - start_time
        try:
            save_test_log(
                person_url=person_s3_url or "",
                dress_url=dress_s3_url or None,
                result_url="",
                model=model_id,
                prompt=used_prompt,
                success=False,
                run_time=run_time
            )
        except Exception:
            pass

        print(f"Gemini API 호출 실패: {exc}")
        traceback.print_exc()
        return JSONResponse(
            {
                "success": False,
                "error": "Gemini call failed",
                "message": f"Gemini 2.5 Flash 호출에 실패했습니다: {str(exc)}"
            },
            status_code=502,
        )

    if not response.candidates:
        run_time = time.time() - start_time
        save_test_log(
            person_url=person_s3_url or "",
            dress_url=dress_s3_url or None,
            result_url="",
            model=model_id,
            prompt=used_prompt,
            success=False,
            run_time=run_time
        )
        return JSONResponse(
            {
                "success": False,
                "error": "No response",
                "message": "Gemini가 결과를 생성하지 못했습니다."
            },
            status_code=500,
        )

    candidate = response.candidates[0]
    parts = getattr(candidate.content, "parts", None)
    if not parts:
        run_time = time.time() - start_time
        save_test_log(
            person_url=person_s3_url or "",
            dress_url=dress_s3_url or None,
            result_url="",
            model=model_id,
            prompt=used_prompt,
            success=False,
            run_time=run_time
        )
        return JSONResponse(
            {
                "success": False,
                "error": "No parts",
                "message": "Gemini 응답에 이미지 데이터가 포함되지 않았습니다."
            },
            status_code=500,
        )

    image_parts: List[bytes] = []
    result_text = ""
    for part in parts:
        if hasattr(part, "inline_data") and part.inline_data:
            data = part.inline_data.data
            if isinstance(data, bytes):
                image_parts.append(data)
            elif isinstance(data, str):
                image_parts.append(base64.b64decode(data))
        if hasattr(part, "text") and part.text:
            result_text += part.text

    if not image_parts:
        run_time = time.time() - start_time
        save_test_log(
            person_url=person_s3_url or "",
            dress_url=dress_s3_url or None,
            result_url="",
            model=model_id,
            prompt=used_prompt,
            success=False,
            run_time=run_time
        )
        return JSONResponse(
            {
                "success": False,
                "error": "No image generated",
                "message": "Gemini가 이미지 결과를 반환하지 않았습니다."
            },
            status_code=500,
        )

    result_img = Image.open(io.BytesIO(image_parts[0]))
    result_buffered = io.BytesIO()
    result_img.save(result_buffered, format="PNG")
    result_s3_url = upload_log_to_s3(result_buffered.getvalue(), model_id, "result") or ""

    run_time = time.time() - start_time
    save_test_log(
        person_url=person_s3_url or "",
        dress_url=dress_s3_url or None,
        result_url=result_s3_url or "",
        model=model_id,
        prompt=used_prompt,
        success=True,
        run_time=run_time
    )

    result_base64 = base64.b64encode(result_buffered.getvalue()).decode()

    return JSONResponse(
        {
            "success": True,
            "person_image": f"data:image/png;base64,{person_base64}",
            "dress_image": f"data:image/png;base64,{dress_base64}",
            "result_image": f"data:image/png;base64,{result_base64}",
            "message": "이미지 합성이 완료되었습니다.",
            "gemini_response": result_text
        }
    )


# ============================================================
# 의상 합성 고품화 기능 주석 처리 (torch/transformers 미사용)
# ============================================================
# @router.post("/api/compose-enhanced", tags=["의상합성 고품화"])
# async def compose_enhanced(
#     person_image: UploadFile = File(..., description="사람 이미지 파일"),
#     dress_image: UploadFile = File(..., description="드레스 이미지 파일"),
#     generate_shoes: str = Form("false", description="구두 생성 여부"),
#     shoes_prompt: Optional[str] = Form(None, description="구두 생성 프롬프트")
# ):
#     """
#     의상합성 개선 통합 파이프라인
#     
#     SegFormer B2 + RTMPose + HR-VITON을 사용한 고품화 합성
#     """
#     start_time = time.time()
#     pipeline_steps = []
#     
#     try:
#         import cv2
#         import torch.nn as nn
#         
#         # 모델 lazy loading
#         from core.model_loader import _load_segformer_b2_models, _load_rtmpose_model
#         segformer_b2_processor, segformer_b2_model = _load_segformer_b2_models()
#         rtmpose_model = _load_rtmpose_model()
        
#         # 이미지 읽기
#         person_contents = await person_image.read()
#         dress_contents = await dress_image.read()
#         
#         person_img = Image.open(io.BytesIO(person_contents)).convert("RGB")
#         dress_img = Image.open(io.BytesIO(dress_contents)).convert("RGB")
#         
#         # 원본 이미지들을 base64로 인코딩
#         person_buffered = io.BytesIO()
#         person_img.save(person_buffered, format="PNG")
#         person_base64 = base64.b64encode(person_buffered.getvalue()).decode()
#         
#         dress_buffered = io.BytesIO()
#         dress_img.save(dress_buffered, format="PNG")
#         dress_base64 = base64.b64encode(dress_buffered.getvalue()).decode()
#         
#         # 이미지 크기 정규화 (512×768)
#         TARGET_WIDTH = 512
#         TARGET_HEIGHT = 768
        
#         # ========== Step 1: RMBG - 인물 배경 제거 ==========
#         person_rgba_img = None
#         
#         try:
#             print(f"[Step 1] 시작: 원본 이미지 크기: {person_img.size}, 모드: {person_img.mode}")
#             
#             # 이미지 크기 정규화 (512×768)
#             person_resized = person_img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
#             person_array = np.array(person_resized)
#             print(f"[Step 1] 정규화 완료: {person_resized.size}, 배열 크기: {person_array.shape}")
#             
#             # SegFormer B2 on LIP 모델 사용
#             if segformer_b2_processor is None or segformer_b2_model is None:
#                 segformer_b2_processor, segformer_b2_model = _load_segformer_b2_models()
#                 print(f"[Step 1] SegFormer B2 Human Parsing 모델 로딩 완료")
#             
#             # SegFormer B2 기반 배경 제거
#             person_inputs = segformer_b2_processor(images=person_resized, return_tensors="pt")
#             with torch.no_grad():
#                 person_outputs = segformer_b2_model(**person_inputs)
#                 person_logits = person_outputs.logits.cpu()
#             
#             person_upsampled = nn.functional.interpolate(
#                 person_logits,
#                 size=(TARGET_HEIGHT, TARGET_WIDTH),
#                 mode="bilinear",
#                 align_corners=False,
#             )
#             person_pred = person_upsampled.argmax(dim=1)[0].numpy()
#             
#             # 배경 마스크 추출 (배경이 아닌 모든 것)
#             bg_mask = (person_pred != 0).astype(np.uint8) * 255
#             bg_mask_pixel_count = np.sum(bg_mask > 0)
#             bg_mask_ratio = bg_mask_pixel_count / (TARGET_HEIGHT * TARGET_WIDTH)
#             print(f"[Step 1] 배경 마스크 생성: 픽셀 수: {bg_mask_pixel_count}, 비율: {bg_mask_ratio:.2%}")
#             
#             if bg_mask_pixel_count == 0:
#                 raise ValueError("배경 마스크가 비어있습니다. 인물이 감지되지 않았습니다.")
#             
#             # OpenCV bitwise AND 적용하여 배경 제거
#             person_array_bgr = cv2.cvtColor(person_array, cv2.COLOR_RGB2BGR)
#             bg_mask_3d = np.stack([bg_mask] * 3, axis=2)
#             result_bgr = cv2.bitwise_and(person_array_bgr, bg_mask_3d)
#             result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
#             
#             # RGBA 이미지 생성
#             person_rgba = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 4), dtype=np.uint8)
#             person_rgba[:, :, :3] = result_rgb
#             person_rgba[:, :, 3] = bg_mask
#             
#             person_rgba_img = Image.fromarray(person_rgba, mode='RGBA')
#             print(f"[Step 1] RGBA 이미지 생성 완료: 크기: {person_rgba_img.size}, 모드: {person_rgba_img.mode}")
#             
#             pipeline_steps.append({"step": "RMBG", "status": "success", 
#                                   "message": f"인물 배경 제거 완료 (마스크 비율: {bg_mask_ratio:.1%})"})
#         except Exception as e:
#             traceback.print_exc()
#             print(f"[Step 1] 에러: {str(e)}")
#             pipeline_steps.append({"step": "RMBG", "status": "skipped", "message": f"스킵됨: {str(e)}"})
#             # Fallback: 원본 이미지 사용 (배경 제거 없이 진행)
#             person_rgba_img = person_img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS).convert("RGBA")
#             print(f"[Step 1] Fallback: 원본 이미지 사용 (RGBA 변환)")
#         
#         # ========== Step 2: Dress Preprocessing - 드레스 배경 제거 + 정렬 ==========
#         dress_ready_img = None
#         
#         try:
#             print(f"[Step 2] 시작: 원본 드레스 이미지 크기: {dress_img.size}, 모드: {dress_img.mode}")
#             
#             # 드레스 배경 제거 (SegFormer 사용)
#             dress_resized = dress_img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
#             dress_array = np.array(dress_resized)
#             print(f"[Step 2] 정규화 완료: {dress_resized.size}, 배열 크기: {dress_array.shape}")
#             
#             # SegFormer B2 Human Parsing 모델이 없으면 초기화
#             if segformer_b2_processor is None or segformer_b2_model is None:
#                 segformer_b2_processor, segformer_b2_model = _load_segformer_b2_models()
#                 print(f"[Step 2] SegFormer B2 Human Parsing 모델 로딩 완료")
#             
#             # SegFormer B2로 드레스 배경 제거
#             dress_inputs = segformer_b2_processor(images=dress_resized, return_tensors="pt")
#             with torch.no_grad():
#                 dress_outputs = segformer_b2_model(**dress_inputs)
#                 dress_logits = dress_outputs.logits.cpu()
#             
#             dress_upsampled = nn.functional.interpolate(
#                 dress_logits,
#                 size=(TARGET_HEIGHT, TARGET_WIDTH),
#                 mode="bilinear",
#                 align_corners=False,
#             )
#             dress_pred = dress_upsampled.argmax(dim=1)[0].numpy()
#             
#             # 드레스 마스크 (배경 제외, 얼굴/머리 제외)
#             dress_mask = ((dress_pred != 0) & (dress_pred != 11) & (dress_pred != 2)).astype(np.uint8) * 255
#             dress_mask_pixel_count = np.sum(dress_mask > 0)
#             dress_mask_ratio = dress_mask_pixel_count / (TARGET_HEIGHT * TARGET_WIDTH)
#             print(f"[Step 2] 드레스 마스크 생성: 픽셀 수: {dress_mask_pixel_count}, 비율: {dress_mask_ratio:.2%}")
#             
#             if dress_mask_pixel_count == 0:
#                 print(f"[Step 2] 경고: 드레스 마스크가 비어있습니다. 전체 이미지를 사용합니다.")
#                 dress_mask = np.ones((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8) * 255
#                 dress_mask_pixel_count = TARGET_HEIGHT * TARGET_WIDTH
#             
#             # 배경 제거 적용
#             dress_array_bgr = cv2.cvtColor(dress_array, cv2.COLOR_RGB2BGR)
#             dress_mask_3d = np.stack([dress_mask] * 3, axis=2)
#             dress_bg_removed = cv2.bitwise_and(dress_array_bgr, dress_mask_3d)
#             dress_bg_removed_rgb = cv2.cvtColor(dress_bg_removed, cv2.COLOR_BGR2RGB)
#             
#             # 간단히 중앙 정렬된 드레스 이미지 생성
#             dress_ready = dress_bg_removed_rgb.copy()
#             
#             # RGBA로 변환
#             dress_rgba = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 4), dtype=np.uint8)
#             dress_rgba[:, :, :3] = dress_ready
#             dress_rgba[:, :, 3] = dress_mask
#             
#             dress_ready_img = Image.fromarray(dress_rgba, mode='RGBA')
#             print(f"[Step 2] RGBA 이미지 생성 완료: 크기: {dress_ready_img.size}, 모드: {dress_ready_img.mode}")
#             
#             pipeline_steps.append({"step": "Dress Preprocessing", "status": "success", 
#                                   "message": f"드레스 정렬 완료 (마스크 비율: {dress_mask_ratio:.1%})"})
#         except Exception as e:
#             traceback.print_exc()
#             print(f"[Step 2] 에러: {str(e)}")
#             pipeline_steps.append({"step": "Dress Preprocessing", "status": "skipped", "message": f"스킵됨: {str(e)}"})
#             # Fallback: 원본 드레스 이미지 사용
#             dress_ready_img = dress_img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS).convert("RGBA")
#             print(f"[Step 2] Fallback: 원본 드레스 이미지 사용 (RGBA 변환)")
#         
#         # ========== Step 2.5: RTMPose - 포즈/키포인트 인식 (Step 3 이전에 실행) ==========
#         keypoints = None
#         keypoints_array = None
#         waist_y = None
#         
#         try:
#             if rtmpose_model is None:
#                 rtmpose_model = _load_rtmpose_model()
#             
#             import mmcv
#             from mmpose.apis import inference_top_down_pose_model
#             
#             # person_rgba에서 RGB 추출
#             if person_rgba_img is None:
#                 raise ValueError("person_rgba_img가 None입니다")
#             
#             person_rgb = person_rgba_img.convert("RGB")
#             img_array = np.array(person_rgb)
#             img_bgr = mmcv.imconvert(img_array, 'RGB', 'BGR')
#             pose_results, _ = inference_top_down_pose_model(rtmpose_model, img_bgr)
#             
#             if pose_results and len(pose_results) > 0:
#                 person_result = pose_results[0]
#                 if 'keypoints' in person_result:
#                     keypoints_array = person_result['keypoints']
#                     keypoints = keypoints_array.tolist()
#                     
#                     # 허리 Y 좌표 계산 (골반 키포인트 11, 12 평균)
#                     if len(keypoints_array) > 12:
#                         pelvis_points = keypoints_array[11:13]
#                         valid_pelvis = pelvis_points[pelvis_points[:, 1] > 0]
#                         if len(valid_pelvis) > 0:
#                             waist_y = int(np.mean(valid_pelvis[:, 1]))
#             
#             pipeline_steps.append({"step": "RTMPose", "status": "success", "message": f"포즈 인식 완료 (허리 Y: {waist_y if waist_y else 'N/A'})"})
#         except Exception as e:
#             traceback.print_exc()
#             print(f"[Step 2.5] RTMPose 에러: {str(e)}")
#             pipeline_steps.append({"step": "RTMPose", "status": "skipped", "message": f"스킵됨: {str(e)}"})
#             keypoints = None
#             waist_y = None
#         
#         # ========== Step 3: SegFormer B2 Human Parsing - 의상 영역 마스크 생성 ==========
#         human_mask = None
#         
#         try:
#             # person_rgba에서 RGB 추출
#             if person_rgba_img is None:
#                 raise ValueError("person_rgba_img가 None입니다")
#             
#             person_rgb = person_rgba_img.convert("RGB")
#             print(f"[Step 3] 입력 이미지 크기: {person_rgb.size}")
#             
#             if segformer_b2_processor is None or segformer_b2_model is None:
#                 segformer_b2_processor, segformer_b2_model = _load_segformer_b2_models()
#                 print(f"[Step 3] SegFormer B2 Human Parsing 모델 로딩 완료")
#             
#             # Device 설정 (이미 _load_segformer_b2_models에서 설정됨)
#             device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#             print(f"[Step 3] Device: {device}")
#             
#             # SegFormer B2 on LIP로 human parsing 수행
#             person_inputs = segformer_b2_processor(images=person_rgb, return_tensors="pt")
#             # 입력 텐서를 device로 이동
#             person_inputs = {k: v.to(device) for k, v in person_inputs.items()}
#             
#             with torch.no_grad():
#                 person_outputs = segformer_b2_model(**person_inputs)
#                 person_logits = person_outputs.logits.cpu()
#             
#             person_upsampled = nn.functional.interpolate(
#                 person_logits,
#                 size=(TARGET_HEIGHT, TARGET_WIDTH),
#                 mode="bilinear",
#                 align_corners=False,
#             )
#             person_pred = person_upsampled.argmax(dim=1)[0].numpy()
#             
#             # 의상 영역 추출 (상의, 하의, 드레스 모두 포함)
#             clothes_mask = ((person_pred == 5) | (person_pred == 6) | (person_pred == 9) | (person_pred == 10) | (person_pred == 13)).astype(np.uint8) * 255
#             
#             # 검증: 의상 마스크 픽셀 합계 확인
#             clothes_mask_ratio = np.sum(clothes_mask > 0) / (TARGET_HEIGHT * TARGET_WIDTH)
#             clothes_mask_pixel_count = np.sum(clothes_mask > 0)
#             print(f"[Step 3] 의상 영역 마스크 픽셀 수: {clothes_mask_pixel_count}, 비율: {clothes_mask_ratio:.2%}")
#             
#             # 의상 영역이 없으면 전체 인물 영역을 의상 영역으로 사용 (fallback)
#             if clothes_mask_ratio < 0.05:
#                 print(f"[Step 3] 경고: 의상 영역이 감지되지 않았습니다. 전체 인물 영역을 의상 영역으로 사용합니다.")
#                 # 전체 인물 영역 마스크 생성 (배경 제외, 하지만 얼굴/손/다리는 보존)
#                 human_mask = (person_pred != 0).astype(np.uint8) * 255
#                 # 얼굴, 머리, 팔, 다리 제외 (의상 영역만 남김)
#                 face_mask = (person_pred == 14).astype(np.uint8)  # 얼굴 (14)
#                 hair_mask = (person_pred == 2).astype(np.uint8)  # 머리 (2)
#                 arms_mask = ((person_pred == 15) | (person_pred == 16)).astype(np.uint8)  # 팔 (15: left_arm, 16: right_arm)
#                 legs_mask = ((person_pred == 17) | (person_pred == 18)).astype(np.uint8)  # 다리 (17: left_leg, 18: right_leg)
#                 preserve_mask = (face_mask | hair_mask | arms_mask | legs_mask)
#                 clothes_mask = (human_mask.astype(np.uint8) - preserve_mask * 255).astype(np.uint8)
#                 clothes_mask = np.clip(clothes_mask, 0, 255)
#                 clothes_mask_pixel_count = np.sum(clothes_mask > 0)
#                 print(f"[Step 3] Fallback: 의상 영역 마스크 픽셀 수: {clothes_mask_pixel_count}")
#             
#             # 의상 영역 마스크를 human_mask 변수에 저장
#             human_mask = clothes_mask
#             
#             # 최종 검증
#             mask_ratio = np.sum(human_mask > 0) / (TARGET_HEIGHT * TARGET_WIDTH)
#             mask_pixel_count = np.sum(human_mask > 0)
#             print(f"[Step 3] 최종 의상 영역 마스크 픽셀 수: {mask_pixel_count}, 비율: {mask_ratio:.2%}")
#             
#             if mask_ratio < 0.05:
#                 raise ValueError(f"의상 영역 마스크 비율이 너무 낮습니다 ({mask_ratio:.2%}). 의상이 감지되지 않았습니다.")
#             
#             pipeline_steps.append({"step": "SegFormer B2 on LIP", "status": "success", 
#                                   "message": f"의상 영역 마스크 생성 완료 (픽셀: {mask_pixel_count}, 비율: {mask_ratio:.1%})"})
#         except Exception as e:
#             traceback.print_exc()
#             print(f"[Step 3] 에러: {str(e)}")
#             pipeline_steps.append({"step": "SegFormer B2 on LIP", "status": "skipped", "message": f"스킵됨: {str(e)}"})
#             # Fallback: 전체 이미지의 중앙 하단 60%를 의상 영역으로 사용
#             human_mask = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8)
#             human_mask[int(TARGET_HEIGHT * 0.4):, :] = 255
#             print(f"[Step 3] Fallback: 이미지 하단 60%를 의상 영역으로 사용")
#         
#         # ========== Step 4: HR-VITON - 의상 영역만 교체 (드레스 입히기) ==========
#         viton_result_img = None
#         
#         try:
#             # None 체크
#             if person_rgba_img is None:
#                 raise ValueError("person_rgba_img가 None입니다")
#             if dress_ready_img is None:
#                 raise ValueError("dress_ready_img가 None입니다")
#             if human_mask is None:
#                 raise ValueError("human_mask가 None입니다")
#             
#             print(f"[Step 4] 입력 검증 완료")
#             print(f"[Step 4] person_rgba_img 크기: {person_rgba_img.size}, 모드: {person_rgba_img.mode}")
#             print(f"[Step 4] dress_ready_img 크기: {dress_ready_img.size}, 모드: {dress_ready_img.mode}")
#             print(f"[Step 4] human_mask 크기: {human_mask.shape}, 픽셀 수: {np.sum(human_mask > 0)}")
#             
#             # 두 이미지를 중앙 정렬
#             person_rgb = person_rgba_img.convert("RGB")
#             dress_rgb = dress_ready_img.convert("RGB")
#             
#             person_array = np.array(person_rgb)
#             dress_array = np.array(dress_rgb)
#             h, w = person_array.shape[:2]
#             
#             print(f"[Step 4] person_array 크기: {person_array.shape}")
#             print(f"[Step 4] dress_array 크기: {dress_array.shape}")
#             
#             # human_mask 크기 확인 및 조정
#             if human_mask.shape != (h, w):
#                 print(f"[Step 4] 경고: human_mask 크기 불일치. 리사이즈 필요: {human_mask.shape} -> ({h}, {w})")
#                 human_mask = cv2.resize(human_mask, (w, h), interpolation=cv2.INTER_NEAREST)
#             
#             # 의상 영역 마스크를 3D로 변환
#             clothes_mask_3d = np.stack([human_mask] * 3, axis=2) / 255.0
#             clothes_mask_pixel_count = np.sum(human_mask > 0)
#             print(f"[Step 4] 의상 영역 마스크 픽셀 수: {clothes_mask_pixel_count}, 비율: {clothes_mask_pixel_count / (h * w):.1%}")
#             
#             # 드레스 알파 채널 추출
#             dress_array_full = np.array(dress_ready_img)
#             if dress_array_full.shape[2] == 4:
#                 dress_alpha = dress_array_full[:, :, 3] / 255.0
#             else:
#                 print(f"[Step 4] 경고: 드레스 이미지에 알파 채널이 없습니다. 전체를 사용합니다.")
#                 dress_alpha = np.ones((h, w), dtype=np.float32)
#             
#             dress_alpha_3d = np.stack([dress_alpha] * 3, axis=2)
#             dress_alpha_pixel_count = np.sum(dress_alpha > 0.5)
#             print(f"[Step 4] 드레스 알파 채널 픽셀 수: {dress_alpha_pixel_count}, 비율: {dress_alpha_pixel_count / (h * w):.1%}")
#             
#             if dress_alpha_pixel_count == 0:
#                 raise ValueError("드레스 알파 채널이 비어있습니다. 드레스가 감지되지 않았습니다.")
#             
#             # 드레스 영역만 추출
#             dress_extracted = dress_array * dress_alpha_3d
#             dress_extracted_pixel_count = np.sum(np.any(dress_extracted > 0, axis=2))
#             print(f"[Step 4] 드레스 추출 영역 픽셀 수: {dress_extracted_pixel_count}")
#             
#             # 인물의 의상 영역만 교체 (얼굴, 손, 다리 등은 보존)
#             # 의상 영역에만 드레스 합성 (의상 마스크 AND 드레스 알파)
#             dress_region_mask = clothes_mask_3d * dress_alpha_3d
#             dress_region_pixel_count = np.sum(dress_region_mask > 0.5)
#             print(f"[Step 4] 드레스 합성 영역 픽셀 수: {dress_region_pixel_count}")
#             
#             # 최종 합성: 의상 영역에 드레스 합성
#             result_array = person_array.copy().astype(np.float32)
#             result_array = result_array * (1 - dress_region_mask) + dress_extracted * dress_region_mask
#             result_array = np.clip(result_array, 0, 255).astype(np.uint8)
#             
#             viton_result_img = Image.fromarray(result_array, mode='RGB')
#             print(f"[Step 4] HR-VITON 합성 완료: 크기: {viton_result_img.size}, 모드: {viton_result_img.mode}")
#             
#             pipeline_steps.append({"step": "HR-VITON", "status": "success", 
#                                   "message": f"드레스 합성 완료 (합성 영역: {dress_region_pixel_count}픽셀)"})
#         except Exception as e:
#             traceback.print_exc()
#             print(f"[Step 4] 에러: {str(e)}")
#             pipeline_steps.append({"step": "HR-VITON", "status": "skipped", "message": f"스킵됨: {str(e)}"})
#             # Fallback: 간단한 알파 블렌딩
#             if person_rgba_img and dress_ready_img:
#                 person_rgb = person_rgba_img.convert("RGB")
#                 dress_rgb = dress_ready_img.convert("RGB")
#                 person_array = np.array(person_rgb)
#                 dress_array = np.array(dress_rgb)
#                 alpha = 0.7
#                 result_array = (alpha * dress_array + (1 - alpha) * person_array).astype(np.uint8)
#                 viton_result_img = Image.fromarray(result_array, mode='RGB')
#                 print(f"[Step 4] Fallback: 간단한 알파 블렌딩 사용")
#             else:
#                 viton_result_img = person_img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
#         
#         # Step 5 (RTMPose)는 이미 Step 2.5로 이동됨
#         
#         # 현재 이미지를 viton_result로 설정
#         if viton_result_img is None:
#             print(f"[Step 4] 경고: viton_result_img가 None입니다. person_rgba_img 사용")
#             current_image = person_rgba_img.convert("RGB") if person_rgba_img else person_img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
#         else:
#             print(f"[Step 4] 합성 완료: 결과 이미지 크기: {viton_result_img.size}, 모드: {viton_result_img.mode}")
#             current_image = viton_result_img
#         
#         # ========== Step 6: Real-ESRGAN - 질감/해상도 업스케일 ==========
#         upscaled_img = None
#         
#         try:
#             print(f"[Step 6] 시작: 입력 이미지 크기: {current_image.size}, 모드: {current_image.mode}")
#             
#             # 512×768 → 1024×1536 으로 업스케일
#             realesrgan_model = _load_realesrgan_model(scale=4)
#             
#             if realesrgan_model is not None:
#                 img_array = np.array(current_image)
#                 img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
#                 # 2배 업스케일 (512×768 → 1024×1536)
#                 output, _ = realesrgan_model.enhance(img_bgr, outscale=2)
#                 output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
#                 
#                 # 픽셀 샤프닝 적용
#                 kernel = np.array([[-1, -1, -1],
#                                   [-1,  9, -1],
#                                   [-1, -1, -1]])
#                 sharpened = cv2.filter2D(output_rgb, -1, kernel)
#                 upscaled_img = Image.fromarray(sharpened)
#                 print(f"[Step 6] Real-ESRGAN 업스케일 완료: 결과 크기: {upscaled_img.size}")
#                 
#                 pipeline_steps.append({"step": "Real-ESRGAN", "status": "success", "message": "업스케일 완료"})
#             else:
#                 # Fallback: OpenCV resize 대체
#                 new_size = (TARGET_WIDTH * 2, TARGET_HEIGHT * 2)  # 1024×1536
#                 upscaled_img = current_image.resize(new_size, Image.Resampling.LANCZOS)
#                 print(f"[Step 6] Fallback: OpenCV resize 사용, 결과 크기: {upscaled_img.size}")
#                 pipeline_steps.append({"step": "Real-ESRGAN", "status": "fallback", "message": "OpenCV resize 사용"})
#         except Exception as e:
#             traceback.print_exc()
#             print(f"[Step 6] 에러: {str(e)}")
#             pipeline_steps.append({"step": "Real-ESRGAN", "status": "skipped", "message": f"스킵됨: {str(e)}"})
#             # Fallback: OpenCV resize
#             new_size = (TARGET_WIDTH * 2, TARGET_HEIGHT * 2)
#             upscaled_img = current_image.resize(new_size, Image.Resampling.LANCZOS)
#             print(f"[Step 6] Fallback: OpenCV resize 사용, 결과 크기: {upscaled_img.size}")
#         
#         # 현재 이미지를 upscaled로 설정
#         if upscaled_img is None:
#             print(f"[Step 6] 경고: upscaled_img가 None입니다. current_image 사용")
#         else:
#             current_image = upscaled_img
#             print(f"[Step 6] 업스케일 완료: 현재 이미지 크기: {current_image.size}")
#         
#         # ========== Step 7: Color Harmonization - 색상/조명 보정 ==========
#         final_result_img = None
#         
#         try:
#             print(f"[Step 7] 시작: 입력 이미지 크기: {current_image.size}, 모드: {current_image.mode}")
#             
#             # HSV 또는 LAB 색공간으로 변환
#             img_array = np.array(current_image)
#             img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
#             
#             # LAB 색공간으로 변환
#             lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
#             l, a, b = cv2.split(lab)
#             
#             # 전체 인물 영역 색상 보정 (human_mask 사용)
#             # 인물 영역 평균 밝기 계산
#             if human_mask is not None:
#                 # human_mask를 현재 이미지 크기로 리사이즈
#                 h, w = img_array.shape[:2]
#                 human_mask_resized = cv2.resize(human_mask, (w, h), interpolation=cv2.INTER_NEAREST) / 255.0
#                 human_mask_3d = np.stack([human_mask_resized] * 3, axis=2)
#                 
#                 # 전체 인물 영역 평균 밝기
#                 person_region = l[human_mask_resized > 0.5]
#                 person_brightness = np.mean(person_region) if len(person_region) > 0 else np.mean(l)
#                 
#                 # 전체 이미지 평균 밝기
#                 overall_brightness = np.mean(l)
#                 
#                 # 전체 인물 영역 밝기 보정
#                 brightness_diff = overall_brightness - person_brightness
#                 
#                 # 채도·명도 조정
#                 l_adjusted = l.copy()
#                 if abs(brightness_diff) > 5:  # 밝기 차이가 5 이상일 때만 조정
#                     # 전체 인물 영역 밝기 조정
#                     l_adjusted = np.clip(l + brightness_diff * 0.3, 0, 255).astype(np.uint8)
#                 
#                 # blend ratio = 0.3(인물) + 0.7(드레스)
#                 l_final = (l_adjusted * 0.3 + l * 0.7).astype(np.uint8)
#                 
#                 lab = cv2.merge([l_final, a, b])
#                 result_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
#                 final_result_img = Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
#             else:
#                 # mask가 없으면 감마 보정만 적용
#                 result_bgr = cv2.convertScaleAbs(img_bgr, alpha=1.1, beta=5)
#                 final_result_img = Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
#             
#             print(f"[Step 7] 색상 보정 완료: 결과 이미지 크기: {final_result_img.size}")
#             pipeline_steps.append({"step": "Color Harmonization", "status": "success", "message": "색상 보정 완료"})
#         except Exception as e:
#             traceback.print_exc()
#             print(f"[Step 7] 에러: {str(e)}")
#             pipeline_steps.append({"step": "Color Harmonization", "status": "skipped", "message": f"스킵됨: {str(e)}"})
#             # Fallback: 감마 보정 (1.1× + beta 5)
#             try:
#                 img_array = np.array(current_image)
#                 img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
#                 result_bgr = cv2.convertScaleAbs(img_bgr, alpha=1.1, beta=5)
#                 final_result_img = Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
#                 print(f"[Step 7] Fallback: 감마 보정 완료")
#             except Exception as fallback_error:
#                 print(f"[Step 7] Fallback 실패: {str(fallback_error)}")
#                 final_result_img = current_image
#         
#         # 최종 결과 이미지 설정
#         if final_result_img is None:
#             print(f"[Step 7] 경고: final_result_img가 None입니다. current_image 사용")
#         else:
#             current_image = final_result_img
#             print(f"[Step 7] 최종 결과 이미지 설정 완료: 크기: {current_image.size}, 모드: {current_image.mode}")
#         
#         # 파이프라인 요약 로그
#         success_count = len([s for s in pipeline_steps if s['status'] == 'success'])
#         warning_count = len([s for s in pipeline_steps if s['status'] == 'warning'])
#         error_count = len([s for s in pipeline_steps if s['status'] == 'error'])
#         skipped_count = len([s for s in pipeline_steps if s['status'] == 'skipped'])
#         print(f"[파이프라인 요약] 성공: {success_count}, 경고: {warning_count}, 에러: {error_count}, 스킵: {skipped_count}")
#         print(f"[파이프라인 요약] 최종 이미지 크기: {current_image.size}, 모드: {current_image.mode}")
#         
#         # 최종 결과 이미지를 base64로 인코딩
#         result_buffered = io.BytesIO()
#         current_image.save(result_buffered, format="PNG")
#         result_base64 = base64.b64encode(result_buffered.getvalue()).decode()
#         
#         run_time = time.time() - start_time
#         print(f"[파이프라인 요약] 총 실행 시간: {run_time:.2f}초")
#         
#         # 로그 저장
#         model_id = "enhanced-compose-pipeline"
#         person_s3_url = upload_log_to_s3(person_buffered.getvalue(), model_id, "person") or ""
#         dress_s3_url = upload_log_to_s3(dress_buffered.getvalue(), model_id, "dress") or ""
#         
#         result_buffered_for_s3 = io.BytesIO()
#         current_image.save(result_buffered_for_s3, format="PNG")
#         result_s3_url = upload_log_to_s3(result_buffered_for_s3.getvalue(), model_id, "result") or ""
#         
#         save_test_log(
#             person_url=person_s3_url or "",
#             dress_url=dress_s3_url or None,
#             result_url=result_s3_url or "",
#             model=model_id,
#             prompt=f"Enhanced pipeline with {len([s for s in pipeline_steps if s['status'] == 'success'])}/7 steps",
#             success=True,
#             run_time=run_time
#         )
#         
#         return JSONResponse({
#             "success": True,
#             "person_image": f"data:image/png;base64,{person_base64}",
#             "dress_image": f"data:image/png;base64,{dress_base64}",
#             "result_image": f"data:image/png;base64,{result_base64}",
#             "pipeline_steps": pipeline_steps,
#             "run_time": round(run_time, 2),
#             "message": f"의상합성 개선 파이프라인 완료 ({len([s for s in pipeline_steps if s['status'] == 'success'])}/7 단계 성공)"
#         })
#         
#     except Exception as e:
#         traceback.print_exc()
#         run_time = time.time() - start_time
#         return JSONResponse({
#             "success": False,
#             "error": str(e),
#             "error_detail": traceback.format_exc(),
#             "pipeline_steps": pipeline_steps,
#             "run_time": round(run_time, 2),
#             "message": f"처리 중 오류 발생: {str(e)}"
#         }, status_code=500)


@router.post("/api/hr-viton-compose", tags=["가상 피팅"])
async def hr_viton_compose(
    person_image: UploadFile = File(..., description="사람 이미지 파일"),
    dress_image: UploadFile = File(..., description="드레스 이미지 파일")
):
    """
    HR-VITON 가상 피팅 - 옷 교체/워핑/합성
    
    사람 이미지에 드레스를 자연스럽게 합성합니다.
    """
    try:
        # 이미지 읽기
        person_contents = await person_image.read()
        dress_contents = await dress_image.read()
        
        person_img = Image.open(io.BytesIO(person_contents)).convert("RGB")
        dress_img = Image.open(io.BytesIO(dress_contents)).convert("RGB")
        
        # 원본 이미지들을 base64로 인코딩
        person_buffered = io.BytesIO()
        person_img.save(person_buffered, format="PNG")
        person_base64 = base64.b64encode(person_buffered.getvalue()).decode()
        
        dress_buffered = io.BytesIO()
        dress_img.save(dress_buffered, format="PNG")
        dress_base64 = base64.b64encode(dress_buffered.getvalue()).decode()
        
        # HR-VITON 구현 (간단한 버전)
        # 실제로는 HR-VITON 저장소의 코드를 사용해야 함
        # 여기서는 기본적인 합성 로직으로 대체
        
        # 이미지 크기 맞추기
        person_array = np.array(person_img)
        dress_array = np.array(dress_img)
        
        # 간단한 합성 (드레스 영역을 사람 이미지에 합성)
        # 실제 HR-VITON은 복잡한 워핑 및 합성 알고리즘 사용
        result_array = person_array.copy()
        
        # 드레스 이미지를 사람 이미지 크기에 맞춰 리사이즈
        dress_resized = dress_img.resize(person_img.size, Image.Resampling.LANCZOS)
        dress_array_resized = np.array(dress_resized)
        
        # 간단한 알파 블렌딩 (실제로는 정교한 워핑 필요)
        alpha = 0.7
        result_array = (alpha * dress_array_resized + (1 - alpha) * result_array).astype(np.uint8)
        
        result_img = Image.fromarray(result_array)
        
        # 결과 이미지를 base64로 인코딩
        result_buffered = io.BytesIO()
        result_img.save(result_buffered, format="PNG")
        result_base64 = base64.b64encode(result_buffered.getvalue()).decode()
        
        return JSONResponse({
            "success": True,
            "person_image": f"data:image/png;base64,{person_base64}",
            "dress_image": f"data:image/png;base64,{dress_base64}",
            "result_image": f"data:image/png;base64,{result_base64}",
            "message": "HR-VITON 가상 피팅 완료 (참고: 실제 HR-VITON 모델 구현 필요)"
        })
        
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"처리 중 오류 발생: {str(e)}"
        }, status_code=500)

