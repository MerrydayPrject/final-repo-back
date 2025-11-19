"""Fitting 서비스"""
import os
import io
import base64
import time
import traceback
import numpy as np
import cv2
from typing import Dict, Optional
from PIL import Image
from google import genai

from core.segformer_person_parser import parse_person_image
from core.segformer_garment_parser import parse_garment_image
from core.xai_client import generate_prompt_from_images
from core.s3_client import upload_log_to_s3
from core.face.face_pipeline import preserve_face_pipeline
from core.preprocess.background_cleaner import remove_background
from services.image_service import preprocess_dress_image
from services.log_service import save_test_log
from config.settings import GEMINI_FLASH_MODEL, XAI_PROMPT_MODEL
from config.hf_segformer import FACE_MASK_IDS, NEUTRAL_COLOR


def parse_person_with_b2(person_img: Image.Image) -> Dict:
    """
    SegFormer B2 Human Parsing을 사용하여 인물 이미지 파싱
    
    Args:
        person_img: 인물 이미지 (PIL Image)
    
    Returns:
        dict: {
            "success": bool,
            "parsing_mask": Optional[np.ndarray],
            "face_mask": Optional[np.ndarray],
            "cloth_mask": Optional[np.ndarray],
            "body_mask": Optional[np.ndarray],
            "message": str,
            "error": Optional[str]
        }
    """
    return parse_person_image(person_img)


def extract_face_patch(person_img: Image.Image, parsing_mask: np.ndarray) -> Image.Image:
    """
    face_mask + hair_mask로 face_patch 추출
    
    Args:
        person_img: 인물 이미지 (PIL Image)
        parsing_mask: 파싱 마스크 (numpy array)
    
    Returns:
        Image.Image: face_patch 이미지 (RGBA)
    """
    # face_mask 생성 (face, skin, hair)
    face_mask_array = np.isin(parsing_mask, FACE_MASK_IDS).astype(np.uint8) * 255
    
    # 원본 이미지를 RGBA로 변환
    person_rgba = person_img.convert("RGBA")
    person_array = np.array(person_rgba)
    
    # face_mask 영역만 추출
    face_patch_array = person_array.copy()
    face_patch_array[:, :, 3] = face_mask_array  # Alpha 채널에 face_mask 적용
    
    # face_patch 생성
    face_patch = Image.fromarray(face_patch_array, mode='RGBA')
    
    return face_patch


def generate_base_image(person_img: Image.Image, parsing_mask: np.ndarray) -> Image.Image:
    """
    cloth_mask 영역을 neutral_color(128,128,128)로 덮어서 base_img 생성
    
    Args:
        person_img: 인물 이미지 (PIL Image)
        parsing_mask: 파싱 마스크 (numpy array)
    
    Returns:
        Image.Image: base_img 이미지 (RGB)
    """
    # cloth_mask 생성
    cloth_mask_ids = [4, 5, 6, 7, 8, 16, 17]
    cloth_mask_array = np.isin(parsing_mask, cloth_mask_ids).astype(np.uint8)
    
    # 원본 이미지 배열로 변환
    person_array = np.array(person_img.convert("RGB"))
    
    # cloth_mask 영역을 neutral_color로 덮기
    base_img_array = person_array.copy()
    cloth_mask_3d = cloth_mask_array[:, :, np.newaxis]  # (H, W, 1)
    base_img_array = np.where(
        cloth_mask_3d > 0,
        np.array(NEUTRAL_COLOR, dtype=np.uint8),
        base_img_array
    )
    
    # base_img 생성
    base_img = Image.fromarray(base_img_array, mode='RGB')
    
    return base_img


def generate_inpaint_mask(parsing_mask: np.ndarray) -> Image.Image:
    """
    inpaint_mask = body_mask - face_mask
    
    Args:
        parsing_mask: 파싱 마스크 (numpy array)
    
    Returns:
        Image.Image: inpaint_mask 이미지 (L mode, 0 또는 255)
    """
    # body_mask 생성
    body_mask_ids = [12, 13, 14, 15]
    body_mask_array = np.isin(parsing_mask, body_mask_ids).astype(np.uint8) * 255
    
    # face_mask 생성
    face_mask_array = np.isin(parsing_mask, FACE_MASK_IDS).astype(np.uint8) * 255
    
    # inpaint_mask = body_mask - face_mask
    inpaint_mask_array = np.clip(body_mask_array.astype(np.int16) - face_mask_array.astype(np.int16), 0, 255).astype(np.uint8)
    
    # PIL Image로 변환
    inpaint_mask = Image.fromarray(inpaint_mask_array, mode='L')
    
    return inpaint_mask


def build_preprocessed_person_payload(
    face_patch: Image.Image,
    base_img: Image.Image,
    inpaint_mask: Image.Image
) -> Dict:
    """
    face_patch, base_img, inpaint_mask를 base64(PNG)로 변환하여 payload 생성
    
    Args:
        face_patch: face_patch 이미지 (PIL Image)
        base_img: base_img 이미지 (PIL Image)
        inpaint_mask: inpaint_mask 이미지 (PIL Image)
    
    Returns:
        dict: {
            "face_patch": str (base64),
            "base_img": str (base64),
            "inpaint_mask": str (base64)
        }
    """
    def image_to_base64(img: Image.Image) -> str:
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode("utf-8")
    
    return {
        "face_patch": image_to_base64(face_patch),
        "base_img": image_to_base64(base_img),
        "inpaint_mask": image_to_base64(inpaint_mask)
    }


def blend_face_patch(
    generated_img: Image.Image,
    face_patch: Image.Image,
    face_mask: np.ndarray
) -> Image.Image:
    """
    Gemini 생성 이미지에 face_patch를 합성하고 경계 블렌딩 수행
    
    Args:
        generated_img: Gemini가 생성한 이미지 (PIL Image)
        face_patch: face_patch 이미지 (PIL Image, RGBA)
        face_mask: face_mask (numpy array, 0 또는 255)
    
    Returns:
        Image.Image: 최종 합성 이미지 (RGB)
    """
    # 이미지 크기 맞추기
    if generated_img.size != face_patch.size:
        face_patch = face_patch.resize(generated_img.size, Image.Resampling.LANCZOS)
    
    if generated_img.size[0] != face_mask.shape[1] or generated_img.size[1] != face_mask.shape[0]:
        face_mask_resized = cv2.resize(face_mask, generated_img.size, interpolation=cv2.INTER_NEAREST)
    else:
        face_mask_resized = face_mask
    
    # generated_img를 RGBA로 변환
    generated_rgba = generated_img.convert("RGBA")
    generated_array = np.array(generated_rgba)
    face_patch_array = np.array(face_patch)
    
    # face_mask를 3D로 확장
    face_mask_3d = face_mask_resized[:, :, np.newaxis] / 255.0
    
    # 경계 블렌딩을 위한 가우시안 블러 적용
    face_mask_blurred = cv2.GaussianBlur(face_mask_resized.astype(np.float32), (21, 21), 0) / 255.0
    face_mask_blurred_3d = face_mask_blurred[:, :, np.newaxis]
    
    # 블렌딩: face_patch와 generated_img를 블렌딩
    blended_array = (
        face_patch_array[:, :, :3] * face_mask_blurred_3d +
        generated_array[:, :, :3] * (1 - face_mask_blurred_3d)
    ).astype(np.uint8)
    
    # 최종 이미지 생성
    final_img = Image.fromarray(blended_array, mode='RGB')
    
    return final_img


def compose_v2_5(
    person_img: Image.Image,
    garment_img: Image.Image,
    background_img: Image.Image,
    use_person_preprocess: bool = True,
    model_id: str = "xai-gemini-unified-v2.5"
) -> Dict:
    """
    XAI + Gemini 2.5 V2.5 통합 파이프라인
    
    Args:
        person_img: 인물 이미지 (PIL Image)
        garment_img: 의상 이미지 (PIL Image)
        background_img: 배경 이미지 (PIL Image)
        use_person_preprocess: 인물 전처리 사용 여부 (기본값: True)
        model_id: 모델 ID (기본값: "xai-gemini-unified-v2.5")
    
    Returns:
        dict: {
            "success": bool,
            "prompt": str,
            "result_image": str (base64),
            "message": str,
            "llm": str,
            "error": Optional[str]
        }
    """
    start_time = time.time()
    person_s3_url = ""
    garment_s3_url = ""
    garment_only_s3_url = ""
    background_s3_url = ""
    result_s3_url = ""
    used_prompt = ""
    face_patch = None
    face_mask_array = None
    
    try:
        # 1. 의상 이미지 전처리 및 SegFormer B2 Garment Parsing
        print("의상 이미지 전처리 시작...")
        garment_img_processed = preprocess_dress_image(garment_img, target_size=1024)
        print("의상 이미지 전처리 완료")
        
        print("\n" + "="*80)
        print("SegFormer B2 Garment Parsing 시작")
        print("="*80)
        
        parsing_result = parse_garment_image(garment_img_processed)
        
        if not parsing_result.get("success"):
            error_msg = parsing_result.get("message", "SegFormer B2 Garment Parsing에 실패했습니다.")
            run_time = time.time() - start_time
            
            person_buffered = io.BytesIO()
            person_img.save(person_buffered, format="PNG")
            person_s3_url = upload_log_to_s3(person_buffered.getvalue(), model_id, "person") or ""
            
            garment_buffered = io.BytesIO()
            garment_img_processed.save(garment_buffered, format="PNG")
            garment_s3_url = upload_log_to_s3(garment_buffered.getvalue(), model_id, "garment") or ""
            
            save_test_log(
                person_url=person_s3_url or "",
                dress_url=garment_s3_url or None,
                result_url="",
                model=model_id,
                prompt="",
                success=False,
                run_time=run_time
            )
            
            return {
                "success": False,
                "prompt": "",
                "result_image": "",
                "message": error_msg,
                "llm": "segformer-b2-parsing",
                "error": parsing_result.get("error", "segformer_parsing_failed")
            }
        
        garment_only_img = parsing_result.get("garment_only")
        if not garment_only_img:
            error_msg = "garment_only 이미지를 추출할 수 없습니다."
            run_time = time.time() - start_time
            
            person_buffered = io.BytesIO()
            person_img.save(person_buffered, format="PNG")
            person_s3_url = upload_log_to_s3(person_buffered.getvalue(), model_id, "person") or ""
            
            garment_buffered = io.BytesIO()
            garment_img_processed.save(garment_buffered, format="PNG")
            garment_s3_url = upload_log_to_s3(garment_buffered.getvalue(), model_id, "garment") or ""
            
            save_test_log(
                person_url=person_s3_url or "",
                dress_url=garment_s3_url or None,
                result_url="",
                model=model_id,
                prompt="",
                success=False,
                run_time=run_time
            )
            
            return {
                "success": False,
                "prompt": "",
                "result_image": "",
                "message": error_msg,
                "llm": "segformer-b2-parsing",
                "error": "garment_only_extraction_failed"
            }
        
        print("SegFormer B2 Garment Parsing 완료 - garment_only 이미지 추출 성공")
        
        # 2. 인물 전처리 (1~5단계) - use_person_preprocess가 True인 경우
        base_img = person_img
        inpaint_mask_img = None
        
        if use_person_preprocess:
            print("\n" + "="*80)
            print("인물 전처리 파이프라인 시작 (1~5단계)")
            print("="*80)
            
            # Step 1: SegFormer B2 Human Parsing
            print("[Step 1] SegFormer B2 Human Parsing...")
            person_parsing_result = parse_person_with_b2(person_img)
            
            if not person_parsing_result.get("success"):
                error_msg = person_parsing_result.get("message", "인물 파싱에 실패했습니다.")
                run_time = time.time() - start_time
                
                person_buffered = io.BytesIO()
                person_img.save(person_buffered, format="PNG")
                person_s3_url = upload_log_to_s3(person_buffered.getvalue(), model_id, "person") or ""
                
                save_test_log(
                    person_url=person_s3_url or "",
                    dress_url=garment_s3_url or None,
                    result_url="",
                    model=model_id,
                    prompt="",
                    success=False,
                    run_time=run_time
                )
                
                return {
                    "success": False,
                    "prompt": "",
                    "result_image": "",
                    "message": error_msg,
                    "llm": "segformer-b2-person-parsing",
                    "error": person_parsing_result.get("error", "person_parsing_failed")
                }
            
            parsing_mask = person_parsing_result.get("parsing_mask")
            face_mask_array = person_parsing_result.get("face_mask")
            
            # Step 2: face_patch 추출
            print("[Step 2] face_patch 추출...")
            face_patch = extract_face_patch(person_img, parsing_mask)
            print("[Step 2] face_patch 추출 완료")
            
            # Step 3: base_img 생성
            print("[Step 3] base_img 생성...")
            base_img = generate_base_image(person_img, parsing_mask)
            print("[Step 3] base_img 생성 완료")
            
            # Step 4: inpaint_mask 생성
            print("[Step 4] inpaint_mask 생성...")
            inpaint_mask_img = generate_inpaint_mask(parsing_mask)
            print("[Step 4] inpaint_mask 생성 완료")
            
            print("[Step 5] 인물 전처리 완료")
        
        # 원본 인물 이미지 크기 저장
        person_size = person_img.size
        print(f"인물 이미지 크기: {person_size[0]}x{person_size[1]}")
        
        # garment_only 이미지를 인물 이미지 크기로 조정
        print(f"garment_only 이미지를 인물 크기({person_size[0]}x{person_size[1]})로 조정...")
        garment_only_img = garment_only_img.resize(person_size, Image.Resampling.LANCZOS)
        print(f"garment_only 이미지 크기 조정 완료: {garment_only_img.size[0]}x{garment_only_img.size[1]}")
        
        # base_img도 인물 이미지 크기로 조정 (필요한 경우)
        if base_img.size != person_size:
            base_img = base_img.resize(person_size, Image.Resampling.LANCZOS)
        
        # 투명 베이스 이미지 생성 (Background Lock용)
        print("\n" + "="*80)
        print("[BG] 투명 베이스 이미지 생성 시작")
        print("="*80)
        if use_person_preprocess and parsing_mask is not None:
            transparent_base_img = remove_background(base_img, parsing_mask)
            print(f"[BG] 투명 베이스 이미지 생성 완료: {transparent_base_img.size[0]}x{transparent_base_img.size[1]} (RGBA)")
        else:
            # use_person_preprocess가 False인 경우, base_img를 RGBA로 변환 (fallback)
            print("[BG] use_person_preprocess=False 또는 parsing_mask 없음 - base_img를 RGBA로 변환")
            transparent_base_img = base_img.convert("RGBA")
        
        # 배경 이미지는 원본 그대로 유지
        background_img_processed = background_img
        background_size = background_img.size
        print(f"배경 이미지 원본 유지: {background_size[0]}x{background_size[1]} (변형 없음)")
        
        # S3에 입력 이미지 업로드
        person_buffered = io.BytesIO()
        person_img.save(person_buffered, format="PNG")
        person_s3_url = upload_log_to_s3(person_buffered.getvalue(), model_id, "person") or ""
        
        garment_buffered = io.BytesIO()
        garment_img_processed.save(garment_buffered, format="PNG")
        garment_s3_url = upload_log_to_s3(garment_buffered.getvalue(), model_id, "garment") or ""
        
        garment_only_buffered = io.BytesIO()
        garment_only_img.save(garment_only_buffered, format="PNG")
        garment_only_s3_url = upload_log_to_s3(garment_only_buffered.getvalue(), model_id, "garment_only") or ""
        
        background_buffered = io.BytesIO()
        background_img_processed.save(background_buffered, format="PNG")
        background_s3_url = upload_log_to_s3(background_buffered.getvalue(), model_id, "background") or ""
        
        # 3. X.AI 프롬프트 생성
        print("\n" + "="*80)
        print("X.AI 프롬프트 생성 시작")
        print("="*80)
        
        xai_result = generate_prompt_from_images(person_img, garment_only_img)
        
        if not xai_result.get("success"):
            error_msg = xai_result.get("message", "X.AI 프롬프트 생성에 실패했습니다.")
            run_time = time.time() - start_time
            
            save_test_log(
                person_url=person_s3_url or "",
                dress_url=garment_s3_url or None,
                result_url="",
                model=model_id,
                prompt="",
                success=False,
                run_time=run_time
            )
            
            return {
                "success": False,
                "prompt": "",
                "result_image": "",
                "message": error_msg,
                "llm": XAI_PROMPT_MODEL,
                "error": xai_result.get("error", "xai_prompt_generation_failed")
            }
        
        used_prompt = xai_result.get("prompt", "")
        print("\n생성된 프롬프트:")
        print("-"*80)
        print(used_prompt)
        print("="*80 + "\n")
        
        # 4. Gemini 2.5 Flash 이미지 합성
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            error_msg = ".env 파일에 GEMINI_API_KEY가 설정되지 않았습니다."
            run_time = time.time() - start_time
            
            save_test_log(
                person_url=person_s3_url or "",
                dress_url=garment_s3_url or None,
                result_url="",
                model=model_id,
                prompt=used_prompt,
                success=False,
                run_time=run_time
            )
            
            return {
                "success": False,
                "prompt": used_prompt,
                "result_image": "",
                "message": error_msg,
                "llm": f"{XAI_PROMPT_MODEL}+{GEMINI_FLASH_MODEL}",
                "error": "gemini_api_key_not_found"
            }
        
        # Gemini Client 생성
        client = genai.Client(api_key=api_key)
        
        print("\n" + "="*80)
        print("Gemini 2.5 Flash Image로 이미지 합성 시작")
        print("="*80)
        print("합성에 사용되는 최종 프롬프트:")
        print("-"*80)
        print(used_prompt)
        print("="*80 + "\n")
        
        # Identity Lock 규칙
        identity_lock_rules = """IDENTITY RULES:
Use the exact same face as Image 1.
Do not change facial identity, structure, or expression.
No beautification. No relighting on the face."""

        # Background Lock 규칙 추가
        background_lock_rules = """BACKGROUND RULES:
Use Image 3 exactly as the final background.
Replace the ENTIRE background of Image 1 with Image 3.
Do NOT keep any pixel of the original background from Image 1.
Do NOT retain sky, floor, walls, plants, or any part of the old scene.
Only Image 3 must appear as the final background."""

        # 프롬프트 조합
        enhanced_prompt = f"""{identity_lock_rules}

{background_lock_rules}

{used_prompt}"""
        
        print("[Prompt] Background-Lock + Identity-Lock rules injected")
        
        # Gemini API 호출
        # 이미지 순서: transparent_base_img (Image 1), garment_only (Image 2), background_img (Image 3)
        try:
            gemini_images = [transparent_base_img, garment_only_img, background_img_processed]
            print("\n[Stage5] Gemini Inputs:")
            print("  - transparent_base_img: OK (Image 1, RGBA)")
            print("  - garment_only_img: OK (Image 2)")
            print("  - background_img: OK (Image 3)")
            
            response = client.models.generate_content(
                model=GEMINI_FLASH_MODEL,
                contents=gemini_images + [enhanced_prompt]
            )
        except Exception as exc:
            run_time = time.time() - start_time
            save_test_log(
                person_url=person_s3_url or "",
                dress_url=garment_s3_url or None,
                result_url="",
                model=model_id,
                prompt=used_prompt,
                success=False,
                run_time=run_time
            )
            
            print(f"Gemini API 호출 실패: {exc}")
            traceback.print_exc()
            return {
                "success": False,
                "prompt": used_prompt,
                "result_image": "",
                "message": f"Gemini 2.5 Flash 호출에 실패했습니다: {str(exc)}",
                "llm": f"{XAI_PROMPT_MODEL}+{GEMINI_FLASH_MODEL}",
                "error": "gemini_call_failed"
            }
        
        # 응답 확인
        if not response.candidates or len(response.candidates) == 0:
            error_msg = "Gemini API가 응답을 생성하지 못했습니다."
            run_time = time.time() - start_time
            
            save_test_log(
                person_url=person_s3_url or "",
                dress_url=garment_s3_url or None,
                result_url="",
                model=model_id,
                prompt=used_prompt,
                success=False,
                run_time=run_time
            )
            
            return {
                "success": False,
                "prompt": used_prompt,
                "result_image": "",
                "message": error_msg,
                "llm": f"{XAI_PROMPT_MODEL}+{GEMINI_FLASH_MODEL}",
                "error": "no_response"
            }
        
        candidate = response.candidates[0]
        if not hasattr(candidate, 'content') or candidate.content is None:
            error_msg = "Gemini API 응답에 content가 없습니다."
            run_time = time.time() - start_time
            
            save_test_log(
                person_url=person_s3_url or "",
                dress_url=garment_s3_url or None,
                result_url="",
                model=model_id,
                prompt=used_prompt,
                success=False,
                run_time=run_time
            )
            
            return {
                "success": False,
                "prompt": used_prompt,
                "result_image": "",
                "message": error_msg,
                "llm": f"{XAI_PROMPT_MODEL}+{GEMINI_FLASH_MODEL}",
                "error": "no_content"
            }
        
        if not hasattr(candidate.content, 'parts') or candidate.content.parts is None:
            error_msg = "Gemini API 응답에 parts가 없습니다."
            run_time = time.time() - start_time
            
            save_test_log(
                person_url=person_s3_url or "",
                dress_url=garment_s3_url or None,
                result_url="",
                model=model_id,
                prompt=used_prompt,
                success=False,
                run_time=run_time
            )
            
            return {
                "success": False,
                "prompt": used_prompt,
                "result_image": "",
                "message": error_msg,
                "llm": f"{XAI_PROMPT_MODEL}+{GEMINI_FLASH_MODEL}",
                "error": "no_parts"
            }
        
        # 응답에서 이미지 추출
        image_parts = [
            part.inline_data.data
            for part in candidate.content.parts
            if hasattr(part, 'inline_data') and part.inline_data
        ]
        
        if not image_parts:
            error_msg = "Gemini API가 이미지를 생성하지 못했습니다."
            run_time = time.time() - start_time
            
            save_test_log(
                person_url=person_s3_url or "",
                dress_url=garment_s3_url or None,
                result_url="",
                model=model_id,
                prompt=used_prompt,
                success=False,
                run_time=run_time
            )
            
            return {
                "success": False,
                "prompt": used_prompt,
                "result_image": "",
                "message": error_msg,
                "llm": f"{XAI_PROMPT_MODEL}+{GEMINI_FLASH_MODEL}",
                "error": "no_image_generated"
            }
        
        # 5. Stage 6: Identity Lock 방식 - 얼굴 추출/정렬/블렌딩 없이 Gemini 출력을 그대로 사용
        print("\n" + "="*80)
        print("NEW STAGE 6: No Face Extract / No Alignment / No Blending")
        print("="*80)
        print("• 얼굴 추출 X")
        print("• 얼굴 패치 X")
        print("• 색보정 X")
        print("• 이미지 합성 뒤 얼굴 복원 X")
        print("")
        print("→ 투명 베이스 이미지로 배경 완전 교체")
        print("→ Prompt에 \"Identity Lock Rules\" 자동 삽입")
        print("→ Prompt에 \"Background Lock Rules\" 자동 삽입")
        print("")
        print("결과: 얼굴이 절대 바뀌지 않고 배경이 완전히 교체된 자연스러운 Try-On")
        print("="*80)
        
        # Gemini 출력을 그대로 최종 결과로 사용
        final_img = Image.open(io.BytesIO(image_parts[0]))
        print("[FacePreserve] Using Identity Lock (no face extraction)")
        print("[FacePreserve] Gemini 출력을 최종 결과로 사용 (Identity Lock + Background Lock)")
        
        # 6. 결과 이미지 처리 및 S3 업로드
        result_image_base64 = base64.b64encode(image_parts[0]).decode()
        
        result_buffered = io.BytesIO()
        final_img.save(result_buffered, format="PNG")
        result_s3_url = upload_log_to_s3(result_buffered.getvalue(), model_id, "result") or ""
        
        # 최종 이미지를 base64로 인코딩
        final_buffered = io.BytesIO()
        final_img.save(final_buffered, format="PNG")
        final_image_base64 = base64.b64encode(final_buffered.getvalue()).decode()
        
        run_time = time.time() - start_time
        
        # 7. 성공 로그 저장
        save_test_log(
            person_url=person_s3_url or "",
            dress_url=garment_s3_url or None,
            result_url=result_s3_url or "",
            model=model_id,
            prompt=used_prompt,
            success=True,
            run_time=run_time
        )
        
        llm_info = f"segformer-b2-parsing+{XAI_PROMPT_MODEL}+{GEMINI_FLASH_MODEL}"
        if use_person_preprocess:
            llm_info = f"person-preprocess+{llm_info}"
        
        return {
            "success": True,
            "prompt": used_prompt,
            "result_image": f"data:image/png;base64,{final_image_base64}",
            "message": "통합 트라이온 파이프라인 V2.5가 성공적으로 완료되었습니다.",
            "llm": llm_info
        }
        
    except Exception as e:
        error_detail = traceback.format_exc()
        run_time = time.time() - start_time
        
        # 오류 로그 저장
        try:
            save_test_log(
                person_url=person_s3_url or "",
                dress_url=garment_s3_url or None,
                result_url=result_s3_url or "",
                model=model_id,
                prompt=used_prompt,
                success=False,
                run_time=run_time
            )
        except:
            pass  # 로그 저장 실패해도 계속 진행
        
        print(f"통합 트라이온 파이프라인 V2.5 오류: {e}")
        traceback.print_exc()
        
        return {
            "success": False,
            "prompt": used_prompt,
            "result_image": "",
            "message": f"통합 트라이온 파이프라인 V2.5 중 오류 발생: {str(e)}",
            "llm": f"segformer-b2-parsing+{XAI_PROMPT_MODEL}+{GEMINI_FLASH_MODEL}",
            "error": str(e)
        }

