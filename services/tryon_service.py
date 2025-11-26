"""통합 트라이온 서비스"""
import os
import io
import base64
import time
import traceback
from typing import Dict, Optional, Tuple
from PIL import Image
from google import genai

from core.xai_client import generate_prompt_from_images
from core.s3_client import upload_log_to_s3
# SegFormer B2 Garment Parsing (HuggingFace Inference API)
from core.segformer_garment_parser import parse_garment_image
from services.image_service import preprocess_dress_image
from services.log_service import save_test_log
from config.settings import GEMINI_FLASH_MODEL, GEMINI_3_FLASH_MODEL, XAI_PROMPT_MODEL
from core.gemini_client import get_gemini_client_pool


async def generate_unified_tryon(
    person_img: Image.Image,
    dress_img: Image.Image,
    background_img: Image.Image,
    model_id: str = "xai-gemini-unified"
) -> Dict:
    """
    통합 트라이온 파이프라인: X.AI 프롬프트 생성 + Gemini 2.5 Flash 이미지 합성 (배경 포함)
    
    Args:
        person_img: 사람 이미지 (PIL Image)
        dress_img: 드레스 이미지 (PIL Image)
        background_img: 배경 이미지 (PIL Image)
        model_id: 모델 ID (기본값: "xai-gemini-unified")
    
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
    dress_s3_url = ""
    background_s3_url = ""
    result_s3_url = ""
    used_prompt = ""
    
    try:
        # 1. 이미지 전처리
        print("드레스 이미지 전처리 시작...")
        dress_img_processed = preprocess_dress_image(dress_img, target_size=1024)
        print("드레스 이미지 전처리 완료")
        
        # 원본 인물 이미지 크기 저장
        person_size = person_img.size
        print(f"인물 이미지 크기: {person_size[0]}x{person_size[1]}")
        
        # 드레스 이미지를 인물 이미지 크기로 조정
        print(f"드레스 이미지를 인물 크기({person_size[0]}x{person_size[1]})로 조정...")
        dress_img_processed = dress_img_processed.resize(person_size, Image.Resampling.LANCZOS)
        print(f"드레스 이미지 크기 조정 완료: {dress_img_processed.size[0]}x{dress_img_processed.size[1]}")
        
        # 배경 이미지는 원본 그대로 유지 (변형 방지)
        background_img_processed = background_img
        background_size = background_img.size
        print(f"배경 이미지 원본 유지: {background_size[0]}x{background_size[1]} (변형 없음)")
        
        # S3에 입력 이미지 업로드
        person_buffered = io.BytesIO()
        person_img.save(person_buffered, format="PNG")
        person_s3_url = upload_log_to_s3(person_buffered.getvalue(), model_id, "person") or ""
        
        dress_buffered = io.BytesIO()
        dress_img_processed.save(dress_buffered, format="PNG")
        dress_s3_url = upload_log_to_s3(dress_buffered.getvalue(), model_id, "dress") or ""
        
        background_buffered = io.BytesIO()
        background_img_processed.save(background_buffered, format="PNG")
        background_s3_url = upload_log_to_s3(background_buffered.getvalue(), model_id, "background") or ""
        
        # 2. X.AI 프롬프트 생성
        print("\n" + "="*80)
        print("X.AI 프롬프트 생성 시작")
        print("="*80)
        
        xai_result = await generate_prompt_from_images(person_img, dress_img_processed)
        
        if not xai_result.get("success"):
            error_msg = xai_result.get("message", "X.AI 프롬프트 생성에 실패했습니다.")
            run_time = time.time() - start_time
            
            save_test_log(
                person_url=person_s3_url or "",
                dress_url=dress_s3_url or None,
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
        
        # 3. Gemini 2.5 Flash 이미지 합성
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            error_msg = ".env 파일에 GEMINI_API_KEY가 설정되지 않았습니다."
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
        print("Gemini 2.5 Flash Image로 이미지 합성 시작 (배경 포함)")
        print("="*80)
        print("합성에 사용되는 최종 프롬프트:")
        print("-"*80)
        print(used_prompt)
        print("="*80 + "\n")
        
        # 배경 관련 지시사항을 프롬프트에 추가
        enhanced_prompt = f"""IDENTITY PRESERVATION RULES:
- The person in Image 1 must remain the same individual.
- Do NOT modify the person's face, identity, head shape, or expression.
- NEVER generate a new face.

{used_prompt}

BACKGROUND RULES:
1. Do NOT modify the background image.
2. Do NOT stretch, crop, distort, or resize the background.
3. Insert the person naturally into the background.
4. Match lighting and perspective.
5. Do NOT modify the face.
6. Only apply the outfit and integrate with shadows."""
        
        # Gemini API 호출 (person(Image 1), dress(Image 2), background(Image 3), text 순서)
        try:
            response = client.models.generate_content(
                model=GEMINI_FLASH_MODEL,
                contents=[person_img, dress_img_processed, background_img_processed, enhanced_prompt]
            )
        except Exception as exc:
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
                dress_url=dress_s3_url or None,
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
                dress_url=dress_s3_url or None,
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
                dress_url=dress_s3_url or None,
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
                dress_url=dress_s3_url or None,
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
        
        # 4. 결과 이미지 처리 및 S3 업로드
        result_image_base64 = base64.b64encode(image_parts[0]).decode()
        
        result_img = Image.open(io.BytesIO(image_parts[0]))
        result_buffered = io.BytesIO()
        result_img.save(result_buffered, format="PNG")
        result_s3_url = upload_log_to_s3(result_buffered.getvalue(), model_id, "result") or ""
        
        run_time = time.time() - start_time
        
        # 5. 성공 로그 저장
        save_test_log(
            person_url=person_s3_url or "",
            dress_url=dress_s3_url or None,
            result_url=result_s3_url or "",
            model=model_id,
            prompt=used_prompt,
            success=True,
            run_time=run_time
        )
        
        return {
            "success": True,
            "prompt": used_prompt,
            "result_image": f"data:image/png;base64,{result_image_base64}",
            "message": "통합 트라이온 파이프라인이 성공적으로 완료되었습니다.",
            "llm": f"{XAI_PROMPT_MODEL}+{GEMINI_FLASH_MODEL}"
        }
        
    except Exception as e:
        error_detail = traceback.format_exc()
        run_time = time.time() - start_time
        
        # 오류 로그 저장
        try:
            save_test_log(
                person_url=person_s3_url or "",
                dress_url=dress_s3_url or None,
                result_url=result_s3_url or "",
                model=model_id,
                prompt=used_prompt,
                success=False,
                run_time=run_time
            )
        except:
            pass  # 로그 저장 실패해도 계속 진행
        
        print(f"통합 트라이온 파이프라인 오류: {e}")
        traceback.print_exc()
        
        return {
            "success": False,
            "prompt": used_prompt,
            "result_image": "",
            "message": f"통합 트라이온 파이프라인 중 오류 발생: {str(e)}",
            "llm": f"{XAI_PROMPT_MODEL}+{GEMINI_FLASH_MODEL}",
            "error": str(e)
        }


async def generate_unified_tryon_v2(
    person_img: Image.Image,
    garment_img: Image.Image,
    background_img: Image.Image,
    model_id: str = "xai-gemini-unified-v2"
) -> Dict:
    """
    통합 트라이온 파이프라인 V2: SegFormer B2 Garment Parsing + X.AI 프롬프트 생성 + Gemini 2.5 Flash 이미지 합성 (배경 포함)
    
    V2는 SegFormer B2 Human Parsing을 먼저 수행하여 garment_only 이미지를 추출한 후,
    해당 이미지로 XAI 프롬프트를 생성하고 Gemini 합성을 수행합니다.
    
    Args:
        person_img: 사람 이미지 (PIL Image)
        garment_img: 의상 이미지 (PIL Image) - SegFormer B2 Parsing 대상
        background_img: 배경 이미지 (PIL Image)
        model_id: 모델 ID (기본값: "xai-gemini-unified-v2")
    
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
    
    try:
        # 1. 의상 이미지 전처리
        print("의상 이미지 전처리 시작...")
        garment_img_processed = preprocess_dress_image(garment_img, target_size=1024)
        print("의상 이미지 전처리 완료")
        
        # 2. SegFormer B2 Garment Parsing - garment_only 이미지 추출
        print("\n" + "="*80)
        print("SegFormer B2 Garment Parsing 시작")
        print("="*80)
        
        parsing_result = parse_garment_image(garment_img_processed)
        
        if not parsing_result.get("success"):
            error_msg = parsing_result.get("message", "SegFormer B2 Garment Parsing에 실패했습니다.")
            run_time = time.time() - start_time
            
            # S3에 입력 이미지 업로드 (실패 로그용)
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
        
        # 원본 인물 이미지 크기 저장
        person_size = person_img.size
        print(f"인물 이미지 크기: {person_size[0]}x{person_size[1]}")
        
        # garment_only 이미지를 인물 이미지 크기로 조정
        print(f"garment_only 이미지를 인물 크기({person_size[0]}x{person_size[1]})로 조정...")
        garment_only_img = garment_only_img.resize(person_size, Image.Resampling.LANCZOS)
        print(f"garment_only 이미지 크기 조정 완료: {garment_only_img.size[0]}x{garment_only_img.size[1]}")
        
        # 배경 이미지는 원본 그대로 유지 (변형 방지)
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
        
        # 3. X.AI 프롬프트 생성 (person_img, garment_only_img 사용)
        print("\n" + "="*80)
        print("X.AI 프롬프트 생성 시작 (V2: garment_only 이미지 사용)")
        print("="*80)
        
        xai_result = await generate_prompt_from_images(person_img, garment_only_img)
        
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
        print("Gemini 2.5 Flash Image로 이미지 합성 시작 (V2: 배경 포함)")
        print("="*80)
        print("합성에 사용되는 최종 프롬프트:")
        print("-"*80)
        print(used_prompt)
        print("="*80 + "\n")
        
        # 배경 관련 지시사항을 프롬프트에 추가
        enhanced_prompt = f"""IDENTITY PRESERVATION RULES:
- The person in Image 1 must remain the same individual.
- Do NOT modify the person's face, identity, head shape, or expression.
- NEVER generate a new face.

{used_prompt}

BACKGROUND RULES:
1. Do NOT modify the background image (Image 3).
2. Do NOT stretch, crop, distort, or resize the background.
3. Insert the person naturally into the background.
4. Match lighting and perspective.
5. Do NOT modify the face.
6. Only apply the outfit and integrate with shadows."""
        
        # Gemini API 호출 (person(Image 1), garment_only(Image 2), background(Image 3), text 순서)
        try:
            response = client.models.generate_content(
                model=GEMINI_FLASH_MODEL,
                contents=[person_img, garment_only_img, background_img_processed, enhanced_prompt]
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
        
        # 5. 결과 이미지 처리 및 S3 업로드
        result_image_base64 = base64.b64encode(image_parts[0]).decode()
        
        result_img = Image.open(io.BytesIO(image_parts[0]))
        result_buffered = io.BytesIO()
        result_img.save(result_buffered, format="PNG")
        result_s3_url = upload_log_to_s3(result_buffered.getvalue(), model_id, "result") or ""
        
        run_time = time.time() - start_time
        
        # 6. 성공 로그 저장
        save_test_log(
            person_url=person_s3_url or "",
            dress_url=garment_s3_url or None,
            result_url=result_s3_url or "",
            model=model_id,
            prompt=used_prompt,
            success=True,
            run_time=run_time
        )
        
        # 배경 이미지 URL도 로그에 포함 (필요 시 확장)
        
        return {
            "success": True,
            "prompt": used_prompt,
            "result_image": f"data:image/png;base64,{result_image_base64}",
            "message": "통합 트라이온 파이프라인 V2가 성공적으로 완료되었습니다.",
            "llm": f"segformer-b2-parsing+{XAI_PROMPT_MODEL}+{GEMINI_FLASH_MODEL}"
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
        
        print(f"통합 트라이온 파이프라인 V2 오류: {e}")
        traceback.print_exc()
        
        return {
            "success": False,
            "prompt": used_prompt,
            "result_image": "",
            "message": f"통합 트라이온 파이프라인 V2 중 오류 발생: {str(e)}",
            "llm": f"segformer-b2-parsing+{XAI_PROMPT_MODEL}+{GEMINI_FLASH_MODEL}",
            "error": str(e)
        }


async def generate_custom_tryon_v2(
    person_img: Image.Image,
    dress_img: Image.Image,
    model_id: str = "xai-gemini-custom-v2"
) -> Dict:
    """
    커스텀 트라이온 파이프라인 V2: SegFormer B2 Garment Parsing + X.AI 프롬프트 생성 + Gemini 2.5 Flash 이미지 합성 (배경 없음)
    
    커스텀 피팅용으로 배경 이미지 없이 동작합니다.
    
    Args:
        person_img: 사람 이미지 (PIL Image)
        dress_img: 드레스 이미지 (PIL Image) - SegFormer B2 Parsing 대상
        model_id: 모델 ID (기본값: "xai-gemini-custom-v2")
    
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
    dress_s3_url = ""
    garment_only_s3_url = ""
    result_s3_url = ""
    used_prompt = ""
    
    try:
        # 1. 의상 이미지 전처리
        print("의상 이미지 전처리 시작...")
        dress_img_processed = preprocess_dress_image(dress_img, target_size=1024)
        print("의상 이미지 전처리 완료")
        
        # 2. SegFormer B2 Garment Parsing - garment_only 이미지 추출
        print("\n" + "="*80)
        print("SegFormer B2 Garment Parsing 시작")
        print("="*80)
        
        parsing_result = parse_garment_image(dress_img_processed)
        
        if not parsing_result.get("success"):
            error_msg = parsing_result.get("message", "SegFormer B2 Garment Parsing에 실패했습니다.")
            run_time = time.time() - start_time
            
            # S3에 입력 이미지 업로드 (실패 로그용)
            person_buffered = io.BytesIO()
            person_img.save(person_buffered, format="PNG")
            person_s3_url = upload_log_to_s3(person_buffered.getvalue(), model_id, "person") or ""
            
            dress_buffered = io.BytesIO()
            dress_img_processed.save(dress_buffered, format="PNG")
            dress_s3_url = upload_log_to_s3(dress_buffered.getvalue(), model_id, "dress") or ""
            
            save_test_log(
                person_url=person_s3_url or "",
                dress_url=dress_s3_url or None,
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
            
            dress_buffered = io.BytesIO()
            dress_img_processed.save(dress_buffered, format="PNG")
            dress_s3_url = upload_log_to_s3(dress_buffered.getvalue(), model_id, "dress") or ""
            
            save_test_log(
                person_url=person_s3_url or "",
                dress_url=dress_s3_url or None,
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
        
        # 원본 인물 이미지 크기 저장
        person_size = person_img.size
        print(f"인물 이미지 크기: {person_size[0]}x{person_size[1]}")
        
        # garment_only 이미지를 인물 이미지 크기로 조정
        print(f"garment_only 이미지를 인물 크기({person_size[0]}x{person_size[1]})로 조정...")
        garment_only_img = garment_only_img.resize(person_size, Image.Resampling.LANCZOS)
        print(f"garment_only 이미지 크기 조정 완료: {garment_only_img.size[0]}x{garment_only_img.size[1]}")
        
        # S3에 입력 이미지 업로드
        person_buffered = io.BytesIO()
        person_img.save(person_buffered, format="PNG")
        person_s3_url = upload_log_to_s3(person_buffered.getvalue(), model_id, "person") or ""
        
        dress_buffered = io.BytesIO()
        dress_img_processed.save(dress_buffered, format="PNG")
        dress_s3_url = upload_log_to_s3(dress_buffered.getvalue(), model_id, "dress") or ""
        
        garment_only_buffered = io.BytesIO()
        garment_only_img.save(garment_only_buffered, format="PNG")
        garment_only_s3_url = upload_log_to_s3(garment_only_buffered.getvalue(), model_id, "garment_only") or ""
        
        # 3. X.AI 프롬프트 생성 (person_img, garment_only_img 사용)
        print("\n" + "="*80)
        print("X.AI 프롬프트 생성 시작 (커스텀 피팅: garment_only 이미지 사용)")
        print("="*80)
        
        xai_result = await generate_prompt_from_images(person_img, garment_only_img)
        
        if not xai_result.get("success"):
            error_msg = xai_result.get("message", "X.AI 프롬프트 생성에 실패했습니다.")
            run_time = time.time() - start_time
            
            save_test_log(
                person_url=person_s3_url or "",
                dress_url=dress_s3_url or None,
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
        
        # 4. Gemini 2.5 Flash 이미지 합성 (배경 없이)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            error_msg = ".env 파일에 GEMINI_API_KEY가 설정되지 않았습니다."
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
        print("Gemini 2.5 Flash Image로 이미지 합성 시작 (커스텀 피팅: 배경 없음)")
        print("="*80)
        print("합성에 사용되는 최종 프롬프트:")
        print("-"*80)
        print(used_prompt)
        print("="*80 + "\n")
        
        # 배경 없이 합성하기 위한 프롬프트 보강
        enhanced_prompt = f"""IDENTITY PRESERVATION RULES:
- The person in Image 1 must remain the same individual.
- Do NOT modify the person's face, identity, head shape, or expression.
- NEVER generate a new face.

{used_prompt}

BACKGROUND RULES:
1. Use a clean, simple white or neutral background.
2. Do NOT add complex backgrounds or scenery.
3. Focus on the person wearing the outfit from Image 2.
4. Maintain natural lighting and shadows on the person only."""
        
        # Gemini API 호출 (person(Image 1), garment_only(Image 2), text 순서) - 배경 없음
        try:
            response = client.models.generate_content(
                model=GEMINI_FLASH_MODEL,
                contents=[person_img, garment_only_img, enhanced_prompt]
            )
        except Exception as exc:
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
                dress_url=dress_s3_url or None,
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
                dress_url=dress_s3_url or None,
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
                dress_url=dress_s3_url or None,
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
                dress_url=dress_s3_url or None,
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
        
        # 5. 결과 이미지 처리 및 S3 업로드
        result_image_base64 = base64.b64encode(image_parts[0]).decode()
        
        result_img = Image.open(io.BytesIO(image_parts[0]))
        result_buffered = io.BytesIO()
        result_img.save(result_buffered, format="PNG")
        result_s3_url = upload_log_to_s3(result_buffered.getvalue(), model_id, "result") or ""
        
        run_time = time.time() - start_time
        
        # 6. 성공 로그 저장
        save_test_log(
            person_url=person_s3_url or "",
            dress_url=dress_s3_url or None,
            result_url=result_s3_url or "",
            model=model_id,
            prompt=used_prompt,
            success=True,
            run_time=run_time
        )
        
        return {
            "success": True,
            "prompt": used_prompt,
            "result_image": f"data:image/png;base64,{result_image_base64}",
            "message": "커스텀 트라이온 파이프라인 V2가 성공적으로 완료되었습니다.",
            "llm": f"segformer-b2-parsing+{XAI_PROMPT_MODEL}+{GEMINI_FLASH_MODEL}"
        }
        
    except Exception as e:
        error_detail = traceback.format_exc()
        run_time = time.time() - start_time
        
        # 오류 로그 저장
        try:
            save_test_log(
                person_url=person_s3_url or "",
                dress_url=dress_s3_url or None,
                result_url=result_s3_url or "",
                model=model_id,
                prompt=used_prompt,
                success=False,
                run_time=run_time
            )
        except:
            pass  # 로그 저장 실패해도 계속 진행
        
        print(f"커스텀 트라이온 파이프라인 V2 오류: {e}")
        traceback.print_exc()
        
        return {
            "success": False,
            "prompt": used_prompt,
            "result_image": "",
            "message": f"커스텀 트라이온 파이프라인 V2 중 오류 발생: {str(e)}",
            "llm": f"segformer-b2-parsing+{XAI_PROMPT_MODEL}+{GEMINI_FLASH_MODEL}",
            "error": str(e)
        }


# ============================================================
# V3 파이프라인 헬퍼 함수
# ============================================================

def decode_base64_to_image(image_data: bytes) -> Image.Image:
    """
    Gemini API 응답에서 이미지 데이터를 추출하여 PIL Image로 변환합니다.
    
    Args:
        image_data: Gemini API 응답의 이미지 바이너리 데이터
    
    Returns:
        Image.Image: 변환된 PIL Image
    """
    return Image.open(io.BytesIO(image_data))


def load_v3_stage2_prompt(xai_prompt: str) -> str:
    """
    V3 Stage 2 프롬프트 템플릿 로드 + X.AI 프롬프트 결합 + 강력한 최종 제약사항 추가
    """
    prompt_path = os.path.join(
        os.path.dirname(__file__), "..", "prompts", "v3", "prompt_stage2_outfit.txt"
    )

    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            template = f.read().strip()

        # 템플릿 + X.AI 분석 + [강력한 최종 제약사항 (샌드위치 기법)]
        combined = (
            template
            + "\n\n--- DETAILED INSTRUCTIONS FROM ANALYSIS ---\n"
            + xai_prompt
            + "\n\n--- FINAL CONSTRAINTS (EXECUTE LAST) ---\n"
            + "1. FACE PRIORITY: The face must match Image 1 EXACTLY. Do not re-render, beautify, or alter facial features.\n"
            + "2. TEXTURE RESET: The texture of the original top (knitted/thick) must be completely removed. Use ONLY the texture from Image 2.\n"
            + "3. NO GHOSTING: The original black pants must NOT be visible under the dress or on the legs. \n"
            + "4. SKIN TEXTURE: If legs or arms are exposed, use high-quality natural skin texture, not clothing texture.\n"
            + "5. SHOE FIX: Original sneakers are BANNED. Render appropriate formal footwear."
        )

        return combined
    except FileNotFoundError:
        print(f"[V3] WARNING: Stage 2 프롬프트 템플릿을 찾을 수 없습니다: {prompt_path}")
        # Fallback: 기본 프롬프트 반환
        return f"""STAGE 2 — STRICT OUTFIT REPLACEMENT
THIS STAGE MUST REMOVE THE ORIGINAL OUTFIT COMPLETELY.
THIS STAGE MUST APPLY THE OUTFIT FROM IMAGE 2 EXACTLY.

DO NOT MODIFY:
- face
- head shape
- expression
- hairstyle
- body shape
- pose
- background
- lighting

TASK (MUST FOLLOW):
Replace ALL of the person's original clothing with the outfit from Image 2.
The original clothing MUST NOT appear in the final image.

--- DETAILED INSTRUCTIONS FROM IMAGE ANALYSIS ---
{xai_prompt}

--- FINAL CONSTRAINTS (EXECUTE LAST) ---
1. FACE PRIORITY: The face must match Image 1 EXACTLY. Do not re-render, beautify, or alter facial features.
2. OUTFIT FIT: Ensure the new outfit fits perfectly. NO traces of the original clothing (pants/top).
3. NO GHOSTING: No mixed outfits or double clothing artifacts.
4. LENGTH LOGIC: If the dress is short, legs/shoes must be realistic. If long, cover everything underneath."""
    except Exception as e:
        print(f"[V3] ERROR loading prompt file: {e}")
        return xai_prompt


def load_v3_stage3_prompt() -> str:
    """
    V3 Stage 3 프롬프트 템플릿을 로드합니다.
    
    Returns:
        str: Stage 3 프롬프트
    """
    prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "v3", "prompt_stage3_background_lighting.txt")
    
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"[V3] WARNING: Stage 3 프롬프트 템플릿을 찾을 수 없습니다: {prompt_path}")
        # Fallback: 기본 프롬프트 반환
        return """LIGHTING CORRECTION STAGE — DO NOT CHANGE IDENTITY OR OUTFIT

Adjust lighting, shadows, and color grading of the subject to match the background.
Do NOT alter the face or outfit.

IDENTITY RULES (NEVER BREAK):
- Do NOT change face shape, skin texture, expression, or identity.
- Preserve the exact face from Stage 2.

OUTFIT RULES:
- Do NOT alter outfit design, texture, color, length, material, or silhouette.

LIGHTING MATCHING REQUIREMENTS:
- Match global illumination to background.
- Adjust brightness, shadow falloff, and rimlight direction.
- Match color temperature (warm/cool) to environment."""
    except Exception as e:
        print(f"[V3] ERROR: Stage 3 프롬프트 로드 실패: {e}")
        return "Adjust lighting to match background. Keep face and outfit unchanged."


# ============================================================
# V4 프롬프트 로드 함수
# ============================================================

def load_v4_stage2_prompt(xai_prompt: str) -> str:
    """
    V4 Stage 2 프롬프트 템플릿 로드 + X.AI 프롬프트 결합 + 강력한 최종 제약사항 추가
    """
    prompt_path = os.path.join(
        os.path.dirname(__file__), "..", "prompts", "v4", "prompt_stage2_outfit.txt"
    )

    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            template = f.read().strip()

        # 템플릿 + X.AI 분석 + [강력한 최종 제약사항 (샌드위치 기법)]
        combined = (
            template
            + "\n\n--- DETAILED INSTRUCTIONS FROM ANALYSIS ---\n"
            + xai_prompt
            + "\n\n--- FINAL CONSTRAINTS (EXECUTE LAST) ---\n"
            + "1. FACE PRIORITY: The face must match Image 1 EXACTLY. Do not re-render, beautify, or alter facial features.\n"
            + "2. TEXTURE RESET: The texture of the original top (knitted/thick) must be completely removed. Use ONLY the texture from Image 2.\n"
            + "3. NO GHOSTING: The original black pants must NOT be visible under the dress or on the legs. \n"
            + "4. SKIN TEXTURE: If legs or arms are exposed, use high-quality natural skin texture, not clothing texture.\n"
            + "5. SHOE FIX: Original sneakers are BANNED. Render appropriate formal footwear."
        )

        return combined
    except FileNotFoundError:
        print(f"[V4] WARNING: Stage 2 프롬프트 템플릿을 찾을 수 없습니다: {prompt_path}")
        # Fallback: 기본 프롬프트 반환
        return f"""STAGE 2 — STRICT OUTFIT REPLACEMENT
THIS STAGE MUST REMOVE THE ORIGINAL OUTFIT COMPLETELY.
THIS STAGE MUST APPLY THE OUTFIT FROM IMAGE 2 EXACTLY.

DO NOT MODIFY:
- face
- head shape
- expression
- hairstyle
- body shape
- pose
- background
- lighting

TASK (MUST FOLLOW):
Replace ALL of the person's original clothing with the outfit from Image 2.
The original clothing MUST NOT appear in the final image.

--- DETAILED INSTRUCTIONS FROM IMAGE ANALYSIS ---
{xai_prompt}

--- FINAL CONSTRAINTS (EXECUTE LAST) ---
1. FACE PRIORITY: The face must match Image 1 EXACTLY. Do not re-render, beautify, or alter facial features.
2. OUTFIT FIT: Ensure the new outfit fits perfectly. NO traces of the original clothing (pants/top).
3. NO GHOSTING: No mixed outfits or double clothing artifacts.
4. LENGTH LOGIC: If the dress is short, legs/shoes must be realistic. If long, cover everything underneath."""
    except Exception as e:
        print(f"[V4] ERROR loading prompt file: {e}")
        return xai_prompt


def load_v4_stage3_prompt() -> str:
    """
    V4 Stage 3 프롬프트 템플릿을 로드합니다.
    
    Returns:
        str: Stage 3 프롬프트
    """
    prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "v4", "prompt_stage3_background_lighting.txt")
    
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"[V4] WARNING: Stage 3 프롬프트 템플릿을 찾을 수 없습니다: {prompt_path}")
        # Fallback: 기본 프롬프트 반환
        return """LIGHTING CORRECTION STAGE — DO NOT CHANGE IDENTITY OR OUTFIT

Adjust lighting, shadows, and color grading of the subject to match the background.
Do NOT alter the face or outfit.

IDENTITY RULES (NEVER BREAK):
- Do NOT change face shape, skin texture, expression, or identity.
- Preserve the exact face from Stage 2.

OUTFIT RULES:
- Do NOT alter outfit design, texture, color, length, material, or silhouette.

LIGHTING MATCHING REQUIREMENTS:
- Match global illumination to background.
- Adjust brightness, shadow falloff, and rimlight direction.
- Match color temperature (warm/cool) to environment."""
    except Exception as e:
        print(f"[V4] ERROR: Stage 3 프롬프트 로드 실패: {e}")
        return "Adjust lighting to match background. Keep face and outfit unchanged."


def load_v4_unified_prompt(xai_prompt: str) -> str:
    """
    V4 통합 프롬프트: Stage 2와 Stage 3 프롬프트를 순서대로 결합
    
    Args:
        xai_prompt: X.AI에서 생성된 프롬프트
    
    Returns:
        str: Stage 2 프롬프트 + Stage 3 프롬프트를 순서대로 결합한 통합 프롬프트
    """
    # Stage 2 프롬프트 로드 (X.AI 프롬프트 포함)
    stage2_prompt = load_v4_stage2_prompt(xai_prompt)
    
    # Stage 3 프롬프트 로드
    stage3_prompt = load_v4_stage3_prompt()
    
    # 순서대로 결합: Stage 2 먼저, 그 다음 Stage 3
    unified_prompt = (
        stage2_prompt
        + "\n\n"
        + "="*80
        + "\n"
        + stage3_prompt
    )
    
    return unified_prompt


# ============================================================
# V3 파이프라인 메인 함수
# ============================================================

async def generate_unified_tryon_v3(
    person_img: Image.Image,
    garment_img: Image.Image,
    background_img: Image.Image,
    model_id: str = "xai-gemini-unified-v3"
) -> Dict:
    """
    통합 트라이온 파이프라인 V3: 2단계 Gemini 플로우
    - Stage 1: 의상 이미지 전처리 + X.AI 프롬프트 생성
    - Stage 2: Gemini로 의상 교체만 수행 (person + garment)
    - Stage 3: Gemini로 배경 합성 + 조명 보정 (dressed_person + background)
    
    Args:
        person_img: 사람 이미지 (PIL Image)
        garment_img: 의상 이미지 (PIL Image)
        background_img: 배경 이미지 (PIL Image)
        model_id: 모델 ID (기본값: "xai-gemini-unified-v3")
    
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
    background_s3_url = ""
    stage2_result_s3_url = ""
    result_s3_url = ""
    used_prompt = ""
    
    try:
        # ============================================================
        # Stage 1: X.AI 프롬프트 생성 준비
        # ============================================================
        print("\n" + "="*80)
        print("V3 파이프라인 시작")
        print("="*80)
        
        # 배경 이미지는 원본 그대로 유지
        background_img_processed = background_img
        
        # S3에 입력 이미지 업로드
        person_buffered = io.BytesIO()
        person_img.save(person_buffered, format="PNG")
        person_s3_url = upload_log_to_s3(person_buffered.getvalue(), model_id, "person") or ""
        
        garment_buffered = io.BytesIO()
        garment_img.save(garment_buffered, format="PNG")
        garment_s3_url = upload_log_to_s3(garment_buffered.getvalue(), model_id, "garment") or ""
        
        background_buffered = io.BytesIO()
        background_img_processed.save(background_buffered, format="PNG")
        background_s3_url = upload_log_to_s3(background_buffered.getvalue(), model_id, "background") or ""
        
        # ============================================================
        # Stage 1: X.AI 프롬프트 생성
        # ============================================================
        print("\n" + "="*80)
        print("[Stage 1] X.AI 프롬프트 생성 시작 (V3: 원본 의상 이미지 사용)")
        print("="*80)
        
        xai_result = generate_prompt_from_images(person_img, garment_img)
        
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
        print("\n[Stage 1] 생성된 프롬프트:")
        print("-"*80)
        print(used_prompt)
        print("="*80 + "\n")
        
        # ============================================================
        # Stage 2: Gemini로 의상 교체만 수행
        # ============================================================
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
        
        client = genai.Client(api_key=api_key)
        
        print("\n" + "="*80)
        print("[Stage 2] Gemini 2.5 Flash - 의상 교체만 수행")
        print("="*80)
        
        # Stage 2 프롬프트 로드
        stage2_prompt = load_v3_stage2_prompt(used_prompt)
        
        print("[Stage 2] 프롬프트:")
        print("-"*80)
        print(stage2_prompt)
        print("="*80 + "\n")
        
        print("[Stage 2] Gemini API 호출 시작...")
        print(f"[Stage 2] 입력 이미지: person_img ({person_img.size[0]}x{person_img.size[1]}), garment_img ({garment_img.size[0]}x{garment_img.size[1]})")
        stage2_start_time = time.time()
        
        try:
            stage2_response = client.models.generate_content(
                model=GEMINI_FLASH_MODEL,
                contents=[person_img, garment_img, stage2_prompt]
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
            
            print(f"[Stage 2] Gemini API 호출 실패: {exc}")
            traceback.print_exc()
            return {
                "success": False,
                "prompt": used_prompt,
                "result_image": "",
                "message": f"Stage 2 Gemini 호출에 실패했습니다: {str(exc)}",
                "llm": f"{XAI_PROMPT_MODEL}+{GEMINI_FLASH_MODEL}",
                "error": "stage2_gemini_call_failed"
            }
        
        stage2_latency = time.time() - stage2_start_time
        print(f"[Stage 2] Gemini API 응답 수신 완료 (지연 시간: {stage2_latency:.2f}초)")
        
        # Stage 2 응답 확인 및 이미지 추출
        if not stage2_response.candidates or len(stage2_response.candidates) == 0:
            error_msg = "Stage 2: Gemini API가 응답을 생성하지 못했습니다."
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
                "error": "stage2_no_response"
            }
        
        stage2_candidate = stage2_response.candidates[0]
        if not hasattr(stage2_candidate, 'content') or stage2_candidate.content is None:
            error_msg = "Stage 2: Gemini API 응답에 content가 없습니다."
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
                "error": "stage2_no_content"
            }
        
        if not hasattr(stage2_candidate.content, 'parts') or stage2_candidate.content.parts is None:
            error_msg = "Stage 2: Gemini API 응답에 parts가 없습니다."
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
                "error": "stage2_no_parts"
            }
        
        # Stage 2 이미지 추출
        stage2_image_parts = [
            part.inline_data.data
            for part in stage2_candidate.content.parts
            if hasattr(part, 'inline_data') and part.inline_data
        ]
        
        if not stage2_image_parts:
            error_msg = "Stage 2: Gemini API가 이미지를 생성하지 못했습니다."
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
                "error": "stage2_no_image_generated"
            }
        
        # Stage 2 결과 이미지 변환
        dressed_person_img = decode_base64_to_image(stage2_image_parts[0])
        print(f"[Stage 2] 의상 교체 완료 - 이미지 크기: {dressed_person_img.size[0]}x{dressed_person_img.size[1]}")
        
        # Stage 2 결과 S3 업로드
        stage2_buffered = io.BytesIO()
        dressed_person_img.save(stage2_buffered, format="PNG")
        stage2_result_s3_url = upload_log_to_s3(stage2_buffered.getvalue(), model_id, "stage2_result") or ""
        
        # ============================================================
        # Stage 3: Gemini로 배경 합성 + 조명 보정
        # ============================================================
        print("\n" + "="*80)
        print("[Stage 3] Gemini 2.5 Flash - 배경 합성 + 조명 보정")
        print("="*80)
        
        # Stage 3 프롬프트 로드
        stage3_prompt = load_v3_stage3_prompt()
        
        print("[Stage 3] 프롬프트:")
        print("-"*80)
        print(stage3_prompt)
        print("="*80 + "\n")
        
        print("[Stage 3] Gemini API 호출 시작...")
        print(f"[Stage 3] 입력 이미지: dressed_person_img ({dressed_person_img.size[0]}x{dressed_person_img.size[1]}), background_img ({background_img_processed.size[0]}x{background_img_processed.size[1]})")
        stage3_start_time = time.time()
        
        try:
            stage3_response = client.models.generate_content(
                model=GEMINI_FLASH_MODEL,
                contents=[dressed_person_img, background_img_processed, stage3_prompt]
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
            
            print(f"[Stage 3] Gemini API 호출 실패: {exc}")
            traceback.print_exc()
            return {
                "success": False,
                "prompt": used_prompt,
                "result_image": "",
                "message": f"Stage 3 Gemini 호출에 실패했습니다: {str(exc)}",
                "llm": f"{XAI_PROMPT_MODEL}+{GEMINI_FLASH_MODEL}",
                "error": "stage3_gemini_call_failed"
            }
        
        stage3_latency = time.time() - stage3_start_time
        print(f"[Stage 3] Gemini API 응답 수신 완료 (지연 시간: {stage3_latency:.2f}초)")
        
        # Stage 3 응답 확인 및 이미지 추출
        if not stage3_response.candidates or len(stage3_response.candidates) == 0:
            error_msg = "Stage 3: Gemini API가 응답을 생성하지 못했습니다."
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
                "error": "stage3_no_response"
            }
        
        stage3_candidate = stage3_response.candidates[0]
        if not hasattr(stage3_candidate, 'content') or stage3_candidate.content is None:
            error_msg = "Stage 3: Gemini API 응답에 content가 없습니다."
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
                "error": "stage3_no_content"
            }
        
        if not hasattr(stage3_candidate.content, 'parts') or stage3_candidate.content.parts is None:
            error_msg = "Stage 3: Gemini API 응답에 parts가 없습니다."
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
                "error": "stage3_no_parts"
            }
        
        # Stage 3 이미지 추출
        stage3_image_parts = [
            part.inline_data.data
            for part in stage3_candidate.content.parts
            if hasattr(part, 'inline_data') and part.inline_data
        ]
        
        if not stage3_image_parts:
            error_msg = "Stage 3: Gemini API가 이미지를 생성하지 못했습니다."
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
                "error": "stage3_no_image_generated"
            }
        
        # Stage 3 결과 이미지 변환
        final_img = decode_base64_to_image(stage3_image_parts[0])
        print(f"[Stage 3] 배경 합성 + 조명 보정 완료 - 이미지 크기: {final_img.size[0]}x{final_img.size[1]}")
        
        # ============================================================
        # 최종 결과 처리 및 S3 업로드
        # ============================================================
        result_image_base64 = base64.b64encode(stage3_image_parts[0]).decode()
        
        result_buffered = io.BytesIO()
        final_img.save(result_buffered, format="PNG")
        result_s3_url = upload_log_to_s3(result_buffered.getvalue(), model_id, "result") or ""
        
        run_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("[V3] 파이프라인 완료")
        print("="*80)
        print(f"전체 실행 시간: {run_time:.2f}초")
        print(f"Stage 2 지연 시간: {stage2_latency:.2f}초")
        print(f"Stage 3 지연 시간: {stage3_latency:.2f}초")
        print(f"최종 이미지 크기: {final_img.size[0]}x{final_img.size[1]}")
        print("="*80 + "\n")
        
        # 성공 로그 저장
        save_test_log(
            person_url=person_s3_url or "",
            dress_url=garment_s3_url or None,
            result_url=result_s3_url or "",
            model=model_id,
            prompt=used_prompt,
            success=True,
            run_time=run_time
        )
        
        return {
            "success": True,
            "prompt": used_prompt,
            "result_image": f"data:image/png;base64,{result_image_base64}",
            "message": "통합 트라이온 파이프라인 V3가 성공적으로 완료되었습니다.",
            "llm": f"{XAI_PROMPT_MODEL}+{GEMINI_FLASH_MODEL}+{GEMINI_FLASH_MODEL}"
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
        
        print(f"통합 트라이온 파이프라인 V3 오류: {e}")
        traceback.print_exc()
        
        return {
            "success": False,
            "prompt": used_prompt,
            "result_image": "",
            "message": f"통합 트라이온 파이프라인 V3 중 오류 발생: {str(e)}",
            "llm": f"{XAI_PROMPT_MODEL}+{GEMINI_FLASH_MODEL}+{GEMINI_FLASH_MODEL}",
            "error": str(e)
        }


# ============================================================
# V4 파이프라인 메인 함수
# ============================================================

async def generate_unified_tryon_v4(
    person_img: Image.Image,
    garment_img: Image.Image,
    background_img: Image.Image,
    model_id: str = "xai-gemini-unified-v4"
) -> Dict:
    """
    통합 트라이온 파이프라인 V4: 통합 Gemini 3 Flash 플로우
    - Stage 1: X.AI 프롬프트 생성
    - Stage 2: Gemini 3 Flash로 의상 교체 + 배경 합성 통합 처리 (person + garment + background)
    
    Args:
        person_img: 사람 이미지 (PIL Image)
        garment_img: 의상 이미지 (PIL Image)
        background_img: 배경 이미지 (PIL Image)
        model_id: 모델 ID (기본값: "xai-gemini-unified-v4")
    
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
    background_s3_url = ""
    result_s3_url = ""
    used_prompt = ""
    
    try:
        # ============================================================
        # Stage 1: X.AI 프롬프트 생성 준비
        # ============================================================
        print("\n" + "="*80)
        print("V4 파이프라인 시작")
        print("="*80)
        
        # 배경 이미지는 원본 그대로 유지
        background_img_processed = background_img
        
        # S3에 입력 이미지 업로드
        # person_buffered = io.BytesIO()
        # person_img.save(person_buffered, format="PNG")
        # person_s3_url = upload_log_to_s3(person_buffered.getvalue(), model_id, "person") or ""
        person_s3_url = ""
        
        # garment_buffered = io.BytesIO()
        # garment_img.save(garment_buffered, format="PNG")
        # garment_s3_url = upload_log_to_s3(garment_buffered.getvalue(), model_id, "garment") or ""
        garment_s3_url = ""
        
        # background_buffered = io.BytesIO()
        # background_img_processed.save(background_buffered, format="PNG")
        # background_s3_url = upload_log_to_s3(background_buffered.getvalue(), model_id, "background") or ""
        background_s3_url = ""
        
        # ============================================================
        # Stage 1: X.AI 프롬프트 생성
        # ============================================================
        print("\n" + "="*80)
        print("[Stage 1] X.AI 프롬프트 생성 시작 (V4: 원본 의상 이미지 사용)")
        print("="*80)
        
        xai_result = await generate_prompt_from_images(person_img, garment_img)
        
        if not xai_result.get("success"):
            error_msg = xai_result.get("message", "X.AI 프롬프트 생성에 실패했습니다.")
            run_time = time.time() - start_time
            
            # save_test_log(
            #     person_url=person_s3_url or "",
            #     dress_url=garment_s3_url or None,
            #     result_url="",
            #     model=model_id,
            #     prompt="",
            #     success=False,
            #     run_time=run_time
            # )
            
            return {
                "success": False,
                "prompt": "",
                "result_image": "",
                "message": error_msg,
                "llm": XAI_PROMPT_MODEL,
                "error": xai_result.get("error", "xai_prompt_generation_failed")
            }
        
        used_prompt = xai_result.get("prompt", "")
        
        # ============================================================
        # Stage 2: Gemini 3 Flash로 의상 교체 + 배경 합성 통합 처리
        # ============================================================
        try:
            client_pool = get_gemini_client_pool()
        except ValueError as e:
            error_msg = f".env 파일에 GEMINI_3_API_KEY가 설정되지 않았습니다: {str(e)}"
            run_time = time.time() - start_time
            
            # save_test_log(
            #     person_url=person_s3_url or "",
            #     dress_url=garment_s3_url or None,
            #     result_url="",
            #     model=model_id,
            #     prompt=used_prompt,
            #     success=False,
            #     run_time=run_time
            # )
            
            return {
                "success": False,
                "prompt": used_prompt,
                "result_image": "",
                "message": error_msg,
                "llm": f"{XAI_PROMPT_MODEL}+{GEMINI_3_FLASH_MODEL}",
                "error": "gemini_api_key_not_found"
            }
        
        print("\n" + "="*80)
        print("[Stage 2] Gemini 3 Flash - 의상 교체 + 배경 합성 통합 처리 (다중 API 키 풀 사용)")
        print("="*80)
        
        # 통합 프롬프트 로드 (Stage 2 + Stage 3 순서대로 결합)
        unified_prompt = load_v4_unified_prompt(used_prompt)
        
        print("[Stage 2] Gemini API 호출 시작 (다중 키 풀 사용)...")
        print(f"[Stage 2] 입력 이미지: person_img ({person_img.size[0]}x{person_img.size[1]}), garment_img ({garment_img.size[0]}x{garment_img.size[1]}), background_img ({background_img_processed.size[0]}x{background_img_processed.size[1]})")
        stage2_start_time = time.time()
        
        try:
            stage2_response = await client_pool.generate_content_with_retry_async(
                model=GEMINI_3_FLASH_MODEL,
                contents=[person_img, garment_img, background_img_processed, unified_prompt]
            )
        except Exception as exc:
            run_time = time.time() - start_time
            # save_test_log(
            #     person_url=person_s3_url or "",
            #     dress_url=garment_s3_url or None,
            #     result_url="",
            #     model=model_id,
            #     prompt=used_prompt,
            #     success=False,
            #     run_time=run_time
            # )
            
            print(f"[Stage 2] Gemini API 호출 실패: {exc}")
            traceback.print_exc()
            return {
                "success": False,
                "prompt": used_prompt,
                "result_image": "",
                "message": f"Stage 2 Gemini 호출에 실패했습니다: {str(exc)}",
                "llm": f"{XAI_PROMPT_MODEL}+{GEMINI_3_FLASH_MODEL}",
                "error": "stage2_gemini_call_failed"
            }
        
        stage2_latency = time.time() - stage2_start_time
        print(f"[Stage 2] Gemini API 응답 수신 완료 (지연 시간: {stage2_latency:.2f}초)")
        
        # Stage 2 응답 확인 및 이미지 추출
        if not stage2_response.candidates or len(stage2_response.candidates) == 0:
            error_msg = "Stage 2: Gemini API가 응답을 생성하지 못했습니다."
            run_time = time.time() - start_time
            
            # save_test_log(
            #     person_url=person_s3_url or "",
            #     dress_url=garment_s3_url or None,
            #     result_url="",
            #     model=model_id,
            #     prompt=used_prompt,
            #     success=False,
            #     run_time=run_time
            # )
            
            return {
                "success": False,
                "prompt": used_prompt,
                "result_image": "",
                "message": error_msg,
                "llm": f"{XAI_PROMPT_MODEL}+{GEMINI_3_FLASH_MODEL}",
                "error": "stage2_no_response"
            }
        
        stage2_candidate = stage2_response.candidates[0]
        if not hasattr(stage2_candidate, 'content') or stage2_candidate.content is None:
            error_msg = "Stage 2: Gemini API 응답에 content가 없습니다."
            run_time = time.time() - start_time
            
            # save_test_log(
            #     person_url=person_s3_url or "",
            #     dress_url=garment_s3_url or None,
            #     result_url="",
            #     model=model_id,
            #     prompt=used_prompt,
            #     success=False,
            #     run_time=run_time
            # )
            
            return {
                "success": False,
                "prompt": used_prompt,
                "result_image": "",
                "message": error_msg,
                "llm": f"{XAI_PROMPT_MODEL}+{GEMINI_3_FLASH_MODEL}",
                "error": "stage2_no_content"
            }
        
        if not hasattr(stage2_candidate.content, 'parts') or stage2_candidate.content.parts is None:
            error_msg = "Stage 2: Gemini API 응답에 parts가 없습니다."
            run_time = time.time() - start_time
            
            # save_test_log(
            #     person_url=person_s3_url or "",
            #     dress_url=garment_s3_url or None,
            #     result_url="",
            #     model=model_id,
            #     prompt=used_prompt,
            #     success=False,
            #     run_time=run_time
            # )
            
            return {
                "success": False,
                "prompt": used_prompt,
                "result_image": "",
                "message": error_msg,
                "llm": f"{XAI_PROMPT_MODEL}+{GEMINI_3_FLASH_MODEL}",
                "error": "stage2_no_parts"
            }
        
        # Stage 2 이미지 추출
        stage2_image_parts = [
            part.inline_data.data
            for part in stage2_candidate.content.parts
            if hasattr(part, 'inline_data') and part.inline_data
        ]
        
        if not stage2_image_parts:
            error_msg = "Stage 2: Gemini API가 이미지를 생성하지 못했습니다."
            run_time = time.time() - start_time
            
            # save_test_log(
            #     person_url=person_s3_url or "",
            #     dress_url=garment_s3_url or None,
            #     result_url="",
            #     model=model_id,
            #     prompt=used_prompt,
            #     success=False,
            #     run_time=run_time
            # )
            
            return {
                "success": False,
                "prompt": used_prompt,
                "result_image": "",
                "message": error_msg,
                "llm": f"{XAI_PROMPT_MODEL}+{GEMINI_3_FLASH_MODEL}",
                "error": "stage2_no_image_generated"
            }
        
        # 최종 결과 이미지 변환
        final_img = decode_base64_to_image(stage2_image_parts[0])
        print(f"[Stage 2] 의상 교체 + 배경 합성 완료 - 이미지 크기: {final_img.size[0]}x{final_img.size[1]}")
        
        # ============================================================
        # 최종 결과 처리 및 S3 업로드
        # ============================================================
        result_image_base64 = base64.b64encode(stage2_image_parts[0]).decode()
        
        # result_buffered = io.BytesIO()
        # final_img.save(result_buffered, format="PNG")
        # result_s3_url = upload_log_to_s3(result_buffered.getvalue(), model_id, "result") or ""
        result_s3_url = ""
        
        run_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("[V4] 파이프라인 완료")
        print("="*80)
        print(f"전체 실행 시간: {run_time:.2f}초")
        print(f"Stage 2 지연 시간: {stage2_latency:.2f}초")
        print(f"최종 이미지 크기: {final_img.size[0]}x{final_img.size[1]}")
        print("="*80 + "\n")
        
        # 성공 로그 저장
        save_test_log(
            person_url=person_s3_url or "",
            dress_url=garment_s3_url or None,
            result_url=result_s3_url or "",
            model=model_id,
            prompt=used_prompt,
            success=True,
            run_time=run_time
        )
        
        return {
            "success": True,
            "prompt": used_prompt,
            "result_image": f"data:image/png;base64,{result_image_base64}",
            "message": "통합 트라이온 파이프라인 V4가 성공적으로 완료되었습니다.",
            "llm": f"{XAI_PROMPT_MODEL}+{GEMINI_3_FLASH_MODEL}+{GEMINI_3_FLASH_MODEL}"
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
        
        print(f"통합 트라이온 파이프라인 V4 오류: {e}")
        traceback.print_exc()
        
        return {
            "success": False,
            "prompt": used_prompt,
            "result_image": "",
            "message": f"통합 트라이온 파이프라인 V4 중 오류 발생: {str(e)}",
            "llm": f"{XAI_PROMPT_MODEL}+{GEMINI_3_FLASH_MODEL}+{GEMINI_3_FLASH_MODEL}",
            "error": str(e)
        }