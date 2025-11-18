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
from config.settings import GEMINI_FLASH_MODEL, XAI_PROMPT_MODEL


def generate_unified_tryon(
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
        
        xai_result = generate_prompt_from_images(person_img, dress_img_processed)
        
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


def generate_unified_tryon_v2(
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

