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
        enhanced_prompt = f"""{used_prompt}

CRITICAL BACKGROUND REQUIREMENTS:
1. PRESERVE BACKGROUND ORIGINAL: The third image is the background. DO NOT stretch, distort, resize, crop, or modify the background image in ANY way. Keep the background EXACTLY as it appears in the original image - same dimensions, same aspect ratio, same quality, same resolution, same details. The background must remain completely unchanged.
2. NO BACKGROUND TRANSFORMATION: Do not resize, warp, or transform the background. Use the background image as-is without any geometric transformations or distortions.
3. NATURAL INTEGRATION: Place the person wearing the dress into this background scene naturally and seamlessly. The person should appear as if they were originally photographed in this background. Adjust ONLY the person to fit the background, not the other way around.
4. LIGHTING MATCHING: Adjust the person's lighting, shadows, and color tones to perfectly match the background environment. The person should look naturally lit by the same light source as the background.
5. PERSPECTIVE ALIGNMENT: Ensure the person's perspective, scale, and position match the background's perspective and depth. The person should appear at the correct distance and angle relative to the background.
6. SHADOW CONSISTENCY: Create realistic shadows for the person that match the background's lighting direction and intensity. Shadows should be cast on the ground or surfaces in the background.
7. COLOR HARMONY: Blend the person's colors with the background's color palette to create a cohesive, natural-looking composition.
8. EDGE BLENDING: Smoothly blend the edges of the person with the background to avoid any visible seams or artifacts.

ABSOLUTE RULE: The background image must remain 100% unchanged - no stretching, no distortion, no resizing, no cropping, no transformation of any kind. Only integrate the person into the existing background without modifying the background itself."""
        
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

