"""CustomV4 통합 트라이온 서비스"""
import io
import base64
import time
import traceback
from typing import Dict
from PIL import Image

from core.xai_client import generate_prompt_from_images
from core.s3_client import upload_log_to_s3
from services.image_service import preprocess_dress_image
from services.log_service import save_test_log
from core.segformer_garment_parser import parse_garment_image_v4
from services.tryon_service import (
    load_v4_unified_prompt,
    decode_base64_to_image
)
from config.settings import GEMINI_3_FLASH_MODEL, XAI_PROMPT_MODEL
from core.gemini_client import get_gemini_client_pool


def generate_unified_tryon_custom_v4(
    person_img: Image.Image,
    garment_img: Image.Image,
    background_img: Image.Image,
    model_id: str = "xai-gemini-unified-custom-v4"
) -> Dict:
    """
    CustomV4 통합 트라이온 파이프라인: 의상 누끼 + X.AI 프롬프트 생성 + 통합 Gemini 3 플로우
    
    CustomV4는 기존 V3 커스텀과 동일한 구조를 가지지만, Gemini 3 Flash 모델을 사용합니다.
    
    파이프라인 단계:
    - Stage 0: 의상 이미지 누끼 처리 (배경 제거)
    - Stage 1: 누끼 처리된 의상 이미지로 X.AI 프롬프트 생성
    - Stage 2: Gemini 3로 의상 교체 + 배경 합성 통합 처리 (person + garment_nukki + background)
    
    Args:
        person_img: 사람 이미지 (PIL Image)
        garment_img: 의상 이미지 (PIL Image)
        background_img: 배경 이미지 (PIL Image)
        model_id: 모델 ID (기본값: "xai-gemini-unified-custom-v4")
    
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
    garment_nukki_s3_url = ""
    background_s3_url = ""
    result_s3_url = ""
    used_prompt = ""
    
    try:
        # ============================================================
        # Stage 0: 의상 이미지 누끼 처리 (배경 제거)
        # ============================================================
        print("\n" + "="*80)
        print("CustomV4 파이프라인 시작")
        print("="*80)
        
        print("\n[Stage 0] 의상 이미지 누끼 처리 시작 (HuggingFace API)...")
        parsing_result = parse_garment_image_v4(garment_img)
        
        if not parsing_result.get("success"):
            error_msg = parsing_result.get("message", "의상 이미지 누끼 처리에 실패했습니다.")
            print(f"[Stage 0] 누끼 처리 실패: {error_msg}")
            print("[Stage 0] 원본 의상 이미지를 그대로 사용합니다.")
            # 누끼 처리 실패 시 원본 이미지를 RGB로 변환하여 사용
            garment_nukki_rgb = garment_img.convert('RGB')
        else:
            garment_nukki_rgb = parsing_result.get("garment_only")
            if garment_nukki_rgb is None:
                print("[Stage 0] 누끼 처리 결과에서 garment_only 이미지를 찾을 수 없습니다.")
                garment_nukki_rgb = garment_img.convert('RGB')
            else:
                print("[Stage 0] 의상 이미지 누끼 처리 완료")
                print(f"[Stage 0] 누끼 처리된 이미지 크기: {garment_nukki_rgb.size[0]}x{garment_nukki_rgb.size[1]}, 모드: {garment_nukki_rgb.mode}")
        
        # ============================================================
        # Stage 1: X.AI 프롬프트 생성 준비
        # ============================================================
        # garment_nukki_rgb는 이미 RGB 모드로 반환됨
        
        # 배경 이미지는 원본 그대로 유지
        background_img_processed = background_img
        
        # S3에 입력 이미지 업로드
        person_buffered = io.BytesIO()
        person_img.save(person_buffered, format="PNG")
        person_s3_url = upload_log_to_s3(person_buffered.getvalue(), model_id, "person") or ""
        
        garment_buffered = io.BytesIO()
        garment_nukki_rgb.save(garment_buffered, format="PNG")
        garment_s3_url = upload_log_to_s3(garment_buffered.getvalue(), model_id, "garment") or ""
        
        garment_nukki_buffered = io.BytesIO()
        garment_nukki_rgb.save(garment_nukki_buffered, format="PNG")
        garment_nukki_s3_url = upload_log_to_s3(garment_nukki_buffered.getvalue(), model_id, "garment_nukki") or ""
        
        background_buffered = io.BytesIO()
        background_img_processed.save(background_buffered, format="PNG")
        background_s3_url = upload_log_to_s3(background_buffered.getvalue(), model_id, "background") or ""
        
        # ============================================================
        # Stage 1: X.AI 프롬프트 생성 (누끼 처리된 의상 이미지 사용)
        # ============================================================
        print("\n" + "="*80)
        print("[Stage 1] X.AI 프롬프트 생성 시작 (CustomV4: 누끼 처리된 의상 이미지 사용)")
        print("="*80)
        
        xai_result = generate_prompt_from_images(person_img, garment_nukki_rgb)
        
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
        # Stage 2: Gemini 3로 의상 교체 + 배경 합성 통합 처리
        # ============================================================
        try:
            client_pool = get_gemini_client_pool()
        except ValueError as e:
            error_msg = f".env 파일에 GEMINI_3_API_KEY가 설정되지 않았습니다: {str(e)}"
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
                "llm": f"{XAI_PROMPT_MODEL}+{GEMINI_3_FLASH_MODEL}",
                "error": "gemini_api_key_not_found"
            }
        
        print("\n" + "="*80)
        print("[Stage 2] Gemini 3 Flash - 의상 교체 + 배경 합성 통합 처리 (다중 API 키 풀 사용)")
        print("="*80)
        
        # 통합 프롬프트 로드 (Stage 2 + Stage 3 순서대로 결합)
        unified_prompt = load_v4_unified_prompt(used_prompt)
        
        print("[Stage 2] 통합 프롬프트:")
        print("-"*80)
        print(unified_prompt)
        print("="*80 + "\n")
        
        print("[Stage 2] Gemini API 호출 시작 (다중 키 풀 사용)...")
        print(f"[Stage 2] 입력 이미지: person_img ({person_img.size[0]}x{person_img.size[1]}), garment_nukki_rgb ({garment_nukki_rgb.size[0]}x{garment_nukki_rgb.size[1]}), background_img ({background_img_processed.size[0]}x{background_img_processed.size[1]})")
        stage2_start_time = time.time()
        
        try:
            stage2_response = client_pool.generate_content_with_retry(
                model=GEMINI_3_FLASH_MODEL,
                contents=[person_img, garment_nukki_rgb, background_img_processed, unified_prompt]
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
                "llm": f"{XAI_PROMPT_MODEL}+{GEMINI_3_FLASH_MODEL}",
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
                "llm": f"{XAI_PROMPT_MODEL}+{GEMINI_3_FLASH_MODEL}",
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
                "llm": f"{XAI_PROMPT_MODEL}+{GEMINI_3_FLASH_MODEL}",
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
        
        result_buffered = io.BytesIO()
        final_img.save(result_buffered, format="PNG")
        result_s3_url = upload_log_to_s3(result_buffered.getvalue(), model_id, "result") or ""
        
        run_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("[CustomV4] 파이프라인 완료")
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
            "message": "CustomV4 파이프라인이 성공적으로 완료되었습니다.",
            "llm": f"{XAI_PROMPT_MODEL}+{GEMINI_3_FLASH_MODEL}",
            "error": None
        }
        
    except Exception as e:
        run_time = time.time() - start_time
        error_detail = traceback.format_exc()
        print(f"[CustomV4] 파이프라인 오류: {e}")
        print(error_detail)
        
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
            "message": f"CustomV4 파이프라인 처리 중 오류가 발생했습니다: {str(e)}",
            "llm": f"{XAI_PROMPT_MODEL}+{GEMINI_3_FLASH_MODEL}",
            "error": "custom_v4_pipeline_error"
        }

