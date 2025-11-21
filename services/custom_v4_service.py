"""CustomV4 통합 트라이온 서비스"""
import os
import io
import base64
import time
import traceback
from typing import Dict
from PIL import Image
from google import genai

from core.xai_client import generate_prompt_from_images
from core.s3_client import upload_log_to_s3
from services.image_service import preprocess_dress_image
from services.log_service import save_test_log
from services.garment_nukki_service import remove_garment_background
from services.tryon_service import (
    load_v4_stage2_prompt,
    load_v4_stage3_prompt,
    decode_base64_to_image
)
from config.settings import GEMINI_3_FLASH_MODEL, XAI_PROMPT_MODEL


def generate_unified_tryon_custom_v4(
    person_img: Image.Image,
    garment_img: Image.Image,
    background_img: Image.Image,
    model_id: str = "xai-gemini-unified-custom-v4"
) -> Dict:
    """
    CustomV4 통합 트라이온 파이프라인: 의상 누끼 + X.AI 프롬프트 생성 + 2단계 Gemini 3 플로우
    
    CustomV4는 기존 V3 커스텀과 동일한 구조를 가지지만, Gemini 3 Flash 모델을 사용합니다.
    
    파이프라인 단계:
    - Stage 0: 의상 이미지 누끼 처리 (배경 제거)
    - Stage 1: 누끼 처리된 의상 이미지로 X.AI 프롬프트 생성
    - Stage 2: Gemini 3로 의상 교체만 수행 (person + garment_nukki)
    - Stage 3: Gemini 3로 배경 합성 + 조명 보정 (dressed_person + background)
    
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
    stage2_result_s3_url = ""
    result_s3_url = ""
    used_prompt = ""
    
    try:
        # ============================================================
        # Stage 0: 의상 이미지 누끼 처리 (배경 제거)
        # ============================================================
        print("\n" + "="*80)
        print("CustomV4 파이프라인 시작")
        print("="*80)
        
        print("\n[Stage 0] 의상 이미지 누끼 처리 시작...")
        try:
            garment_nukki = remove_garment_background(garment_img)
            print("[Stage 0] 의상 이미지 누끼 처리 완료")
            print(f"[Stage 0] 누끼 처리된 이미지 크기: {garment_nukki.size[0]}x{garment_nukki.size[1]}, 모드: {garment_nukki.mode}")
        except Exception as e:
            print(f"[Stage 0] 누끼 처리 실패: {e}")
            print("[Stage 0] 원본 의상 이미지를 그대로 사용합니다.")
            # 누끼 처리 실패 시 원본 이미지를 RGB로 변환하여 사용
            garment_nukki = garment_img.convert('RGB')
        
        # ============================================================
        # Stage 1: X.AI 프롬프트 생성 준비
        # ============================================================
        # 누끼 처리된 이미지를 RGB로 변환
        if garment_nukki.mode == 'RGBA':
            # RGBA를 RGB로 변환 (흰색 배경에 합성)
            white_bg = Image.new('RGB', garment_nukki.size, (255, 255, 255))
            white_bg.paste(garment_nukki, mask=garment_nukki.split()[3] if garment_nukki.mode == 'RGBA' else None)
            garment_nukki_rgb = white_bg
        else:
            garment_nukki_rgb = garment_nukki.convert('RGB')
        
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
        garment_nukki.save(garment_nukki_buffered, format="PNG")
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
        # Stage 2: Gemini 3로 의상 교체만 수행
        # ============================================================
        api_key = os.getenv("GEMINI_3_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            error_msg = ".env 파일에 GEMINI_3_API_KEY 또는 GEMINI_API_KEY가 설정되지 않았습니다."
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
        
        client = genai.Client(api_key=api_key)
        
        print("\n" + "="*80)
        print("[Stage 2] Gemini 3 Flash - 의상 교체만 수행")
        print("="*80)
        
        # Stage 2 프롬프트 로드
        stage2_prompt = load_v4_stage2_prompt(used_prompt)
        
        print("[Stage 2] 프롬프트:")
        print("-"*80)
        print(stage2_prompt)
        print("="*80 + "\n")
        
        print("[Stage 2] Gemini API 호출 시작...")
        print(f"[Stage 2] 입력 이미지: person_img ({person_img.size[0]}x{person_img.size[1]}), garment_img ({garment_nukki_rgb.size[0]}x{garment_nukki_rgb.size[1]})")
        stage2_start_time = time.time()
        
        try:
            stage2_response = client.models.generate_content(
                model=GEMINI_3_FLASH_MODEL,
                contents=[person_img, garment_nukki_rgb, stage2_prompt]
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
        
        # Stage 2 결과 이미지 변환
        dressed_person_img = decode_base64_to_image(stage2_image_parts[0])
        print(f"[Stage 2] 의상 교체 완료 - 이미지 크기: {dressed_person_img.size[0]}x{dressed_person_img.size[1]}")
        
        # Stage 2 결과 S3 업로드
        stage2_buffered = io.BytesIO()
        dressed_person_img.save(stage2_buffered, format="PNG")
        stage2_result_s3_url = upload_log_to_s3(stage2_buffered.getvalue(), model_id, "stage2_result") or ""
        
        # ============================================================
        # Stage 3: Gemini 3로 배경 합성 + 조명 보정
        # ============================================================
        print("\n" + "="*80)
        print("[Stage 3] Gemini 3 Flash - 배경 합성 + 조명 보정")
        print("="*80)
        
        # Stage 3 프롬프트 로드
        stage3_prompt = load_v4_stage3_prompt()
        
        print("[Stage 3] 프롬프트:")
        print("-"*80)
        print(stage3_prompt)
        print("="*80 + "\n")
        
        print("[Stage 3] Gemini API 호출 시작...")
        print(f"[Stage 3] 입력 이미지: dressed_person_img ({dressed_person_img.size[0]}x{dressed_person_img.size[1]}), background_img ({background_img_processed.size[0]}x{background_img_processed.size[1]})")
        stage3_start_time = time.time()
        
        try:
            stage3_response = client.models.generate_content(
                model=GEMINI_3_FLASH_MODEL,
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
                "llm": f"{XAI_PROMPT_MODEL}+{GEMINI_3_FLASH_MODEL}",
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
                "llm": f"{XAI_PROMPT_MODEL}+{GEMINI_3_FLASH_MODEL}",
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
                "llm": f"{XAI_PROMPT_MODEL}+{GEMINI_3_FLASH_MODEL}",
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
                "llm": f"{XAI_PROMPT_MODEL}+{GEMINI_3_FLASH_MODEL}",
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
                "llm": f"{XAI_PROMPT_MODEL}+{GEMINI_3_FLASH_MODEL}",
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
        print("[CustomV4] 파이프라인 완료")
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

