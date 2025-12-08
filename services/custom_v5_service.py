"""CustomV5 통합 트라이온 서비스 - 누끼 + V5 프롬프트 (X.AI 제거)"""
import io
import base64
import time
import traceback
from typing import Dict
from PIL import Image

from core.s3_client import upload_log_to_s3
from services.log_service import save_test_log
from core.segformer_garment_parser import parse_garment_image_v4
from services.tryon_service import (
    load_v5_unified_prompt,
    decode_base64_to_image
)
from config.settings import GEMINI_3_FLASH_MODEL
from core.gemini_client import get_gemini_client_pool


async def generate_unified_tryon_custom_v5(
    person_img: Image.Image,
    garment_img: Image.Image,
    background_img: Image.Image,
    model_id: str = "gemini-unified-custom-v5"
) -> Dict:
    """
    CustomV5 통합 트라이온 파이프라인: 의상 누끼 + V5 프롬프트 + Gemini 3 Flash 직접 처리
    
    CustomV5는 V5 파이프라인에 누끼 기능을 추가한 버전입니다.
    X.AI 프롬프트 생성 없이 Gemini 3 Flash가 직접 처리합니다.
    
    파이프라인 단계:
    - Stage 0: 의상 이미지 누끼 처리 (배경 제거)
    - Stage 1: Gemini 3 Flash로 의상 교체 + 배경 합성 통합 처리 (person + garment_nukki + background)
    
    Args:
        person_img: 사람 이미지 (PIL Image)
        garment_img: 의상 이미지 (PIL Image)
        background_img: 배경 이미지 (PIL Image)
        model_id: 모델 ID (기본값: "gemini-unified-custom-v5")
    
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
        print("CustomV5 파이프라인 시작")
        print("="*80)
        
        print("\n[Stage 0] 의상 이미지 누끼 처리 시작 (HuggingFace API)...")
        parsing_result = await parse_garment_image_v4(garment_img)
        
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
        # Stage 1 준비: 배경 이미지 처리
        # ============================================================
        # 배경 이미지는 원본 그대로 유지
        background_img_processed = background_img
        
        # S3 업로드 (현재 비활성화)
        person_s3_url = ""
        garment_s3_url = ""
        garment_nukki_s3_url = ""
        background_s3_url = ""
        
        # ============================================================
        # Stage 1: Gemini 3 Flash로 의상 교체 + 배경 합성 통합 처리
        # ============================================================
        try:
            client_pool = get_gemini_client_pool()
        except ValueError as e:
            error_msg = f".env 파일에 GEMINI_3_API_KEY가 설정되지 않았습니다: {str(e)}"
            run_time = time.time() - start_time
            
            return {
                "success": False,
                "prompt": "",
                "result_image": "",
                "message": error_msg,
                "llm": GEMINI_3_FLASH_MODEL,
                "error": "gemini_api_key_not_found"
            }
        
        print("\n" + "="*80)
        print("[Stage 1] Gemini 3 Flash - 의상 교체 + 배경 합성 통합 처리 (CustomV5: 누끼 처리된 의상 사용)")
        print("="*80)
        
        # V5 통합 프롬프트 로드 (X.AI 분석 없이 정적 프롬프트만 사용)
        unified_prompt = load_v5_unified_prompt()
        used_prompt = unified_prompt
        
        print("[Stage 1] Gemini API 호출 시작 (다중 키 풀 사용)...")
        print(f"[Stage 1] 입력 이미지: person_img ({person_img.size[0]}x{person_img.size[1]}), garment_nukki_rgb ({garment_nukki_rgb.size[0]}x{garment_nukki_rgb.size[1]}), background_img ({background_img_processed.size[0]}x{background_img_processed.size[1]})")
        stage1_start_time = time.time()
        
        try:
            stage1_response = await client_pool.generate_content_with_retry_async(
                model=GEMINI_3_FLASH_MODEL,
                contents=[person_img, garment_nukki_rgb, background_img_processed, unified_prompt]
            )
        except Exception as exc:
            run_time = time.time() - start_time
            
            print(f"[Stage 1] Gemini API 호출 실패: {exc}")
            traceback.print_exc()
            return {
                "success": False,
                "prompt": used_prompt,
                "result_image": "",
                "message": f"Stage 1 Gemini 호출에 실패했습니다: {str(exc)}",
                "llm": GEMINI_3_FLASH_MODEL,
                "error": "stage1_gemini_call_failed"
            }
        
        stage1_latency = time.time() - stage1_start_time
        print(f"[Stage 1] Gemini API 응답 수신 완료 (지연 시간: {stage1_latency:.2f}초)")
        
        # Stage 1 응답 확인 및 이미지 추출
        if not stage1_response.candidates or len(stage1_response.candidates) == 0:
            error_msg = "Stage 1: Gemini API가 응답을 생성하지 못했습니다."
            run_time = time.time() - start_time
            
            return {
                "success": False,
                "prompt": used_prompt,
                "result_image": "",
                "message": error_msg,
                "llm": GEMINI_3_FLASH_MODEL,
                "error": "stage1_no_response"
            }
        
        stage1_candidate = stage1_response.candidates[0]
        if not hasattr(stage1_candidate, 'content') or stage1_candidate.content is None:
            error_msg = "Stage 1: Gemini API 응답에 content가 없습니다."
            run_time = time.time() - start_time
            
            return {
                "success": False,
                "prompt": used_prompt,
                "result_image": "",
                "message": error_msg,
                "llm": GEMINI_3_FLASH_MODEL,
                "error": "stage1_no_content"
            }
        
        if not hasattr(stage1_candidate.content, 'parts') or stage1_candidate.content.parts is None:
            error_msg = "Stage 1: Gemini API 응답에 parts가 없습니다."
            run_time = time.time() - start_time
            
            return {
                "success": False,
                "prompt": used_prompt,
                "result_image": "",
                "message": error_msg,
                "llm": GEMINI_3_FLASH_MODEL,
                "error": "stage1_no_parts"
            }
        
        # Stage 1 이미지 추출
        stage1_image_parts = [
            part.inline_data.data
            for part in stage1_candidate.content.parts
            if hasattr(part, 'inline_data') and part.inline_data
        ]
        
        if not stage1_image_parts:
            error_msg = "Stage 1: Gemini API가 이미지를 생성하지 못했습니다."
            run_time = time.time() - start_time
            
            return {
                "success": False,
                "prompt": used_prompt,
                "result_image": "",
                "message": error_msg,
                "llm": GEMINI_3_FLASH_MODEL,
                "error": "stage1_no_image_generated"
            }
        
        # 최종 결과 이미지 변환
        final_img = decode_base64_to_image(stage1_image_parts[0])
        print(f"[Stage 1] 의상 교체 + 배경 합성 완료 - 이미지 크기: {final_img.size[0]}x{final_img.size[1]}")
        
        # ============================================================
        # 결과 이미지 크기 고정 (960x1280, 비율 유지 리사이즈 후 중앙 크롭)
        # ============================================================
        target_width = 960
        target_height = 1280
        target_ratio = target_width / target_height  # 0.75
        
        original_width, original_height = final_img.size
        original_ratio = original_width / original_height
        
        # 비율 유지하면서 리사이즈 (긴 쪽 기준)
        if original_ratio > target_ratio:
            # 원본이 더 넓음 - 높이를 기준으로 리사이즈
            new_height = target_height
            new_width = int(original_width * (target_height / original_height))
        else:
            # 원본이 더 높음 - 너비를 기준으로 리사이즈
            new_width = target_width
            new_height = int(original_height * (target_width / original_width))
        
        # 리사이즈
        resized_img = final_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        print(f"[CustomV5] 비율 유지 리사이즈 완료 - 이미지 크기: {resized_img.size[0]}x{resized_img.size[1]}")
        
        # 중앙 크롭하여 960x1280으로 맞춤
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        
        final_img = resized_img.crop((left, top, right, bottom))
        print(f"[CustomV5] 중앙 크롭 완료 - 최종 이미지 크기: {final_img.size[0]}x{final_img.size[1]}")
        
        # ============================================================
        # 최종 결과 처리
        # ============================================================
        # 크기 고정된 이미지를 base64로 인코딩
        img_buffer = io.BytesIO()
        final_img.save(img_buffer, format="PNG")
        result_image_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # S3 업로드 (현재 비활성화)
        result_s3_url = ""
        
        run_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("[CustomV5] 파이프라인 완료")
        print("="*80)
        print(f"전체 실행 시간: {run_time:.2f}초")
        print(f"Stage 1 지연 시간: {stage1_latency:.2f}초")
        print(f"최종 이미지 크기: {final_img.size[0]}x{final_img.size[1]} (고정 크기: 960x1280)")
        print("="*80 + "\n")
        
        return {
            "success": True,
            "prompt": used_prompt,
            "result_image": f"data:image/png;base64,{result_image_base64}",
            "message": "CustomV5 파이프라인이 성공적으로 완료되었습니다.",
            "llm": GEMINI_3_FLASH_MODEL,
            "error": None
        }
        
    except Exception as e:
        run_time = time.time() - start_time
        error_detail = traceback.format_exc()
        print(f"[CustomV5] 파이프라인 오류: {e}")
        print(error_detail)
        
        return {
            "success": False,
            "prompt": used_prompt,
            "result_image": "",
            "message": f"CustomV5 파이프라인 처리 중 오류가 발생했습니다: {str(e)}",
            "llm": GEMINI_3_FLASH_MODEL,
            "error": "custom_v5_pipeline_error"
        }

