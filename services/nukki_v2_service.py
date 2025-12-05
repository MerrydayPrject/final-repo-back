"""누끼V2 서비스 - Gemini3 + OpenAI 동시 실행"""
import asyncio
import base64
import time
import traceback
from typing import Dict, Optional, Tuple
from io import BytesIO
from PIL import Image

from core.gemini_client import get_gemini_client_pool
from core.openai_image_client import (
    generate_ghost_mannequin_openai_with_image,
    load_prompt_from_file
)
from core.s3_client import upload_log_to_s3
from services.log_service import save_test_log
from config.settings import GEMINI_3_FLASH_MODEL


def image_to_base64(img: Image.Image) -> str:
    """PIL Image를 base64 문자열로 변환"""
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


def base64_to_bytes(base64_str: str) -> bytes:
    """base64 문자열을 bytes로 변환 (data URL 형식 지원)"""
    if base64_str.startswith("data:"):
        # data:image/png;base64, 형식 처리
        base64_str = base64_str.split(",", 1)[1]
    return base64.b64decode(base64_str)


async def process_with_gemini3(
    dress_img: Image.Image,
    prompt: str
) -> Dict:
    """
    Gemini3 모델로 Ghost Mannequin 생성
    
    Args:
        dress_img: 드레스 이미지
        prompt: 프롬프트
    
    Returns:
        dict: 처리 결과
    """
    start_time = time.time()
    model_name = "gemini-3"
    
    try:
        print(f"[NukkiV2] Gemini3 처리 시작")
        
        # Gemini 클라이언트 풀 가져오기
        pool = get_gemini_client_pool()
        
        # Gemini API 호출 (이미지 + 프롬프트)
        response = await pool.generate_content_with_retry_async(
            model=GEMINI_3_FLASH_MODEL,
            contents=[dress_img, prompt]
        )
        
        # 응답에서 이미지 추출
        result_image = None
        if response.candidates and len(response.candidates) > 0:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    image_data = part.inline_data.data
                    mime_type = part.inline_data.mime_type or "image/png"
                    image_b64 = base64.b64encode(image_data).decode("utf-8")
                    result_image = f"data:{mime_type};base64,{image_b64}"
                    break
        
        run_time = time.time() - start_time
        
        if result_image:
            print(f"[NukkiV2] Gemini3 처리 완료 (시간: {run_time:.2f}초)")
            return {
                "success": True,
                "result_image": result_image,
                "model": model_name,
                "run_time": run_time,
                "message": f"Gemini3 Ghost Mannequin 생성 완료 ({run_time:.2f}초)",
                "error": None
            }
        else:
            return {
                "success": False,
                "result_image": None,
                "model": model_name,
                "run_time": run_time,
                "message": "Gemini3 응답에 이미지가 없습니다.",
                "error": "no_image_in_response"
            }
            
    except Exception as e:
        run_time = time.time() - start_time
        print(f"[NukkiV2] Gemini3 처리 실패: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "result_image": None,
            "model": model_name,
            "run_time": run_time,
            "message": f"Gemini3 처리 중 오류: {str(e)}",
            "error": str(e)
        }


async def process_with_openai(
    dress_img: Image.Image
) -> Dict:
    """
    OpenAI DALL-E-3 모델로 Ghost Mannequin 생성
    
    Args:
        dress_img: 드레스 이미지
    
    Returns:
        dict: 처리 결과
    """
    start_time = time.time()
    model_name = "dall-e-3"
    
    try:
        print(f"[NukkiV2] OpenAI 처리 시작")
        
        result = await generate_ghost_mannequin_openai_with_image(dress_img)
        
        run_time = time.time() - start_time
        result["run_time"] = run_time
        result["model"] = model_name
        
        if result["success"]:
            print(f"[NukkiV2] OpenAI 처리 완료 (시간: {run_time:.2f}초)")
            result["message"] = f"OpenAI Ghost Mannequin 생성 완료 ({run_time:.2f}초)"
        else:
            print(f"[NukkiV2] OpenAI 처리 실패: {result.get('error')}")
        
        return result
        
    except Exception as e:
        run_time = time.time() - start_time
        print(f"[NukkiV2] OpenAI 처리 실패: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "result_image": None,
            "model": model_name,
            "run_time": run_time,
            "message": f"OpenAI 처리 중 오류: {str(e)}",
            "error": str(e)
        }


async def process_nukki_v2(
    dress_img: Image.Image,
    save_to_s3: bool = True,
    save_to_db: bool = True
) -> Dict:
    """
    누끼V2 메인 처리 함수 - Gemini3 + OpenAI 동시 실행
    
    Args:
        dress_img: 드레스 이미지 (PIL Image)
        save_to_s3: S3에 저장 여부
        save_to_db: MySQL에 로그 저장 여부
    
    Returns:
        dict: {
            "success": bool (최소 1개 성공 시 True),
            "gemini3": dict (Gemini3 결과),
            "openai": dict (OpenAI 결과),
            "input_url": str (입력 이미지 S3 URL),
            "message": str
        }
    """
    print(f"[NukkiV2] 두 모델 동시 처리 시작")
    
    # 프롬프트 로드
    prompt = load_prompt_from_file()
    
    # 입력 이미지 S3 업로드
    input_url = None
    if save_to_s3:
        try:
            input_bytes = BytesIO()
            dress_img.save(input_bytes, format="PNG")
            input_bytes = input_bytes.getvalue()
            
            input_url = upload_log_to_s3(
                file_content=input_bytes,
                model_id="nukki-v2",
                image_type="input"
            )
            print(f"[NukkiV2] 입력 이미지 S3 업로드 완료: {input_url}")
        except Exception as e:
            print(f"[NukkiV2] 입력 이미지 S3 업로드 실패: {e}")
    
    # 두 모델 동시 실행
    gemini_task = process_with_gemini3(dress_img, prompt)
    openai_task = process_with_openai(dress_img)
    
    results = await asyncio.gather(gemini_task, openai_task, return_exceptions=True)
    
    gemini_result = results[0] if not isinstance(results[0], Exception) else {
        "success": False,
        "result_image": None,
        "model": "gemini-3",
        "run_time": 0,
        "message": f"예외 발생: {str(results[0])}",
        "error": str(results[0])
    }
    
    openai_result = results[1] if not isinstance(results[1], Exception) else {
        "success": False,
        "result_image": None,
            "model": "dall-e-3",
        "run_time": 0,
        "message": f"예외 발생: {str(results[1])}",
        "error": str(results[1])
    }
    
    # 결과 이미지 S3 업로드 및 DB 저장
    if save_to_s3:
        # Gemini3 결과 저장
        if gemini_result.get("success") and gemini_result.get("result_image"):
            try:
                result_bytes = base64_to_bytes(gemini_result["result_image"])
                gemini_result_url = upload_log_to_s3(
                    file_content=result_bytes,
                    model_id="gemini-3",
                    image_type="result"
                )
                gemini_result["result_url"] = gemini_result_url
                print(f"[NukkiV2] Gemini3 결과 S3 업로드 완료: {gemini_result_url}")
                
                # DB 저장
                if save_to_db:
                    save_test_log(
                        person_url=input_url or "",
                        result_url=gemini_result_url or "",
                        model="nukki-v2-gemini-3",
                        prompt=prompt[:500],  # 프롬프트 길이 제한
                        success=True,
                        run_time=gemini_result.get("run_time", 0),
                        dress_url=input_url
                    )
            except Exception as e:
                print(f"[NukkiV2] Gemini3 결과 저장 실패: {e}")
        elif save_to_db and not gemini_result.get("success"):
            # 실패 로그 저장
            try:
                save_test_log(
                    person_url=input_url or "",
                    result_url="",
                    model="nukki-v2-gemini-3",
                    prompt=prompt[:500],
                    success=False,
                    run_time=gemini_result.get("run_time", 0),
                    dress_url=input_url
                )
            except Exception as e:
                print(f"[NukkiV2] Gemini3 실패 로그 저장 실패: {e}")
        
        # OpenAI 결과 저장
        if openai_result.get("success") and openai_result.get("result_image"):
            try:
                result_bytes = base64_to_bytes(openai_result["result_image"])
                openai_result_url = upload_log_to_s3(
                    file_content=result_bytes,
                    model_id="dall-e-3",
                    image_type="result"
                )
                openai_result["result_url"] = openai_result_url
                print(f"[NukkiV2] OpenAI 결과 S3 업로드 완료: {openai_result_url}")
                
                # DB 저장
                if save_to_db:
                    save_test_log(
                        person_url=input_url or "",
                        result_url=openai_result_url or "",
                        model="nukki-v2-dall-e-3",
                        prompt=prompt[:500],
                        success=True,
                        run_time=openai_result.get("run_time", 0),
                        dress_url=input_url
                    )
            except Exception as e:
                print(f"[NukkiV2] OpenAI 결과 저장 실패: {e}")
        elif save_to_db and not openai_result.get("success"):
            # 실패 로그 저장
            try:
                save_test_log(
                    person_url=input_url or "",
                    result_url="",
                    model="nukki-v2-gpt-image-1",
                    prompt=prompt[:500],
                    success=False,
                    run_time=openai_result.get("run_time", 0),
                    dress_url=input_url
                )
            except Exception as e:
                print(f"[NukkiV2] OpenAI 실패 로그 저장 실패: {e}")
    
    # 최소 1개 성공 시 전체 성공
    overall_success = gemini_result.get("success", False) or openai_result.get("success", False)
    
    if overall_success:
        success_count = sum([gemini_result.get("success", False), openai_result.get("success", False)])
        message = f"{success_count}/2 모델 성공"
    else:
        message = "모든 모델 처리 실패"
    
    print(f"[NukkiV2] 처리 완료: {message}")
    
    return {
        "success": overall_success,
        "gemini3": gemini_result,
        "openai": openai_result,
        "input_url": input_url,
        "message": message
    }

