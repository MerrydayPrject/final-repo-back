"""프롬프트 테스트 서비스 - V4 파이프라인 기반"""
import io
import os
import base64
import time
import httpx
import traceback
from typing import Dict, List, Optional
from PIL import Image
from io import BytesIO

from core.s3_client import upload_log_to_s3
from core.segformer_garment_parser import parse_garment_image_v4
from config.settings import GEMINI_3_FLASH_MODEL, XAI_PROMPT_MODEL, XAI_API_KEY, XAI_API_BASE_URL
from core.gemini_client import get_gemini_client_pool
from services.log_service import save_test_log


# 프롬프트 파일 경로
PROMPTS_V4_DIR = os.path.join(os.getcwd(), "prompts", "v4")
PROMPTS_V5_DIR = os.path.join(os.getcwd(), "prompts", "v5")


def get_v4_prompt_files() -> Dict[str, List[str]]:
    """
    V4 프롬프트 파일 목록 반환
    
    Returns:
        dict: {
            "stage1": ["stage1_default.txt", "stage1_test.txt"],
            "stage2": ["prompt_stage2_outfit.txt", "stage2_test.txt"],
            "stage3": ["prompt_stage3_background_lighting.txt", "stage3_test.txt"]
        }
    """
    stage1_files = []
    stage2_files = []
    stage3_files = []
    
    try:
        for filename in os.listdir(PROMPTS_V4_DIR):
            if filename.endswith(".txt"):
                if "stage1" in filename:
                    stage1_files.append(filename)
                elif "stage2" in filename:
                    stage2_files.append(filename)
                elif "stage3" in filename:
                    stage3_files.append(filename)
    except Exception as e:
        print(f"프롬프트 파일 목록 조회 오류: {e}")
    
    return {
        "stage1": sorted(stage1_files),
        "stage2": sorted(stage2_files),
        "stage3": sorted(stage3_files)
    }


def get_v5_prompt_files() -> List[str]:
    """
    V5 프롬프트 파일 목록 반환
    
    Returns:
        list: ["prompt_unified.txt", ...]
    """
    files = []
    
    try:
        if os.path.exists(PROMPTS_V5_DIR):
            for filename in os.listdir(PROMPTS_V5_DIR):
                if filename.endswith(".txt"):
                    files.append(filename)
    except Exception as e:
        print(f"V5 프롬프트 파일 목록 조회 오류: {e}")
    
    return sorted(files)


def get_v5_prompt_content(filename: str) -> Optional[str]:
    """
    V5 프롬프트 파일 내용 반환
    
    Args:
        filename: 프롬프트 파일명
    
    Returns:
        str: 프롬프트 내용 또는 None
    """
    try:
        filepath = os.path.join(PROMPTS_V5_DIR, filename)
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"V5 프롬프트 파일 읽기 오류: {e}")
        return None


def get_prompt_content(filename: str) -> Optional[str]:
    """
    특정 프롬프트 파일 내용 반환
    
    Args:
        filename: 프롬프트 파일명
    
    Returns:
        str: 프롬프트 내용 또는 None
    """
    try:
        filepath = os.path.join(PROMPTS_V4_DIR, filename)
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"프롬프트 파일 읽기 오류: {e}")
        return None


def build_unified_prompt(xai_prompt: str, stage2_prompt: str, stage3_prompt: str) -> str:
    """
    통합 프롬프트 생성 (Stage 2 + Stage 3)
    
    Args:
        xai_prompt: X.AI가 생성한 의상 분석 프롬프트
        stage2_prompt: Stage 2 프롬프트 템플릿
        stage3_prompt: Stage 3 프롬프트 템플릿
    
    Returns:
        str: 통합 프롬프트
    """
    # Stage 2 프롬프트에 xai_prompt 삽입 (템플릿에 {xai_prompt} 플레이스홀더가 있으면)
    if "{xai_prompt}" in stage2_prompt:
        stage2_final = stage2_prompt.format(xai_prompt=xai_prompt)
    else:
        # 플레이스홀더가 없으면 끝에 추가
        stage2_final = stage2_prompt + "\n\n--- DETAILED INSTRUCTIONS FROM IMAGE ANALYSIS ---\n" + xai_prompt
    
    # Stage 2 + Stage 3 결합
    unified_prompt = (
        stage2_final
        + "\n\n"
        + "=" * 80
        + "\n"
        + stage3_prompt
    )
    
    return unified_prompt


async def generate_prompt_with_custom_system(
    person_img: Image.Image,
    dress_img: Image.Image,
    system_prompt: str,
    model: Optional[str] = None
) -> Dict:
    """
    커스텀 시스템 프롬프트로 X.AI 프롬프트 생성
    
    Args:
        person_img: 사람 이미지 (PIL Image)
        dress_img: 드레스 이미지 (PIL Image)
        system_prompt: 커스텀 시스템 프롬프트
        model: 사용할 모델 ID (기본값: "grok-2-vision-1212")
    
    Returns:
        dict: 생성 결과 (success, prompt, error, message)
    """
    if not XAI_API_KEY:
        return {
            "success": False,
            "error": "API key not found",
            "message": "XAI_API_KEY가 설정되지 않았습니다."
        }
    
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    model_to_use = model or XAI_PROMPT_MODEL
    
    # 이미지를 base64로 인코딩
    def image_to_base64(img: Image.Image) -> str:
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode("utf-8")
    
    person_b64 = image_to_base64(person_img)
    dress_b64 = image_to_base64(dress_img)
    
    person_data_url = f"data:image/png;base64,{person_b64}"
    dress_data_url = f"data:image/png;base64,{dress_b64}"
    
    user_message = "Analyze Image 1 (person) and Image 2 (outfit), then generate the prompt following the exact structure provided."
    
    payload = {
        "model": model_to_use,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {"type": "image_url", "image_url": {"url": person_data_url}},
                    {"type": "text", "text": "Image 2 (dress):"},
                    {"type": "image_url", "image_url": {"url": dress_data_url}}
                ]
            }
        ],
        "max_tokens": 2000,
        "temperature": 0.3
    }
    
    try:
        print(f"[x.ai API] 커스텀 프롬프트 생성 요청 시작 (모델: {model_to_use})")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{XAI_API_BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            )
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                prompt_text = result["choices"][0]["message"]["content"].strip()
                return {
                    "success": True,
                    "prompt": prompt_text,
                    "model": model_to_use,
                    "message": f"x.ai로 프롬프트 생성 완료 (모델: {model_to_use})"
                }
            else:
                return {"success": False, "error": "응답 형식 오류", "message": "API 응답에 프롬프트가 없습니다."}
        else:
            error_detail = response.text
            try:
                error_json = response.json()
                error_detail = error_json.get("error", {}).get("message", response.text)
            except:
                pass
            return {"success": False, "error": f"API 오류: {response.status_code}", "message": error_detail}
            
    except httpx.TimeoutException:
        return {"success": False, "error": "요청 시간 초과", "message": "프롬프트 생성 요청이 시간 초과되었습니다."}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e), "message": f"프롬프트 생성 중 오류 발생: {str(e)}"}


async def run_v4_pipeline_with_prompt(
    person_img: Image.Image,
    garment_nukki_rgb: Image.Image,
    background_img: Image.Image,
    unified_prompt: str,
    model_id: str
) -> Dict:
    """
    커스텀 프롬프트로 V4 파이프라인 실행
    """
    try:
        client_pool = get_gemini_client_pool()
    except ValueError as e:
        return {
            "success": False,
            "result_image": "",
            "message": f"Gemini API 키 설정 오류: {str(e)}",
            "error": "gemini_api_key_not_found"
        }
    
    try:
        response = await client_pool.generate_content_with_retry_async(
            model=GEMINI_3_FLASH_MODEL,
            contents=[person_img, garment_nukki_rgb, background_img, unified_prompt]
        )
    except Exception as exc:
        return {
            "success": False,
            "result_image": "",
            "message": f"Gemini API 호출 실패: {str(exc)}",
            "error": "gemini_call_failed"
        }
    
    if not response.candidates or len(response.candidates) == 0:
        return {"success": False, "result_image": "", "message": "Gemini API가 응답을 생성하지 못했습니다.", "error": "no_response"}
    
    candidate = response.candidates[0]
    if not hasattr(candidate, 'content') or candidate.content is None:
        return {"success": False, "result_image": "", "message": "Gemini API 응답에 content가 없습니다.", "error": "no_content"}
    
    if not hasattr(candidate.content, 'parts') or candidate.content.parts is None:
        return {"success": False, "result_image": "", "message": "Gemini API 응답에 parts가 없습니다.", "error": "no_parts"}
    
    image_parts = [
        part.inline_data.data
        for part in candidate.content.parts
        if hasattr(part, 'inline_data') and part.inline_data
    ]
    
    if not image_parts:
        return {"success": False, "result_image": "", "message": "Gemini API가 이미지를 생성하지 못했습니다.", "error": "no_image_generated"}
    
    result_image_base64 = base64.b64encode(image_parts[0]).decode()
    
    return {
        "success": True,
        "result_image": f"data:image/png;base64,{result_image_base64}",
        "result_bytes": image_parts[0],
        "message": "성공"
    }


async def run_single_pipeline(
    person_img: Image.Image,
    garment_img: Image.Image,
    background_img: Image.Image,
    stage1_filename: str,
    stage2_filename: str,
    stage3_filename: str
) -> Dict:
    """
    선택한 프롬프트로 단일 V4 파이프라인 실행
    
    Args:
        person_img: 인물 이미지
        garment_img: 의상 이미지
        background_img: 배경 이미지
        stage1_filename: Stage 1 프롬프트 파일명 (Grok 시스템 프롬프트)
        stage2_filename: Stage 2 프롬프트 파일명 (의상 교체)
        stage3_filename: Stage 3 프롬프트 파일명 (배경/조명)
    
    Returns:
        dict: {
            "success": bool,
            "xai_prompt": str,
            "result": {...},
            "message": str,
            "run_time": float
        }
    """
    start_time = time.time()
    
    # ============================================================
    # Stage 0: 의상 이미지 누끼 처리
    # ============================================================
    print("\n" + "=" * 80)
    print("[Prompt Test] Stage 0: 의상 이미지 누끼 처리")
    print("=" * 80)
    
    parsing_result = await parse_garment_image_v4(garment_img)
    
    if not parsing_result.get("success"):
        print("[Stage 0] 누끼 처리 실패, 원본 이미지 사용")
        garment_nukki_rgb = garment_img.convert('RGB')
    else:
        garment_nukki_rgb = parsing_result.get("garment_only")
        if garment_nukki_rgb is None:
            garment_nukki_rgb = garment_img.convert('RGB')
        else:
            print(f"[Stage 0] 누끼 처리 완료: {garment_nukki_rgb.size}")
    
    # ============================================================
    # Stage 1: X.AI 프롬프트 생성 (커스텀 시스템 프롬프트 사용)
    # ============================================================
    print("\n" + "=" * 80)
    print(f"[Prompt Test] Stage 1: X.AI 프롬프트 생성 (프롬프트 파일: {stage1_filename})")
    print("=" * 80)
    
    # Stage 1 시스템 프롬프트 로드
    stage1_system_prompt = get_prompt_content(stage1_filename) or ""
    if not stage1_system_prompt:
        return {
            "success": False,
            "xai_prompt": "",
            "result": None,
            "message": f"Stage 1 프롬프트 파일을 찾을 수 없습니다: {stage1_filename}",
            "error": "stage1_prompt_not_found"
        }
    
    xai_result = await generate_prompt_with_custom_system(
        person_img, garment_nukki_rgb, stage1_system_prompt
    )
    
    if not xai_result.get("success"):
        return {
            "success": False,
            "xai_prompt": "",
            "result": None,
            "message": f"X.AI 프롬프트 생성 실패: {xai_result.get('message', 'Unknown error')}",
            "error": "xai_prompt_failed"
        }
    
    xai_prompt = xai_result.get("prompt", "")
    print(f"[Stage 1] X.AI 프롬프트 생성 완료 (길이: {len(xai_prompt)})")
    
    # ============================================================
    # Stage 2 & 3 프롬프트 로드
    # ============================================================
    stage2_prompt = get_prompt_content(stage2_filename) or ""
    stage3_prompt = get_prompt_content(stage3_filename) or ""
    
    if not stage2_prompt:
        return {
            "success": False,
            "xai_prompt": xai_prompt,
            "result": None,
            "message": f"Stage 2 프롬프트 파일을 찾을 수 없습니다: {stage2_filename}",
            "error": "stage2_prompt_not_found"
        }
    
    if not stage3_prompt:
        return {
            "success": False,
            "xai_prompt": xai_prompt,
            "result": None,
            "message": f"Stage 3 프롬프트 파일을 찾을 수 없습니다: {stage3_filename}",
            "error": "stage3_prompt_not_found"
        }
    
    # 통합 프롬프트 생성
    unified_prompt = build_unified_prompt(xai_prompt, stage2_prompt, stage3_prompt)
    
    print(f"[Prompt Test] 사용 프롬프트: {stage1_filename} / {stage2_filename} / {stage3_filename}")
    
    # ============================================================
    # S3 업로드 - 입력 이미지
    # ============================================================
    print("\n[Prompt Test] S3 업로드 - 입력 이미지")
    
    person_buffered = io.BytesIO()
    person_img.save(person_buffered, format="PNG")
    person_s3_url = upload_log_to_s3(person_buffered.getvalue(), "prompt-test", "person") or ""
    
    garment_buffered = io.BytesIO()
    garment_nukki_rgb.save(garment_buffered, format="PNG")
    garment_s3_url = upload_log_to_s3(garment_buffered.getvalue(), "prompt-test", "garment") or ""
    
    background_buffered = io.BytesIO()
    background_img.save(background_buffered, format="PNG")
    background_s3_url = upload_log_to_s3(background_buffered.getvalue(), "prompt-test", "background") or ""
    
    # ============================================================
    # Gemini 실행
    # ============================================================
    print("\n" + "=" * 80)
    print("[Prompt Test] Gemini 파이프라인 실행")
    print("=" * 80)
    
    result = await run_v4_pipeline_with_prompt(
        person_img, garment_nukki_rgb, background_img,
        unified_prompt, "prompt-test"
    )
    
    run_time = time.time() - start_time
    
    # ============================================================
    # S3 업로드 및 로그 저장 - 결과 이미지
    # ============================================================
    if result.get("success") and result.get("result_bytes"):
        result_s3_url = upload_log_to_s3(
            result["result_bytes"], "prompt-test", "result"
        ) or ""
        
        # 모델명에 사용한 프롬프트 파일 정보 포함
        model_name = f"prompt-test({stage1_filename}/{stage2_filename}/{stage3_filename})"
        
        save_test_log(
            person_url=person_s3_url,
            dress_url=garment_s3_url,
            result_url=result_s3_url,
            model=model_name[:100],  # 모델명 길이 제한
            prompt=unified_prompt[:2000],
            success=True,
            run_time=run_time
        )
    
    # result_bytes 제거 (응답에서 불필요)
    if "result_bytes" in result:
        del result["result_bytes"]
    
    print("\n" + "=" * 80)
    print(f"[Prompt Test] 완료 - 총 실행 시간: {run_time:.2f}초")
    print("=" * 80 + "\n")
    
    return {
        "success": True,
        "xai_prompt": xai_prompt,
        "result": result,
        "message": "프롬프트 테스트 완료",
        "run_time": run_time,
        "prompts_used": {
            "stage1": stage1_filename,
            "stage2": stage2_filename,
            "stage3": stage3_filename
        }
    }


# ============================================================
# V5 파이프라인 실행 함수 (X.AI 제거)
# ============================================================

async def run_v5_pipeline(
    person_img: Image.Image,
    garment_img: Image.Image,
    background_img: Image.Image,
    prompt_filename: str
) -> Dict:
    """
    V5 파이프라인 실행 (X.AI 제거, Gemini 직접 처리)
    
    Args:
        person_img: 인물 이미지
        garment_img: 의상 이미지
        background_img: 배경 이미지
        prompt_filename: V5 통합 프롬프트 파일명
    
    Returns:
        dict: {
            "success": bool,
            "result": {...},
            "message": str,
            "run_time": float,
            "prompt_used": str
        }
    """
    start_time = time.time()
    
    # ============================================================
    # V5 프롬프트 로드
    # ============================================================
    print("\n" + "=" * 80)
    print(f"[V5 Prompt Test] 시작 - 프롬프트 파일: {prompt_filename}")
    print("=" * 80)
    
    unified_prompt = get_v5_prompt_content(prompt_filename)
    if not unified_prompt:
        return {
            "success": False,
            "result": None,
            "message": f"V5 프롬프트 파일을 찾을 수 없습니다: {prompt_filename}",
            "error": "v5_prompt_not_found"
        }
    
    print(f"[V5] 통합 프롬프트 로드 완료 (길이: {len(unified_prompt)})")
    
    # ============================================================
    # Gemini 실행 (X.AI 없이 직접)
    # ============================================================
    print("\n" + "=" * 80)
    print("[V5 Prompt Test] Gemini 파이프라인 실행 (직접 처리)")
    print("=" * 80)
    
    try:
        client_pool = get_gemini_client_pool()
    except ValueError as e:
        return {
            "success": False,
            "result": None,
            "message": f"Gemini API 키 설정 오류: {str(e)}",
            "error": "gemini_api_key_not_found"
        }
    
    try:
        response = await client_pool.generate_content_with_retry_async(
            model=GEMINI_3_FLASH_MODEL,
            contents=[person_img, garment_img, background_img, unified_prompt]
        )
    except Exception as exc:
        run_time = time.time() - start_time
        return {
            "success": False,
            "result": {"success": False, "message": str(exc)},
            "message": f"Gemini API 호출 실패: {str(exc)}",
            "run_time": run_time,
            "prompt_used": prompt_filename,
            "error": "gemini_call_failed"
        }
    
    # 응답 확인 및 이미지 추출
    if not response.candidates or len(response.candidates) == 0:
        run_time = time.time() - start_time
        return {
            "success": False,
            "result": {"success": False, "message": "Gemini API가 응답을 생성하지 못했습니다."},
            "message": "Gemini API가 응답을 생성하지 못했습니다.",
            "run_time": run_time,
            "prompt_used": prompt_filename,
            "error": "no_response"
        }
    
    candidate = response.candidates[0]
    if not hasattr(candidate, 'content') or candidate.content is None:
        run_time = time.time() - start_time
        return {
            "success": False,
            "result": {"success": False, "message": "Gemini API 응답에 content가 없습니다."},
            "message": "Gemini API 응답에 content가 없습니다.",
            "run_time": run_time,
            "prompt_used": prompt_filename,
            "error": "no_content"
        }
    
    if not hasattr(candidate.content, 'parts') or candidate.content.parts is None:
        run_time = time.time() - start_time
        return {
            "success": False,
            "result": {"success": False, "message": "Gemini API 응답에 parts가 없습니다."},
            "message": "Gemini API 응답에 parts가 없습니다.",
            "run_time": run_time,
            "prompt_used": prompt_filename,
            "error": "no_parts"
        }
    
    image_parts = [
        part.inline_data.data
        for part in candidate.content.parts
        if hasattr(part, 'inline_data') and part.inline_data
    ]
    
    if not image_parts:
        run_time = time.time() - start_time
        return {
            "success": False,
            "result": {"success": False, "message": "Gemini API가 이미지를 생성하지 못했습니다."},
            "message": "Gemini API가 이미지를 생성하지 못했습니다.",
            "run_time": run_time,
            "prompt_used": prompt_filename,
            "error": "no_image_generated"
        }
    
    result_image_base64 = base64.b64encode(image_parts[0]).decode()
    
    run_time = time.time() - start_time
    
    # ============================================================
    # S3 업로드 및 로그 저장
    # ============================================================
    print("\n[V5 Prompt Test] S3 업로드 및 로그 저장")
    
    person_buffered = io.BytesIO()
    person_img.save(person_buffered, format="PNG")
    person_s3_url = upload_log_to_s3(person_buffered.getvalue(), "v5-prompt-test", "person") or ""
    
    garment_buffered = io.BytesIO()
    garment_img.save(garment_buffered, format="PNG")
    garment_s3_url = upload_log_to_s3(garment_buffered.getvalue(), "v5-prompt-test", "garment") or ""
    
    result_s3_url = upload_log_to_s3(image_parts[0], "v5-prompt-test", "result") or ""
    
    model_name = f"v5-prompt-test({prompt_filename})"
    
    save_test_log(
        person_url=person_s3_url,
        dress_url=garment_s3_url,
        result_url=result_s3_url,
        model=model_name[:100],
        prompt=unified_prompt[:2000],
        success=True,
        run_time=run_time
    )
    
    print("\n" + "=" * 80)
    print(f"[V5 Prompt Test] 완료 - 총 실행 시간: {run_time:.2f}초")
    print("=" * 80 + "\n")
    
    return {
        "success": True,
        "result": {
            "success": True,
            "result_image": f"data:image/png;base64,{result_image_base64}",
            "message": "성공"
        },
        "message": "V5 프롬프트 테스트 완료",
        "run_time": run_time,
        "prompt_used": prompt_filename
    }


# ============================================================
# 커스텀 프롬프트 파이프라인 (사용자 직접 입력/수정)
# ============================================================

async def run_custom_prompt_pipeline(
    person_img: Image.Image,
    garment_img: Image.Image,
    background_img: Image.Image,
    stage1_prompt: str,
    stage2_prompt: str,
    stage3_prompt: str
) -> Dict:
    """
    사용자가 직접 입력/수정한 프롬프트로 V4 파이프라인 실행
    
    Args:
        person_img: 인물 이미지
        garment_img: 의상 이미지
        background_img: 배경 이미지
        stage1_prompt: Stage 1 프롬프트 내용
        stage2_prompt: Stage 2 프롬프트 내용
        stage3_prompt: Stage 3 프롬프트 내용
    
    Returns:
        dict: 실행 결과
    """
    start_time = time.time()
    
    # ============================================================
    # Stage 0: 의상 이미지 누끼 처리
    # ============================================================
    print("\n" + "=" * 80)
    print("[Custom Prompt Test] Stage 0: 의상 이미지 누끼 처리")
    print("=" * 80)
    
    parsing_result = await parse_garment_image_v4(garment_img)
    
    if not parsing_result.get("success"):
        print("[Stage 0] 누끼 처리 실패, 원본 이미지 사용")
        garment_nukki_rgb = garment_img.convert('RGB')
    else:
        garment_nukki_rgb = parsing_result.get("garment_only")
        if garment_nukki_rgb is None:
            garment_nukki_rgb = garment_img.convert('RGB')
        else:
            print(f"[Stage 0] 누끼 처리 완료: {garment_nukki_rgb.size}")
    
    # ============================================================
    # Stage 1: X.AI 프롬프트 생성 (커스텀 시스템 프롬프트 사용)
    # ============================================================
    print("\n" + "=" * 80)
    print("[Custom Prompt Test] Stage 1: X.AI 프롬프트 생성")
    print("=" * 80)
    
    if not stage1_prompt.strip():
        return {
            "success": False,
            "xai_prompt": "",
            "result": None,
            "message": "Stage 1 프롬프트가 비어있습니다.",
            "error": "stage1_prompt_empty"
        }
    
    xai_result = await generate_prompt_with_custom_system(
        person_img, garment_nukki_rgb, stage1_prompt
    )
    
    if not xai_result.get("success"):
        return {
            "success": False,
            "xai_prompt": "",
            "result": None,
            "message": f"X.AI 프롬프트 생성 실패: {xai_result.get('message', 'Unknown error')}",
            "error": "xai_prompt_failed"
        }
    
    xai_prompt = xai_result.get("prompt", "")
    print(f"[Stage 1] X.AI 프롬프트 생성 완료 (길이: {len(xai_prompt)})")
    
    # ============================================================
    # Stage 2 & 3 통합 프롬프트 생성
    # ============================================================
    if not stage2_prompt.strip():
        return {
            "success": False,
            "xai_prompt": xai_prompt,
            "result": None,
            "message": "Stage 2 프롬프트가 비어있습니다.",
            "error": "stage2_prompt_empty"
        }
    
    if not stage3_prompt.strip():
        return {
            "success": False,
            "xai_prompt": xai_prompt,
            "result": None,
            "message": "Stage 3 프롬프트가 비어있습니다.",
            "error": "stage3_prompt_empty"
        }
    
    unified_prompt = build_unified_prompt(xai_prompt, stage2_prompt, stage3_prompt)
    
    print("[Custom Prompt Test] 커스텀 프롬프트 사용")
    
    # ============================================================
    # Gemini 실행
    # ============================================================
    print("\n" + "=" * 80)
    print("[Custom Prompt Test] Gemini 파이프라인 실행")
    print("=" * 80)
    
    result = await run_v4_pipeline_with_prompt(
        person_img, garment_nukki_rgb, background_img,
        unified_prompt, "custom-prompt-test"
    )
    
    run_time = time.time() - start_time
    
    # result_bytes 제거
    if "result_bytes" in result:
        del result["result_bytes"]
    
    print("\n" + "=" * 80)
    print(f"[Custom Prompt Test] 완료 - 총 실행 시간: {run_time:.2f}초")
    print("=" * 80 + "\n")
    
    return {
        "success": True,
        "xai_prompt": xai_prompt,
        "result": result,
        "message": "커스텀 프롬프트 테스트 완료",
        "run_time": run_time,
        "prompts_used": {
            "stage1": "custom",
            "stage2": "custom",
            "stage3": "custom"
        }
    }


async def run_v5_custom_prompt_pipeline(
    person_img: Image.Image,
    garment_img: Image.Image,
    background_img: Image.Image,
    unified_prompt: str
) -> Dict:
    """
    사용자가 직접 입력/수정한 프롬프트로 V5 파이프라인 실행
    
    Args:
        person_img: 인물 이미지
        garment_img: 의상 이미지
        background_img: 배경 이미지
        unified_prompt: 통합 프롬프트 내용
    
    Returns:
        dict: 실행 결과
    """
    start_time = time.time()
    
    # ============================================================
    # V5 커스텀 프롬프트 실행
    # ============================================================
    print("\n" + "=" * 80)
    print("[V5 Custom Prompt Test] 시작")
    print("=" * 80)
    
    if not unified_prompt.strip():
        return {
            "success": False,
            "result": None,
            "message": "통합 프롬프트가 비어있습니다.",
            "error": "unified_prompt_empty"
        }
    
    print(f"[V5] 커스텀 통합 프롬프트 사용 (길이: {len(unified_prompt)})")
    
    # ============================================================
    # Gemini 실행 (X.AI 없이 직접)
    # ============================================================
    print("\n" + "=" * 80)
    print("[V5 Custom Prompt Test] Gemini 파이프라인 실행 (직접 처리)")
    print("=" * 80)
    
    try:
        client_pool = get_gemini_client_pool()
    except ValueError as e:
        return {
            "success": False,
            "result": None,
            "message": f"Gemini API 키 설정 오류: {str(e)}",
            "error": "gemini_api_key_not_found"
        }
    
    try:
        response = await client_pool.generate_content_with_retry_async(
            model=GEMINI_3_FLASH_MODEL,
            contents=[person_img, garment_img, background_img, unified_prompt]
        )
    except Exception as exc:
        run_time = time.time() - start_time
        return {
            "success": False,
            "result": {"success": False, "message": str(exc)},
            "message": f"Gemini API 호출 실패: {str(exc)}",
            "run_time": run_time,
            "prompt_used": "custom",
            "error": "gemini_call_failed"
        }
    
    # 응답 확인 및 이미지 추출
    if not response.candidates or len(response.candidates) == 0:
        run_time = time.time() - start_time
        return {
            "success": False,
            "result": {"success": False, "message": "Gemini API가 응답을 생성하지 못했습니다."},
            "message": "Gemini API가 응답을 생성하지 못했습니다.",
            "run_time": run_time,
            "prompt_used": "custom",
            "error": "no_response"
        }
    
    candidate = response.candidates[0]
    if not hasattr(candidate, 'content') or candidate.content is None:
        run_time = time.time() - start_time
        return {
            "success": False,
            "result": {"success": False, "message": "Gemini API 응답에 content가 없습니다."},
            "message": "Gemini API 응답에 content가 없습니다.",
            "run_time": run_time,
            "prompt_used": "custom",
            "error": "no_content"
        }
    
    if not hasattr(candidate.content, 'parts') or candidate.content.parts is None:
        run_time = time.time() - start_time
        return {
            "success": False,
            "result": {"success": False, "message": "Gemini API 응답에 parts가 없습니다."},
            "message": "Gemini API 응답에 parts가 없습니다.",
            "run_time": run_time,
            "prompt_used": "custom",
            "error": "no_parts"
        }
    
    image_parts = [
        part.inline_data.data
        for part in candidate.content.parts
        if hasattr(part, 'inline_data') and part.inline_data
    ]
    
    if not image_parts:
        run_time = time.time() - start_time
        return {
            "success": False,
            "result": {"success": False, "message": "Gemini API가 이미지를 생성하지 못했습니다."},
            "message": "Gemini API가 이미지를 생성하지 못했습니다.",
            "run_time": run_time,
            "prompt_used": "custom",
            "error": "no_image_generated"
        }
    
    result_image_base64 = base64.b64encode(image_parts[0]).decode()
    
    run_time = time.time() - start_time
    
    # ============================================================
    # S3 업로드 및 로그 저장
    # ============================================================
    print("\n[V5 Custom Prompt Test] S3 업로드 및 로그 저장")
    
    person_buffered = io.BytesIO()
    person_img.save(person_buffered, format="PNG")
    person_s3_url = upload_log_to_s3(person_buffered.getvalue(), "v5-prompt-test", "person") or ""
    
    garment_buffered = io.BytesIO()
    garment_img.save(garment_buffered, format="PNG")
    garment_s3_url = upload_log_to_s3(garment_buffered.getvalue(), "v5-prompt-test", "garment") or ""
    
    result_s3_url = upload_log_to_s3(image_parts[0], "v5-prompt-test", "result") or ""
    
    model_name = "v5-prompt-test(custom)"
    
    save_test_log(
        person_url=person_s3_url,
        dress_url=garment_s3_url,
        result_url=result_s3_url,
        model=model_name[:100],
        prompt=unified_prompt[:2000],
        success=True,
        run_time=run_time
    )
    
    print("\n" + "=" * 80)
    print(f"[V5 Custom Prompt Test] 완료 - 총 실행 시간: {run_time:.2f}초")
    print("=" * 80 + "\n")
    
    return {
        "success": True,
        "result": {
            "success": True,
            "result_image": f"data:image/png;base64,{result_image_base64}",
            "message": "성공"
        },
        "message": "V5 커스텀 프롬프트 테스트 완료",
        "run_time": run_time,
        "prompt_used": "custom"
    }
