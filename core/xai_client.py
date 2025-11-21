"""x.ai API 클라이언트"""
import os
import base64
import requests
import traceback
from typing import Dict, Optional
from io import BytesIO
from PIL import Image

from config.settings import XAI_API_KEY, XAI_API_BASE_URL, XAI_IMAGE_MODEL, XAI_PROMPT_MODEL
from config.prompts import COMMON_PROMPT_REQUIREMENT


def generate_image_from_text(
    prompt: str,
    model: Optional[str] = None,
    n: int = 1
) -> Dict:
    """
    x.ai API를 사용하여 텍스트 프롬프트로 이미지 생성
    
    Args:
        prompt: 이미지 생성 프롬프트
        model: 사용할 모델 ID (기본값: "grok-2-image")
        n: 생성할 이미지 수 (기본값: 1)
    
    Returns:
        dict: 생성 결과 (success, result_image, error, message)
    
    Note:
        x.ai API는 size 파라미터를 지원하지 않습니다.
        유효한 파라미터: prompt (필수), model (선택, 기본값: "grok-2-image"), n (선택)
    """
    if not XAI_API_KEY:
        error_msg = (
            "XAI_API_KEY가 설정되지 않았습니다!\n\n"
            "해결 방법:\n"
            "1. final-repo-back/.env 파일 생성 또는 수정\n"
            "2. 다음 줄 추가: XAI_API_KEY=xai-your_api_key_here\n"
            "3. https://x.ai 에서 API 키 발급\n"
            "4. 서버 재시작"
        )
        print(f"[x.ai API] 오류: {error_msg}")
        return {
            "success": False,
            "error": "API key not found",
            "message": error_msg
        }
    
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # API 요청 데이터 (x.ai는 size 파라미터를 지원하지 않음)
    # 모델이 지정되지 않으면 기본값 "grok-2-image" 사용
    model_to_use = model or XAI_IMAGE_MODEL
    
    payload = {
        "prompt": prompt,
        "model": model_to_use,
        "n": n
    }
    
    try:
        print(f"[x.ai API] 요청 시작 - 프롬프트: {prompt[:50]}...")
        print(f"[x.ai API] 엔드포인트: {XAI_API_BASE_URL}/images/generations")
        print(f"[x.ai API] API 키 설정: {'O' if XAI_API_KEY else 'X'}")
        
        response = requests.post(
            f"{XAI_API_BASE_URL}/images/generations",
            headers=headers,
            json=payload,
            timeout=60  # 이미지 생성은 시간이 걸릴 수 있음
        )
        
        print(f"[x.ai API] 응답 상태 코드: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # 응답 형식 확인 (OpenAI 스타일과 유사할 것으로 예상)
            # 일반적으로 {"data": [{"url": "...", "b64_json": "..."}]} 형식
            if "data" in result and len(result["data"]) > 0:
                image_data = result["data"][0]
                
                # base64 인코딩된 이미지가 있는 경우
                if "b64_json" in image_data:
                    image_base64 = image_data["b64_json"]
                    result_image = f"data:image/png;base64,{image_base64}"
                # URL이 있는 경우 다운로드하여 base64로 변환
                elif "url" in image_data:
                    image_url = image_data["url"]
                    img_response = requests.get(image_url, timeout=30)
                    if img_response.status_code == 200:
                        image_bytes = img_response.content
                        image_base64 = base64.b64encode(image_bytes).decode()
                        result_image = f"data:image/png;base64,{image_base64}"
                    else:
                        return {
                            "success": False,
                            "error": f"이미지 다운로드 실패: {img_response.status_code}",
                            "message": "생성된 이미지를 다운로드할 수 없습니다."
                        }
                else:
                    return {
                        "success": False,
                        "error": "응답 형식 오류",
                        "message": "API 응답에 이미지 데이터가 없습니다."
                    }
                
                print(f"[x.ai API] 성공! 이미지 생성 완료 (모델: {model_to_use})")
                return {
                    "success": True,
                    "result_image": result_image,
                    "model": model_to_use,
                    "message": f"x.ai로 이미지 생성 완료 (모델: {model_to_use})"
                }
            else:
                return {
                    "success": False,
                    "error": "응답 형식 오류",
                    "message": "API 응답 형식이 예상과 다릅니다."
                }
        else:
            error_detail = response.text
            try:
                error_json = response.json()
                error_detail = error_json.get("error", {}).get("message", response.text)
                print(f"[x.ai API] 오류 응답: {error_json}")
            except:
                print(f"[x.ai API] 원시 오류: {response.text}")
            
            return {
                "success": False,
                "error": f"API 오류: {response.status_code}",
                "message": error_detail or f"x.ai API 호출 실패 (상태 코드: {response.status_code})"
            }
            
    except requests.exceptions.Timeout:
        print(f"[x.ai API] 타임아웃 오류")
        return {
            "success": False,
            "error": "요청 시간 초과",
            "message": "이미지 생성 요청이 시간 초과되었습니다. 다시 시도해주세요."
        }
    except Exception as e:
        print(f"[x.ai API] 예외 발생: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "message": f"이미지 생성 중 오류 발생: {str(e)}"
        }


def generate_prompt_from_images(
    person_img: Image.Image,
    dress_img: Image.Image,
    model: Optional[str] = None
) -> Dict:
    """
    x.ai API를 사용하여 이미지 기반 프롬프트 생성
    
    Args:
        person_img: 사람 이미지 (PIL Image)
        dress_img: 드레스 이미지 (PIL Image)
        model: 사용할 모델 ID (기본값: "grok-2-vision-1212")
    
    Returns:
        dict: 생성 결과 (success, prompt, error, message)
    """
    if not XAI_API_KEY:
        error_msg = (
            "XAI_API_KEY가 설정되지 않았습니다!\n\n"
            "해결 방법:\n"
            "1. final-repo-back/.env 파일 생성 또는 수정\n"
            "2. 다음 줄 추가: XAI_API_KEY=xai-your_api_key_here\n"
            "3. https://x.ai 에서 API 키 발급\n"
            "4. 서버 재시작"
        )
        print(f"[x.ai API] 오류: {error_msg}")
        return {
            "success": False,
            "error": "API key not found",
            "message": error_msg
        }
    
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 모델이 지정되지 않으면 기본값 사용
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
    
    # 시스템 프롬프트 (VISUAL OVERWRITE 섹션 추가 및 잔상 제거 강화)
    system_prompt = f"""
You are an expert AI fashion director creating a precise image generation prompt.
Your goal is to visualize the person from Image 1 wearing the NEW outfit from Image 2.

COMMON REQUIREMENT:
{COMMON_PROMPT_REQUIREMENT}

**STEP 1: ANALYZE INPUTS**
1. **Identify Old Clothes (Image 1):** Note the specific texture/color (e.g., white sweater, black pants). These MUST be the primary targets for destruction.
2. **Analyze New Outfit (Image 2):** Length, Style, and Footwear needs.

**STEP 2: GENERATE THE PROMPT**

--- PROMPT STRUCTURE ---

**OPENING:**
"A photorealistic full-body shot of the person from Image 1, now wearing the outfit from Image 2."

**VISUAL OVERWRITE (CRITICAL):**
"The original [White/Thick] top is completely replaced by the [New Material] of the new outfit.
The original [Black/Dark] bottom is completely erased.
The new outfit creates a brand new silhouette, ignoring the bulk/wrinkles of the old clothing."

**OUTFIT DESCRIPTION:**
"The person is dressed in a [Color] [Material] [Outfit Name] exactly as shown in Image 2.
The outfit features [Sleeve details], [Neckline details]."

**LENGTH & BODY LOGIC (SELECT ONE BLOCK):**

> **[OPTION A: SHORT / KNEE-LENGTH / MIDI]**
> "The hemline ends at the [Thigh/Knee], explicitly revealing the legs.
> **LEGS:** The area below the hemline shows **NATURAL BARE SKIN**.
> **ERASURE:** The original black pants are GONE. Do NOT render black textures on the legs.
> **SHOES:** The original sneakers are replaced with [High Heels/Sandals] showing skin texture on the feet."

> **[OPTION B: LONG / FLOOR-LENGTH / MAXI]**
> "The skirt is floor-length and opaque.
> **COVERAGE:** The fabric falls to the floor, completely blocking the view of the legs and original shoes.
> **SILHOUETTE:** The skirt flares out naturally, overriding the shape of the original pants. No gap between legs should be visible if the dress is a full gown."

**POSE & INTEGRATION:**
"Maintain the crossed-arms pose, but adapt the new sleeves/bodice to fold naturally around the arms.
Ensure lighting and shadows match the environment."

**SAFETY & VOCABULARY:**
- Use: "Bare legs", "Skin texture", "Replaced by", "Concealed".
- Do NOT use: "Nude", "Naked", "Undress".

Output ONLY the final prompt text.
"""
    
    # 사용자 메시지
    user_message = "Analyze Image 1 (person) and Image 2 (outfit), then generate the prompt following the exact structure provided."
    
    # API 요청 데이터 (OpenAI 스타일의 chat completions 형식)
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
                    {
                        "type": "text",
                        "text": user_message
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": person_data_url
                        }
                    },
                    {
                        "type": "text",
                        "text": "Image 2 (dress):"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": dress_data_url
                        }
                    }
                ]
            }
        ],
        "max_tokens": 2000,
        "temperature": 0.3
    }
    
    try:
        print(f"[x.ai API] 프롬프트 생성 요청 시작 (모델: {model_to_use})")
        print(f"[x.ai API] 엔드포인트: {XAI_API_BASE_URL}/chat/completions")
        print(f"[x.ai API] API 키 설정: {'O' if XAI_API_KEY else 'X'}")
        
        response = requests.post(
            f"{XAI_API_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        print(f"[x.ai API] 응답 상태 코드: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # 응답에서 프롬프트 추출
            if "choices" in result and len(result["choices"]) > 0:
                prompt_text = result["choices"][0]["message"]["content"].strip()
                
                print(f"[x.ai API] 성공! 프롬프트 생성 완료 (모델: {model_to_use})")
                print(f"[x.ai API] 생성된 프롬프트 길이: {len(prompt_text)}자")
                
                return {
                    "success": True,
                    "prompt": prompt_text,
                    "model": model_to_use,
                    "message": f"x.ai로 프롬프트 생성 완료 (모델: {model_to_use})"
                }
            else:
                return {
                    "success": False,
                    "error": "응답 형식 오류",
                    "message": "API 응답에 프롬프트가 없습니다."
                }
        else:
            error_detail = response.text
            try:
                error_json = response.json()
                error_detail = error_json.get("error", {}).get("message", response.text)
                print(f"[x.ai API] 오류 응답: {error_json}")
            except:
                print(f"[x.ai API] 원시 오류: {response.text}")
            
            return {
                "success": False,
                "error": f"API 오류: {response.status_code}",
                "message": error_detail or f"x.ai API 호출 실패 (상태 코드: {response.status_code})"
            }
            
    except requests.exceptions.Timeout:
        print(f"[x.ai API] 타임아웃 오류")
        return {
            "success": False,
            "error": "요청 시간 초과",
            "message": "프롬프트 생성 요청이 시간 초과되었습니다. 다시 시도해주세요."
        }
    except Exception as e:
        print(f"[x.ai API] 예외 발생: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "message": f"프롬프트 생성 중 오류 발생: {str(e)}"
        }

