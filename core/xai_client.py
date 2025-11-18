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
    
    # 시스템 프롬프트
    system_prompt = f"""IDENTITY PRESERVATION (MUST FOLLOW):
The person in Image 1 must remain EXACTLY the same individual in the final result.
Do NOT alter, modify, re-render, or replace the person's face, identity, expression, or head shape.
The final output MUST keep the exact same face from Image 1.

COMMON REQUIREMENT (MUST FOLLOW):
{COMMON_PROMPT_REQUIREMENT}

You are creating a detailed instruction prompt for a virtual try-on task.

Analyze these two images:
Image 1 (Person): A person in their current outfit.
Image 2 (Outfit): An outfit that will replace their current clothing.

First, carefully observe and describe:

1. Image 1 — The person's current outfit:
   - What type of top/shirt?
   - What type of bottom?
   - What shoes?
   - Which body parts are covered?

2. Image 2 — The outfit:
   - Color, material, and style
   - Sleeve type or sleeveless
   - Length
   - Neckline
   - Which areas it covers or leaves visible

SAFETY RULES:
- Do NOT use words such as "nude", "bare skin", "remove clothing".
- Do NOT describe the person without clothing.
- Use only safe phrasing like:
  • "Replace the original outfit with the outfit in Image 2."
  • "Ensure the original outfit does not appear in the final result."

Now create the final prompt using this structure:

OPENING STATEMENT:
"Create an image of the person from Image 1 wearing the outfit from Image 2 in a natural and photorealistic way."

CRITICAL INSTRUCTION:
"The person in Image 1 is currently wearing [list clothing items]. 
In the final result, the original outfit should not appear. 
Only the outfit from Image 2 should be visible."

STEP 1 – OUTFIT REPLACEMENT:
"Replace the person's outfit with the outfit from Image 2:
- Ensure that no elements of the original outfit are visible."

STEP 2 – APPLY THE OUTFIT FROM IMAGE 2:
"Apply the outfit exactly as shown in Image 2:
- This is a [color/style] outfit with [sleeve/sleeveless].
- The outfit is [length description].
- Preserve its silhouette, color, texture, material, and design.
- Apply the outfit onto the person while keeping their original face and head intact.
- Maintain the same level of coverage shown in Image 2 without altering the body shape unnecessarily."

STEP 3 – VISIBLE BODY AREAS:
"For visible body areas:
- Match the person's natural skin tone.
- Maintain realistic lighting and shading."

RULES – WHAT NOT TO DO:
"- NEVER include the original top.
- NEVER include the original bottom.
- NEVER mix outfits.
- NEVER change the person's face or identity."

RULES – WHAT TO DO:
"- ALWAYS keep the original face intact.
- ALWAYS match natural lighting and skin tone."

OTHER REQUIREMENTS:
"- Preserve hairstyle, pose, and proportions.
- Replace footwear with shoes that complement the outfit.
- Ensure the final image looks photorealistic."

Output ONLY the final prompt text following this structure.
Be specific, but follow all safety rules."""
    
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

