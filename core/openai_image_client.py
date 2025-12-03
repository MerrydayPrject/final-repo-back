"""OpenAI 이미지 생성 클라이언트 (gpt-image-1)"""
import base64
import traceback
from typing import Dict, Optional
from io import BytesIO
from PIL import Image
from openai import OpenAI

from config.settings import OPENAI_API_KEY, OPENAI_IMAGE_MODEL


def load_prompt_from_file(prompt_path: str = "prompts/nukki/ghost_mannequin.txt") -> str:
    """
    프롬프트 파일에서 텍스트를 로드합니다.
    
    Args:
        prompt_path: 프롬프트 파일 경로
    
    Returns:
        프롬프트 텍스트
    """
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"[OpenAI Image] 프롬프트 파일 로드 오류: {e}")
        # 기본 프롬프트 반환
        return """You are an expert AI Fashion Retoucher specialized in creating 'Ghost Mannequin' (Invisible Mannequin) images for high-end e-commerce.
Reconstruct the clothing from the provided user image into a professional 'Ghost Mannequin' product shot.
Completely remove the model while preserving the 3D volume and natural draping of the clothing.
Background: Pure solid grey (#f2f2f2)."""


def image_to_base64(img: Image.Image) -> str:
    """PIL Image를 base64 문자열로 변환"""
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


async def generate_ghost_mannequin_openai_with_image(
    dress_img: Image.Image,
    model: Optional[str] = None
) -> Dict:
    """
    OpenAI gpt-image-1을 사용하여 입력 이미지 기반 Ghost Mannequin 이미지 생성
    (Vision API로 이미지 분석 후 images.generate로 생성)
    
    Args:
        dress_img: 드레스 이미지 (PIL Image)
        model: 사용할 모델 ID (기본값: gpt-image-1)
    
    Returns:
        dict: 생성 결과
    """
    if not OPENAI_API_KEY:
        error_msg = (
            "OPENAI_API_KEY가 설정되지 않았습니다!\n\n"
            "해결 방법:\n"
            "1. final-repo-back/.env 파일 생성 또는 수정\n"
            "2. 다음 줄 추가: OPENAI_API_KEY=sk-your_api_key_here\n"
            "3. https://platform.openai.com 에서 API 키 발급\n"
            "4. 서버 재시작"
        )
        print(f"[OpenAI Image] 오류: {error_msg}")
        return {
            "success": False,
            "result_image": None,
            "model": None,
            "message": error_msg,
            "error": "api_key_not_found"
        }
    
    model_to_use = model or OPENAI_IMAGE_MODEL
    
    try:
        print(f"[OpenAI Image] Ghost Mannequin 이미지 생성 시작 (모델: {model_to_use})")
        
        # 프롬프트 로드
        base_prompt = load_prompt_from_file()
        
        # 이미지를 base64로 인코딩
        dress_b64 = image_to_base64(dress_img)
        dress_data_url = f"data:image/png;base64,{dress_b64}"
        
        # OpenAI 클라이언트 생성
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Vision API로 이미지 분석하여 상세 프롬프트 생성
        print(f"[OpenAI Image] Vision API로 이미지 분석 중...")
        vision_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert fashion image analyzer. Analyze the clothing image and create a detailed description for Ghost Mannequin generation. Focus on: color, fabric texture, style, silhouette, neckline, sleeves, length, and any decorative details."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this dress/clothing image and create a detailed description. Include: exact color, fabric type, style (A-line, mermaid, etc.), neckline, sleeve type, length, and decorative elements. Be specific and detailed."
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
            max_tokens=500
        )
        
        # 분석된 설명 추출
        dress_description = vision_response.choices[0].message.content
        print(f"[OpenAI Image] 이미지 분석 완료: {dress_description[:100]}...")
        
        # 최종 프롬프트 생성
        final_prompt = f"""Create a professional Ghost Mannequin / Invisible Mannequin product shot:

{base_prompt}

Dress Description:
{dress_description}

IMPORTANT:
- Create the dress as if worn by an invisible mannequin
- Show the 3D form and volume of the garment
- Pure solid grey background (#f2f2f2)
- Professional e-commerce quality
- Front view, centered
- Preserve all fabric details, texture, and colors exactly as described"""
        
        print(f"[OpenAI Image] 이미지 생성 시작...")
        
        # 이미지 생성 요청
        response = client.images.generate(
            model=model_to_use,
            prompt=final_prompt,
            n=1,
            size="1024x1024"
        )
        
        # 응답에서 이미지 추출
        if response.data and len(response.data) > 0:
            # URL에서 이미지 다운로드
            image_url = response.data[0].url
            if image_url:
                import httpx
                async with httpx.AsyncClient() as http_client:
                    img_response = await http_client.get(image_url)
                    if img_response.status_code == 200:
                        image_b64 = base64.b64encode(img_response.content).decode("utf-8")
                        result_image = f"data:image/png;base64,{image_b64}"
                        
                        print(f"[OpenAI Image] Ghost Mannequin 이미지 생성 완료 (모델: {model_to_use})")
                        return {
                            "success": True,
                            "result_image": result_image,
                            "model": model_to_use,
                            "message": f"OpenAI {model_to_use}로 Ghost Mannequin 생성 완료",
                            "error": None
                        }
            
            # b64_json이 있는 경우
            if hasattr(response.data[0], 'b64_json') and response.data[0].b64_json:
                image_b64 = response.data[0].b64_json
                result_image = f"data:image/png;base64,{image_b64}"
                
                print(f"[OpenAI Image] Ghost Mannequin 이미지 생성 완료 (모델: {model_to_use})")
                return {
                    "success": True,
                    "result_image": result_image,
                    "model": model_to_use,
                    "message": f"OpenAI {model_to_use}로 Ghost Mannequin 생성 완료",
                    "error": None
                }
        
        return {
            "success": False,
            "result_image": None,
            "model": model_to_use,
            "message": "API 응답에 이미지 데이터가 없습니다.",
            "error": "no_image_data"
        }
            
    except Exception as e:
        print(f"[OpenAI Image] 예외 발생: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "result_image": None,
            "model": model_to_use,
            "message": f"Ghost Mannequin 생성 중 오류 발생: {str(e)}",
            "error": str(e)
        }
