"""프롬프트 생성 라우터"""
from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import Optional
import os
import base64
import io
import traceback
from PIL import Image
from urllib.parse import urlparse
import requests
import boto3
from botocore.exceptions import ClientError

from core.llm_clients import generate_custom_prompt_from_images, _build_gpt4o_prompt_inputs, _extract_gpt4o_prompt
from config.settings import GPT4O_MODEL_NAME, GEMINI_PROMPT_MODEL
from services.image_service import preprocess_dress_image
from openai import OpenAI

router = APIRouter()


@router.post("/api/gemini/generate-prompt", tags=["프롬프트 생성"])
async def generate_prompt(
    person_image: UploadFile = File(..., description="사람 이미지 파일"),
    dress_image: Optional[UploadFile] = File(None, description="드레스 이미지 파일"),
    dress_url: Optional[str] = Form(None, description="드레스 이미지 URL (S3 또는 로컬)")
):
    """
    이미지를 분석하여 맞춤 프롬프트만 생성합니다.
    
    사용자가 프롬프트를 확인한 후 compose-dress API를 호출할 수 있습니다.
    """
    try:
        llm_info = {"llm": GEMINI_PROMPT_MODEL}
        # API 키 확인
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return JSONResponse({**llm_info, 
                "success": False,
                "error": "API key not found",
                "message": ".env 파일에 GEMINI_API_KEY가 설정되지 않았습니다."
            }, status_code=500)
        
        # 사람 이미지 읽기
        person_contents = await person_image.read()
        person_img = Image.open(io.BytesIO(person_contents))
        
        # 드레스 이미지 처리
        dress_img = None
        if dress_image:
            dress_contents = await dress_image.read()
            dress_img = Image.open(io.BytesIO(dress_contents))
        elif dress_url:
            try:
                if not dress_url.startswith('http'):
                    return JSONResponse({**llm_info, 
                        "success": False,
                        "error": "Invalid dress URL",
                        "message": f"유효하지 않은 드레스 URL입니다."
                    }, status_code=400)
                
                parsed_url = urlparse(dress_url)
                aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
                aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
                region = os.getenv("AWS_REGION", "ap-northeast-2")
                
                if not all([aws_access_key, aws_secret_key]):
                    response = requests.get(dress_url, timeout=10)
                    response.raise_for_status()
                    dress_img = Image.open(io.BytesIO(response.content))
                else:
                    s3_client = boto3.client(
                        's3',
                        aws_access_key_id=aws_access_key,
                        aws_secret_access_key=aws_secret_key,
                        region_name=region
                    )
                    
                    if '.s3.' in parsed_url.netloc or '.s3-' in parsed_url.netloc:
                        bucket_name = parsed_url.netloc.split('.')[0]
                        s3_key = parsed_url.path.lstrip('/')
                    else:
                        path_parts = parsed_url.path.lstrip('/').split('/', 1)
                        if len(path_parts) == 2:
                            bucket_name, s3_key = path_parts
                        else:
                            raise ValueError(f"S3 URL 형식을 파싱할 수 없습니다.")
                    
                    s3_response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
                    image_data = s3_response['Body'].read()
                    dress_img = Image.open(io.BytesIO(image_data))
                    
            except Exception as e:
                print(f"드레스 이미지 다운로드 오류: {e}")
                return JSONResponse({**llm_info, 
                    "success": False,
                    "error": "Image download failed",
                    "message": f"드레스 이미지를 다운로드할 수 없습니다: {str(e)}"
                }, status_code=400)
        else:
            return JSONResponse({**llm_info, 
                "success": False,
                "error": "No dress image provided",
                "message": "드레스 이미지 파일 또는 URL이 필요합니다."
            }, status_code=400)
        
        # 드레스 이미지 전처리
        print("드레스 이미지 전처리 시작...")
        dress_img = preprocess_dress_image(dress_img, target_size=1024)
        print("드레스 이미지 전처리 완료")
        
        # 맞춤 프롬프트 생성
        print("\n" + "="*80)
        print("이미지 분석 및 프롬프트 생성")
        print("="*80)
        
        custom_prompt = await generate_custom_prompt_from_images(person_img, dress_img, api_key)
        
        if custom_prompt:
            return JSONResponse({**llm_info, 
                "success": True,
                "prompt": custom_prompt,
                "message": "프롬프트가 성공적으로 생성되었습니다."
            })
        else:
            # 기본 프롬프트 반환
            default_prompt = """Create an image of the woman from Image 1 wearing the dress from Image 2.

CRITICAL INSTRUCTIONS:
- Extract ONLY the dress design, pattern, color, and style from Image 2
- COMPLETELY IGNORE the background, pose, body position, and any other visual context from Image 2
- Apply the dress onto the woman's body from Image 1
- Maintain the woman's face, facial features, and posture from Image 1 exactly as they are
- The clothing from Image 1 should NOT be reflected in the final image
- Use a pure white background (#FFFFFF)
- DO NOT replicate or reference any pose, stance, or positioning from the dress image
- Focus solely on transferring the dress garment itself onto the woman from Image 1

CRITICAL - SKIN EXPOSURE RULES:
- If Image 1 woman wears long sleeves but Image 2 dress is sleeveless → Generate natural bare arms with skin
- If Image 1 woman wears pants but Image 2 dress is short → Generate natural bare legs with skin
- If Image 1 woman covers shoulders but Image 2 dress is strapless → Generate natural bare shoulders with skin
- Any body part that will be EXPOSED by the new dress MUST show natural skin tone, NOT the original clothing
- Example: Woman in long-sleeve shirt wearing sleeveless dress = bare arms visible with natural skin
- Example: Woman in jeans wearing short dress = bare legs visible with natural skin

MANDATORY FOOTWEAR CHANGE - THIS IS CRITICAL:
- You MUST completely replace the footwear with elegant high heels or formal dress shoes
- NEVER use sneakers, casual shoes, or athletic footwear
- NEVER keep white sneakers or any casual footwear from Image 1
- For a black dress: generate black high heels or black formal pumps
- For colored dresses: generate heels that match or complement the dress color
- The shoes must be formal, elegant, and appropriate for a cocktail dress or evening gown
- The heel height should be appropriate for formal wear (3-4 inches)
- This footwear change is NON-NEGOTIABLE and must be applied"""
            
            return JSONResponse({**llm_info, 
                "success": True,
                "prompt": default_prompt,
                "message": "맞춤 프롬프트 생성 실패. 기본 프롬프트를 사용하세요.",
                "is_default": True
            })
            
    except Exception as e:
        print(f"프롬프트 생성 API 오류: {str(e)}")
        traceback.print_exc()
        return JSONResponse({**llm_info, 
            "success": False,
            "error": str(e),
            "message": f"프롬프트 생성 중 오류 발생: {str(e)}"
        }, status_code=500)


@router.post("/api/gpt4o-gemini/generate-prompt", tags=["프롬프트 생성"])
async def generate_gpt4o_prompt(
    person_image: UploadFile = File(..., description="사람 이미지 파일"),
    dress_image: UploadFile = File(..., description="드레스 이미지 파일"),
):
    """
    GPT-4o를 사용한 프롬프트 생성
    """
    try:
        llm_info = {"llm": GPT4O_MODEL_NAME}
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            return JSONResponse(
                {
                    **llm_info,
                    "success": False,
                    "error": "API key not found",
                    "message": ".env 파일에 OPENAI_API_KEY가 설정되지 않았습니다."
                },
                status_code=500,
            )

        person_bytes = await person_image.read()
        dress_bytes = await dress_image.read()

        if not person_bytes or not dress_bytes:
            return JSONResponse(
                {
                    **llm_info,
                    "success": False,
                    "error": "Invalid input",
                    "message": "사람 이미지와 드레스 이미지를 모두 업로드해주세요."
                },
                status_code=400,
            )

        person_b64 = base64.b64encode(person_bytes).decode("utf-8")
        dress_b64 = base64.b64encode(dress_bytes).decode("utf-8")
        person_mime = person_image.content_type or "image/png"
        dress_mime = dress_image.content_type or "image/png"
        person_data_url = f"data:{person_mime};base64,{person_b64}"
        dress_data_url = f"data:{dress_mime};base64,{dress_b64}"

        client = OpenAI(api_key=openai_api_key)

        request_input = _build_gpt4o_prompt_inputs(person_data_url, dress_data_url)

        try:
            response = client.responses.create(
                model=GPT4O_MODEL_NAME,
                input=request_input,
                max_output_tokens=600,
            )
        except Exception as exc:
            print(f"GPT-4o API 호출 실패: {exc}")
            traceback.print_exc()
            return JSONResponse(
                {
                    **llm_info,
                    "success": False,
                    "error": "OpenAI call failed",
                    "message": f"GPT-4o 호출에 실패했습니다: {str(exc)}"
                },
                status_code=502,
            )

        prompt_text = _extract_gpt4o_prompt(response)
        
        return JSONResponse({**llm_info, "success": True, "prompt": prompt_text})
    except Exception as exc:
        print(f"GPT-4o 프롬프트 생성 중 오류: {exc}")
        traceback.print_exc()
        return JSONResponse(
            {
                **llm_info,
                "success": False,
                "error": str(exc),
                "message": f"프롬프트 생성 중 예상치 못한 오류가 발생했습니다: {str(exc)}"
            },
            status_code=500,
        )

