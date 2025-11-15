"""LLM 클라이언트 (Gemini, GPT-4o)"""
from typing import Dict, List, Any, Optional
from PIL import Image
from google import genai
from openai import OpenAI
import traceback

from config.settings import GPT4O_MODEL_NAME, GPT4O_V2_MODEL_NAME, GEMINI_PROMPT_MODEL
from config.prompts import COMMON_PROMPT_REQUIREMENT


def _build_gpt4o_prompt_inputs(person_data_url: str, dress_data_url: str) -> List[Dict[str, Any]]:
    """GPT-4o 프롬프트 입력 생성"""
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "You are creating a detailed instruction prompt for a virtual try-on task.\n\n"
                        "COMMON REQUIREMENT (MUST FOLLOW):\n"
                        + COMMON_PROMPT_REQUIREMENT +
                        "\n\n"
                        "Analyze these two images:\n"
                        "Image 1 (Person): A woman in her current outfit\n"
                        "Image 2 (Dress): A formal dress/gown that will replace her current outfit\n\n"
                        "First, carefully observe and describe:\n"
                        "1. Image 1 - List ALL clothing items she is wearing:\n"
                        "   - What type of top/shirt? (long sleeves, short sleeves, or sleeveless?)\n"
                        "   - What type of bottom? (pants, jeans, skirt, shorts?)\n"
                        "   - What shoes is she wearing?\n"
                        "   - Which body parts are currently covered by clothing?\n\n"
                        "2. Image 2 - Describe the dress in detail:\n"
                        "   - What color and style is the dress?\n"
                        "   - Does it have sleeves, or is it sleeveless?\n"
                        "   - What is the length? (short, knee-length, floor-length?)\n"
                        "   - What is the neckline style?\n"
                        "   - Which body parts will the dress cover, and which will be exposed?\n\n"
                        "Now, create a detailed prompt using this EXACT structure:\n\n"
                        "OPENING STATEMENT:\n"
                        "\"You are performing a virtual try-on task. Create an image of the woman from Image 1 wearing the dress from Image 2.\"\n\n"
                        "CRITICAL INSTRUCTION:\n"
                        "\"The woman in Image 1 is currently wearing [list specific items: e.g., a long-sleeved shirt, jeans, and sneakers]. "
                        "You MUST completely remove and erase ALL of this original clothing before applying the new dress. "
                        "The original clothing must be 100% invisible in the final result.\"\n\n"
                        "STEP 1 - REMOVE ALL ORIGINAL CLOTHING:\n"
                        "List each specific item to remove:\n"
                        "\"Delete and erase from Image 1:\n"
                        "- The [specific top description] (including all sleeves)\n"
                        "- The [specific bottom description]\n"
                        "- The [specific shoes description]\n"
                        "- Any other visible clothing items\n\n"
                        "Treat the original clothing as if it never existed. The woman should be conceptually nude before you apply the dress.\"\n\n"
                        "STEP 2 - APPLY THE DRESS FROM IMAGE 2:\n"
                        "Describe the dress application:\n"
                        "\"Take ONLY the dress garment from Image 2 and apply it to the woman's body:\n"
                        "- This is a [color] [style] dress that is [sleeveless/has sleeves/etc.]\n"
                        "- The dress is [length description]\n"
                        "- Copy the exact dress design, color, pattern, and style from Image 2\n"
                        "- Maintain the same coverage as shown in Image 2\n"
                        "- Fit the dress naturally to her body shape and pose from Image 1\"\n\n"
                        "STEP 3 - GENERATE NATURAL SKIN FOR EXPOSED BODY PARTS:\n"
                        "For each body part that will be exposed, write specific instructions:\n\n"
                        "\"For every body part that is NOT covered by the dress, you must generate natural skin:\n\n"
                        "[If applicable] If the dress is sleeveless:\n"
                        "- Generate natural BARE ARMS with realistic skin\n"
                        "- Match the exact skin tone from her face, neck, and hands in Image 1\n"
                        "- Include realistic skin texture with natural color variations, shadows, and highlights\n"
                        "- IMPORTANT: Do NOT show any fabric from the original [sleeve description]\n\n"
                        "[If applicable] If the dress is short or knee-length:\n"
                        "- Generate natural BARE LEGS with realistic skin\n"
                        "- Match the exact skin tone from her face, neck, and hands in Image 1\n"
                        "- Include realistic skin texture with natural color variations, shadows, and highlights\n"
                        "- IMPORTANT: Do NOT show any fabric from the original [pants/jeans description]\n\n"
                        "[If applicable] If the dress exposes shoulders or back:\n"
                        "- Generate natural BARE SHOULDERS/BACK with realistic skin\n"
                        "- Match the exact skin tone from her face, neck, and hands in Image 1\n"
                        "- IMPORTANT: Do NOT show any fabric from the original clothing\"\n\n"
                        "RULES - WHAT NOT TO DO:\n"
                        "\"- NEVER keep any part of the [original top] from Image 1\n"
                        "- NEVER keep any part of the [original bottom] from Image 1\n"
                        "- NEVER keep the original sleeves on arms that should show skin\n"
                        "- NEVER show original clothing fabric where skin should be visible\n"
                        "- NEVER mix elements from the original outfit with the new dress\"\n\n"
                        "RULES - WHAT TO DO:\n"
                        "\"- ALWAYS show natural skin on body parts not covered by the dress\n"
                        "- ALWAYS match skin tone to the visible skin in her face/neck/hands from Image 1\n"
                        "- ALWAYS ensure the original clothing is completely erased before applying the dress\n"
                        "- ALWAYS maintain consistent and realistic skin texture on exposed areas\"\n\n"
                        "OTHER REQUIREMENTS:\n"
                        "\"- Preserve her face, facial features, hair, and body pose exactly as in Image 1\n"
                        "- Use a pure white background\n"
                        "- Replace footwear with elegant heels that match or complement the dress color\n"
                        "- The final image should look photorealistic and natural\"\n\n"
                        "Output ONLY the final prompt text with this complete structure. "
                        "Be extremely specific about which clothing items to remove and which body parts need natural skin generation."
                    ),
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Analyze Image 1 (person) and Image 2 (dress), then generate the prompt following the exact structure provided.",
                },
                {"type": "input_image", "image_url": person_data_url},
                {"type": "input_image", "image_url": dress_data_url},
            ],
        },
    ]


def _extract_gpt4o_prompt(response: Any) -> str:
    """GPT-4o 응답에서 프롬프트 추출"""
    try:
        prompt_text = response.output_text  # type: ignore[attr-defined]
    except AttributeError:
        prompt_text = ""
    
    prompt_text = (prompt_text or "").strip()
    if not prompt_text:
        raise ValueError("GPT-4o가 유효한 프롬프트를 반환하지 않았습니다.")
    return prompt_text


async def generate_custom_prompt_from_images(person_img: Image.Image, dress_img: Image.Image, api_key: str) -> Optional[str]:
    """
    이미지를 분석하여 맞춤 프롬프트를 생성합니다.
    
    Args:
        person_img: 사람 이미지 (PIL Image)
        dress_img: 드레스 이미지 (PIL Image)
        api_key: Gemini API 키
    
    Returns:
        생성된 맞춤 프롬프트 문자열 또는 None
    """
    try:
        print("이미지 분석 시작...")
        client = genai.Client(api_key=api_key)
        
        analysis_prompt = f"""You are creating a detailed instruction prompt for a virtual try-on task.

COMMON REQUIREMENT (MUST FOLLOW):
{COMMON_PROMPT_REQUIREMENT}

Analyze these two images:
Image 1 (Person): A woman in her current outfit
Image 2 (Dress): A formal dress/gown that will replace her current outfit

First, carefully observe and describe:
1. Image 1 - List ALL clothing items she is wearing:
   - What type of top/shirt? (long sleeves, short sleeves, or sleeveless?)
   - What type of bottom? (pants, jeans, skirt, shorts?)
   - What shoes is she wearing?
   - Which body parts are currently covered by clothing?

2. Image 2 - Describe the dress in detail:
   - What color and style is the dress?
   - Does it have sleeves, or is it sleeveless?
   - What is the length? (short, knee-length, floor-length?)
   - What is the neckline style?
   - Which body parts will the dress cover, and which will be exposed?

Now, create a detailed prompt using this EXACT structure:

OPENING STATEMENT:
"You are performing a virtual try-on task. Create an image of the woman from Image 1 wearing the dress from Image 2."

CRITICAL INSTRUCTION:
"The woman in Image 1 is currently wearing [list specific items: e.g., a long-sleeved shirt, jeans, and sneakers]. You MUST completely remove and erase ALL of this original clothing before applying the new dress. The original clothing must be 100% invisible in the final result."

STEP 1 - REMOVE ALL ORIGINAL CLOTHING:
List each specific item to remove:
"Delete and erase from Image 1:
- The [specific top description] (including all sleeves)
- The [specific bottom description]
- The [specific shoes description]
- Any other visible clothing items

Treat the original clothing as if it never existed. The woman should be conceptually nude before you apply the dress."

STEP 2 - APPLY THE DRESS FROM IMAGE 2:
Describe the dress application:
"Take ONLY the dress garment from Image 2 and apply it to the woman's body:
- This is a [color] [style] dress that is [sleeveless/has sleeves/etc.]
- The dress is [length description]
- Copy the exact dress design, color, pattern, and style from Image 2
- Maintain the same coverage as shown in Image 2
- Fit the dress naturally to her body shape and pose from Image 1"

STEP 3 - GENERATE NATURAL SKIN FOR EXPOSED BODY PARTS:
For each body part that will be exposed, write specific instructions:

"For every body part that is NOT covered by the dress, you must generate natural skin:

[If applicable] If the dress is sleeveless:
- Generate natural BARE ARMS with realistic skin
- Match the exact skin tone from her face, neck, and hands in Image 1
- Include realistic skin texture with natural color variations, shadows, and highlights
- IMPORTANT: Do NOT show any fabric from the original [sleeve description]

[If applicable] If the dress is short or knee-length:
- Generate natural BARE LEGS with realistic skin
- Match the exact skin tone from her face, neck, and hands in Image 1
- Include realistic skin texture with natural color variations, shadows, and highlights
- IMPORTANT: Do NOT show any fabric from the original [pants/jeans description]

[If applicable] If the dress exposes shoulders or back:
- Generate natural BARE SHOULDERS/BACK with realistic skin
- Match the exact skin tone from her face, neck, and hands in Image 1
- IMPORTANT: Do NOT show any fabric from the original clothing"

RULES - WHAT NOT TO DO:
"- NEVER keep any part of the [original top] from Image 1
- NEVER keep any part of the [original bottom] from Image 1
- NEVER keep the original sleeves on arms that should show skin
- NEVER show original clothing fabric where skin should be visible
- NEVER mix elements from the original outfit with the new dress"

RULES - WHAT TO DO:
"- ALWAYS show natural skin on body parts not covered by the dress
- ALWAYS match skin tone to the visible skin in her face/neck/hands from Image 1
- ALWAYS ensure the original clothing is completely erased before applying the dress
- ALWAYS maintain consistent and realistic skin texture on exposed areas"

OTHER REQUIREMENTS:
"- Preserve her face, facial features, hair, and body pose exactly as in Image 1
- Use a pure white background
- Replace footwear with elegant heels that match or complement the dress color
- The final image should look photorealistic and natural"

Output ONLY the final prompt text with this complete structure. Be extremely specific about which clothing items to remove and which body parts need natural skin generation."""

        response = client.models.generate_content(
            model=GEMINI_PROMPT_MODEL,
            contents=[person_img, dress_img, analysis_prompt]
        )
        
        # 생성된 프롬프트 추출
        custom_prompt = ""
        if response.candidates and len(response.candidates) > 0:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    custom_prompt += part.text
        
        if custom_prompt:
            print(f"맞춤 프롬프트 생성 완료 (길이: {len(custom_prompt)}자)")
            print("\n" + "="*80)
            print("생성된 맞춤 프롬프트:")
            print("="*80)
            print(custom_prompt)
            print("="*80 + "\n")
            return custom_prompt
        else:
            print("프롬프트 생성 실패, 기본 프롬프트 사용")
            return None
            
    except Exception as e:
        print(f"프롬프트 생성 중 오류: {str(e)}")
        traceback.print_exc()
        return None


def _build_gpt4o_v2_short_prompt_inputs(person_data_url: str, dress_data_url: str) -> List[Dict[str, Any]]:
    """GPT-4o-V2 short prompt 입력 생성 (x.ai 최적화)"""
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "You generate concise photorealistic image prompts optimized for x.ai image models. "
                        "Analyze Image 1 (person) and Image 2 (dress). "
                        "Create a single English prompt under 1024 characters describing a photorealistic image where: "
                        "– The same person from Image 1 appears with identical face, hairstyle, body proportions, pose and background. "
                        "– The original clothing is not present. "
                        "– The person is wearing the dress shown in Image 2, matching color, fabric, shine, silhouette and style. "
                        "– Natural skin appears on any exposed areas. "
                        "– The person is wearing elegant heels that match the dress. "
                        "Output only one compact descriptive paragraph. "
                        "No lists, no steps, no commentary."
                    ),
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Create a short photorealistic prompt for x.ai using Image 1 as the person and Image 2 as the dress reference.",
                },
                {"type": "input_image", "image_url": person_data_url},
                {"type": "input_image", "image_url": dress_data_url},
            ],
        },
    ]


def call_gpt4o_v2_short_prompt(person_data_url: str, dress_data_url: str, api_key: str) -> str:
    """
    GPT-4o-V2를 사용하여 x.ai 최적화 short prompt 생성
    
    Args:
        person_data_url: 사람 이미지 data URL
        dress_data_url: 드레스 이미지 data URL
        api_key: OpenAI API 키
    
    Returns:
        생성된 short prompt 문자열 (최대 1024자)
    """
    try:
        client = OpenAI(api_key=api_key)
        
        request_input = _build_gpt4o_v2_short_prompt_inputs(person_data_url, dress_data_url)
        
        response = client.responses.create(
            model=GPT4O_V2_MODEL_NAME,
            input=request_input,
            max_output_tokens=400,  # 1024자 제한을 고려한 토큰 수
        )
        
        # 응답에서 프롬프트 추출
        prompt_text = ""
        try:
            prompt_text = response.output_text  # type: ignore[attr-defined]
        except AttributeError:
            prompt_text = ""
        
        prompt_text = (prompt_text or "").strip()
        if not prompt_text:
            raise ValueError("GPT-4o-V2가 유효한 프롬프트를 반환하지 않았습니다.")
        
        # 최종 1024자로 truncation (공백 포함)
        if len(prompt_text) > 1024:
            prompt_text = prompt_text[:1024].rsplit(' ', 1)[0]  # 단어 단위로 자르기
        
        print(f"GPT-4o-V2 short prompt 생성 완료 (길이: {len(prompt_text)}자)")
        return prompt_text
        
    except Exception as e:
        print(f"GPT-4o-V2 short prompt 생성 중 오류: {str(e)}")
        traceback.print_exc()
        raise

