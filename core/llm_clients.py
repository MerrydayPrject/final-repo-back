"""LLM 클라이언트 (Gemini, GPT-4o)"""
from typing import Dict, List, Any, Optional
from PIL import Image
from google import genai
from openai import OpenAI
import traceback

from config.settings import GPT4O_MODEL_NAME, GEMINI_PROMPT_MODEL


def _build_gpt4o_prompt_inputs(person_data_url: str, dress_data_url: str) -> List[Dict[str, Any]]:
    """GPT-4o 프롬프트 입력 생성"""
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "You are a professional visual prompt engineer specialized in AI outfit try-on. "
                        "Your job is to analyze two reference images: (1) a person photo, and (2) a clothing photo. "
                        "Then, write a detailed English prompt for a generative image model (e.g., Gemini 2.5 Flash) "
                        "that will replace only the person's outfit with the clothing from the second image. "
                        "Rules: Keep the same person's face, body shape, pose, hairstyle, and background exactly. "
                        "Do NOT change facial expression, body proportions, or lighting. "
                        "Describe the new outfit (color, texture, fabric, style) based on the clothing image. "
                        "Make it photorealistic and naturally blended, as if the person was originally photographed wearing it. "
                        "Return ONLY the final prompt text, no explanations."
                    ),
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Analyze the outfit replacement request.",
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
        
        analysis_prompt = """You are creating a detailed instruction prompt for a virtual try-on task.

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

