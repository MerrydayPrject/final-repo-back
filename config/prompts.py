"""프롬프트 템플릿"""

# 공통 프롬프트 요구사항 (모든 모델에 공통 적용)
COMMON_PROMPT_REQUIREMENT = (
    "IDENTITY RULE (NON-NEGOTIABLE):\n"
    "- Keep the person's face exactly as it appears in Image 1.\n"
    "- Do NOT alter, modify, or re-render the face. The final output MUST use the exact same face.\n\n"

    "OUTFIT RULE:\n"
    "- Ensure the outfit from Image 2 is applied accurately (Color, Texture, Material, EXACT length).\n"
    "- **IGNORE ORIGINAL CLOTHING VOLUME:** The original outfit's thickness (e.g., thick sweater bulk) must be completely ignored. Rerender the body shape to fit the new outfit's tightness/looseness.\n\n"

    "FOOTWEAR RULE (MANDATORY):\n"
    "- YOU MUST COMPLETELY REPLACE original sneakers/boots.\n"
    "- Force generation of [High Heels, Sandals, or Elegant Flats] matching the outfit.\n"
    "- If the new outfit reveals the feet, ensure they look natural and elegant, NOT like the original bulky shoes."
)

# 기본 드레스 합성 프롬프트
GEMINI_DEFAULT_COMPOSITION_PROMPT = """You are performing a virtual try-on task. Create an image of the woman from Image 1 wearing the dress from Image 2.

CRITICAL INSTRUCTION - READ CAREFULLY:
The woman in Image 1 is currently wearing clothing (shirt, pants, sleeves, shoes, etc.). You MUST completely remove and erase ALL of this original clothing before applying the new dress. Think of this as a two-step process: first remove all existing clothes, then dress her in the new outfit. The original clothing must be 100% invisible in the final result.

STEP 1 - REMOVE ALL ORIGINAL CLOTHING:
Delete and erase from Image 1:
- The shirt/top (including all sleeves)
- The pants/jeans/bottoms
- The shoes/sneakers
- Any other visible clothing items

Treat the original clothing as if it never existed. The woman should be conceptually nude before you apply the dress.

STEP 2 - APPLY THE DRESS FROM IMAGE 2:
Take ONLY the dress garment from Image 2 and apply it to the woman's body:
- Copy the exact dress design, color, pattern, and style from Image 2
- Maintain the same coverage as shown in Image 2 (if sleeveless in Image 2, result must be sleeveless)
- Fit the dress naturally to her body shape and pose from Image 1
- DO NOT copy the background, pose, or any other elements from Image 2

STEP 3 - GENERATE NATURAL SKIN FOR EXPOSED BODY PARTS:
For every body part that is NOT covered by the dress, you must generate natural skin:

If the dress is sleeveless (no sleeves):
- Generate natural BARE ARMS with realistic skin
- Match the exact skin tone from her face, neck, and hands in Image 1
- Include realistic skin texture with natural color variations, shadows, and highlights
- IMPORTANT: Do NOT show any fabric from the original shirt sleeves

If the dress is short or knee-length:
- Generate natural BARE LEGS with realistic skin
- Match the exact skin tone from her face, neck, and hands in Image 1
- Include realistic skin texture with natural color variations, shadows, and highlights  
- IMPORTANT: Do NOT show any fabric from the original pants

If the dress exposes shoulders or back:
- Generate natural BARE SHOULDERS/BACK with realistic skin
- Match the exact skin tone from her face, neck, and hands in Image 1
- IMPORTANT: Do NOT show any fabric from the original clothing

RULES - WHAT NOT TO DO:
- NEVER keep any part of the shirt/top from Image 1
- NEVER keep any part of the pants/jeans from Image 1
- NEVER keep the original sleeves on arms that should show skin
- NEVER show original clothing fabric where skin should be visible
- NEVER mix elements from the original outfit with the new dress

RULES - WHAT TO DO:
- ALWAYS show natural skin on body parts not covered by the dress
- ALWAYS match skin tone to the visible skin in her face/neck/hands from Image 1
- ALWAYS ensure the original clothing is completely erased before applying the dress
- ALWAYS maintain consistent and realistic skin texture on exposed areas

OTHER REQUIREMENTS:
- Preserve her face, facial features, hair, and body pose exactly as in Image 1
- Use a pure white background
- Replace footwear with elegant heels that match or complement the dress color
- The final image should look photorealistic and natural"""

