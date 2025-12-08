"""드레스 판별 서비스."""
import os
import json
import base64
import io
import time
import re
from typing import Dict, Optional

from PIL import Image
from openai import OpenAI

try:
    from openai import RateLimitError
except ImportError:
    RateLimitError = Exception

from config.settings import GPT4O_MODEL_NAME


class DressCheckService:
    """OpenAI 비전 모델을 사용한 드레스 판별 서비스."""

    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        self.client = OpenAI(api_key=self.openai_api_key)

    def _image_to_base64(self, image: Image.Image) -> str:
        """PIL 이미지를 base64 문자열로 변환."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _build_prompt(self, mode: str) -> str:
        """프롬프트 텍스트 로드."""
        filename = "dress_check_fast.txt" if mode == "fast" else "dress_check_accurate.txt"
        path = os.path.join(os.getcwd(), "prompts", "dress_check", filename)

        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return """
다음 이미지를 분석해 JSON으로만 응답하라:

{
  "dress": true 또는 false,
  "confidence": 0.0 ~ 1.0,
  "category": "웨딩드레스 종류 또는 비드레스 종류"
}
            """.strip()

    def check_dress(self, image: Image.Image, model="gpt-4o-mini", mode="fast") -> Dict:
        """이미지가 드레스인지 판별."""
        import traceback

        try:
            img_base64 = self._image_to_base64(image)
            prompt = self._build_prompt(mode)
            model_name = GPT4O_MODEL_NAME if model == "gpt-4o" else "gpt-4o-mini"

            # --- OpenAI API ---
            max_retries = 5
            response = None

            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{img_base64}"
                                        },
                                    },
                                ],
                            }
                        ],
                        response_format={"type": "json_object"},
                        max_tokens=200,
                    )
                    break
                except Exception as e:
                    if "429" in str(e) or "rate limit" in str(e).lower():
                        time.sleep(0.4 * (2**attempt))
                        continue
                    raise

            if response is None:
                raise RuntimeError("OpenAI 응답 없음")

            # --- JSON 파싱 ---
            response_text = response.choices[0].message.content.strip()

            try:
                result = json.loads(response_text)
            except Exception:
                match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if match:
                    result = json.loads(match.group())
                else:
                    raise ValueError("JSON 파싱 실패")

            # --- 안전 검증 ---
            dress = bool(result.get("dress", False))
            confidence = float(result.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
            category = str(result.get("category", "비드레스" if not dress else "드레스"))

            # 드레스가 아닌데 category가 '웨딩드레스'로 잘못 오는 케이스 방지
            if not dress:
                category = "비드레스"

            return {
                "dress": dress,
                "confidence": confidence,
                "category": category,
            }

        except Exception:
            print("드레스 판별 예외 발생:")
            traceback.print_exc()
            return {
                "dress": False,
                "confidence": 0.0,
                "category": "오류",
            }

    def check_wedding_dress(self, image: Image.Image, model="gpt-4o-mini", mode="fast", threshold=0.7) -> Dict:
        """웨딩드레스 여부 추가 판별.

        조건:
        - 드레스 판정(dress=True)
        - 신뢰도 threshold 이상
        - 카테고리에 '웨딩/브라이덜/wedding/bridal/신부' 키워드 포함
        """
        raw = self.check_dress(image, model, mode)

        dress = raw.get("dress", False)
        confidence = float(raw.get("confidence", 0.0))
        category = raw.get("category", "") or ""
        category_lower = category.lower()

        # 카테고리 키워드 확인 (한글/영문)
        wedding_keywords = ("웨딩", "브라이덜", "신부", "wedding", "bridal")
        non_wedding_keywords = (
            "캐주얼",
            "하객",
            "데일리",
            "정장",
            "투피스",
            "세트",
            "바지",
            "팬츠",
            "점프수트",
            "슬랙스",
            "셔츠",
            "블라우스",
            "니트",
            "점퍼",
            "코트",
            "자켓",
        )
        has_wedding_keyword = any(k.lower() in category_lower for k in wedding_keywords)
        has_non_wedding_keyword = any(k.lower() in category_lower for k in non_wedding_keywords)

        is_wedding = (
            dress
            and confidence >= threshold
            and has_wedding_keyword
            and not has_non_wedding_keyword
        )

        return {
            "is_wedding_dress": is_wedding,
            "confidence": confidence,
            "category": category,
            "raw": raw,
        }


# --- 싱글톤 ---
_service_instance: Optional[DressCheckService] = None

def get_dress_check_service() -> DressCheckService:
    global _service_instance
    if _service_instance is None:
        _service_instance = DressCheckService()
    return _service_instance
