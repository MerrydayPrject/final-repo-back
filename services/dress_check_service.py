"""드레스 판별 서비스"""
import os
import json
import base64
import io
from typing import Dict, Optional
from PIL import Image
from openai import OpenAI

from config.settings import GPT4O_MODEL_NAME


class DressCheckService:
    """드레스 판별 서비스"""

    def __init__(self):
        """서비스 초기화"""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

        self.client = OpenAI(api_key=self.openai_api_key)

    def _image_to_base64(self, image: Image.Image) -> str:
        """PIL Image를 base64 문자열로 변환"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    def _build_prompt(self, mode: str) -> str:
        """프롬프트 생성"""
        rejection_rules = """
**거부 기준:**
- 속옷, 란제리, 수영복, 비키니, 야한/노출이 심한 의류 등은 모두 드레스로 취급하지 않습니다.
- 위 항목으로 판단되면 dress=false, confidence는 0.0~0.2의 낮은 값으로 설정하고, category는 거부 사유를 짧게 설명하세요.
- 거부되는 경우 rejected=true, rejection_reason에 간단한 이유를 작성하세요."""

        base_response = """{
    "dress": true 또는 false,
    "confidence": 0.0부터 1.0 사이의 숫자,
    "category": "드레스 스타일 또는 일반 옷 종류 또는 거부 사유",
    "rejected": true 또는 false,
    "rejection_reason": "거부 사유 (거부인 경우에만)"
}"""

        if mode == "fast":
            return f"""이 이미지를 분석하여 다음 정보를 JSON 형식으로 반환해주세요:
{base_response}

이미지가 드레스(웨딩드레스, 파티드레스 등)인지, 아니면 일반 옷(상의, 하의, 아우터 등)인지 판별해주세요.
드레스인 경우 dress를 true로, 일반 옷인 경우 false로 설정하세요.
confidence는 판별의 확신도를 0.0~1.0 사이의 숫자로 표현하세요.
category는 드레스인 경우 스타일(예: "벨라인", "A라인", "머메이드" 등), 일반 옷인 경우 종류(예: "상의", "하의", "아우터" 등)를 한 줄로 작성하세요.{rejection_rules}

반드시 JSON 형식만 반환하고, 다른 설명은 포함하지 마세요."""
        else:  # accurate
            return f"""이미지를 자세히 분석하여 다음 정보를 정확하게 JSON 형식으로 반환해주세요:

{base_response}

**판별 기준:**
- 드레스: 한 벌로 구성된 여성용 의류로, 상의와 하의가 하나로 연결된 형태. 웨딩드레스, 파티드레스, 원피스 등이 포함됩니다.
- 일반 옷: 상의, 하의, 아우터 등 드레스가 아닌 의류
{rejection_rules}

**분석 항목:**
1. 의류의 형태와 구조를 자세히 관찰
2. 드레스의 특징(원피스 형태, 웨딩/파티 스타일 등) 확인
3. 일반 옷의 특징(상하 분리, 재킷/코트 등) 확인
4. 판별 확신도 평가

**응답 형식:**
- dress: 드레스면 true, 일반 옷이면 false
- confidence: 판별 확신도 (0.0: 매우 불확실, 1.0: 매우 확실)
- category:
  * 드레스인 경우: "벨라인", "A라인", "머메이드", "프린세스", "슬림", "트럼펫", "미니드레스" 등
  * 일반 옷인 경우: "상의", "하의", "아우터", "세트" 등
  * 거부인 경우: "거부: 속옷/노출 의류로 판정 거부"와 같이 짧게 사유를 명시
- rejected: 거부 시 true, 그 외 false
- rejection_reason: 거부 시 간단한 이유"""

    def check_dress(
        self,
        image: Image.Image,
        model: str = "gpt-4o-mini",  # 사용할 모델 (gpt-4o-mini 또는 gpt-4o)
        mode: str = "fast"           # 모드 (fast 또는 accurate)
    ) -> Dict:
        """
        이미지가 드레스인지 판별

        Args:
            image: PIL Image 객체
            model: 사용할 모델 (gpt-4o-mini 또는 gpt-4o)
            mode: 모드 (fast 또는 accurate)

        Returns:
            {
                "dress": bool,
                "confidence": float,
                "category": str,
                "rejected": bool,
                "rejection_reason": str
            }
        """
        try:
            # 이미지를 base64로 변환
            img_base64 = self._image_to_base64(image)

            # 프롬프트 생성
            prompt = self._build_prompt(mode)

            # 모델명 설정
            model_name = GPT4O_MODEL_NAME if model == "gpt-4o" else "gpt-4o-mini"

            # OpenAI API 호출
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=200
            )

            # 응답 파싱
            response_text = response.choices[0].message.content.strip()

            # JSON 파싱
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{[^}]+\}', response_text)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    raise ValueError("JSON 형식의 응답을 받지 못했습니다.")

            # 결과 검증 및 정규화
            dress = bool(result.get("dress", False))
            confidence = float(result.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            category = str(result.get("category", "알 수 없음"))
            rejected = bool(result.get("rejected", False))
            rejection_reason = str(result.get("rejection_reason", ""))

            rejection_keywords = [
                "속옷", "란제리", "underwear", "lingerie",
                "수영복", "비키니", "노출", "거부", "rejected"
            ]

            if any(k in category.lower() for k in rejection_keywords) or any(k in rejection_reason.lower() for k in rejection_keywords):
                rejected = True

            if rejected:
                dress = False
                confidence = min(confidence, 0.2)
                if not rejection_reason:
                    rejection_reason = category if category else "부적절한 이미지로 판정 거부"

            return {
                "dress": dress,
                "confidence": confidence,
                "category": category,
                "rejected": rejected,
                "rejection_reason": rejection_reason
            }

        except Exception as e:
            print(f"드레스 판별 오류: {str(e)}")
            return {
                "dress": False,
                "confidence": 0.0,
                "category": "오류 발생",
                "rejected": False,
                "rejection_reason": "서비스 오류",
            }


# 전역 인스턴스
_service_instance: Optional[DressCheckService] = None


def get_dress_check_service() -> DressCheckService:
    """전역 DressCheckService 인스턴스 반환"""
    global _service_instance

    if _service_instance is None:
        _service_instance = DressCheckService()

    return _service_instance
