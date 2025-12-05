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
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    def _build_prompt(self, mode: str) -> str:
        """프롬프트 텍스트 로드 (파일 없으면 기본값 반환)."""
        prompt_filename = "dress_check_fast.txt" if mode == "fast" else "dress_check_accurate.txt"
        prompt_path = os.path.join(os.getcwd(), "prompts", "dress_check", prompt_filename)
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"WARNING: 드레스 체크 프롬프트 파일을 찾을 수 없습니다: {prompt_path}")
            return (
                "{\n"
                '    "dress": true 또는 false,\n'
                '    "confidence": 0.0~1.0,\n'
                '    "category": "드레스일 경우 카테고리, 비드레스일 경우 종류"\n'
                "}"
            )
        except Exception as e:
            print(f"ERROR: 드레스 체크 프롬프트 로드 실패: {e}")
            return (
                "{\n"
                '    "dress": true 또는 false,\n'
                '    "confidence": 0.0~1.0,\n'
                '    "category": "드레스일 경우 카테고리, 비드레스일 경우 종류"\n'
                "}"
            )

    def check_dress(
        self,
        image: Image.Image,
        model: str = "gpt-4o-mini",
        mode: str = "fast",
    ) -> Dict:
        """이미지가 드레스인지 판별한다. 예외 발생 시 로그를 남기고 기본값 반환."""
        import traceback

        try:
            # 이미지 base64 변환
            img_base64 = self._image_to_base64(image)

            # 프롬프트 구성
            prompt = self._build_prompt(mode)

            # 모델명 설정
            model_name = GPT4O_MODEL_NAME if model == "gpt-4o" else "gpt-4o-mini"

            # OpenAI 호출 (재시도 포함)
            max_retries = 5
            retry_delay = 0.5
            response = None

            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                            ]
                        }],
                        response_format={"type": "json_object"},
                        max_tokens=200,
                    )
                    break
                except Exception as e:
                    error_msg = str(e)
                    is_rate_limit = (
                        "429" in error_msg
                        or "rate limit" in error_msg.lower()
                        or "rate_limit" in error_msg.lower()
                        or isinstance(e, RateLimitError)
                        if RateLimitError != Exception
                        else False
                    )
                    if is_rate_limit and attempt < max_retries - 1:
                        wait_time = retry_delay * (2**attempt)
                        print(f"Rate limit 오류 발생. {wait_time:.2f}초 뒤 재시도.. (시도 {attempt+1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    raise

            if response is None:
                raise RuntimeError("OpenAI 응답이 비어 있습니다.")

            # 디버그용 원본 출력
            print("OpenAI 응답 원본:", response)

            # JSON 파싱
            response_text = response.choices[0].message.content.strip()
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                json_match = re.search(r"\{[^}]+\}", response_text)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    raise ValueError("JSON 형식의 응답을 받지 못했습니다.")

            dress = bool(result.get("dress", False))
            confidence = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
            category = str(result.get("category", "알 수 없음"))

            return {"dress": dress, "confidence": confidence, "category": category}

        except Exception:
            print("드레스 판별 예외 발생:")
            traceback.print_exc()
            return {"dress": False, "confidence": 0.0, "category": "오류 발생"}


# 싱글톤 인스턴스
_service_instance: Optional[DressCheckService] = None

def get_dress_check_service() -> DressCheckService:
    """싱글톤 DressCheckService 인스턴스 반환."""
    global _service_instance
    if _service_instance is None:
        _service_instance = DressCheckService()
    return _service_instance
