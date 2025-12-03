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
        """프롬프트 생성 (파일에서 로드)"""
        # 프롬프트 파일 경로 설정
        if mode == "fast":
            prompt_filename = "dress_check_fast.txt"
        else:  # accurate
            prompt_filename = "dress_check_accurate.txt"
        
        prompt_path = os.path.join(os.getcwd(), "prompts", "dress_check", prompt_filename)
        
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"WARNING: 드레스 체크 프롬프트 파일을 찾을 수 없습니다: {prompt_path}")
            # Fallback: 기본 프롬프트 반환
            if mode == "fast":
                return """이미지를 분석하여 다음 정보를 JSON 형식으로 반환해주세요:
{
    "dress": true 또는 false,
    "confidence": 0.0부터 1.0 사이의 숫자,
    "category": "드레스 스타일 또는 일반 옷 종류"
}

이미지가 드레스(웨딩드레스, 파티드레스 등)인지, 아니면 일반 옷(상의, 하의, 아우터 등)인지 판별해주세요.
드레스인 경우 dress를 true로, 일반 옷인 경우 false로 설정하세요.
confidence는 판별의 확신도를 0.0~1.0 사이의 숫자로 표현하세요.
category는 드레스인 경우 스타일(예: "벨라인", "A라인", "머메이드" 등), 일반 옷인 경우 종류(예: "상의", "하의", "아우터" 등)를 한글로 작성하세요.

반드시 JSON 형식만 반환하고, 다른 설명은 포함하지 마세요."""
            else:  # accurate
                return """이미지를 자세히 분석하여 다음 정보를 정확하게 JSON 형식으로 반환해주세요:

{
    "dress": true 또는 false,
    "confidence": 0.0부터 1.0 사이의 숫자,
    "category": "드레스 스타일 또는 일반 옷 종류"
}

**판별 기준:**
- 드레스: 한 벌로 구성된 여성용 의류로, 상의와 하의가 하나로 연결된 형태. 웨딩드레스, 파티드레스, 원피스 등이 포함됩니다.
- 일반 옷: 상의, 하의, 아우터 등 드레스가 아닌 의류

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

반드시 JSON 형식만 반환하고, 다른 설명이나 주석은 포함하지 마세요."""
        except Exception as e:
            print(f"ERROR: 드레스 체크 프롬프트 로드 실패: {e}")
            # Fallback: 기본 프롬프트 반환
            return """이미지를 분석하여 다음 정보를 JSON 형식으로 반환해주세요:
{
    "dress": true 또는 false,
    "confidence": 0.0부터 1.0 사이의 숫자,
    "category": "드레스 스타일 또는 일반 옷 종류"
}

반드시 JSON 형식만 반환하고, 다른 설명은 포함하지 마세요."""
    
    def check_dress(
        self, 
        image: Image.Image, 
        model: str = "gpt-4o-mini",
        mode: str = "fast"
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
                "category": str
            }
        """
        try:
            # 이미지를 base64로 변환
            img_base64 = self._image_to_base64(image)
            
            # 프롬프트 생성
            prompt = self._build_prompt(mode)
            
            # 모델명 설정
            if model == "gpt-4o":
                model_name = GPT4O_MODEL_NAME
            else:
                model_name = "gpt-4o-mini"
            
            # OpenAI API 호출
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=200
            )
            
            # 응답 파싱
            response_text = response.choices[0].message.content.strip()
            
            # JSON 파싱 시도
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # JSON이 아닌 경우, JSON 부분만 추출 시도
                import re
                json_match = re.search(r'\{[^}]+\}', response_text)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    raise ValueError("JSON 형식의 응답을 받지 못했습니다.")
            
            # 결과 검증 및 정규화
            dress = bool(result.get("dress", False))
            confidence = float(result.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))  # 0.0~1.0 범위로 제한
            category = str(result.get("category", "알 수 없음"))
            
            return {
                "dress": dress,
                "confidence": confidence,
                "category": category
            }
            
        except Exception as e:
            print(f"드레스 판별 오류: {str(e)}")
            # 오류 발생 시 기본값 반환
            return {
                "dress": False,
                "confidence": 0.0,
                "category": "오류 발생"
            }


# 전역 인스턴스
_service_instance: Optional[DressCheckService] = None


def get_dress_check_service() -> DressCheckService:
    """전역 DressCheckService 인스턴스 반환"""
    global _service_instance
    
    if _service_instance is None:
        _service_instance = DressCheckService()
    
    return _service_instance

