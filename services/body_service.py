"""체형 분석 서비스"""
import os
from typing import Dict, List, Optional
from PIL import Image
from google import genai

from body_analysis_test.database import (
    get_multiple_body_definitions,
    format_body_type_info_for_prompt
)


def determine_body_features(body_type: Dict, bmi: float, height: float, measurements: Dict) -> List[str]:
    """
    체형 라인, BMI, 키를 종합하여 체형 특징 판단
    
    Args:
        body_type: 체형 타입 (X라인, A라인 등)
        bmi: BMI 수치
        height: 키 (cm)
        measurements: 체형 측정값
    
    Returns:
        List[str]: 체형 특징 리스트
    """
    features = []
    
    # 키 관련 판단 (DB 키워드 유지, 사용자 표시는 프롬프트에서 부드럽게 처리)
    if height:
        if height < 160:
            features.append('키가 작은 체형')  # DB 키워드 유지
        elif height >= 170:
            features.append('키가 큰 체형')  # DB 키워드 유지
    
    # BMI 관련 판단
    if bmi:
        if bmi < 18.5:
            features.append('마른 체형')
        elif bmi >= 25:
            # DB 조회용으로는 포함 (사용자 표시에서는 제외)
            features.append('복부가 신경 쓰이는 체형')
    
    # 체형 라인 기반 판단
    body_line = body_type.get('type', '')
    
    # 어깨/엉덩이 비율로 어깨 넓은지 좁은지 판단
    shoulder_hip_ratio = measurements.get('shoulder_hip_ratio', 1.0)
    if shoulder_hip_ratio > 1.6:
        features.append('어깨가 넓은 체형')  # DB 키워드 유지
    elif shoulder_hip_ratio < 1.3:
        features.append('어깨가 좁은 체형')  # DB 키워드 유지
    
    # 허리 비율로 허리 짧은지 판단
    waist_hip_ratio = measurements.get('waist_hip_ratio', 1.0)
    if waist_hip_ratio > 1.2:
        features.append('허리가 짧은 체형')  # DB 키워드 유지
    
    # X라인은 글래머러스한 체형으로 판단
    if body_line == 'X라인':
        features.append('글래머러스한 체형')
    
    # 중복 제거
    return list(set(features))


async def analyze_body_with_gemini(
    image: Image.Image, 
    measurements: Dict, 
    body_type: Dict,
    bmi: Optional[float] = None,
    height: Optional[float] = None,
    body_features: List[str] = None
):
    """
    Gemini API로 체형 상세 분석
    DB에서 체형별 정의를 조회하여 프롬프트에 포함
    """
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("GEMINI_API_KEY가 설정되지 않았습니다.")
            return None
        
        client = genai.Client(api_key=api_key)
        
        # DB에서 체형별 정의 조회 (체형 특징 기반)
        db_definitions = []
        if body_features:
            print(f"[DEBUG] 체형 특징: {body_features}")
            db_definitions = get_multiple_body_definitions(body_features)
            print(f"[DEBUG] DB에서 조회한 체형 정의 개수: {len(db_definitions)}")
            if db_definitions:
                for def_item in db_definitions:
                    print(f"[DEBUG] - {def_item.get('body_feature')}: 추천={def_item.get('recommended_dresses')}, 피해야할={def_item.get('avoid_dresses')}")
        
        db_info_text = format_body_type_info_for_prompt(db_definitions)
        print(f"[DEBUG] DB 정의 정보가 프롬프트에 포함됨: {len(db_info_text) > 0}")
        
        # BMI 및 키 정보 텍스트 생성 (사용자에게 표시되는 부분은 부드럽게)
        bmi_info = ""
        if bmi is not None and height is not None:
            # 체형 특징을 부드러운 표현으로 변환 (부정적 표현 제거)
            soft_features = []
            for feature in body_features:
                if feature == '키가 작은 체형':
                    soft_features.append('컴팩트한 체형')
                elif feature == '허리가 짧은 체형':
                    soft_features.append('상체 비율이 짧은 체형')
                elif feature == '복부가 신경 쓰이는 체형':
                    # 부정적 표현이므로 제외
                    continue
                else:
                    soft_features.append(feature)
            
            bmi_info = f"\n**사용자 정보**:\n- 키: {height}cm\n- BMI: {bmi:.1f}\n"
            if soft_features:
                bmi_info += f"- 판단된 체형 특징: {', '.join(soft_features)}\n"
        
        detected_body_type = body_type.get('type', 'unknown')
        
        # 긴 프롬프트는 생략하고 핵심만 포함 (전체는 main.py에서 확인 가능)
        prompt = f"""
**⚠️ 매우 중요: 객관적 데이터(BMI, 키, 몸무게, 체형 특징)를 최우선으로 반영하고, DB 정의 정보를 적극 활용하여 정확한 분석을 제공하세요.**

이미지를 자세히 관찰하고 체형을 분석해주세요. 아래 정보들을 종합하여 최적의 분석 결과를 도출해주세요.

**분석 우선순위 (반드시 지켜주세요)**:
1. **최우선: BMI 및 체형 특징 판별 결과** - BMI, 키, 몸무게는 객관적 데이터이므로 반드시 반영
2. **최우선: DB 체형별 정의 정보** - DB에 저장된 체형별 장점, 단점, 추천/피해야 할 스타일을 적극 활용
3. 참고: 이미지 직접 관찰 - 실제로 보이는 체형 특징을 확인
4. 참고: 랜드마크 기반 체형 라인 판별 결과 (수치는 부정확할 수 있음)

**⚠️ 매우 중요**: 
- BMI가 25 이상이면 과체중이므로 **절대 슬림 드레스를 추천하지 마세요**. 벨라인, A라인 등이 적합합니다.
- BMI가 18.5 미만이면 저체중이므로 **절대 슬림 드레스를 추천하지 마세요**. 프린세스, 머메이드 등이 적합합니다.
- 슬림 드레스는 **BMI 18.5~25 사이의 균형잡힌 체형**에만 추천하세요.
- DB 정의에 명시된 "피해야 할 드레스"는 **반드시 피해야 할 스타일**입니다.

---

**1. 랜드마크 기반 체형 라인 판별 결과** (참고용, 부정확할 수 있음):
- 체형 라인: {detected_body_type}에 가깝습니다
- 어깨/엉덩이 비율: {measurements.get('shoulder_hip_ratio', 1.0):.2f}
- 허리/어깨 비율: {measurements.get('waist_shoulder_ratio', 1.0):.2f}
- 허리/엉덩이 비율: {measurements.get('waist_hip_ratio', 1.0):.2f}

**⚠️ 주의**: 위 수치는 랜드마크 기반 추정치로 **매우 부정확할 수 있습니다**. 실제 체형 판단은 **반드시 이미지를 직접 관찰**하여 하세요.

---

**2. BMI 및 키 기반 체형 특징 판별 결과** (⚠️ 최우선 반영):
{bmi_info if bmi_info else "- 키/몸무게 정보가 제공되지 않았습니다."}

**⚠️ BMI 기반 추천 규칙 (반드시 지켜주세요)**:
- BMI < 18.5: 프린세스, 머메이드 추천 / 슬림 절대 금지
- BMI 18.5~25: 슬림, 벨라인, A라인 추천 가능
- BMI ≥ 25: 벨라인, A라인 추천 / 슬림 절대 금지
- BMI ≥ 30: 벨라인, A라인 추천 / 슬림, 머메이드 절대 금지

---

**3. DB 체형별 정의 정보** (⚠️ 최우선 반영, 적극 활용):
{db_info_text if db_info_text else "- 체형 특징이 판별되지 않아 DB 정의 정보가 없습니다."}

**⚠️ 매우 중요**: 
- 위 DB 정의 정보는 **체형별 전문 지식**이므로 **반드시 적극 활용**하세요.
- DB 정의에 명시된 "추천 드레스"는 해당 체형 특징에 **가장 적합한 스타일**입니다.
- DB 정의에 명시된 "피해야 할 드레스"는 해당 체형 특징에 **절대 부적합한 스타일**이므로 **반드시 피해야 합니다**.
- DB 정의의 "장점"과 "스타일 팁"을 참고하여 분석하세요.

---

**4. 이미지 직접 관찰 (참고용)**:
**이미지를 관찰하여 위의 BMI 판별 결과와 DB 정의 정보가 실제 이미지와 일치하는지 확인하세요.**

**성별 판별 지침**
- 이미지를 보고 성별을 추정하세요.
- **남성으로 보이면** 체형 분석만 제공하고 드레스 추천은 생략하세요. 문장 앞에 굳이 성별을 언급할 필요는 없습니다.
- **여성으로 보이면** "여성입니다", "여성으로 보입니다" 같은 문장을 쓰지 말고 바로 체형 특징을 설명하며 드레스 추천을 포함하세요.

**최종 분석 지침** (위의 모든 정보를 종합하여):

1. **⚠️ 최우선: BMI 및 체형 특징 판별 결과를 반영**하여 이 사람의 체형 특징을 정확하게 파악하세요:
   - BMI 수치를 기반으로 적합한 드레스를 판단하세요
   - 판별된 체형 특징을 반드시 반영하세요
   - 이미지 관찰로 위 판별 결과가 실제와 일치하는지 확인하세요
   - **BMI가 25 이상이면 슬림 드레스를 절대 추천하지 마세요**
   - **DB 정의에 명시된 "피해야 할 드레스"는 절대 추천하지 마세요**
   - **⚠️ "과체중", "저체중", "비만" 같은 표현은 절대 사용하지 마세요. BMI 수치만 참고하세요.**

2. **여성인 경우에만** BMI 판별 결과와 DB 정의 정보를 기반으로 드레스 스타일을 추천하세요:
   - **⚠️ 최우선: BMI 수치를 반영**하여 적합한 드레스를 추천하세요
   - **⚠️ 최우선: DB 정의에 명시된 "추천 드레스"를 우선 추천**하세요
   - **⚠️ BMI ≥ 25이면 슬림 드레스를 절대 추천하지 마세요**
   - **⚠️ BMI < 18.5이면 슬림 드레스를 절대 추천하지 마세요**
   - **⚠️ DB 정의에 명시된 "피해야 할 드레스"는 절대 추천하지 마세요**
   
   **BMI 기반 추천 우선순위**:
   - **BMI ≥ 25**: 벨라인 > A라인 > 트럼펫 (슬림 절대 금지)
   - **BMI 18.5~25**: 벨라인, A라인, 슬림, 프린세스 (상황에 따라)
   - **BMI < 18.5**: 프린세스 > 머메이드 (슬림 절대 금지)

3. **⚠️ BMI 판별 결과와 DB 정의 정보를 최우선으로 반영하고, 이미지 관찰로 확인하여 최종 판단하세요.**

다음을 자연스러운 문장으로 설명해주세요:

1. **이미지를 직접 관찰한 실제 체형 특징**을 구체적이고 상세하게 설명하세요 (최소 3-4문장):
   - 통통함, 마름, 근육질, 볼륨 분포 등 실제로 보이는 특징을 자세히 설명
   - **⚠️ 매우 중요: 존댓말을 사용하고, 부드럽고 건설적인 표현을 사용하세요.**
   - **⚠️ 핵심 원칙**:
     - **단점을 직접적으로 말하지 말고, "이렇게 보완할 수 있다"는 식으로 부드럽게 표현하세요.**
     - **장점은 "이렇게 살리는게 좋은 방법이다"는 식으로 긍정적으로 표현하세요.**

2. **여성인 경우에만** 실제 이미지에서 관찰한 체형 특징을 바탕으로 드레스 스타일을 2개 상세하고 친절하게 설명하세요 (각 스타일당 최소 2-3문장):
   
   **⚠️ 매우 중요**: 
   - 남성인 경우 이 항목은 완전히 생략하세요.
   - **최우선: BMI 수치와 체형 특징 판별 결과를 반영**하여 드레스를 추천하세요.
   - **최우선: DB 정의에 명시된 "추천 드레스"를 우선 추천**하세요.
   - **BMI ≥ 25이면 슬림 드레스를 절대 추천하지 마세요.**
   - **DB 정의에 명시된 "피해야 할 드레스"는 절대 추천하지 마세요.**
   - **⚠️ 존댓말을 사용하고, 각 스타일이 왜 어울리는지 구체적이고 상세하게 설명하세요.**
   - **⚠️ 핵심 원칙: 단점을 보완하는 방식으로, 장점을 살리는 방식으로 표현하세요.**
   
   - **⚠️ 매우 중요: 추천할 드레스 스타일은 반드시 다음 7가지 카테고리 중에서만 선택하세요 (다른 스타일은 절대 추천하지 마세요):**
     - 벨라인 (벨트라인, 하이웨이스트 포함) - 허리 라인 강조, 복부 가려줌
     - 머메이드 (물고기 실루엣) - 커브 강조
     - 프린세스 (프린세스라인) - 볼륨 추가
     - A라인 (에이라인) - 하체 볼륨 커버
     - 슬림 (스트레이트, H라인 포함) - 깔끔한 라인 (BMI ≥ 25인 경우 절대 추천 금지)
     - 트럼펫 (플레어 실루엣) - 플레어로 균형
     - 미니드레스 - 활동적이고 젊은 느낌

3. **여성인 경우에만** 피해야 할 드레스 스타일을 부드럽고 친절하게 설명하세요 (최소 2-3문장). 최대 2개까지 언급하고, 각 스타일을 피해야 하는 이유를 구체적으로 설명하세요.
   **⚠️ 매우 중요**: 
   - 남성인 경우 이 항목은 완전히 생략하세요.
   - **최우선: BMI 수치를 반영**하여 피해야 할 드레스를 판단하세요.
   - **BMI ≥ 25이면 반드시 슬림 드레스를 피해야 할 스타일로 언급**하세요.
   - **최우선: DB 정의에 명시된 "피해야 할 드레스"를 반드시 언급**하세요.
   - 피해야 할 스타일도 위의 카테고리 중에서만 언급하세요.
   - **⚠️ 존댓말을 사용하고, 부드럽게 설명하세요.**
   - **⚠️ 핵심 원칙: "피해야 한다"고 직접적으로 말하지 말고, "이렇게 보완하는 것이 더 좋은 방법이다"는 식으로 건설적으로 표현하세요.**

반드시 지켜야 할 사항:
- **남성 사진인 경우 드레스 추천 문장(추천 1, 추천 2, 피해야 할 등)은 절대 작성하지 마세요. 체형 분석만 제공하세요.**
- **⚠️ 최우선: BMI 수치와 체형 특징 판별 결과를 반영하여 정확한 분석을 제공하세요.**
- **⚠️ 최우선: DB 정의에 명시된 추천/피해야 할 드레스를 적극 활용하세요.**
- **BMI ≥ 25이면 슬림 드레스를 절대 추천하지 마세요.**
- **BMI < 18.5이면 슬림 드레스를 절대 추천하지 마세요.**
- **DB 정의에 명시된 "피해야 할 드레스"는 절대 추천하지 마세요.**
- 랜드마크 수치는 참고용일 뿐이며 매우 부정확할 수 있습니다.
- 이미지 관찰은 BMI 판별 결과와 DB 정의 정보가 실제와 일치하는지 확인하는 용도입니다.
- 스타일링 팁, 액세서리 추천, 색상 추천, 코디 팁 등은 절대 포함하지 마세요.
- 여성인 경우에만 추천 드레스 스타일명과 피해야 할 드레스 스타일명은 반드시 위의 카테고리 중에서만 선택하세요.
- 별도의 리스트나 항목으로 나열하지 말고, 자연스러운 문단 형식으로 설명해주세요.
"""
        
        # Gemini API 호출
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[image, prompt]
        )
        
        # 응답 파싱
        analysis_text = response.text
        
        # 상세 분석만 반환
        return {
            "detailed_analysis": analysis_text
        }
        
    except Exception as e:
        print(f"Gemini 분석 오류: {e}")
        return None

