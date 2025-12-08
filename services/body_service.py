"""체형 분석 서비스"""
import os
from typing import Dict, List, Optional
from PIL import Image
from google import genai
from pathlib import Path

from services.body_analysis_database import (
    get_multiple_body_definitions,
    format_body_type_info_for_prompt
)


def load_body_line_definitions() -> Dict[str, str]:
    """
    라인별 체형 정의를 DB에서 읽어와서 딕셔너리로 반환
    DB에서 읽기 실패 시 파일에서 읽기 (대체)
    
    Returns:
        Dict[str, str]: {라인명: 정의 텍스트}
    """
    from services.database import get_db_connection
    
    # 우선 DB에서 읽기
    connection = get_db_connection()
    if connection:
        try:
            with connection.cursor() as cursor:
                # body_type_definitions 테이블에서 라인별 정의 조회
                cursor.execute("""
                    SELECT body_feature, definition_text 
                    FROM body_type_definitions
                    WHERE body_feature IN ('A라인', 'H라인', 'X라인', 'O라인')
                    AND definition_text IS NOT NULL
                    ORDER BY 
                        CASE body_feature
                            WHEN 'A라인' THEN 1
                            WHEN 'H라인' THEN 2
                            WHEN 'X라인' THEN 3
                            WHEN 'O라인' THEN 4
                        END
                """)
                results = cursor.fetchall()
                
                if results:
                    definitions = {}
                    for row in results:
                        definitions[row['body_feature']] = row['definition_text']
                    
                    print(f"[INFO] 라인별 체형 정의 로드 완료 (DB): {list(definitions.keys())}")
                    for line_name, definition in definitions.items():
                        print(f"[INFO] - {line_name}: {len(definition)}자")
                    
                    return definitions
                    
        except Exception as e:
            print(f"[WARN] 라인별 체형 정의 DB 조회 실패: {e}")
            import traceback
            print(traceback.format_exc())
        finally:
            connection.close()
    
    # DB 실패 시 파일에서 읽기 (대체)
    print("[INFO] DB에서 읽기 실패, 파일에서 읽기 시도...")
    base_dir = Path(__file__).parent.parent
    definition_file = base_dir / "body_line_definitions.md"
    
    if not definition_file.exists():
        print(f"[WARN] 라인별 체형 정의 파일을 찾을 수 없습니다: {definition_file}")
        return {}
    
    try:
        with open(definition_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 마크다운 파싱하여 라인별로 분리
        definitions = {}
        current_line = None
        current_text = []
        
        for line in content.split('\n'):
            # 라인 헤더 감지 (## X 라인, ## H 라인 등)
            if line.startswith('## '):
                # 이전 라인 저장
                if current_line and current_text:
                    definitions[current_line] = '\n'.join(current_text).strip()
                
                # 새 라인 시작
                line_name = line.replace('## ', '').strip()
                # "X 라인 (X-Line) – 모래시계형" -> "X라인"
                if '라인' in line_name:
                    parts = line_name.split('라인')
                    if parts:
                        line_char = parts[0].strip()
                        current_line = line_char + '라인'
                    else:
                        current_line = line_name
                else:
                    current_line = line_name
                current_text = []
            elif line.startswith('---'):
                # 구분선 무시
                continue
            elif current_line:
                # 빈 줄이 아닌 경우 추가
                current_text.append(line)
        
        # 마지막 라인 저장
        if current_line and current_text:
            definitions[current_line] = '\n'.join(current_text).strip()
        
        print(f"[INFO] 라인별 체형 정의 로드 완료 (파일): {list(definitions.keys())}")
        for line_name, definition in definitions.items():
            print(f"[INFO] - {line_name}: {len(definition)}자")
        
        return definitions
        
    except Exception as e:
        print(f"[WARN] 라인별 체형 정의 파일 읽기 실패: {e}")
        import traceback
        print(traceback.format_exc())
        return {}


def format_body_line_definitions_for_prompt(definitions: Dict[str, str]) -> str:
    """
    라인별 정의를 Gemini 프롬프트에 포함할 형식으로 변환
    
    Args:
        definitions: load_body_line_definitions()의 반환값
    
    Returns:
        str: 프롬프트에 포함할 텍스트
    """
    if not definitions:
        return ""
    
    parts = []
    parts.append("**라인별 체형 정의** (참고용으로만 사용, 대체 정의가 있으면 대체 정의를 우선):")
    parts.append("")
    
    for line_name, definition in definitions.items():
        parts.append(f"### {line_name}")
        parts.append(definition)
        parts.append("")
    
    return "\n".join(parts)


async def classify_body_line_with_gemini(
    image: Image.Image,
    measurements: Dict,
    body_line_definitions: Dict[str, str]
) -> Dict:
    """
    Gemini API로 체형 라인 분류 (라인별 정의 포함)
    
    Args:
        image: 사용자 이미지
        measurements: MediaPipe 측정값
        body_line_definitions: 라인별 체형 정의 (파일에서 읽은 것 또는 DB에서)
    
    Returns:
        Dict: {"body_line": "A라인", "confidence": "high/medium/low"}
    """
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("[Gemini] GEMINI_API_KEY가 설정되지 않았습니다.")
            return {"body_line": "H라인", "confidence": "low"}
        
        client = genai.Client(api_key=api_key)
        
        # 라인별 정의를 프롬프트 형식으로 변환
        line_definitions_text = format_body_line_definitions_for_prompt(body_line_definitions)
        
        # 측정값 기반 가능성 분석 (참고용)
        shoulder_hip_ratio = measurements.get('shoulder_hip_ratio', 1.0)
        waist_hip_ratio = measurements.get('waist_hip_ratio', 1.0)
        waist_shoulder_ratio = measurements.get('waist_shoulder_ratio', 1.0)
        
        prompt = f"""
당신은 체형 라인 분류 전문가입니다.

**MediaPipe 기반 측정값**:
- 어깨/엉덩이 비율: {shoulder_hip_ratio:.2f}
- 허리/어깨 비율: {waist_shoulder_ratio:.2f}
- 허리/엉덩이 비율: {waist_hip_ratio:.2f}

**⚠️ 매우 중요: 모든 라인을 동등하게 비교하여 판단하세요. H라인에만 편향되지 마세요.**

---

{line_definitions_text}

---

**판별 원칙 (매우 중요 – 모든 라인을 동등하게 비교)**

**H라인 판별 (엄격한 기준 – 정말 일자 체형일 때만)**  
- 어깨·허리·골반 폭이 서로 거의 같아야 합니다 (**차이가 8% 이내일 때만 H라인으로 판단**).  
- 골반/엉덩이/허벅지 쪽에 특별히 넓어 보이는 부분이 없어야 합니다.  
- 허리선이 거의 일자여야 합니다 (살짝만 들어가거나 아예 일자).  
- 전체적으로 직사각형형/일자형 느낌이어야 합니다.  
- **사진에서 어깨는 카메라 각도 때문에 실제보다 넓게 나오는 경우가 많으므로,  
  어깨가 넓어 보인다는 이유만으로 H라인을 선택하지 마세요.**  
- **하체(골반/엉덩이/허벅지)가 조금이라도 더 넓어 보이거나, 허리가 눈에 띄게 들어가 보이면 H라인이 아니라 다른 라인(A/X/O)을 우선 고려하세요.**  
- **이 기준들이 모두 명확하게 만족될 때만 H라인으로 판별하세요.**

**A라인 판별 (하체 발달형에 중점 – 어깨가 약간 넓어도 괜찮음)**  
- 하체(골반/엉덩이/허벅지)가 상체보다 넓거나 볼륨 있어 보이는지 확인하세요.  
- 실루엣이 위→아래로 갈수록 넓어지는 형태인가?  
- 골반/엉덩이/허벅지 폭이 상체 폭보다 넓거나, 하체에 시선이 더 많이 가는가?  
- **어깨가 다소 넓어 보이더라도, 골반/엉덩이/허벅지 쪽이 더 넓거나 볼륨 있어 보이면 A라인을 선택하세요.**  
- **사진에서 어깨는 기본적으로 넓게 나오는 경우가 많으므로, “어깨가 좁게 나와야만 A라인”이라고 생각하지 마세요. 하체가 충분히 발달되어 보이면 A라인 후보입니다.**  
- **H라인과 A라인이 애매하게 느껴질 경우, 하체가 조금이라도 넓어 보이면 H라인이 아니라 A라인을 우선 선택하세요.**

**O라인 판별**  
- 허리선이 모호하고 둥근 실루엣인가?  
- 허리 둘레가 가슴/엉덩이와 거의 비슷하거나 더 큰가?  
- 복부·허리 주변에 볼륨이 많이 모여 있는가?  

**X라인 판별**  
- 어깨와 골반 너비가 비슷한가?  
- 허리가 가슴/엉덩이보다 확실히 작은가? (적당히 잘록)  
- 정면에서 보면 "X"자 형태인가?  
- **위 조건들이 모두 만족될 때만 X라인입니다.**

**최종 판별 원칙**  
- **어깨 폭만 보지 말고, 하체(골반/엉덩이/허벅지)의 볼륨과 실루엣까지 모두 비교하여 가장 적합한 라인을 선택하세요.**  
- **H라인은 “정말 일자에 가까운 체형”일 때만 선택하고, 애매하면 A/X/O 라인을 우선 고려하세요.**  
- 이미지 직접 관찰이 가장 중요합니다.  
- 측정값은 참고용일 뿐입니다.

**응답 형식**:
다음 형식으로만 응답하세요:
```
BODY_LINE: [A라인|H라인|X라인|O라인]
CONFIDENCE: [high|medium|low]
```

예시:
```
BODY_LINE: H라인
CONFIDENCE: high
```

추가 설명은 쓰지 마세요.
"""
        
        # Gemini API 호출
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[image, prompt]
        )
        
        # 응답 파싱
        response_text = response.text.strip()
        print(f"[Gemini 라인 분류] 응답: {response_text[:200]}")
        
        # BODY_LINE과 CONFIDENCE 추출
        body_line = "H라인"  # 기본값
        confidence = "low"  # 기본값
        
        for line in response_text.split('\n'):
            if 'BODY_LINE:' in line.upper():
                line_value = line.split(':')[1].strip() if ':' in line else ""
                if line_value in ['A라인', 'H라인', 'X라인', 'O라인']:
                    body_line = line_value
            elif 'CONFIDENCE:' in line.upper():
                conf_value = line.split(':')[1].strip().lower() if ':' in line else ""
                if conf_value in ['high', 'medium', 'low']:
                    confidence = conf_value
        
        print(f"[Gemini 라인 분류] 최종 결과: {body_line} (신뢰도: {confidence})")
        
        return {
            "body_line": body_line,
            "confidence": confidence
        }
        
    except Exception as e:
        print(f"Gemini 라인 분류 오류: {e}")
        import traceback
        print(traceback.format_exc())
        return {"body_line": "H라인", "confidence": "low"}


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
**⚠️ 매우 중요: 이미지에서 보이는 실제 체형을 최우선으로 판단하고, BMI는 참고용으로만 사용하세요. 마르고 비율 좋은 사람들은 이미지만 봐도 알 수 있습니다.**

이미지를 자세히 관찰하고 체형을 분석해주세요. 아래 정보들을 종합하여 최적의 분석 결과를 도출해주세요.

**분석 우선순위 (반드시 지켜주세요)**:
1. **최우선: 이미지 직접 관찰** - 실제로 보이는 체형 특징을 솔직하게 판단하세요. 마르고 비율 좋은 사람들은 이미지만 봐도 알 수 있습니다.
2. **참고: DB 체형별 정의 정보** - DB에 저장된 체형별 장점, 단점, 추천/피해야 할 스타일을 참고하세요.
3. **참고: BMI 및 체형 특징 판별 결과** - BMI는 참고용일 뿐이며, 이미지에서 보이는 실제 체형이 더 중요합니다.
4. **참고: 랜드마크 기반 체형 라인 판별 결과** - 수치는 부정확할 수 있으므로 참고용으로만 사용하세요.

**⚠️ 매우 중요**: 
- **이미지에서 보이는 실제 체형을 최우선으로 판단하세요. BMI는 참고용일 뿐입니다.**
- **이미지에서 마르고 비율 좋게 보이면, BMI와 관계없이 다양한 스타일(슬림, 머메이드, 프린세스, A라인 등)을 추천하세요.**
- **이미지에서 통통하거나 복부가 보이면, 벨라인, A라인 등 체형 보완 스타일을 추천하세요.**
- DB 정의에 명시된 "피해야 할 드레스"는 **반드시 피해야 할 스타일**입니다.

---

**1. 랜드마크 기반 체형 라인 판별 결과** (참고용, 부정확할 수 있음):
- 체형 라인: {detected_body_type}에 가깝습니다
- 어깨/엉덩이 비율: {measurements.get('shoulder_hip_ratio', 1.0):.2f}
- 허리/어깨 비율: {measurements.get('waist_shoulder_ratio', 1.0):.2f}
- 허리/엉덩이 비율: {measurements.get('waist_hip_ratio', 1.0):.2f}

**⚠️ 주의**: 위 수치는 랜드마크 기반 추정치로 **매우 부정확할 수 있습니다**. 실제 체형 판단은 **반드시 이미지를 직접 관찰**하여 하세요.

---

**2. BMI 및 키 기반 체형 특징 판별 결과** (⚠️ 참고용, 이미지 관찰이 더 중요):
{bmi_info if bmi_info else "- 키/몸무게 정보가 제공되지 않았습니다."}

**⚠️ BMI는 참고용일 뿐입니다. 이미지에서 보이는 실제 체형을 우선 판단하세요.**
- BMI는 참고용 정보일 뿐이며, **이미지에서 보이는 실제 체형이 더 중요합니다.**
- **이미지에서 마르고 비율 좋게 보이면, BMI와 관계없이 다양한 스타일을 추천하세요.**
- **이미지에서 통통하거나 복부가 보이면, 체형 보완 스타일을 추천하세요.**

**⚠️ 벨라인 추천 원칙 (매우 중요)**:
- **벨라인은 이미지에서 체형 보완이 필요한 분들(통통하게 보이거나 복부가 보이는 체형 등)에게만 추천하세요.**
- **이미지에서 마르고 비율 좋게 보이는 분들에게는 벨라인 대신 슬림, 머메이드, 프린세스, A라인 등 다양한 스타일을 추천하세요.**
- 벨라인은 허리 라인 강조와 복부 가려주는 기능이 있으므로, 체형 보완이 필요하지 않은 마른 체형에는 다양한 스타일이 더 적합합니다.

---

**3. DB 체형별 정의 정보** (⚠️ 참고용, 적극 활용):
{db_info_text if db_info_text else "- 체형 특징이 판별되지 않아 DB 정의 정보가 없습니다."}

**⚠️ 매우 중요**: 
- 위 DB 정의 정보는 **체형별 전문 지식**이므로 **반드시 적극 활용**하세요.
- DB 정의에 명시된 "추천 드레스"는 해당 체형 특징에 **가장 적합한 스타일**입니다.
- DB 정의에 명시된 "피해야 할 드레스"는 해당 체형 특징에 **절대 부적합한 스타일**이므로 **반드시 피해야 합니다**.
- DB 정의의 "장점"과 "스타일 팁"을 참고하여 분석하세요.

---

**4. 이미지 직접 관찰 (⚠️ 최우선)**:
**이미지를 자세히 관찰하여 실제 체형 특징을 솔직하게 판단하세요. 마르고 비율 좋은 사람들은 이미지만 봐도 알 수 있습니다. BMI나 다른 정보보다 이미지에서 보이는 실제 체형이 더 중요합니다.**

**성별 판별 지침**
- 이미지를 보고 성별을 추정하세요.
- **남성으로 보이면** 체형 분석만 제공하고 드레스 추천은 생략하세요. 문장 앞에 굳이 성별을 언급할 필요는 없습니다.
- **여성으로 보이면** "여성입니다", "여성으로 보입니다" 같은 문장을 쓰지 말고 바로 체형 특징을 설명하며 드레스 추천을 포함하세요.

**최종 분석 지침** (위의 모든 정보를 종합하여):

1. **⚠️ 최우선: 이미지에서 보이는 실제 체형 특징을 솔직하게 판단하세요**:
   - **이미지를 직접 관찰하여 실제로 보이는 체형을 판단하세요. 마르고 비율 좋은 사람들은 이미지만 봐도 알 수 있습니다.**
   - **이미지에서 마르고 비율 좋게 보이면, BMI와 관계없이 다양한 스타일을 추천하세요.**
   - **이미지에서 통통하거나 복부가 보이면, 체형 보완 스타일을 추천하세요.**
   - BMI는 참고용일 뿐이며, 이미지 관찰이 더 중요합니다.
   - **DB 정의에 명시된 "피해야 할 드레스"는 절대 추천하지 마세요**
   - **⚠️ "과체중", "저체중", "비만" 같은 표현은 절대 사용하지 마세요.**

2. **여성인 경우에만** 이미지에서 보이는 실제 체형을 기반으로 드레스 스타일을 추천하세요:
   - **⚠️ 최우선: 이미지에서 보이는 실제 체형을 기반으로 추천하세요.**
   - **⚠️ 이미지에서 마르고 비율 좋게 보이면, 슬림, 머메이드, 프린세스, A라인 등 다양한 스타일을 추천하세요.**
   - **⚠️ 이미지에서 통통하거나 복부가 보이면, 벨라인, A라인 등 체형 보완 스타일을 추천하세요.**
   - **⚠️ DB 정의에 명시된 "피해야 할 드레스"는 절대 추천하지 마세요**
   - BMI는 참고용일 뿐이며, 이미지 관찰이 더 중요합니다.
   
   **이미지 기반 추천 원칙**:
   - **이미지에서 마르고 비율 좋게 보이는 경우**: 슬림, 머메이드, 프린세스, A라인 등 다양한 스타일 추천 / **벨라인은 체형 보완이 필요한 경우에만 추천**
   - **이미지에서 통통하거나 복부가 보이는 경우**: 벨라인 > A라인 > 트럼펫 (슬림은 피하는 것이 좋음)
   
   **⚠️ 벨라인 추천 원칙 (매우 중요)**:
   - **벨라인은 이미지에서 체형 보완이 필요한 분들(통통하게 보이거나 복부가 보이는 체형 등)에게만 추천하세요.**
   - **이미지에서 마르고 비율 좋게 보이는 분들에게는 벨라인 대신 슬림, 머메이드, 프린세스, A라인 등 다양한 스타일을 추천하세요.**

3. **⚠️ 이미지에서 보이는 실제 체형을 최우선으로 판단하고, DB 정의 정보를 참고하여 최종 판단하세요. BMI는 참고용일 뿐입니다.**

**⚠️ 매우 중요: 전체 분석을 정확히 7줄로 작성해주세요. 너무 짧지도 길지도 않게 적절한 길이로 작성하세요.**

다음을 자연스러운 문장으로 설명해주세요 (전체 정확히 7줄):

1. **이미지를 직접 관찰한 실제 체형 특징**을 설명하세요 (2-3문장):
   - 통통함, 마름, 근육질, 볼륨 분포 등 실제로 보이는 특징을 구체적으로 설명
   - **⚠️ 매우 중요: 존댓말을 사용하고, 부드럽고 건설적인 표현을 사용하세요.**
   - **⚠️ 핵심 원칙**:
     - **단점을 직접적으로 말하지 말고, "이렇게 보완할 수 있다"는 식으로 부드럽게 표현하세요.**
     - **장점은 "이렇게 살리는게 좋은 방법이다"는 식으로 긍정적으로 표현하세요.**

2. **여성인 경우에만** 실제 이미지에서 관찰한 체형 특징을 바탕으로 드레스 스타일을 2개 설명하세요 (각 스타일당 1-2문장, 총 2-4문장):
   
   **⚠️ 매우 중요**: 
   - 남성인 경우 이 항목은 완전히 생략하세요.
   - **최우선: 이미지에서 보이는 실제 체형을 기반으로 드레스를 추천하세요.**
   - **이미지에서 마르고 비율 좋게 보이면, BMI와 관계없이 다양한 스타일을 추천하세요.**
   - **이미지에서 통통하거나 복부가 보이면, 체형 보완 스타일을 추천하세요.**
   - **DB 정의에 명시된 "피해야 할 드레스"는 절대 추천하지 마세요.**
   - **⚠️ 벨라인 추천 원칙: 벨라인은 이미지에서 체형 보완이 필요한 분들(통통하게 보이거나 복부가 보이는 체형 등)에게만 추천하고, 마르고 비율 좋게 보이는 분들에게는 벨라인 대신 슬림, 머메이드, 프린세스, A라인 등 다양한 스타일을 추천하세요.**
   - **⚠️ 존댓말을 사용하고, 각 스타일이 왜 어울리는지 구체적으로 설명하세요.**
   - **⚠️ 핵심 원칙: 단점을 보완하는 방식으로, 장점을 살리는 방식으로 표현하세요.**
   
   - **⚠️ 매우 중요: 추천할 드레스 스타일은 반드시 다음 7가지 카테고리 중에서만 선택하세요 (다른 스타일은 절대 추천하지 마세요):**
     - 벨라인 (벨트라인, 하이웨이스트 포함) - 허리 라인 강조, 복부 가려줌 (**이미지에서 체형 보완이 필요한 경우에만 추천, 통통하게 보이거나 복부가 보이는 체형**)
     - 머메이드 (물고기 실루엣) - 커브 강조 (마른 체형에 적합)
     - 프린세스 (프린세스라인) - 볼륨 추가 (마른 체형에 적합)
     - A라인 (에이라인) - 하체 볼륨 커버 (모든 체형에 적합)
     - 슬림 (스트레이트, H라인 포함) - 깔끔한 라인 (이미지에서 마르고 비율 좋게 보이는 체형에 적합, 통통하게 보이는 경우 추천 금지)
     - 트럼펫 (플레어 실루엣) - 플레어로 균형
     - 미니드레스 - 활동적이고 젊은 느낌 (마른 체형에 적합)

3. **여성인 경우에만** 피해야 할 드레스 스타일을 부드럽고 친절하게 설명하세요 (1-2문장). 최대 2개까지 언급하고, 각 스타일을 피해야 하는 이유를 구체적으로 설명하세요. **전체 분석이 정확히 9줄이 되도록 문장 수를 조절하세요.**
   **⚠️ 매우 중요**: 
   - 남성인 경우 이 항목은 완전히 생략하세요.
   - **최우선: 이미지에서 보이는 실제 체형을 기반으로 피해야 할 드레스를 판단하세요.**
   - **이미지에서 통통하거나 복부가 보이면 슬림 드레스를 피해야 할 스타일로 언급하세요.**
   - **최우선: DB 정의에 명시된 "피해야 할 드레스"를 반드시 언급**하세요.
   - 피해야 할 스타일도 위의 카테고리 중에서만 언급하세요.
   - **⚠️ 존댓말을 사용하고, 부드럽게 설명하세요.**
   - **⚠️ 핵심 원칙: "피해야 한다"고 직접적으로 말하지 말고, "이렇게 보완하는 것이 더 좋은 방법이다"는 식으로 건설적으로 표현하세요.**

반드시 지켜야 할 사항:
- **남성 사진인 경우 드레스 추천 문장(추천 1, 추천 2, 피해야 할 등)은 절대 작성하지 마세요. 체형 분석만 제공하세요.**
- **⚠️ 최우선: 이미지에서 보이는 실제 체형을 기반으로 정확한 분석을 제공하세요. 마르고 비율 좋은 사람들은 이미지만 봐도 알 수 있습니다.**
- **⚠️ 이미지에서 보이는 실제 체형이 BMI보다 더 중요합니다. BMI는 참고용일 뿐입니다.**
- **⚠️ DB 정의에 명시된 추천/피해야 할 드레스를 적극 활용하세요.**
- **이미지에서 통통하거나 복부가 보이면 슬림 드레스를 추천하지 마세요.**
- **DB 정의에 명시된 "피해야 할 드레스"는 절대 추천하지 마세요.**
- 랜드마크 수치는 참고용일 뿐이며 매우 부정확할 수 있습니다.
- BMI는 참고용일 뿐이며, 이미지 관찰이 더 중요합니다.
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

