"""
체형 분석 기능 테스트 스크립트
독립적으로 테스트 가능한 버전 (8002 포트)
"""
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import json
from typing import Dict
from dotenv import load_dotenv
from google import genai
from body_analysis import BodyAnalysisService
from pathlib import Path

# .env 파일 로드 (상위 디렉토리에서도 찾기)
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    # 현재 디렉토리에서도 시도
    load_dotenv()

# FastAPI 앱 초기화 (테스트용)
app = FastAPI(
    title="체형 분석 API (테스트)",
    description="MediaPipe와 Gemini를 사용한 체형 분석 테스트 API",
    version="1.0.0-test"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8000", "http://localhost:8001", "http://localhost:8002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 및 템플릿 설정
static_dir = os.path.join(os.path.dirname(__file__), "static")
templates_dir = os.path.join(os.path.dirname(__file__), "templates")

if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
if os.path.exists(templates_dir):
    templates = Jinja2Templates(directory=templates_dir)
else:
    templates = None

# 전역 변수
body_analysis_service = None

@app.on_event("startup")
async def load_services():
    """서비스 초기화"""
    global body_analysis_service
    try:
        print("체형 분석 서비스 초기화 중...")
        body_analysis_service = BodyAnalysisService()
        if body_analysis_service.is_initialized:
            print("✅ 체형 분석 서비스 초기화 완료!")
        else:
            print("⚠️  체형 분석 서비스 초기화 실패")
    except Exception as e:
        print(f"❌ 서비스 초기화 오류: {e}")

@app.get("/favicon.ico")
async def favicon():
    """Favicon 요청 처리 (404 방지)"""
    return Response(status_code=204)  # No Content

@app.get("/")
async def root():
    """테스트 페이지"""
    if templates:
        return templates.TemplateResponse("body_analysis_test.html", {"request": {}})
    return HTMLResponse("""
    <html>
        <head><title>체형 분석 테스트 서버</title></head>
        <body>
            <h1>체형 분석 테스트 서버</h1>
            <p>API 문서: <a href="/docs">/docs</a></p>
            <p>포트: 8002</p>
        </body>
    </html>
    """)

@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {
        "status": "healthy",
        "service": "body_analysis_test",
        "port": 8002,
        "body_analysis_service": body_analysis_service.is_initialized if body_analysis_service else False,
        "version": "1.0.0-test"
    }

@app.post("/api/analyze-body")
async def analyze_body(file: UploadFile = File(..., description="전신 이미지 파일")):
    """
    전신 이미지 체형 분석 (테스트용)
    
    MediaPipe Pose Landmarker로 포즈 랜드마크를 추출하고,
    체형 비율을 계산한 후 Gemini API로 상세 분석을 수행합니다.
    """
    try:
        # 체형 분석 서비스 확인
        if not body_analysis_service or not body_analysis_service.is_initialized:
            return JSONResponse({
                "success": False,
                "error": "Body analysis service not initialized",
                "message": "체형 분석 서비스가 초기화되지 않았습니다. 모델 파일을 확인해주세요."
            }, status_code=500)
        
        # 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 1. 포즈 랜드마크 추출
        landmarks = body_analysis_service.extract_landmarks(image)
        
        if landmarks is None:
            return JSONResponse({
                "success": False,
                "error": "No pose detected",
                "message": "이미지에서 포즈를 감지할 수 없습니다. 전신이 보이는 이미지를 업로드해주세요."
            }, status_code=400)
        
        # 2. 체형 측정값 계산
        measurements = body_analysis_service.calculate_measurements(landmarks)
        
        # 3. 체형 타입 분류 (랜드마크 기반)
        body_type = body_analysis_service.classify_body_type(measurements)
        
        # 4. Gemini API로 상세 분석
        gemini_analysis = None
        try:
            gemini_analysis = await analyze_body_with_gemini(image, measurements, body_type)
        except Exception as e:
            print(f"Gemini 분석 실패: {e}")
        
        return JSONResponse({
            "success": True,
            "body_analysis": {
                "body_type": body_type['type'],
                "measurements": measurements,
                "body_type_category": body_type
            },
            "pose_landmarks": {
                "total_landmarks": len(landmarks),
                "detected_landmarks": landmarks
            },
            "gemini_analysis": gemini_analysis,
            "message": "체형 분석이 완료되었습니다."
        })
        
    except Exception as e:
        import traceback
        print(f"체형 분석 오류: {e}")
        print(traceback.format_exc())
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"체형 분석 중 오류 발생: {str(e)}"
        }, status_code=500)

async def analyze_body_with_gemini(image: Image.Image, measurements: Dict, body_type: Dict):
    """
    Gemini API로 체형 상세 분석
    """
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("GEMINI_API_KEY가 설정되지 않았습니다.")
            return None
        
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
이 이미지를 자세히 관찰하고 체형을 분석해주세요.

**핵심 원칙**: 이미지를 직접 보고 실제 체형 특징을 솔직하게 판단하세요. 랜드마크 기반 수치는 참고용일 뿐입니다.

랜드마크 기반 체형 라인 (참고용):
- 체형 라인: {body_type.get('type', 'unknown')}에 가깝습니다
- 어깨/엉덩이 비율: {measurements.get('shoulder_hip_ratio', 1.0):.2f}
- 허리/어깨 비율: {measurements.get('waist_shoulder_ratio', 1.0):.2f}
- 허리/엉덩이 비율: {measurements.get('waist_hip_ratio', 1.0):.2f}

**중요**: 위 수치는 랜드마크 기반 추정치로 정확하지 않을 수 있습니다. 실제 체형 판단은 이미지를 직접 관찰하여 하세요.

**먼저 이미지에서 이 사람의 성별을 판단하세요.**

**성별이 남성인 경우:**
- 체형 특징만 분석하세요 (통통함, 마름, 근육질, 상체/하체 비율, 전체적인 체형 인상 등)
- 드레스 추천은 절대 하지 마세요.

**성별이 여성인 경우:**
- 체형 특징을 분석하고
- 드레스 스타일 추천을 포함하세요.

**분석 지침**:
1. **이미지를 직접 관찰**하여 이 사람의 실제 체형 특징을 솔직하게 파악하세요:
   - 통통한지, 마른지, 근육질인지
   - 상체/하체 볼륨 분포 (상체가 큰지, 하체가 큰지, 균형잡혔는지)
   - 허리 라인이 명확한지, 직선적인지
   - 전체적인 체형 인상 (건강한 느낌, 날씬한 느낌, 볼륨감 있는 느낌 등)

2. **여성인 경우에만** 실제 체형 특징에 맞는 드레스 스타일을 추천하세요:
   - 예: 통통한 체형이면 → 벨라인(허리 강조) > 머메이드(볼륨 강조되므로 부적합)
   - 예: 마른 체형이면 → 슬림, 머메이드 등 다양한 스타일 가능
   - 예: 하체가 큰 체형이면 → A라인, 트럼펫(하체 커버) > 슬림(하체 노출)
   - 예: 상체가 큰 체형이면 → A라인(하체 볼륨로 균형) > 프린세스(상체 더 강조)

3. **위의 랜드마크 기반 라인 타입과 실제 이미지 관찰 결과가 다를 수 있습니다. 실제 이미지에서 보이는 체형 특징을 우선하세요.**

다음을 자연스러운 문장으로 설명해주세요:

1. **이미지를 직접 관찰한 실제 체형 특징**을 구체적으로 설명하세요:
   - 통통함, 마름, 근육질, 볼륨 분포 등 실제로 보이는 특징
   - 체형의 장점과 특징적인 부분

2. **여성인 경우에만** 실제 체형 특징에 맞는 드레스 스타일을 2-3개 구체적으로 제시하고, 각 스타일이 왜 어울리는지 이유를 설명 안에 포함하세요.
   **중요**: 
   - 남성인 경우 이 항목은 완전히 생략하세요.
   - 추천할 드레스 스타일은 반드시 다음 카테고리 중에서만 선택하세요:
     - 벨라인 (벨트라인, 하이웨이스트 포함)
     - 머메이드 (물고기 실루엣)
     - 프린세스 (프린세스라인)
     - A라인 (에이라인)
     - 슬림 (스트레이트, H라인 포함)
     - 트럼펫 (플레어 실루엣)
   - **실제 체형 특징을 고려하여 추천하세요** (예: 통통하면 벨라인 > 머메이드, 하체가 크면 A라인 > 슬림)
   - 예: "이 체형은 하체 볼륨이 있어 A라인 드레스가 잘 어울리며, 그 이유는 하체를 자연스럽게 커버하면서 균형잡힌 실루엣을 만들어주기 때문입니다."

3. **여성인 경우에만** 실제 체형 특징에 맞지 않는 드레스 스타일을 2개 구체적으로 제시하고, 각 스타일을 피해야 하는 이유를 설명 안에 포함하세요.
   **중요**: 
   - 남성인 경우 이 항목은 완전히 생략하세요.
   - 피해야 할 스타일도 위의 카테고리 중에서만 언급하세요.
   - **실제 체형 특징을 고려하여 추천하세요** (예: 통통하면 머메이드는 피하는 것이 좋으며, 그 이유는 볼륨을 더 강조하기 때문입니다)
   - 예: "슬림 드레스는 피하는 것이 좋으며, 그 이유는 이 체형의 하체 볼륨을 그대로 노출시켜 불균형해 보일 수 있기 때문입니다."

반드시 지켜야 할 사항:
- **남성 사진인 경우 드레스 추천은 절대 하지 마세요. 체형 분석만 제공하세요.**
- **이미지를 직접 보고 실제 체형 특징을 솔직하게 판단하세요. 랜드마크 수치는 참고용일 뿐입니다.**
- 스타일링 팁, 액세서리 추천, 색상 추천, 코디 팁 등은 절대 포함하지 마세요.
- 여성인 경우에만 추천 드레스 스타일명과 피해야 할 드레스 스타일명은 반드시 위의 6개 카테고리(벨라인, 머메이드, 프린세스, A라인, 슬림, 트럼펫) 중에서만 선택하세요.
- 다른 드레스 스타일명(랩 드레스, 엠파이어 드레스 등)은 언급하지 마세요.
- 별도의 리스트나 항목으로 나열하지 말고, 자연스러운 문단 형식으로 설명해주세요.
- **실제 체형 특징에 맞는 실용적이고 솔직한 추천을 해주세요.**
"""
        
        # Gemini API 호출 (원래 스타일: 이미지와 프롬프트 직접 전달)
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[image, prompt]
        )
        
        # 응답 파싱
        analysis_text = response.text
        
        # 상세 분석만 반환 (별도 리스트 제거)
        return {
            "detailed_analysis": analysis_text
        }
        
    except Exception as e:
        print(f"Gemini 분석 오류: {e}")
        return None

if __name__ == "__main__":
    import uvicorn
    import sys
    
    # 포트 번호 확인 (기본값: 8002)
    port = 8002
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"잘못된 포트 번호: {sys.argv[1]}. 기본 포트 8002 사용")
    
    print("=" * 50)
    print("체형 분석 테스트 서버 시작...")
    print(f"접속 주소: http://localhost:{port}")
    print(f"API 문서: http://localhost:{port}/docs")
    print(f"테스트 페이지: http://localhost:{port}/")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=port)

