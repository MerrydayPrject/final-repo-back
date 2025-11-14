"""웹 인터페이스 라우터"""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse, tags=["Web Interface"])
async def home(request: Request):
    """
    메인 웹 인터페이스
    
    테스트 페이지 선택 페이지를 반환합니다.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/nukki", response_class=HTMLResponse, tags=["Web Interface"])
async def nukki_service(request: Request):
    """
    웨딩드레스 누끼 서비스
    
    웨딩드레스를 입은 인물 이미지에서 드레스만 추출하는 서비스 페이지를 반환합니다.
    """
    return templates.TemplateResponse("nukki.html", {"request": request})


@router.get("/body-analysis", response_class=HTMLResponse, tags=["Web Interface"])
async def body_analysis_page(request: Request):
    """
    체형 분석 웹 페이지
    """
    return templates.TemplateResponse("body_analysis.html", {"request": request})


@router.get("/gemini-test", response_class=HTMLResponse, tags=["Web Interface"])
async def gemini_test_page(request: Request):
    """
    Gemini 이미지 합성 테스트 페이지
    
    사람 이미지와 드레스 이미지를 업로드하여 합성 결과를 테스트할 수 있는 페이지
    """
    return templates.TemplateResponse("gemini_test.html", {"request": request})


@router.get("/3d-conversion", response_class=HTMLResponse, tags=["Web Interface"])
async def conversion_3d_page(request: Request):
    """3D 이미지 변환 페이지"""
    return templates.TemplateResponse("3d_conversion.html", {"request": request})


@router.get("/model-comparison", response_class=HTMLResponse, tags=["Web Interface"])
async def model_comparison_page(request: Request):
    """
    모델 비교 테스트 페이지
    
    여러 모델의 합성 기능을 동시에 비교할 수 있는 페이지
    """
    return templates.TemplateResponse("model-comparison.html", {"request": request})


@router.get("/llm-model", response_class=HTMLResponse, tags=["Web Interface"])
async def llm_model_page(request: Request):
    """
    LLM 모델 테스트 페이지
    
    프롬프트 생성용과 이미지 생성용 LLM 모델을 선택하여 테스트할 수 있는 페이지
    """
    return templates.TemplateResponse("llm_model.html", {"request": request})


@router.get("/admin", response_class=HTMLResponse, tags=["관리자"])
async def admin_page(request: Request):
    """
    관리자 페이지
    
    로그 목록과 통계를 확인할 수 있는 관리자 페이지
    """
    return templates.TemplateResponse("admin.html", {"request": request})


@router.get("/admin/dress-insert", response_class=HTMLResponse, tags=["관리자"])
async def dress_insert_page(request: Request):
    """
    드레스 이미지 삽입 관리자 페이지
    """
    return templates.TemplateResponse("dress_insert.html", {"request": request})


@router.get("/admin/dress-manage", response_class=HTMLResponse, tags=["관리자"])
async def dress_manage_page(request: Request):
    """
    드레스 관리자 페이지
    
    드레스 정보 목록 조회 및 추가가 가능한 관리자 페이지
    """
    return templates.TemplateResponse("dress_manage.html", {"request": request})


@router.get("/favicon.ico")
async def favicon():
    """파비콘 제공"""
    return FileResponse("static/favicon.ico")

