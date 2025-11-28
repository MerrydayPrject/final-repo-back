"""웹 인터페이스 라우터"""
from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from config.auth_middleware import get_current_user
from core.supabase_client import is_admin_user as check_admin

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse, tags=["Web Interface"])
async def home(request: Request):
    """
    메인 웹 인터페이스
    
    테스트 페이지 선택 페이지를 반환합니다.
    로그인 상태를 확인하여 로그인 폼 또는 관리자 메뉴를 표시합니다.
    """
    # 로그인 상태 확인 (선택적 - 실패해도 계속 진행)
    user_data = await get_current_user(request)
    is_authenticated = user_data is not None and check_admin(user_data) if user_data else False
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "is_authenticated": is_authenticated,
        "user_email": user_data.get("email") if user_data else None
    })


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


@router.get("/body-generation", response_class=HTMLResponse, tags=["Web Interface"])
async def body_generation_page(request: Request):
    """
    체형 생성 웹 페이지
    
    얼굴이 포함된 이미지에서 얼굴+목을 보존하고 기본 체형을 생성하여 전신 이미지로 변환하는 페이지
    """
    return templates.TemplateResponse("body_generation.html", {"request": request})


@router.get("/gemini-test", response_class=HTMLResponse, tags=["Web Interface"])
async def gemini_test_page(request: Request):
    """
    Gemini 이미지 합성 테스트 페이지
    
    사람 이미지와 드레스 이미지를 업로드하여 합성 결과를 테스트할 수 있는 페이지
    """
    return templates.TemplateResponse("gemini_test.html", {"request": request})


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


@router.get("/image-filter-sticker", response_class=HTMLResponse, tags=["Web Interface"])
async def image_filter_sticker_page(request: Request):
    """
    이미지 필터 & 프레임 페이지
    
    이미지에 필터와 프레임을 적용하여 꾸밀 수 있는 페이지
    """
    return templates.TemplateResponse("image_filter_frame.html", {"request": request})


@router.get("/admin", response_class=HTMLResponse, tags=["관리자"])
async def admin_page(request: Request):
    """
    관리자 페이지
    
    로그 목록과 통계를 확인할 수 있는 관리자 페이지
    페이지 접근은 허용하고, JavaScript에서 토큰 확인 후 API 호출 시 인증을 확인합니다.
    """
    # 페이지는 인증 없이 접근 가능 (JavaScript에서 토큰 확인)
    return templates.TemplateResponse("admin.html", {"request": request})


@router.get("/admin/dress-insert", response_class=HTMLResponse, tags=["관리자"])
async def dress_insert_page(request: Request):
    """
    드레스 이미지 삽입 관리자 페이지
    페이지 접근은 허용하고, JavaScript에서 토큰 확인 후 API 호출 시 인증을 확인합니다.
    """
    # 페이지는 인증 없이 접근 가능 (JavaScript에서 토큰 확인)
    return templates.TemplateResponse("dress_insert.html", {"request": request})


@router.get("/admin/dress-manage", response_class=HTMLResponse, tags=["관리자"])
async def dress_manage_page(request: Request):
    """
    드레스 관리자 페이지
    
    드레스 정보 목록 조회 및 추가가 가능한 관리자 페이지
    페이지 접근은 허용하고, JavaScript에서 토큰 확인 후 API 호출 시 인증을 확인합니다.
    """
    # 페이지는 인증 없이 접근 가능 (JavaScript에서 토큰 확인)
    return templates.TemplateResponse("dress_manage.html", {"request": request})


@router.get("/document-logs", response_class=HTMLResponse, tags=["관리자"])
async def document_logs_page(request: Request):
    """
    문서용로그 페이지
    
    합성 결과를 이미지로 확인할 수 있는 문서용 로그 페이지
    """
    return templates.TemplateResponse("document_logs.html", {"request": request})


@router.get("/dress-batch-test", response_class=HTMLResponse, tags=["관리자"])
async def dress_batch_test_page(request: Request):
    """
    드레스 판별 테스트 페이지
    
    여러 이미지를 배치로 업로드하여 드레스 여부를 판별하는 페이지
    """
    return templates.TemplateResponse("dress-batch-test.html", {"request": request})


@router.get("/favicon.ico")
async def favicon():
    """파비콘 제공"""
    return FileResponse("static/favicon.ico")

