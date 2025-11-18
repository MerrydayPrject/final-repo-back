"""FastAPI 메인 애플리케이션"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from config.cors import CORS_ORIGINS, CORS_CREDENTIALS, CORS_METHODS, CORS_HEADERS
from core.model_loader import load_models

# 디렉토리 생성
Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)

# FastAPI 앱 초기화
app = FastAPI(
    title="의류 세그멘테이션 API",
    description="SegFormer 모델을 사용한 고급 의류 세그멘테이션 서비스. 웨딩드레스를 포함한 다양한 의류 항목을 감지하고 배경을 제거할 수 있습니다.",
    version="1.0.0",
    contact={
        "name": "API Support",
        "url": "https://github.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_CREDENTIALS,
    allow_methods=CORS_METHODS,
    allow_headers=CORS_HEADERS,
)

# 정적 파일 및 템플릿 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount(
    "/body-analysis-static",
    StaticFiles(directory="body_analysis_test/static"),
    name="body_analysis_static"
)
templates = Jinja2Templates(directory="templates")

# 라우터 등록
from routers import (
    info, web, segmentation, composition, prompt, 
    body_analysis, admin, dress_management, image_processing,
    proxy, models, tryon_router, body_generation, fitting_router
)
# conversion_3d는 숫자로 시작하므로 importlib 사용
import importlib
conversion_3d_router = importlib.import_module('routers.conversion_3d')

app.include_router(info.router)
app.include_router(web.router)
app.include_router(segmentation.router)
app.include_router(composition.router)
app.include_router(prompt.router)
app.include_router(body_analysis.router)
app.include_router(admin.router)
app.include_router(dress_management.router)
app.include_router(image_processing.router)
app.include_router(conversion_3d_router.router)
app.include_router(proxy.router)
app.include_router(models.router)
app.include_router(tryon_router.router)
app.include_router(body_generation.router)
app.include_router(fitting_router.router)

# Startup 이벤트
@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 모델 로드"""
    await load_models()