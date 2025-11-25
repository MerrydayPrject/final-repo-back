"""FastAPI 메인 애플리케이션"""

from dotenv import load_dotenv    
load_dotenv() 

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import importlib
import logging
from pathlib import Path

from config.cors import CORS_ORIGINS, CORS_CREDENTIALS, CORS_METHODS, CORS_HEADERS
from core.model_loader import load_all_models  # 모델 로딩 함수 import

# 로깅 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 디렉토리 생성
Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)

# FastAPI 앱 초기화
app = FastAPI(
    title="의류 세그멘테이션 및 생성 API",
    description="SegFormer, Stable Diffusion, ControlNet을 사용한 고급 의류 관련 서비스",
    version="1.1.0",
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
    proxy, models, generation
)

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
app.include_router(generation.router)

# Startup 이벤트
@app.on_event("startup")
def startup_event():  # async와 await 제거
    """서버 시작 시 모든 모델을 동기적으로 로드합니다."""
    logger.info("서버 시작, 모든 모델 로딩 시작...")
    try:
        load_all_models()  # await를 제거하여 일반 함수로 호출
        logger.info("모든 모델 로딩 완료.")
    except Exception as e:
        logger.error(f"모델 로딩 중 치명적인 오류 발생: {e}", exc_info=True)