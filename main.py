from fastapi import FastAPI, File, UploadFile, Request, Query, Form
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import csv
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import io
import base64
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import os
import time
import pymysql
from datetime import datetime
from dotenv import load_dotenv
import json
import boto3
from botocore.exceptions import ClientError
import requests
from urllib.parse import urlparse, unquote

# .env 파일 로드
load_dotenv()

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
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # 프론트엔드 주소들
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# 디렉토리 생성
Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)
Path("uploads").mkdir(exist_ok=True)
# images 폴더는 S3 사용으로 불필요

# 정적 파일 및 템플릿 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 전역 변수로 모델 저장
processor = None
model = None

# 새 모델들의 전역 변수 (lazy loading)
segformer_b2_processor = None
segformer_b2_model = None
rtmpose_model = None
realesrgan_model = None
sdxl_pipeline = None

# 레이블 정보
LABELS = {
    0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses",
    4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "Dress",
    8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face",
    12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm",
    16: "Bag", 17: "Scarf"
}

# ===================== DB 연결 함수 =====================

def get_db_connection():
    """MySQL 데이터베이스 연결 반환"""
    try:
        connection = pymysql.connect(
            host=os.getenv("MYSQL_HOST", "localhost"),
            port=int(os.getenv("MYSQL_PORT", 3306)),
            user=os.getenv("MYSQL_USER", "devuser"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=os.getenv("MYSQL_DATABASE", "marryday"),
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        return connection
    except Exception as e:
        print(f"DB 연결 오류: {e}")
        return None

def init_database():
    """데이터베이스 테이블 생성"""
    connection = get_db_connection()
    if not connection:
        print("DB 연결 실패 - 테이블 생성 건너뜀")
        return
    
    try:
        with connection.cursor() as cursor:
            # dresses 테이블 생성
            create_dresses_table = """
            CREATE TABLE IF NOT EXISTS dresses (
                idx INT AUTO_INCREMENT PRIMARY KEY,
                dress_name VARCHAR(255) NOT NULL UNIQUE,
                file_name VARCHAR(255) NOT NULL,
                style VARCHAR(255) NOT NULL,
                url TEXT NOT NULL,
                INDEX idx_file_name (file_name),
                INDEX idx_style (style)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """
            cursor.execute(create_dresses_table)
            
            # 기존 테이블에 UNIQUE 제약 조건 추가 (테이블이 이미 존재하는 경우)
            try:
                cursor.execute("ALTER TABLE dresses ADD UNIQUE KEY uk_dress_name (dress_name)")
                print("UNIQUE 제약 조건 추가 완료: dress_name")
            except Exception as e:
                # 이미 제약 조건이 존재하거나 테이블이 없는 경우는 무시
                if "Duplicate key name" not in str(e) and "Unknown column" not in str(e):
                    print(f"UNIQUE 제약 조건 추가 시도: {e}")
            
            connection.commit()
            print("DB 테이블 생성 완료: dresses")
            
            # result_logs 테이블 생성
            create_result_logs_table = """
            CREATE TABLE IF NOT EXISTS result_logs (
                idx INT AUTO_INCREMENT PRIMARY KEY,
                person_url TEXT NOT NULL COMMENT '인물 이미지 (Input 1)',
                dress_url TEXT COMMENT '의상 이미지 (Input 2)',
                result_url TEXT NOT NULL COMMENT '결과 이미지',
                model VARCHAR(255) NOT NULL COMMENT '사용된 AI 모델명',
                prompt TEXT NOT NULL COMMENT '사용된 AI 명령어',
                success BOOLEAN NOT NULL COMMENT '실행 성공 (TRUE/FALSE)',
                run_time DOUBLE NOT NULL COMMENT '실행 시간 (초)',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_model (model),
                INDEX idx_success (success),
                INDEX idx_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """
            cursor.execute(create_result_logs_table)
            connection.commit()
            print("DB 테이블 생성 완료: result_logs")
    except Exception as e:
        print(f"테이블 생성 오류: {e}")
    finally:
        connection.close()

def save_uploaded_image(image: Image.Image, prefix: str) -> str:
    """이미지를 파일 시스템에 저장"""
    timestamp = int(time.time() * 1000)
    filename = f"{prefix}_{timestamp}.png"
    filepath = Path("uploads") / filename
    image.save(filepath)
    return str(filepath)

def load_category_rules() -> List[dict]:
    """카테고리 규칙 JSON 파일 로드"""
    rules_file = Path("category_rules.json")
    if not rules_file.exists():
        # 기본 규칙으로 파일 생성
        default_rules = [
            {"prefix": "A", "style": "A라인"},
            {"prefix": "Mini", "style": "미니드레스"},
            {"prefix": "B", "style": "벨라인"},
            {"prefix": "P", "style": "프린세스"}
        ]
        with open(rules_file, "w", encoding="utf-8") as f:
            json.dump(default_rules, f, ensure_ascii=False, indent=2)
        return default_rules
    
    try:
        with open(rules_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"카테고리 규칙 로드 오류: {e}")
        return []

def save_category_rules(rules: List[dict]) -> bool:
    """카테고리 규칙 JSON 파일 저장"""
    try:
        rules_file = Path("category_rules.json")
        with open(rules_file, "w", encoding="utf-8") as f:
            json.dump(rules, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"카테고리 규칙 저장 오류: {e}")
        return False

def detect_style_from_filename(filename: str) -> Optional[str]:
    """
    이미지 파일명에서 스타일을 감지 (JSON 규칙 기반)
    
    Args:
        filename: 이미지 파일명 (예: "Adress1.jpg", "Mini_dress.png")
    
    Returns:
        감지된 스타일 문자열 또는 None (감지 실패 시)
    """
    rules = load_category_rules()
    filename_upper = filename.upper()
    
    # 규칙을 우선순위대로 확인 (긴 prefix 우선)
    sorted_rules = sorted(rules, key=lambda x: len(x["prefix"]), reverse=True)
    
    for rule in sorted_rules:
        prefix_upper = rule["prefix"].upper()
        # prefix로 시작하거나 포함하는지 확인
        if filename_upper.startswith(prefix_upper) or prefix_upper in filename_upper:
            return rule["style"]
    
    return None

# Pydantic 모델
class LabelInfo(BaseModel):
    """레이블 정보 모델"""
    id: int = Field(..., description="레이블 ID")
    name: str = Field(..., description="레이블 이름")
    percentage: float = Field(..., description="이미지 내 해당 레이블이 차지하는 비율 (%)")

class SegmentationResponse(BaseModel):
    """세그멘테이션 응답 모델"""
    success: bool = Field(..., description="처리 성공 여부")
    original_image: str = Field(..., description="원본 이미지 (base64)")
    result_image: str = Field(..., description="결과 이미지 (base64)")
    detected_labels: List[LabelInfo] = Field(..., description="감지된 레이블 목록")
    message: str = Field(..., description="처리 결과 메시지")

class ErrorResponse(BaseModel):
    """에러 응답 모델"""
    success: bool = Field(False, description="처리 성공 여부")
    error: str = Field(..., description="에러 메시지")
    message: str = Field(..., description="사용자 친화적 에러 메시지")

@app.on_event("startup")
async def load_model():
    """애플리케이션 시작 시 모델 로드 및 DB 초기화"""
    import asyncio
    global processor, model
    print("SegFormer 모델 로딩 중...")
    # 동기 블로킹 작업을 별도 스레드에서 실행
    loop = asyncio.get_event_loop()
    processor = await loop.run_in_executor(None, SegformerImageProcessor.from_pretrained, "mattmdjaga/segformer_b2_clothes")
    model = await loop.run_in_executor(None, AutoModelForSemanticSegmentation.from_pretrained, "mattmdjaga/segformer_b2_clothes")
    model.eval()
    print("모델 로딩 완료!")
    
    # DB 초기화
    print("데이터베이스 초기화 중...")
    await loop.run_in_executor(None, init_database)

@app.get("/", response_class=HTMLResponse, tags=["Web Interface"])
async def home(request: Request):
    """
    메인 웹 인터페이스
    
    테스트 페이지 선택 페이지를 반환합니다.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/nukki", response_class=HTMLResponse, tags=["Web Interface"])
async def nukki_service(request: Request):
    """
    웨딩드레스 누끼 서비스
    
    웨딩드레스를 입은 인물 이미지에서 드레스만 추출하는 서비스 페이지를 반환합니다.
    """
    return templates.TemplateResponse("nukki.html", {"request": request})

@app.get("/labels", tags=["정보"])
async def get_labels():
    """
    사용 가능한 모든 레이블 목록 조회
    
    SegFormer 모델이 감지할 수 있는 18개 의류/신체 부위 레이블 목록을 반환합니다.
    
    Returns:
        dict: 레이블 ID를 키로, 레이블 이름을 값으로 하는 딕셔너리
    """
    return {
        "labels": LABELS,
        "total_labels": len(LABELS),
        "description": "SegFormer B2 모델이 감지할 수 있는 레이블 목록"
    }

@app.post("/api/segment", tags=["세그멘테이션"])
async def segment_dress(file: UploadFile = File(..., description="세그멘테이션할 이미지 파일")):
    """
    드레스 세그멘테이션 (웨딩드레스 누끼)
    
    업로드된 이미지에서 드레스(레이블 7)를 감지하고 배경을 제거합니다.
    
    Args:
        file: 업로드할 이미지 파일 (JPG, PNG, GIF, WEBP 등)
    
    Returns:
        JSONResponse: 원본 이미지, 누끼 결과 이미지(투명 배경), 감지 정보
        
    Raises:
        500: 이미지 처리 중 오류 발생
    """
    try:
        # 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        original_size = image.size
        
        # 원본 이미지를 base64로 인코딩
        buffered_original = io.BytesIO()
        image.save(buffered_original, format="PNG")
        original_base64 = base64.b64encode(buffered_original.getvalue()).decode()
        
        # 모델 추론
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.cpu()
        
        # 업샘플링
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=original_size[::-1],  # (height, width)
            mode="bilinear",
            align_corners=False,
        )
        
        # 세그멘테이션 마스크 생성
        pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
        
        # 드레스 마스크 생성 (레이블 7: Dress)
        dress_mask = (pred_seg == 7).astype(np.uint8) * 255
        
        # 원본 이미지를 numpy 배열로 변환
        image_array = np.array(image)
        
        # 누끼 이미지 생성 (RGBA)
        result_image = np.zeros((image_array.shape[0], image_array.shape[1], 4), dtype=np.uint8)
        result_image[:, :, :3] = image_array  # RGB 채널
        result_image[:, :, 3] = dress_mask    # 알파 채널
        
        # PIL 이미지로 변환
        result_pil = Image.fromarray(result_image, mode='RGBA')
        
        # 결과 이미지를 base64로 인코딩
        buffered_result = io.BytesIO()
        result_pil.save(buffered_result, format="PNG")
        result_base64 = base64.b64encode(buffered_result.getvalue()).decode()
        
        # 드레스가 감지되었는지 확인
        dress_pixels = int(np.sum(pred_seg == 7))
        total_pixels = int(pred_seg.size)
        dress_percentage = float((dress_pixels / total_pixels) * 100)
        
        return JSONResponse({
            "success": True,
            "original_image": f"data:image/png;base64,{original_base64}",
            "result_image": f"data:image/png;base64,{result_base64}",
            "dress_detected": bool(dress_pixels > 0),
            "dress_percentage": round(dress_percentage, 2),
            "message": f"드레스 영역: {dress_percentage:.2f}% 감지됨" if dress_pixels > 0 else "드레스가 감지되지 않았습니다."
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"처리 중 오류 발생: {str(e)}"
        }, status_code=500)

@app.get("/health", tags=["정보"])
async def health_check():
    """
    서버 상태 확인
    
    서버와 모델의 로딩 상태를 확인합니다.
    
    Returns:
        dict: 서버 상태 및 모델 로딩 여부
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None and processor is not None,
        "model_name": "mattmdjaga/segformer_b2_clothes",
        "version": "1.0.0"
    }

@app.get("/test", tags=["테스트"])
async def test_endpoint():
    """
    간단한 테스트 엔드포인트
    
    서버가 정상적으로 응답하는지 확인합니다.
    """
    return {
        "message": "서버가 정상적으로 작동 중입니다!",
        "timestamp": time.time()
    }

@app.post("/api/segment-custom", tags=["세그멘테이션"])
async def segment_custom(
    file: UploadFile = File(..., description="세그멘테이션할 이미지 파일"),
    labels: str = Query(..., description="추출할 레이블 ID (쉼표로 구분, 예: 4,5,6,7)")
):
    """
    커스텀 레이블 세그멘테이션
    
    지정한 레이블들만 추출하여 배경을 제거합니다.
    
    Args:
        file: 업로드할 이미지 파일
        labels: 추출할 레이블 ID (쉼표로 구분)
                예: "7" (드레스만), "4,5,6,7" (상의, 치마, 바지, 드레스)
    
    Returns:
        JSONResponse: 원본 이미지, 선택한 레이블만 추출한 결과 이미지
        
    Example:
        - labels="7": 드레스만 추출
        - labels="4,6": 상의와 바지만 추출
        - labels="1,2,11": 모자, 머리, 얼굴만 추출
    """
    try:
        # 레이블 파싱
        label_ids = [int(l.strip()) for l in labels.split(",")]
        
        # 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        original_size = image.size
        
        # 원본 이미지를 base64로 인코딩
        buffered_original = io.BytesIO()
        image.save(buffered_original, format="PNG")
        original_base64 = base64.b64encode(buffered_original.getvalue()).decode()
        
        # 모델 추론
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.cpu()
        
        # 업샘플링
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=original_size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        
        # 세그멘테이션 마스크 생성
        pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
        
        # 선택한 레이블들의 마스크 생성
        combined_mask = np.zeros_like(pred_seg, dtype=bool)
        for label_id in label_ids:
            combined_mask |= (pred_seg == label_id)
        
        mask = combined_mask.astype(np.uint8) * 255
        
        # 원본 이미지를 numpy 배열로 변환
        image_array = np.array(image)
        
        # 누끼 이미지 생성 (RGBA)
        result_image = np.zeros((image_array.shape[0], image_array.shape[1], 4), dtype=np.uint8)
        result_image[:, :, :3] = image_array
        result_image[:, :, 3] = mask
        
        # PIL 이미지로 변환
        result_pil = Image.fromarray(result_image, mode='RGBA')
        
        # 결과 이미지를 base64로 인코딩
        buffered_result = io.BytesIO()
        result_pil.save(buffered_result, format="PNG")
        result_base64 = base64.b64encode(buffered_result.getvalue()).decode()
        
        # 각 레이블의 픽셀 수 계산
        detected_labels = []
        total_pixels = int(pred_seg.size)
        for label_id in label_ids:
            pixels = int(np.sum(pred_seg == label_id))
            if pixels > 0:
                detected_labels.append({
                    "id": label_id,
                    "name": LABELS.get(label_id, "Unknown"),
                    "percentage": round((pixels / total_pixels) * 100, 2)
                })
        
        total_detected = int(np.sum(combined_mask))
        
        return JSONResponse({
            "success": True,
            "original_image": f"data:image/png;base64,{original_base64}",
            "result_image": f"data:image/png;base64,{result_base64}",
            "requested_labels": [{"id": lid, "name": LABELS.get(lid, "Unknown")} for lid in label_ids],
            "detected_labels": detected_labels,
            "total_percentage": round((total_detected / total_pixels) * 100, 2),
            "message": f"{len(detected_labels)}개의 레이블 감지됨" if detected_labels else "선택한 레이블이 감지되지 않았습니다."
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"처리 중 오류 발생: {str(e)}"
        }, status_code=500)

@app.post("/api/analyze", tags=["분석"])
async def analyze_image(file: UploadFile = File(..., description="분석할 이미지 파일")):
    """
    이미지 전체 분석
    
    이미지에서 모든 레이블을 감지하고 각 레이블의 비율을 분석합니다.
    누끼 처리 없이 분석 정보만 반환합니다.
    
    Args:
        file: 분석할 이미지 파일
    
    Returns:
        JSONResponse: 감지된 모든 레이블과 비율 정보
    """
    try:
        # 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        original_size = image.size
        
        # 모델 추론
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.cpu()
        
        # 업샘플링
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=original_size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        
        # 세그멘테이션 마스크 생성
        pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
        
        # 각 레이블의 픽셀 수 계산
        total_pixels = int(pred_seg.size)
        detected_labels = []
        
        for label_id, label_name in LABELS.items():
            pixels = int(np.sum(pred_seg == label_id))
            percentage = round((pixels / total_pixels) * 100, 2)
            if pixels > 0:
                detected_labels.append({
                    "id": label_id,
                    "name": label_name,
                    "pixels": pixels,
                    "percentage": percentage
                })
        
        # 비율 순으로 정렬
        detected_labels.sort(key=lambda x: x["percentage"], reverse=True)
        
        return JSONResponse({
            "success": True,
            "image_size": {"width": original_size[0], "height": original_size[1]},
            "total_pixels": total_pixels,
            "detected_labels": detected_labels,
            "total_detected": len(detected_labels),
            "message": f"총 {len(detected_labels)}개의 레이블 감지됨"
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"처리 중 오류 발생: {str(e)}"
        }, status_code=500)

@app.post("/api/remove-background", tags=["세그멘테이션"])
async def remove_background(file: UploadFile = File(..., description="배경을 제거할 이미지 파일")):
    """
    전체 배경 제거 (인물만 추출)
    
    배경(레이블 0)을 제거하고 인물과 의류만 남깁니다.
    
    Args:
        file: 배경을 제거할 이미지 파일
    
    Returns:
        JSONResponse: 배경이 제거된 이미지 (투명 배경)
    """
    try:
        # 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        original_size = image.size
        
        # 원본 이미지를 base64로 인코딩
        buffered_original = io.BytesIO()
        image.save(buffered_original, format="PNG")
        original_base64 = base64.b64encode(buffered_original.getvalue()).decode()
        
        # 모델 추론
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.cpu()
        
        # 업샘플링
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=original_size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        
        # 세그멘테이션 마스크 생성
        pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
        
        # 배경이 아닌 모든 것을 포함하는 마스크
        mask = (pred_seg != 0).astype(np.uint8) * 255
        
        # 원본 이미지를 numpy 배열로 변환
        image_array = np.array(image)
        
        # 누끼 이미지 생성 (RGBA)
        result_image = np.zeros((image_array.shape[0], image_array.shape[1], 4), dtype=np.uint8)
        result_image[:, :, :3] = image_array
        result_image[:, :, 3] = mask
        
        # PIL 이미지로 변환
        result_pil = Image.fromarray(result_image, mode='RGBA')
        
        # 결과 이미지를 base64로 인코딩
        buffered_result = io.BytesIO()
        result_pil.save(buffered_result, format="PNG")
        result_base64 = base64.b64encode(buffered_result.getvalue()).decode()
        
        # 배경이 아닌 픽셀 수 계산
        foreground_pixels = int(np.sum(pred_seg != 0))
        total_pixels = int(pred_seg.size)
        foreground_percentage = round((foreground_pixels / total_pixels) * 100, 2)
        
        return JSONResponse({
            "success": True,
            "original_image": f"data:image/png;base64,{original_base64}",
            "result_image": f"data:image/png;base64,{result_base64}",
            "foreground_percentage": foreground_percentage,
            "message": f"배경 제거 완료 (인물 영역: {foreground_percentage}%)"
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"처리 중 오류 발생: {str(e)}"
        }, status_code=500)

# ===================== 프롬프트 생성 헬퍼 함수 =====================

def preprocess_dress_image(dress_img: Image.Image, target_size: int = 1024) -> Image.Image:
    """
    드레스 이미지를 전처리하여 배경 정보를 제거하고 중앙 정렬합니다.
    
    Args:
        dress_img: 원본 드레스 이미지 (PIL Image)
        target_size: 출력 이미지 크기 (정사각형)
    
    Returns:
        전처리된 드레스 이미지 (흰색 배경에 중앙 정렬)
    """
    # RGB로 변환 (투명도 채널이 있을 경우를 대비)
    if dress_img.mode == 'RGBA':
        # 알파 채널을 사용하여 드레스 영역 감지
        alpha = dress_img.split()[3]
        bbox = alpha.getbbox()  # 투명하지 않은 영역의 경계 상자
        
        if bbox:
            # 드레스 영역만 크롭
            dress_cropped = dress_img.crop(bbox)
        else:
            dress_cropped = dress_img
    else:
        dress_cropped = dress_img
    
    # 드레스 이미지 크기 조정 (비율 유지) - 더 크게 표시
    dress_cropped.thumbnail((target_size * 0.95, target_size * 0.95), Image.Resampling.LANCZOS)
    
    # 흰색 배경 생성
    white_bg = Image.new('RGB', (target_size, target_size), (255, 255, 255))
    
    # 드레스를 중앙에 배치
    dress_rgb = dress_cropped.convert('RGB')
    offset_x = (target_size - dress_rgb.width) // 2
    offset_y = (target_size - dress_rgb.height) // 2
    
    # RGBA 모드인 경우 알파 채널을 마스크로 사용
    if dress_cropped.mode == 'RGBA':
        white_bg.paste(dress_rgb, (offset_x, offset_y), dress_cropped.split()[3])
    else:
        white_bg.paste(dress_rgb, (offset_x, offset_y))
    
    return white_bg

async def generate_custom_prompt_from_images(person_img: Image.Image, dress_img: Image.Image, api_key: str) -> Optional[str]:
    """
    이미지를 분석하여 맞춤 프롬프트를 생성합니다.
    
    Args:
        person_img: 사람 이미지 (PIL Image)
        dress_img: 드레스 이미지 (PIL Image)
        api_key: Gemini API 키
    
    Returns:
        생성된 맞춤 프롬프트 문자열 또는 None
    """
    try:
        print("🔍 이미지 분석 시작...")
        client = genai.Client(api_key=api_key)
        
        analysis_prompt = """Analyze these two images carefully:

Image 1 (Person): A woman in her current outfit
Image 2 (Dress): A formal dress/gown

Your task: Create a detailed instruction for virtual try-on that will dress the woman from Image 1 in the dress from Image 2.

First, describe what you see:
1. In Image 1 - What clothing is the woman wearing? (be specific: tops, bottoms, shoes, sleeves)
2. In Image 2 - What does the dress look like? (color, style, length, neckline, sleeves or sleeveless)

Then, create a prompt with these requirements:

CRITICAL - SKIN EXPOSURE RULES:
- Compare the clothing coverage in Image 1 vs Image 2
- If Image 1 has long sleeves but Image 2 dress is sleeveless → Generate natural bare arms with skin
- If Image 1 has pants/jeans but Image 2 dress is short → Generate natural bare legs with skin
- If Image 1 covers shoulders but Image 2 dress is strapless → Generate natural bare shoulders with skin
- Any body part that will be EXPOSED by the new dress MUST show natural skin, NOT the original clothing
- Example: Woman in long-sleeve shirt wearing sleeveless dress = bare arms visible
- Example: Woman in jeans wearing short dress = bare legs visible

OTHER REQUIREMENTS:
- Remove ALL clothing items from Image 1 that you identified
- Apply the dress from Image 2 onto the woman (exact color, style, design)
- Replace footwear with elegant heels matching the dress color
- Keep the woman's face, hair, body shape, and pose from Image 1
- Use white background
- Full body visible from head to toe

Output ONLY the final prompt instructions, nothing else. Start with "Create an image of the woman from Image 1 wearing the dress from Image 2." and continue with specific details based on what you observed, including skin exposure instructions."""
        
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[person_img, dress_img, analysis_prompt]
        )
        
        # 생성된 프롬프트 추출
        custom_prompt = ""
        if response.candidates and len(response.candidates) > 0:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    custom_prompt += part.text
        
        if custom_prompt:
            print(f"✅ 맞춤 프롬프트 생성 완료 (길이: {len(custom_prompt)}자)")
            return custom_prompt
        else:
            print("⚠️ 프롬프트 생성 실패")
            return None
            
    except Exception as e:
        print(f"❌ 프롬프트 생성 중 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

@app.post("/api/generate-prompt", tags=["프롬프트 생성"])
async def generate_prompt(
    person_image: UploadFile = File(..., description="사람 이미지 파일"),
    dress_image: Optional[UploadFile] = File(None, description="드레스 이미지 파일"),
    dress_url: Optional[str] = Form(None, description="드레스 이미지 URL (S3 또는 로컬)")
):
    """
    이미지를 분석하여 맞춤 프롬프트만 생성합니다.
    
    사용자가 프롬프트를 확인한 후 compose-dress API를 호출할 수 있습니다.
    
    Args:
        person_image: 사람 이미지 파일
        dress_image: 드레스 이미지 파일
        dress_url: 드레스 이미지 URL
    
    Returns:
        JSONResponse: 생성된 프롬프트
    """
    try:
        # API 키 확인
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return JSONResponse({
                "success": False,
                "error": "API key not found",
                "message": ".env 파일에 GEMINI_API_KEY가 설정되지 않았습니다."
            }, status_code=500)
        
        # 사람 이미지 읽기
        person_contents = await person_image.read()
        person_img = Image.open(io.BytesIO(person_contents))
        
        # 드레스 이미지 처리
        dress_img = None
        if dress_image:
            dress_contents = await dress_image.read()
            dress_img = Image.open(io.BytesIO(dress_contents))
        elif dress_url:
            try:
                if not dress_url.startswith('http'):
                    return JSONResponse({
                        "success": False,
                        "error": "Invalid dress URL",
                        "message": f"유효하지 않은 드레스 URL입니다."
                    }, status_code=400)
                
                parsed_url = urlparse(dress_url)
                aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
                aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
                region = os.getenv("AWS_REGION", "ap-northeast-2")
                
                if not all([aws_access_key, aws_secret_key]):
                    response = requests.get(dress_url, timeout=10)
                    response.raise_for_status()
                    dress_img = Image.open(io.BytesIO(response.content))
                else:
                    s3_client = boto3.client(
                        's3',
                        aws_access_key_id=aws_access_key,
                        aws_secret_access_key=aws_secret_key,
                        region_name=region
                    )
                    
                    if '.s3.' in parsed_url.netloc or '.s3-' in parsed_url.netloc:
                        bucket_name = parsed_url.netloc.split('.')[0]
                        s3_key = parsed_url.path.lstrip('/')
                    else:
                        path_parts = parsed_url.path.lstrip('/').split('/', 1)
                        if len(path_parts) == 2:
                            bucket_name, s3_key = path_parts
                        else:
                            raise ValueError(f"S3 URL 형식을 파싱할 수 없습니다.")
                    
                    s3_response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
                    image_data = s3_response['Body'].read()
                    dress_img = Image.open(io.BytesIO(image_data))
                    
            except Exception as e:
                print(f"드레스 이미지 다운로드 오류: {e}")
                return JSONResponse({
                    "success": False,
                    "error": "Image download failed",
                    "message": f"드레스 이미지를 다운로드할 수 없습니다: {str(e)}"
                }, status_code=400)
        else:
            return JSONResponse({
                "success": False,
                "error": "No dress image provided",
                "message": "드레스 이미지 파일 또는 URL이 필요합니다."
            }, status_code=400)
        
        # 드레스 이미지 전처리
        print("드레스 이미지 전처리 시작...")
        dress_img = preprocess_dress_image(dress_img, target_size=1024)
        print("드레스 이미지 전처리 완료")
        
        # 맞춤 프롬프트 생성
        print("\n" + "="*80)
        print("🔍 이미지 분석 및 프롬프트 생성")
        print("="*80)
        
        custom_prompt = await generate_custom_prompt_from_images(person_img, dress_img, api_key)
        
        if custom_prompt:
            return JSONResponse({
                "success": True,
                "prompt": custom_prompt,
                "message": "프롬프트가 성공적으로 생성되었습니다."
            })
        else:
            # 기본 프롬프트 반환 (Image 1 = Person, Image 2 = Dress)
            default_prompt = """IMPORTANT: You must preserve the person's identity completely.

Task: Apply ONLY the dress from the second image onto the person from the first image.

STRICT REQUIREMENTS:
1. PRESERVE EXACTLY: The person's face, facial features, skin tone, hair, and body proportions from the first image
2. PRESERVE EXACTLY: The person's pose, stance, and body position from the first image
3. PRESERVE EXACTLY: The background and lighting from the person's image (first image)
4. CHANGE ONLY: Replace the person's clothing with the dress from the second image
5. The dress should fit naturally on the person's body shape
6. Maintain realistic shadows and fabric draping on the dress
7. Keep the person's hands, arms, legs exactly as they are in the original (first image)

CRITICAL - SKIN EXPOSURE RULES:
- If the person in the first image wears long sleeves but the dress in the second image is sleeveless → Generate natural bare arms with skin
- If the person in the first image wears pants but the dress in the second image is short → Generate natural bare legs with skin
- If the person in the first image covers shoulders but the dress in the second image is strapless → Generate natural bare shoulders with skin
- Any body part that will be EXPOSED by the new dress MUST show natural skin tone, NOT the original clothing
- Example: Woman in long-sleeve shirt wearing sleeveless dress = bare arms visible with natural skin
- Example: Woman in jeans wearing short dress = bare legs visible with natural skin

MANDATORY FOOTWEAR CHANGE:
- Replace footwear with elegant high heels or formal dress shoes matching the dress color
- NEVER keep sneakers or casual footwear from the first image

DO NOT change the person's appearance, face, body type, or any physical features from the first image.
ONLY apply the dress design, color, and style from the second image onto the existing person."""
            
            return JSONResponse({
                "success": True,
                "prompt": default_prompt,
                "message": "맞춤 프롬프트 생성 실패. 기본 프롬프트를 사용하세요.",
                "is_default": True
            })
            
    except Exception as e:
        print(f"❌ 프롬프트 생성 API 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"프롬프트 생성 중 오류 발생: {str(e)}"
        }, status_code=500)

@app.post("/api/compose-dress", tags=["Gemini 이미지 합성"])
async def compose_dress(
    person_image: UploadFile = File(..., description="사람 이미지 파일"),
    dress_image: Optional[UploadFile] = File(None, description="드레스 이미지 파일"),
    dress_url: Optional[str] = Form(None, description="드레스 이미지 URL (S3 또는 로컬)"),
    model_name: Optional[str] = Form(None, description="모델명"),
    prompt: Optional[str] = Form(None, description="AI 명령어 (프롬프트)")
):
    """
    Gemini API를 사용한 사람과 드레스 이미지 합성
    
    사람 이미지와 드레스 이미지를 받아서 Gemini API를 통해
    사람이 드레스를 입은 것처럼 합성된 이미지를 생성합니다.
    
    Args:
        person_image: 사람 이미지 파일
        dress_image: 드레스 이미지 파일
        model_name: 사용된 모델명 (선택사항, 기본값: "gemini-compose")
        prompt: AI 명령어 (선택사항, 기본 프롬프트 사용)
    
    Returns:
        JSONResponse: 합성된 이미지 (base64)
    """
    person_image_path = None
    dress_image_path = None
    result_image_path = None
    start_time = time.time()
    model_id = model_name or "gemini-compose"
    
    # 기본 프롬프트
    default_prompt = """IMPORTANT: You must preserve the person's identity completely.

Task: Apply ONLY the dress from the second image onto the person from the first image.

STRICT REQUIREMENTS:
1. PRESERVE EXACTLY: The person's face, facial features, skin tone, hair, and body proportions from the first image
2. PRESERVE EXACTLY: The person's pose, stance, and body position from the first image
3. PRESERVE EXACTLY: The background and lighting from the person's image (first image)
4. CHANGE ONLY: Replace the person's clothing with the dress from the second image
5. The dress should fit naturally on the person's body shape
6. Maintain realistic shadows and fabric draping on the dress
7. Keep the person's hands, arms, legs exactly as they are in the original (first image)

DO NOT change the person's appearance, face, body type, or any physical features from the first image.
ONLY apply the dress design, color, and style from the second image onto the existing person."""
    
    text_input = prompt or default_prompt
    used_prompt = prompt or default_prompt
    success = False
    person_s3_url = ""
    dress_s3_url = ""
    result_s3_url = ""
    
    try:
        # .env에서 API 키 가져오기
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            error_msg = ".env 파일에 GEMINI_API_KEY가 설정되지 않았습니다."
            return JSONResponse({
                "success": False,
                "error": "API key not found",
                "message": error_msg
            }, status_code=500)
        
        # 사람 이미지 읽기
        person_contents = await person_image.read()
        person_img = Image.open(io.BytesIO(person_contents))
        
        # 드레스 이미지 처리: 파일 또는 URL
        dress_img = None
        if dress_image:
            # 파일로 업로드된 경우
            dress_contents = await dress_image.read()
            dress_img = Image.open(io.BytesIO(dress_contents))
        elif dress_url:
            # S3 URL에서 드레스 이미지 다운로드 (AWS 자격 증명 사용)
            try:
                if not dress_url.startswith('http'):
                    return JSONResponse({
                        "success": False,
                        "error": "Invalid dress URL",
                        "message": f"유효하지 않은 드레스 URL입니다. HTTP(S) URL이 필요합니다: {dress_url}"
                    }, status_code=400)
                
                # S3 URL 파싱하여 bucket과 key 추출
                # URL 형식: https://bucket.s3.region.amazonaws.com/key 또는 https://s3.region.amazonaws.com/bucket/key
                parsed_url = urlparse(dress_url)
                
                # AWS S3 클라이언트 생성
                aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
                aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
                region = os.getenv("AWS_REGION", "ap-northeast-2")
                
                if not all([aws_access_key, aws_secret_key]):
                    # AWS 자격 증명이 없으면 일반 HTTP 요청 시도 (퍼블릭 버킷용)
                    print(f"AWS 자격 증명 없음, HTTP 요청 시도: {dress_url}")
                    response = requests.get(dress_url, timeout=10)
                    response.raise_for_status()
                    dress_img = Image.open(io.BytesIO(response.content))
                else:
                    # boto3를 사용하여 S3에서 다운로드
                    s3_client = boto3.client(
                        's3',
                        aws_access_key_id=aws_access_key,
                        aws_secret_access_key=aws_secret_key,
                        region_name=region
                    )
                    
                    # URL에서 bucket과 key 추출
                    # 형식 1: https://bucket.s3.region.amazonaws.com/key
                    # 형식 2: https://s3.region.amazonaws.com/bucket/key
                    if '.s3.' in parsed_url.netloc or '.s3-' in parsed_url.netloc:
                        # bucket.s3.region.amazonaws.com 형식
                        bucket_name = parsed_url.netloc.split('.')[0]
                        s3_key = parsed_url.path.lstrip('/')
                    else:
                        # 다른 형식 - path에서 bucket/key 추출
                        path_parts = parsed_url.path.lstrip('/').split('/', 1)
                        if len(path_parts) == 2:
                            bucket_name, s3_key = path_parts
                        else:
                            raise ValueError(f"S3 URL 형식을 파싱할 수 없습니다: {dress_url}")
                    
                    print(f"S3 다운로드 시도: bucket={bucket_name}, key={s3_key}")
                    
                    # S3에서 객체 가져오기
                    s3_response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
                    image_data = s3_response['Body'].read()
                    dress_img = Image.open(io.BytesIO(image_data))
                    print(f"S3 URL에서 드레스 이미지 다운로드 성공: {dress_url}")
                
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                print(f"S3 ClientError ({error_code}): {e}")
                return JSONResponse({
                    "success": False,
                    "error": f"S3 access denied ({error_code})",
                    "message": f"S3 접근 권한 오류: {str(e)}"
                }, status_code=400)
            except requests.exceptions.RequestException as e:
                print(f"S3 HTTP 다운로드 오류: {e}")
                return JSONResponse({
                    "success": False,
                    "error": "S3 download failed",
                    "message": f"S3에서 드레스 이미지를 다운로드할 수 없습니다: {str(e)}"
                }, status_code=400)
            except Exception as e:
                print(f"드레스 이미지 처리 오류: {e}")
                import traceback
                traceback.print_exc()
                return JSONResponse({
                    "success": False,
                    "error": "Image processing failed",
                    "message": f"드레스 이미지를 처리할 수 없습니다: {str(e)}"
                }, status_code=400)
        else:
            return JSONResponse({
                "success": False,
                "error": "No dress image provided",
                "message": "드레스 이미지 파일 또는 URL이 필요합니다."
            }, status_code=400)
        
        # 입력 이미지들을 파일 시스템에 저장
        person_image_path = save_uploaded_image(person_img, "person")
        dress_image_path = save_uploaded_image(dress_img, "dress")
        
        # S3에 입력 이미지 업로드
        person_buffered = io.BytesIO()
        person_img.save(person_buffered, format="PNG")
        person_s3_url = upload_log_to_s3(person_buffered.getvalue(), model_id, "person") or ""
        
        dress_buffered = io.BytesIO()
        dress_img.save(dress_buffered, format="PNG")
        dress_s3_url = upload_log_to_s3(dress_buffered.getvalue(), model_id, "dress") or ""
        
        # 원본 이미지들을 base64로 변환
        person_base64 = base64.b64encode(person_buffered.getvalue()).decode()
        dress_base64 = base64.b64encode(dress_buffered.getvalue()).decode()
        
        # Gemini Client 생성 (공식 문서와 동일한 방식)
        client = genai.Client(api_key=api_key)
        
        # Gemini API 호출 (person, dress, text 순서)
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[person_img, dress_img, text_input]
        )
        
        # 응답 확인
        if not response.candidates or len(response.candidates) == 0:
            error_msg = "Gemini API가 응답을 생성하지 못했습니다. 이미지가 안전 정책에 위배되거나 모델이 이미지를 생성할 수 없습니다."
            run_time = time.time() - start_time
            
            # 실패 로그 저장
            save_test_log(
                person_url=person_s3_url or "",
                dress_url=dress_s3_url or None,
                result_url="",
                model=model_id,
                prompt=used_prompt,
                success=False,
                run_time=run_time
            )
            
            return JSONResponse({
                "success": False,
                "error": "No response",
                "message": error_msg
            }, status_code=500)
        
        # 응답에서 이미지 추출 (예시 코드와 동일한 방식)
        image_parts = [
            part.inline_data.data
            for part in response.candidates[0].content.parts
            if hasattr(part, 'inline_data') and part.inline_data
        ]
        
        # 텍스트 응답도 추출
        result_text = ""
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'text') and part.text:
                result_text += part.text
        
        if image_parts:
            # 첫 번째 이미지를 base64로 인코딩
            result_image_base64 = base64.b64encode(image_parts[0]).decode()
            
            # 결과 이미지를 파일 시스템에 저장
            result_img = Image.open(io.BytesIO(image_parts[0]))
            result_image_path = save_uploaded_image(result_img, "result")
            
            # S3에 결과 이미지 업로드
            result_buffered = io.BytesIO()
            result_img.save(result_buffered, format="PNG")
            result_s3_url = upload_log_to_s3(result_buffered.getvalue(), model_id, "result") or ""
            
            success = True
            run_time = time.time() - start_time
            
            # 성공 로그 저장
            save_test_log(
                person_url=person_s3_url or "",
                dress_url=dress_s3_url or None,
                result_url=result_s3_url or "",
                model=model_id,
                prompt=used_prompt,
                success=True,
                run_time=run_time
            )
            
            return JSONResponse({
                "success": True,
                "person_image": f"data:image/png;base64,{person_base64}",
                "dress_image": f"data:image/png;base64,{dress_base64}",
                "result_image": f"data:image/png;base64,{result_image_base64}",
                "message": "이미지 합성이 완료되었습니다.",
                "gemini_response": result_text
            })
        else:
            run_time = time.time() - start_time
            
            # 실패 로그 저장
            save_test_log(
                person_url=person_s3_url or "",
                dress_url=dress_s3_url or None,
                result_url="",
                model=model_id,
                prompt=used_prompt,
                success=False,
                run_time=run_time
            )
            
            return JSONResponse({
                "success": False,
                "error": "No image generated",
                "message": "Gemini API가 이미지를 생성하지 못했습니다. 응답: " + result_text,
                "gemini_response": result_text
            }, status_code=500)
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        run_time = time.time() - start_time
        
        # 오류 로그 저장
        try:
            save_test_log(
                person_url=person_s3_url or "",
                dress_url=dress_s3_url or None,
                result_url=result_s3_url or "",
                model=model_id,
                prompt=used_prompt,
                success=False,
                run_time=run_time
            )
        except:
            pass  # 로그 저장 실패해도 계속 진행
        
        return JSONResponse({
            "success": False,
            "error": str(e),
            "error_detail": error_detail,
            "message": f"이미지 합성 중 오류 발생: {str(e)}"
        }, status_code=500)

@app.get("/gemini-test", response_class=HTMLResponse, tags=["Web Interface"])
async def gemini_test_page(request: Request):
    """
    Gemini 이미지 합성 테스트 페이지
    
    사람 이미지와 드레스 이미지를 업로드하여 합성 결과를 테스트할 수 있는 페이지
    """
    return templates.TemplateResponse("gemini_test.html", {"request": request})

# ===================== S3 업로드 함수 =====================

def upload_to_s3(file_content: bytes, file_name: str, content_type: str = "image/png", folder: str = "dresses") -> Optional[str]:
    """
    S3에 파일 업로드
    
    Args:
        file_content: 파일 내용 (bytes)
        file_name: 파일명
        content_type: MIME 타입
        folder: S3 폴더 경로 (기본값: "dresses")
    
    Returns:
        S3 URL 또는 None (실패 시)
    """
    try:
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
        region = os.getenv("AWS_REGION", "ap-northeast-2")
        
        if not all([aws_access_key, aws_secret_key, bucket_name]):
            print("AWS S3 설정이 완료되지 않았습니다.")
            return None
        
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )
        
        # S3에 업로드
        s3_key = f"{folder}/{file_name}"
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=file_content,
            ContentType=content_type
        )
        
        # S3 URL 생성
        s3_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{s3_key}"
        return s3_url
        
    except ClientError as e:
        print(f"S3 업로드 오류: {e}")
        return None
    except Exception as e:
        print(f"S3 업로드 중 예상치 못한 오류: {e}")
        return None

def upload_log_to_s3(file_content: bytes, model_id: str, image_type: str, content_type: str = "image/png") -> Optional[str]:
    """
    S3 logs 폴더에 테스트 이미지 업로드 (별도 S3 계정/버킷 사용)
    
    Args:
        file_content: 파일 내용 (bytes)
        model_id: 모델 ID
        image_type: 이미지 타입 (person, dress, result)
        content_type: MIME 타입
    
    Returns:
        S3 URL 또는 None (실패 시)
    """
    try:
        # 별도 S3 계정 환경변수 사용
        aws_access_key = os.getenv("LOGS_AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("LOGS_AWS_SECRET_ACCESS_KEY")
        bucket_name = os.getenv("LOGS_AWS_S3_BUCKET_NAME")
        region = os.getenv("LOGS_AWS_REGION", "ap-northeast-2")
        
        if not all([aws_access_key, aws_secret_key, bucket_name]):
            print("로그용 S3 설정이 완료되지 않았습니다. (LOGS_AWS_*)")
            return None
        
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )
        
        # 타임스탬프 기반 파일명 생성
        timestamp = int(time.time() * 1000)
        file_name = f"{timestamp}_{model_id}_{image_type}.png"
        s3_key = f"logs/{file_name}"
        
        # S3에 업로드
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=file_content,
            ContentType=content_type
        )
        
        # S3 URL 생성
        s3_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{s3_key}"
        return s3_url
        
    except ClientError as e:
        print(f"로그용 S3 업로드 오류: {e}")
        return None
    except Exception as e:
        print(f"로그용 S3 업로드 중 예상치 못한 오류: {e}")
        return None

def save_test_log(
    person_url: str,
    result_url: str,
    model: str,
    prompt: str,
    success: bool,
    run_time: float,
    dress_url: Optional[str] = None
) -> bool:
    """
    테스트 기록을 MySQL에 저장
    
    Args:
        person_url: 인물 이미지 S3 URL
        result_url: 결과 이미지 S3 URL
        model: 사용된 AI 모델명
        prompt: 사용된 AI 명령어
        success: 실행 성공 여부
        run_time: 실행 시간 (초)
        dress_url: 의상 이미지 S3 URL (선택사항)
    
    Returns:
        저장 성공 여부 (True/False)
    """
    connection = get_db_connection()
    if not connection:
        print("DB 연결 실패 - 테스트 로그 저장 건너뜀")
        return False
    
    try:
        with connection.cursor() as cursor:
            insert_query = """
            INSERT INTO result_logs (person_url, dress_url, result_url, model, prompt, success, run_time)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (
                person_url,
                dress_url,
                result_url,
                model,
                prompt,
                success,
                run_time
            ))
            connection.commit()
            print(f"테스트 로그 저장 완료: {model}")
            return True
    except Exception as e:
        print(f"테스트 로그 저장 오류: {e}")
        connection.rollback()
        return False
    finally:
        connection.close()

def delete_from_s3(file_name: str) -> bool:
    """
    S3에서 파일 삭제
    
    Args:
        file_name: 삭제할 파일명
    
    Returns:
        삭제 성공 여부 (True/False)
    """
    try:
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
        region = os.getenv("AWS_REGION", "ap-northeast-2")
        
        if not all([aws_access_key, aws_secret_key, bucket_name]):
            print("AWS S3 설정이 완료되지 않았습니다.")
            return False
        
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )
        
        # S3 키 생성 (업로드 시와 동일한 형식)
        s3_key = f"dresses/{file_name}"
        
        # S3에서 삭제
        s3_client.delete_object(
            Bucket=bucket_name,
            Key=s3_key
        )
        
        print(f"S3에서 이미지 삭제 완료: {s3_key}")
        return True
        
    except ClientError as e:
        print(f"S3 삭제 오류: {e}")
        return False
    except Exception as e:
        print(f"S3 삭제 중 예상치 못한 오류: {e}")
        return False

# ===================== 카테고리 규칙 API =====================

@app.get("/api/admin/category-rules", tags=["카테고리 규칙"])
async def get_category_rules():
    """
    카테고리 규칙 목록 조회
    """
    try:
        rules = load_category_rules()
        return JSONResponse({
            "success": True,
            "data": rules,
            "total": len(rules)
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"규칙 조회 중 오류 발생: {str(e)}"
        }, status_code=500)

@app.post("/api/admin/category-rules", tags=["카테고리 규칙"])
async def add_category_rule(request: Request):
    """
    새 카테고리 규칙 추가
    """
    try:
        body = await request.json()
        prefix = body.get("prefix")
        style = body.get("style")
        
        if not prefix or not style:
            return JSONResponse({
                "success": False,
                "error": "Missing required fields",
                "message": "prefix와 style은 필수 입력 항목입니다."
            }, status_code=400)
        
        rules = load_category_rules()
        
        # 중복 체크
        if any(rule["prefix"].upper() == prefix.upper() for rule in rules):
            return JSONResponse({
                "success": False,
                "error": "Duplicate prefix",
                "message": f"접두사 '{prefix}'가 이미 존재합니다."
            }, status_code=400)
        
        # 새 규칙 추가
        rules.append({"prefix": prefix, "style": style})
        
        if save_category_rules(rules):
            return JSONResponse({
                "success": True,
                "data": {"prefix": prefix, "style": style},
                "message": "카테고리 규칙이 추가되었습니다."
            })
        else:
            return JSONResponse({
                "success": False,
                "error": "Save failed",
                "message": "규칙 저장에 실패했습니다."
            }, status_code=500)
            
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"규칙 추가 중 오류 발생: {str(e)}"
        }, status_code=500)

@app.delete("/api/admin/category-rules", tags=["카테고리 규칙"])
async def delete_category_rule(request: Request):
    """
    카테고리 규칙 삭제
    """
    try:
        body = await request.json()
        prefix = body.get("prefix")
        
        if not prefix:
            return JSONResponse({
                "success": False,
                "error": "Missing prefix",
                "message": "삭제할 접두사를 입력해주세요."
            }, status_code=400)
        
        rules = load_category_rules()
        
        # 규칙 찾아서 삭제
        filtered_rules = [r for r in rules if r["prefix"].upper() != prefix.upper()]
        
        if len(filtered_rules) == len(rules):
            return JSONResponse({
                "success": False,
                "error": "Rule not found",
                "message": f"접두사 '{prefix}'에 해당하는 규칙을 찾을 수 없습니다."
            }, status_code=404)
        
        if save_category_rules(filtered_rules):
            return JSONResponse({
                "success": True,
                "message": f"접두사 '{prefix}' 규칙이 삭제되었습니다."
            })
        else:
            return JSONResponse({
                "success": False,
                "error": "Save failed",
                "message": "규칙 저장에 실패했습니다."
            }, status_code=500)
            
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"규칙 삭제 중 오류 발생: {str(e)}"
        }, status_code=500)

# ===================== 드레스 업로드 API =====================

@app.post("/api/admin/dresses/upload", tags=["드레스 관리"])
async def upload_dresses(
    files: List[UploadFile] = File(...),
    styles: str = Form(...)
):
    """
    여러 드레스 이미지를 업로드하고 S3에 저장
    
    Args:
        files: 업로드할 이미지 파일 리스트
        styles: 각 파일별 스타일 정보 (JSON 문자열, 예: '[{"file":"image1.png","style":"A라인"},...]')
    """
    try:
        # styles JSON 파싱
        styles_data = json.loads(styles)
        styles_dict = {item["file"]: item["style"] for item in styles_data}
        
        results = []
        success_count = 0
        fail_count = 0
        
        connection = get_db_connection()
        if not connection:
            return JSONResponse({
                "success": False,
                "error": "Database connection failed",
                "message": "데이터베이스 연결에 실패했습니다."
            }, status_code=500)
        
        try:
            for file in files:
                try:
                    # 파일 내용 읽기
                    file_content = await file.read()
                    file_name = file.filename
                    
                    # 파일명 처리
                    file_stem = Path(file_name).stem  # 확장자 제외
                    file_ext = Path(file_name).suffix  # 확장자
                    
                    # 스타일 가져오기 (수동 선택 또는 자동 감지)
                    style = styles_dict.get(file_name)
                    if not style:
                        # 자동 감지 시도
                        style = detect_style_from_filename(file_name)
                        if not style:
                            results.append({
                                "file_name": file_name,
                                "success": False,
                                "error": "스타일을 감지할 수 없습니다."
                            })
                            fail_count += 1
                            continue
                    
                    # S3 업로드
                    content_type = file.content_type or "image/png"
                    s3_url = upload_to_s3(file_content, file_name, content_type)
                    
                    if not s3_url:
                        results.append({
                            "file_name": file_name,
                            "success": False,
                            "error": "S3 업로드 실패"
                        })
                        fail_count += 1
                        continue
                    
                    # DB 저장
                    with connection.cursor() as cursor:
                        # dress_name 중복 체크
                        cursor.execute("SELECT idx FROM dresses WHERE dress_name = %s", (file_stem,))
                        if cursor.fetchone():
                            results.append({
                                "file_name": file_name,
                                "success": False,
                                "error": f"드레스명 '{file_stem}'이 이미 존재합니다. 같은 이름의 드레스는 추가할 수 없습니다."
                            })
                            fail_count += 1
                            continue
                        
                        # file_name 중복 체크
                        cursor.execute("SELECT idx FROM dresses WHERE file_name = %s", (file_name,))
                        if cursor.fetchone():
                            results.append({
                                "file_name": file_name,
                                "success": False,
                                "error": "이미 존재하는 파일명입니다."
                            })
                            fail_count += 1
                            continue
                        
                        # 삽입
                        try:
                            cursor.execute(
                                "INSERT INTO dresses (dress_name, file_name, style, url) VALUES (%s, %s, %s, %s)",
                                (file_stem, file_name, style, s3_url)
                            )
                            connection.commit()
                        except pymysql.IntegrityError as e:
                            # UNIQUE 제약 조건 위반 처리
                            if "dress_name" in str(e).lower() or "Duplicate entry" in str(e):
                                results.append({
                                    "file_name": file_name,
                                    "success": False,
                                    "error": f"드레스명 '{file_stem}'이 이미 존재합니다. 같은 이름의 드레스는 추가할 수 없습니다."
                                })
                                fail_count += 1
                                continue
                            raise
                        
                        results.append({
                            "file_name": file_name,
                            "dress_name": file_stem,
                            "style": style,
                            "url": s3_url,
                            "success": True
                        })
                        success_count += 1
                        
                except Exception as e:
                    results.append({
                        "file_name": file.filename if hasattr(file, 'filename') else "unknown",
                        "success": False,
                        "error": str(e)
                    })
                    fail_count += 1
            
            return JSONResponse({
                "success": True,
                "results": results,
                "summary": {
                    "total": len(files),
                    "success": success_count,
                    "failed": fail_count
                },
                "message": f"{success_count}개 이미지 업로드 완료, {fail_count}개 실패"
            })
            
        except Exception as e:
            connection.rollback()
            return JSONResponse({
                "success": False,
                "error": str(e),
                "message": f"업로드 중 오류 발생: {str(e)}"
            }, status_code=500)
        finally:
            connection.close()
            
    except json.JSONDecodeError:
        return JSONResponse({
            "success": False,
            "error": "Invalid JSON",
            "message": "styles 파라미터가 올바른 JSON 형식이 아닙니다."
        }, status_code=400)
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"업로드 중 오류 발생: {str(e)}"
        }, status_code=500)

# ===================== S3 이미지 프록시 =====================

def get_s3_image(file_name: str) -> Optional[bytes]:
    """
    S3에서 이미지 다운로드
    
    Args:
        file_name: 파일명 (예: "Adress1.JPG")
    
    Returns:
        이미지 바이트 데이터 또는 None (실패 시)
    """
    try:
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
        region = os.getenv("AWS_REGION", "ap-northeast-2")
        
        if not all([aws_access_key, aws_secret_key, bucket_name]):
            print("AWS S3 설정이 완료되지 않았습니다.")
            return None
        
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )
        
        # S3에서 파일 다운로드
        s3_key = f"dresses/{file_name}"
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            return response['Body'].read()
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                print(f"S3에 파일이 없습니다: {s3_key}")
            else:
                print(f"S3 다운로드 오류: {e}")
            return None
    except Exception as e:
        print(f"S3 이미지 다운로드 중 예상치 못한 오류: {e}")
        return None

@app.get("/api/proxy-image", tags=["이미지 프록시"])
async def proxy_image_by_url(url: str = Query(..., description="S3 이미지 URL")):
    """
    S3 URL로 이미지 프록시 (썸네일용)
    
    프론트엔드에서 S3 이미지를 직접 로드할 때 CORS 문제를 해결하기 위한 프록시
    
    Args:
        url: S3 이미지 전체 URL (예: https://bucket.s3.region.amazonaws.com/key)
    
    Returns:
        이미지 바이너리 데이터
    """
    try:
        if not url.startswith('http'):
            return Response(
                content="Invalid URL",
                status_code=400,
                media_type="text/plain",
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, OPTIONS",
                    "Access-Control-Allow-Headers": "*"
                }
            )
        
        # URL 파싱
        parsed_url = urlparse(url)
        
        # AWS S3 클라이언트 생성
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        region = os.getenv("AWS_REGION", "ap-northeast-2")
        
        if not all([aws_access_key, aws_secret_key]):
            # AWS 자격 증명이 없으면 일반 HTTP 요청 시도
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image_data = response.content
        else:
            # boto3를 사용하여 S3에서 다운로드
            s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=region
            )
            
            # URL에서 bucket과 key 추출
            if '.s3.' in parsed_url.netloc or '.s3-' in parsed_url.netloc:
                bucket_name = parsed_url.netloc.split('.')[0]
                s3_key = parsed_url.path.lstrip('/')
            else:
                path_parts = parsed_url.path.lstrip('/').split('/', 1)
                if len(path_parts) == 2:
                    bucket_name, s3_key = path_parts
                else:
                    raise ValueError(f"S3 URL 형식을 파싱할 수 없습니다: {url}")
            
            # S3에서 객체 가져오기
            s3_response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            image_data = s3_response['Body'].read()
        
        # 파일 확장자로 MIME 타입 결정
        file_ext = Path(url).suffix.lower()
        content_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        content_type = content_type_map.get(file_ext, 'image/jpeg')
        
        return Response(
            content=image_data,
            media_type=content_type,
            headers={
                "Cache-Control": "public, max-age=3600",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "*"
            }
        )
    except ClientError as e:
        print(f"S3 프록시 ClientError: {e}")
        return Response(
            content="S3 access error",
            status_code=403,
            media_type="text/plain",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "*"
            }
        )
    except Exception as e:
        print(f"이미지 프록시 오류: {e}")
        import traceback
        traceback.print_exc()
        return Response(
            content=f"Error: {str(e)}",
            status_code=500,
            media_type="text/plain",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "*"
            }
        )

@app.options("/api/images/{file_name:path}", tags=["이미지 프록시"])
async def proxy_s3_image_options(file_name: str):
    """
    CORS preflight 요청 처리
    """
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "3600"
        }
    )

@app.get("/api/images/{file_name:path}", tags=["이미지 프록시"])
async def proxy_s3_image(file_name: str):
    """
    S3 이미지를 프록시로 제공 (CORS 지원)
    
    프론트엔드에서 fetch로 이미지를 가져올 수 있도록 CORS 헤더를 포함합니다.
    모든 이미지는 S3에서 가져옵니다.
    
    Args:
        file_name: 이미지 파일명 (예: "Adress1.JPG")
    
    Returns:
        이미지 파일 또는 404 에러
    """
    try:
        # S3에서 이미지 가져오기
        image_data = get_s3_image(file_name)
        
        if not image_data:
            return Response(
                content="Image not found",
                status_code=404,
                media_type="text/plain",
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, OPTIONS",
                    "Access-Control-Allow-Headers": "*"
                }
            )
        
        # 파일 확장자로 MIME 타입 결정
        file_ext = Path(file_name).suffix.lower()
        content_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        content_type = content_type_map.get(file_ext, 'image/jpeg')
        
        return Response(
            content=image_data,
            media_type=content_type,
            headers={
                "Cache-Control": "public, max-age=3600",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "*"
            }
        )
    except Exception as e:
        print(f"이미지 프록시 오류: {e}")
        return Response(
            content=f"Error: {str(e)}",
            status_code=500,
            media_type="text/plain",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "*"
            }
        )

# ===================== 드레스 관리 API =====================

@app.get("/api/admin/dresses", tags=["드레스 관리"])
async def get_dresses(
    page: int = Query(1, ge=1, description="페이지 번호"),
    limit: int = Query(20, ge=1, le=10000, description="페이지당 항목 수")
):
    """
    드레스 목록 조회 (페이징 지원)
    
    데이터베이스에 저장된 드레스 정보를 페이지별로 반환합니다.
    
    Args:
        page: 페이지 번호 (기본값: 1)
        limit: 페이지당 항목 수 (기본값: 20, 최대: 10000)
    
    Returns:
        JSONResponse: 드레스 목록 및 페이지네이션 정보
    """
    try:
        connection = get_db_connection()
        if not connection:
            return JSONResponse({
                "success": False,
                "error": "Database connection failed",
                "message": "데이터베이스 연결에 실패했습니다."
            }, status_code=500)
        
        try:
            with connection.cursor() as cursor:
                # 전체 건수 조회
                cursor.execute("SELECT COUNT(*) as total FROM dresses")
                total = cursor.fetchone()['total']
                
                # 총 페이지 수 계산
                total_pages = (total + limit - 1) // limit if total > 0 else 0
                
                # 오프셋 계산
                offset = (page - 1) * limit
                
                # 페이징된 데이터 조회
                cursor.execute("""
                    SELECT idx as id, file_name as image_name, style, url
                    FROM dresses
                    ORDER BY idx DESC
                    LIMIT %s OFFSET %s
                """, (limit, offset))
                dresses = cursor.fetchall()
                
                return JSONResponse({
                    "success": True,
                    "data": dresses,
                    "pagination": {
                        "page": page,
                        "limit": limit,
                        "total": total,
                        "total_pages": total_pages
                    },
                    "message": f"{len(dresses)}개의 드레스를 찾았습니다."
                })
        finally:
            connection.close()
            
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"드레스 목록 조회 중 오류 발생: {str(e)}"
        }, status_code=500)

@app.post("/api/admin/dresses", tags=["드레스 관리"])
async def add_dress(request: Request):
    """
    드레스 추가 (S3 URL 또는 이미지명 입력)
    
    이미지명과 스타일, S3 URL을 받아서 데이터베이스에 추가합니다.
    모든 이미지는 S3에 저장되어 있어야 합니다.
    """
    try:
        body = await request.json()
        image_name = body.get("image_name")
        style = body.get("style")
        url = body.get("url")  # S3 URL
        
        if not image_name or not style:
            return JSONResponse({
                "success": False,
                "error": "Missing required fields",
                "message": "image_name과 style은 필수 입력 항목입니다."
            }, status_code=400)
        
        # 파일명에서 스타일 자동 감지 (검증용)
        detected_style = detect_style_from_filename(image_name)
        if detected_style and detected_style != style:
            return JSONResponse({
                "success": False,
                "error": "Style mismatch",
                "message": f"파일명에서 감지된 스타일({detected_style})과 입력한 스타일({style})이 일치하지 않습니다."
            }, status_code=400)
        
        connection = get_db_connection()
        if not connection:
            return JSONResponse({
                "success": False,
                "error": "Database connection failed",
                "message": "데이터베이스 연결에 실패했습니다."
            }, status_code=500)
        
        try:
            with connection.cursor() as cursor:
                # dress_name 추출 (확장자 제외)
                dress_name = Path(image_name).stem
                
                # dress_name 중복 체크
                cursor.execute("SELECT idx FROM dresses WHERE dress_name = %s", (dress_name,))
                if cursor.fetchone():
                    return JSONResponse({
                        "success": False,
                        "error": "Duplicate dress name",
                        "message": f"드레스명 '{dress_name}'이 이미 존재합니다. 같은 이름의 드레스는 추가할 수 없습니다."
                    }, status_code=400)
                
                # file_name 중복 체크
                cursor.execute("SELECT idx FROM dresses WHERE file_name = %s", (image_name,))
                if cursor.fetchone():
                    return JSONResponse({
                        "success": False,
                        "error": "Duplicate file name",
                        "message": f"이미지명 '{image_name}'이 이미 존재합니다."
                    }, status_code=400)
                
                # URL이 제공되지 않으면 기본 S3 URL 생성
                if not url:
                    bucket_name = os.getenv("AWS_S3_BUCKET_NAME", "marryday1")
                    region = os.getenv("AWS_REGION", "ap-northeast-2")
                    image_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/dresses/{image_name}"
                else:
                    image_url = url
                
                # 삽입
                try:
                    cursor.execute(
                        "INSERT INTO dresses (dress_name, file_name, style, url) VALUES (%s, %s, %s, %s)",
                        (dress_name, image_name, style, image_url)
                    )
                    connection.commit()
                except pymysql.IntegrityError as e:
                    # UNIQUE 제약 조건 위반 처리
                    if "dress_name" in str(e).lower() or "Duplicate entry" in str(e):
                        return JSONResponse({
                            "success": False,
                            "error": "Duplicate dress name",
                            "message": f"드레스명 '{dress_name}'이 이미 존재합니다. 같은 이름의 드레스는 추가할 수 없습니다."
                        }, status_code=400)
                    raise
                
                return JSONResponse({
                    "success": True,
                    "message": f"드레스 '{image_name}'가 성공적으로 추가되었습니다.",
                    "data": {
                        "image_name": image_name,
                        "style": style,
                        "dress_name": dress_name
                    }
                })
        finally:
            connection.close()
            
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"드레스 추가 중 오류 발생: {str(e)}"
        }, status_code=500)

@app.delete("/api/admin/dresses/{dress_id}", tags=["드레스 관리"])
async def delete_dress(dress_id: int):
    """
    드레스 삭제
    
    S3의 이미지와 데이터베이스의 레코드를 모두 삭제합니다.
    """
    try:
        connection = get_db_connection()
        if not connection:
            return JSONResponse({
                "success": False,
                "error": "Database connection failed",
                "message": "데이터베이스 연결에 실패했습니다."
            }, status_code=500)
        
        try:
            with connection.cursor() as cursor:
                # 드레스 정보 조회
                cursor.execute("SELECT file_name, url FROM dresses WHERE idx = %s", (dress_id,))
                dress = cursor.fetchone()
                
                if not dress:
                    return JSONResponse({
                        "success": False,
                        "error": "Dress not found",
                        "message": f"드레스 ID {dress_id}를 찾을 수 없습니다."
                    }, status_code=404)
                
                file_name = dress['file_name']
                url = dress['url']
                
                # S3에서 이미지 삭제 시도 (실패해도 계속 진행)
                s3_deleted = False
                if url and url.startswith('https://'):
                    # S3 URL인 경우 삭제 시도
                    s3_deleted = delete_from_s3(file_name)
                
                # 데이터베이스에서 삭제
                cursor.execute("DELETE FROM dresses WHERE idx = %s", (dress_id,))
                connection.commit()
                
                return JSONResponse({
                    "success": True,
                    "message": f"드레스 '{file_name}'가 성공적으로 삭제되었습니다.",
                    "data": {
                        "dress_id": dress_id,
                        "file_name": file_name,
                        "image_deleted": s3_deleted
                    }
                })
        finally:
            connection.close()
            
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"드레스 삭제 중 오류 발생: {str(e)}"
        }, status_code=500)

@app.get("/api/admin/dresses/export", tags=["드레스 관리"])
async def export_dresses(format: str = Query("json", description="내보내기 형식 (json, csv)")):
    """
    드레스 테이블 정보 내보내기
    
    Args:
        format: 내보내기 형식 (json 또는 csv)
    
    Returns:
        CSV 또는 JSON 형식의 파일 다운로드
    """
    try:
        connection = get_db_connection()
        if not connection:
            return JSONResponse({
                "success": False,
                "error": "Database connection failed",
                "message": "데이터베이스 연결에 실패했습니다."
            }, status_code=500)
        
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT idx as id, dress_name, file_name, style, url
                    FROM dresses
                    ORDER BY idx DESC
                """)
                dresses = cursor.fetchall()
                
                if format.lower() == "csv":
                    # CSV 형식으로 내보내기
                    output = io.StringIO()
                    writer = csv.DictWriter(output, fieldnames=["id", "dress_name", "file_name", "style", "url"])
                    writer.writeheader()
                    writer.writerows(dresses)
                    
                    csv_content = output.getvalue()
                    
                    return Response(
                        content=csv_content,
                        media_type="text/csv; charset=utf-8",
                        headers={
                            "Content-Disposition": f"attachment; filename=dresses_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        }
                    )
                else:
                    # JSON 형식으로 내보내기
                    json_content = json.dumps(dresses, ensure_ascii=False, indent=2)
                    
                    return Response(
                        content=json_content,
                        media_type="application/json; charset=utf-8",
                        headers={
                            "Content-Disposition": f"attachment; filename=dresses_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        }
                    )
        finally:
            connection.close()
            
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"데이터 내보내기 중 오류 발생: {str(e)}"
        }, status_code=500)

@app.post("/api/admin/dresses/import", tags=["드레스 관리"])
async def import_dresses(file: UploadFile = File(...)):
    """
    드레스 테이블 정보 가져오기
    
    Args:
        file: 업로드할 JSON 또는 CSV 파일
    
    Returns:
        가져오기 결과 (성공/실패 개수)
    """
    try:
        # 파일 내용 읽기
        file_content = await file.read()
        file_name = file.filename.lower()
        
        # 파일 확장자 확인
        if file_name.endswith('.json'):
            # JSON 파싱
            try:
                data = json.loads(file_content.decode('utf-8'))
            except json.JSONDecodeError as e:
                return JSONResponse({
                    "success": False,
                    "error": "Invalid JSON",
                    "message": f"JSON 파싱 오류: {str(e)}"
                }, status_code=400)
        elif file_name.endswith('.csv'):
            # CSV 파싱
            try:
                csv_content = file_content.decode('utf-8')
                csv_reader = csv.DictReader(io.StringIO(csv_content))
                data = list(csv_reader)
            except Exception as e:
                return JSONResponse({
                    "success": False,
                    "error": "Invalid CSV",
                    "message": f"CSV 파싱 오류: {str(e)}"
                }, status_code=400)
        else:
            return JSONResponse({
                "success": False,
                "error": "Unsupported file type",
                "message": "지원하는 파일 형식은 JSON 또는 CSV입니다."
            }, status_code=400)
        
        if not data:
            return JSONResponse({
                "success": False,
                "error": "Empty file",
                "message": "파일이 비어있습니다."
            }, status_code=400)
        
        connection = get_db_connection()
        if not connection:
            return JSONResponse({
                "success": False,
                "error": "Database connection failed",
                "message": "데이터베이스 연결에 실패했습니다."
            }, status_code=500)
        
        success_count = 0
        fail_count = 0
        results = []
        
        try:
            with connection.cursor() as cursor:
                for row in data:
                    try:
                        # 데이터 추출 (id는 무시, 자동 증가)
                        dress_name = row.get('dress_name') or row.get('dressName')
                        file_name = row.get('file_name') or row.get('fileName')
                        style = row.get('style')
                        url = row.get('url')
                        
                        # 필수 필드 확인
                        if not all([dress_name, file_name, style]):
                            results.append({
                                "row": row,
                                "success": False,
                                "error": "필수 필드가 누락되었습니다 (dress_name, file_name, style 필요)"
                            })
                            fail_count += 1
                            continue
                        
                        # dress_name 중복 체크
                        cursor.execute("SELECT idx FROM dresses WHERE dress_name = %s", (dress_name,))
                        if cursor.fetchone():
                            results.append({
                                "row": row,
                                "success": False,
                                "error": f"드레스명 '{dress_name}'이 이미 존재합니다."
                            })
                            fail_count += 1
                            continue
                        
                        # file_name 중복 체크
                        cursor.execute("SELECT idx FROM dresses WHERE file_name = %s", (file_name,))
                        if cursor.fetchone():
                            results.append({
                                "row": row,
                                "success": False,
                                "error": f"파일명 '{file_name}'이 이미 존재합니다."
                            })
                            fail_count += 1
                            continue
                        
                        # URL이 없으면 기본 S3 URL 생성
                        if not url:
                            bucket_name = os.getenv("AWS_S3_BUCKET_NAME", "marryday1")
                            region = os.getenv("AWS_REGION", "ap-northeast-2")
                            url = f"https://{bucket_name}.s3.{region}.amazonaws.com/dresses/{file_name}"
                        
                        # 삽입
                        try:
                            cursor.execute(
                                "INSERT INTO dresses (dress_name, file_name, style, url) VALUES (%s, %s, %s, %s)",
                                (dress_name, file_name, style, url)
                            )
                            connection.commit()
                            
                            results.append({
                                "row": row,
                                "success": True,
                                "dress_name": dress_name
                            })
                            success_count += 1
                        except pymysql.IntegrityError as e:
                            # UNIQUE 제약 조건 위반 처리
                            if "dress_name" in str(e).lower() or "Duplicate entry" in str(e):
                                results.append({
                                    "row": row,
                                    "success": False,
                                    "error": f"드레스명 '{dress_name}'이 이미 존재합니다."
                                })
                                fail_count += 1
                                continue
                            raise
                    except Exception as e:
                        results.append({
                            "row": row,
                            "success": False,
                            "error": str(e)
                        })
                        fail_count += 1
                        continue
            
            return JSONResponse({
                "success": True,
                "summary": {
                    "total": len(data),
                    "success": success_count,
                    "failed": fail_count
                },
                "results": results,
                "message": f"{success_count}개 항목 추가 완료, {fail_count}개 실패"
            })
        finally:
            connection.close()
            
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"데이터 가져오기 중 오류 발생: {str(e)}"
        }, status_code=500)

@app.get("/admin/dress-insert", response_class=HTMLResponse, tags=["관리자"])
async def dress_insert_page(request: Request):
    """
    드레스 이미지 삽입 관리자 페이지
    """
    return templates.TemplateResponse("dress_insert.html", {"request": request})

# ===================== 관리자 로그 API =====================

@app.get("/api/admin/stats", tags=["관리자"])
async def get_admin_stats():
    """
    관리자 통계 정보 조회
    
    result_logs 테이블에서 통계 정보를 조회합니다.
    
    Returns:
        JSONResponse: 전체, 성공, 실패, 성공률, 평균 처리 시간, 오늘 건수
    """
    try:
        connection = get_db_connection()
        if not connection:
            return JSONResponse({
                "success": False,
                "error": "Database connection failed",
                "message": "데이터베이스 연결에 실패했습니다."
            }, status_code=500)
        
        try:
            with connection.cursor() as cursor:
                # 전체 건수
                cursor.execute("SELECT COUNT(*) as total FROM result_logs")
                total = cursor.fetchone()['total']
                
                # 성공 건수
                cursor.execute("SELECT COUNT(*) as success FROM result_logs WHERE success = TRUE")
                success = cursor.fetchone()['success']
                
                # 실패 건수
                cursor.execute("SELECT COUNT(*) as failed FROM result_logs WHERE success = FALSE")
                failed = cursor.fetchone()['failed']
                
                # 평균 처리 시간
                cursor.execute("SELECT AVG(run_time) as avg_time FROM result_logs")
                avg_time_result = cursor.fetchone()
                avg_time = avg_time_result['avg_time'] if avg_time_result['avg_time'] else 0.0
                
                # 오늘 건수 (created_at 필드가 있으면 사용, 없으면 전체 건수로 대체)
                today = 0
                try:
                    cursor.execute("""
                        SELECT COUNT(*) as today 
                        FROM result_logs 
                        WHERE DATE(created_at) = CURDATE()
                    """)
                    today = cursor.fetchone()['today']
                except Exception as e:
                    # created_at 필드가 없으면 오늘 건수를 0으로 설정
                    print(f"created_at 필드 없음, 오늘 건수 조회 건너뜀: {e}")
                    today = 0
                
                # 성공률 계산
                success_rate = round((success / total * 100), 2) if total > 0 else 0.0
                
                return JSONResponse({
                    "success": True,
                    "data": {
                        "total": total,
                        "success": success,
                        "failed": failed,
                        "success_rate": success_rate,
                        "average_processing_time": round(avg_time, 2),
                        "today": today
                    }
                })
        finally:
            connection.close()
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"통계 조회 오류: {error_detail}")
        return JSONResponse({
            "success": False,
            "error": str(e),
            "error_detail": error_detail,
            "message": f"통계 조회 중 오류 발생: {str(e)}"
        }, status_code=500)

@app.get("/api/admin/logs", tags=["관리자"])
async def get_admin_logs(
    page: int = Query(1, ge=1, description="페이지 번호"),
    limit: int = Query(20, ge=1, le=100, description="페이지당 항목 수"),
    model: Optional[str] = Query(None, description="모델명으로 검색")
):
    """
    관리자 로그 목록 조회
    
    result_logs 테이블에서 로그 목록을 조회합니다.
    
    Args:
        page: 페이지 번호 (기본값: 1)
        limit: 페이지당 항목 수 (기본값: 20, 최대: 100)
        model: 모델명으로 검색 (선택사항)
    
    Returns:
        JSONResponse: 로그 목록 및 페이지네이션 정보
    """
    try:
        connection = get_db_connection()
        if not connection:
            return JSONResponse({
                "success": False,
                "error": "Database connection failed",
                "message": "데이터베이스 연결에 실패했습니다."
            }, status_code=500)
        
        try:
            with connection.cursor() as cursor:
                # 검색 조건에 따른 WHERE 절 생성
                where_clause = ""
                params = []
                
                if model:
                    where_clause = "WHERE model LIKE %s"
                    params.append(f"%{model}%")
                
                # 전체 건수 조회
                count_query = f"SELECT COUNT(*) as total FROM result_logs {where_clause}"
                cursor.execute(count_query, params)
                total = cursor.fetchone()['total']
                
                # 총 페이지 수 계산
                total_pages = (total + limit - 1) // limit if total > 0 else 0
                
                # 오프셋 계산
                offset = (page - 1) * limit
                
                # 로그 목록 조회
                query = f"""
                    SELECT 
                        idx as id,
                        model,
                        run_time,
                        result_url
                    FROM result_logs
                    {where_clause}
                    ORDER BY idx DESC
                    LIMIT %s OFFSET %s
                """
                query_params = params + [limit, offset]
                cursor.execute(query, query_params)
                
                logs = cursor.fetchall()
                
                # 데이터 형식 변환
                for log in logs:
                    log['processing_time'] = log['run_time']
                    log['model_name'] = log['model']
                
                return JSONResponse({
                    "success": True,
                    "data": logs,
                    "pagination": {
                        "page": page,
                        "limit": limit,
                        "total": total,
                        "total_pages": total_pages
                    }
                })
        finally:
            connection.close()
            
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"로그 조회 중 오류 발생: {str(e)}"
        }, status_code=500)

@app.get("/api/admin/logs/{log_id}", tags=["관리자"])
async def get_admin_log_detail(log_id: int):
    """
    관리자 로그 상세 정보 조회
    
    특정 로그의 상세 정보를 조회합니다.
    
    Args:
        log_id: 로그 ID (idx)
    
    Returns:
        JSONResponse: 로그 상세 정보 (result_url 포함)
    """
    try:
        connection = get_db_connection()
        if not connection:
            return JSONResponse({
                "success": False,
                "error": "Database connection failed",
                "message": "데이터베이스 연결에 실패했습니다."
            }, status_code=500)
        
        try:
            with connection.cursor() as cursor:
                # created_at 필드 포함해서 쿼리 시도 (없으면 제외)
                try:
                    cursor.execute("""
                        SELECT 
                            idx as id,
                            person_url,
                            dress_url,
                            result_url,
                            model,
                            prompt,
                            success,
                            run_time,
                            COALESCE(DATE_FORMAT(created_at, '%%Y-%%m-%%d %%H:%%i:%%s'), '') as created_at
                        FROM result_logs
                        WHERE idx = %s
                    """, (log_id,))
                except Exception as e:
                    # created_at 필드가 없으면 다시 쿼리 (없이)
                    print(f"created_at 필드 포함 쿼리 실패, 재시도: {e}")
                    cursor.execute("""
                        SELECT 
                            idx as id,
                            person_url,
                            dress_url,
                            result_url,
                            model,
                            prompt,
                            success,
                            run_time
                        FROM result_logs
                        WHERE idx = %s
                    """, (log_id,))
                
                log = cursor.fetchone()
                
                if not log:
                    return JSONResponse({
                        "success": False,
                        "error": "Log not found",
                        "message": f"로그 ID {log_id}를 찾을 수 없습니다."
                    }, status_code=404)
                
                # 데이터 형식 변환 및 안전 처리
                # None 값을 빈 문자열로 변환
                result_data = {
                    'id': log.get('id') or 0,
                    'person_url': log.get('person_url') or '',
                    'dress_url': log.get('dress_url') or '',
                    'result_url': log.get('result_url') or '',
                    'model': log.get('model') or '',
                    'model_name': log.get('model') or '',
                    'prompt': log.get('prompt') or '',
                    'success': log.get('success', False),
                    'run_time': log.get('run_time') or 0.0,
                    'processing_time': log.get('run_time') or 0.0,
                    'created_at': log.get('created_at') or ''
                }
                
                return JSONResponse({
                    "success": True,
                    "data": result_data
                })
        finally:
            connection.close()
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"로그 상세 조회 오류: {error_detail}")
        return JSONResponse({
            "success": False,
            "error": str(e),
            "error_detail": error_detail,
            "message": f"로그 상세 조회 중 오류 발생: {str(e)}"
        }, status_code=500)

@app.get("/api/admin/s3-image-proxy", tags=["관리자"])
async def get_s3_image_proxy(url: str = Query(..., description="S3 이미지 URL")):
    """
    S3 이미지 프록시 엔드포인트
    
    CORS 문제를 우회하기 위해 백엔드에서 S3 이미지를 다운로드하여 제공합니다.
    
    Args:
        url: S3 이미지 URL
    
    Returns:
        Response: 이미지 바이너리 데이터
    """
    try:
        # URL 디코딩
        decoded_url = unquote(url)
        
        # S3 URL 검증
        if not decoded_url.startswith('https://') or 's3' not in decoded_url.lower():
            return JSONResponse({
                "success": False,
                "error": "Invalid URL",
                "message": "유효하지 않은 S3 URL입니다."
            }, status_code=400)
        
        # S3에서 이미지 다운로드
        # S3 버킷이 공개되어 있지 않으면 AWS 자격 증명 사용
        aws_access_key = os.getenv("LOGS_AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("LOGS_AWS_SECRET_ACCESS_KEY")
        
        if aws_access_key and aws_secret_key:
            # boto3를 사용하여 S3에서 직접 다운로드
            try:
                bucket_name = os.getenv("LOGS_AWS_S3_BUCKET_NAME")
                region = os.getenv("LOGS_AWS_REGION", "ap-northeast-2")
                
                # URL에서 S3 키 추출
                # 예: https://bucket.s3.region.amazonaws.com/logs/file.png
                parsed_url = urlparse(decoded_url)
                
                # S3 URL 형식 확인
                # 형식 1: https://bucket.s3.region.amazonaws.com/path
                # 형식 2: https://s3.region.amazonaws.com/bucket/path
                if bucket_name in parsed_url.netloc:
                    # 형식 1: bucket.s3.region.amazonaws.com
                    s3_key = parsed_url.path.lstrip('/')
                elif parsed_url.netloc.startswith('s3.'):
                    # 형식 2: s3.region.amazonaws.com/bucket/path
                    path_parts = parsed_url.path.lstrip('/').split('/', 1)
                    if path_parts[0] == bucket_name and len(path_parts) > 1:
                        s3_key = path_parts[1]
                    else:
                        s3_key = parsed_url.path.lstrip('/')
                else:
                    # 기본: 경로에서 버킷명 제거
                    s3_key = parsed_url.path.lstrip('/')
                    if s3_key.startswith(bucket_name + '/'):
                        s3_key = s3_key[len(bucket_name) + 1:]
                
                s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    region_name=region
                )
                
                # S3에서 객체 가져오기
                response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
                image_data = response['Body'].read()
                content_type = response.get('ContentType', 'image/png')
                
                return Response(
                    content=image_data,
                    media_type=content_type,
                    headers={
                        "Cache-Control": "public, max-age=3600",
                        "Access-Control-Allow-Origin": "*"
                    }
                )
            except Exception as e:
                print(f"S3 직접 다운로드 실패, HTTP 요청 시도: {e}")
        
        # boto3 실패 시 HTTP 요청으로 시도
        headers = {}
        if aws_access_key and aws_secret_key:
            # AWS 서명이 필요한 경우는 boto3만 사용 가능
            pass
        
        response = requests.get(decoded_url, timeout=10, headers=headers)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', 'image/png')
        
        return Response(
            content=response.content,
            media_type=content_type,
            headers={
                "Cache-Control": "public, max-age=3600",
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except requests.exceptions.RequestException as e:
        return JSONResponse({
            "success": False,
            "error": "Image download failed",
            "message": f"이미지 다운로드 실패: {str(e)}"
        }, status_code=500)
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"S3 이미지 프록시 오류: {error_detail}")
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"이미지 프록시 중 오류 발생: {str(e)}"
        }, status_code=500)

@app.get("/admin", response_class=HTMLResponse, tags=["관리자"])
async def admin_page(request: Request):
    """
    관리자 페이지
    
    로그 목록과 통계를 확인할 수 있는 관리자 페이지
    """
    return templates.TemplateResponse("admin.html", {"request": request})

@app.get("/admin/dress-manage", response_class=HTMLResponse, tags=["관리자"])
async def dress_manage_page(request: Request):
    """
    드레스 관리자 페이지
    
    드레스 정보 목록 조회 및 추가가 가능한 관리자 페이지
    """
    return templates.TemplateResponse("dress_manage.html", {"request": request})

@app.get("/model-comparison", response_class=HTMLResponse, tags=["Web Interface"])
async def model_comparison_page(request: Request):
    """
    모델 비교 테스트 페이지
    
    여러 모델의 합성 기능을 동시에 비교할 수 있는 페이지
    """
    return templates.TemplateResponse("model-comparison.html", {"request": request})

@app.get("/api/models", tags=["모델 관리"])
async def get_models():
    """
    모델 목록 조회
    
    models_config.json 파일에서 모델 정보를 읽어서 반환합니다.
    
    Returns:
        JSONResponse: 모델 목록
    """
    try:
        config_file = Path("models_config.json")
        if not config_file.exists():
            return JSONResponse({
                "success": False,
                "error": "Config file not found",
                "message": "models_config.json 파일을 찾을 수 없습니다."
            }, status_code=404)
        
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        return JSONResponse({
            "success": True,
            "models": config.get("models", []),
            "total": len(config.get("models", []))
        })
    except json.JSONDecodeError as e:
        return JSONResponse({
            "success": False,
            "error": "Invalid JSON",
            "message": f"models_config.json 파일 형식이 올바르지 않습니다: {str(e)}"
        }, status_code=500)
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"모델 목록을 불러오는 중 오류 발생: {str(e)}"
        }, status_code=500)

@app.post("/api/models", tags=["모델 관리"])
async def add_model(model_data: dict):
    """
    새 모델 추가
    
    models_config.json 파일에 새 모델 정보를 추가합니다.
    
    Args:
        model_data: 모델 정보 (id, name, description, endpoint, method, input_type, inputs, category)
    
    Returns:
        JSONResponse: 추가 결과
    """
    try:
        config_file = Path("models_config.json")
        
        # 기존 설정 파일 읽기
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {"models": []}
        
        # 중복 체크
        existing_ids = [m.get("id") for m in config.get("models", [])]
        if model_data.get("id") in existing_ids:
            return JSONResponse({
                "success": False,
                "error": "Duplicate ID",
                "message": "이미 존재하는 모델 ID입니다."
            }, status_code=400)
        
        # 새 모델 추가
        config.setdefault("models", []).append(model_data)
        
        # 파일에 저장
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        return JSONResponse({
            "success": True,
            "message": "모델이 성공적으로 추가되었습니다.",
            "model": model_data
        })
    except json.JSONDecodeError as e:
        return JSONResponse({
            "success": False,
            "error": "Invalid JSON",
            "message": f"models_config.json 파일 형식이 올바르지 않습니다: {str(e)}"
        }, status_code=500)
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"모델 추가 중 오류 발생: {str(e)}"
        }, status_code=500)

# ===================== 새로운 모델 API 엔드포인트 =====================

@app.post("/api/segment-b0", tags=["세그멘테이션"])
async def segment_b0(file: UploadFile = File(..., description="세그멘테이션할 이미지 파일")):
    """
    SegFormer B0 세그멘테이션 (배경 제거/옷 영역 인식)
    
    matei-dorian/segformer-b0-finetuned-human-parsing 모델을 사용하여
    이미지에서 배경을 제거하고 옷 영역을 인식합니다.
    
    Args:
        file: 업로드할 이미지 파일
    
    Returns:
        JSONResponse: 원본 이미지, 결과 이미지 (배경 제거), 감지 정보
    """
    global segformer_b0_processor, segformer_b0_model
    
    try:
        # 모델 lazy loading
        if segformer_b0_processor is None or segformer_b0_model is None:
            try:
                from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
                segformer_b0_processor = SegformerImageProcessor.from_pretrained("matei-dorian/segformer-b0-finetuned-human-parsing")
                segformer_b0_model = AutoModelForSemanticSegmentation.from_pretrained("matei-dorian/segformer-b0-finetuned-human-parsing")
                segformer_b0_model.eval()
                print("SegFormer B0 모델 로딩 완료!")
            except Exception as e:
                return JSONResponse({
                    "success": False,
                    "error": f"모델 로딩 실패: {str(e)}",
                    "message": "SegFormer B0 모델을 로드할 수 없습니다."
                }, status_code=500)
        
        # 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        original_size = image.size
        
        # 원본 이미지를 base64로 인코딩
        buffered_original = io.BytesIO()
        image.save(buffered_original, format="PNG")
        original_base64 = base64.b64encode(buffered_original.getvalue()).decode()
        
        # 모델 추론
        inputs = segformer_b0_processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = segformer_b0_model(**inputs)
            logits = outputs.logits.cpu()
        
        # 업샘플링
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=original_size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        
        # 세그멘테이션 마스크 생성 (배경이 아닌 모든 것)
        pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
        mask = (pred_seg != 0).astype(np.uint8) * 255
        
        # 원본 이미지를 numpy 배열로 변환
        image_array = np.array(image)
        
        # 누끼 이미지 생성 (RGBA)
        result_image = np.zeros((image_array.shape[0], image_array.shape[1], 4), dtype=np.uint8)
        result_image[:, :, :3] = image_array
        result_image[:, :, 3] = mask
        
        # PIL 이미지로 변환
        result_pil = Image.fromarray(result_image, mode='RGBA')
        
        # 결과 이미지를 base64로 인코딩
        buffered_result = io.BytesIO()
        result_pil.save(buffered_result, format="PNG")
        result_base64 = base64.b64encode(buffered_result.getvalue()).decode()
        
        # 전경 픽셀 비율 계산
        foreground_pixels = int(np.sum(pred_seg != 0))
        total_pixels = int(pred_seg.size)
        foreground_percentage = round((foreground_pixels / total_pixels) * 100, 2)
        
        return JSONResponse({
            "success": True,
            "original_image": f"data:image/png;base64,{original_base64}",
            "result_image": f"data:image/png;base64,{result_base64}",
            "foreground_percentage": foreground_percentage,
            "message": f"SegFormer B0 세그멘테이션 완료 (전경 영역: {foreground_percentage}%)"
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"처리 중 오류 발생: {str(e)}"
        }, status_code=500)

@app.post("/api/pose-estimation", tags=["포즈 인식"])
async def pose_estimation(file: UploadFile = File(..., description="포즈 인식할 이미지 파일")):
    """
    RTMPose-s 포즈/관절 키포인트 인식
    
    인체의 포즈와 관절 키포인트를 인식하여 위치 정보를 반환합니다.
    
    Args:
        file: 업로드할 이미지 파일
    
    Returns:
        JSONResponse: 원본 이미지, 키포인트 좌표, 시각화된 이미지
    """
    global rtmpose_model
    
    try:
        # 모델 lazy loading
        if rtmpose_model is None:
            try:
                from mmpose.apis import init_model, inference_top_down_pose_model
                import mmcv
                
                config_file = 'configs/rtmpose/rtmpose-s_8xb256-420e_coco-256x192.py'
                checkpoint_file = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth'
                
                device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                rtmpose_model = init_model(config_file, checkpoint_file, device=device)
                print("RTMPose-s 모델 로딩 완료!")
            except Exception as e:
                return JSONResponse({
                    "success": False,
                    "error": f"모델 로딩 실패: {str(e)}",
                    "message": "RTMPose-s 모델을 로드할 수 없습니다. mmpose 설치 및 설정을 확인하세요."
                }, status_code=500)
        
        # 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 원본 이미지를 base64로 인코딩
        buffered_original = io.BytesIO()
        image.save(buffered_original, format="PNG")
        original_base64 = base64.b64encode(buffered_original.getvalue()).decode()
        
        # mmcv 형식으로 변환
        import mmcv
        img_array = np.array(image)
        img_bgr = mmcv.imconvert(img_array, 'RGB', 'BGR')
        
        # 포즈 추론
        from mmpose.apis import inference_top_down_pose_model
        pose_results, _ = inference_top_down_pose_model(rtmpose_model, img_bgr)
        
        # 키포인트를 이미지에 시각화
        from mmpose.visualization import draw_skeleton_and_kp
        
        vis_img = draw_skeleton_and_kp(
            img_array,
            pose_results,
            kp_thr=0.3,
            skeleton_style='mmpose'
        )
        
        # 결과 이미지를 base64로 인코딩
        vis_pil = Image.fromarray(vis_img)
        buffered_result = io.BytesIO()
        vis_pil.save(buffered_result, format="PNG")
        result_base64 = base64.b64encode(buffered_result.getvalue()).decode()
        
        # 키포인트 정보 추출
        keypoints = []
        if pose_results and len(pose_results) > 0:
            for person in pose_results:
                if 'keypoints' in person:
                    keypoints.append(person['keypoints'].tolist())
        
        return JSONResponse({
            "success": True,
            "original_image": f"data:image/png;base64,{original_base64}",
            "result_image": f"data:image/png;base64,{result_base64}",
            "keypoints": keypoints,
            "num_persons": len(keypoints),
            "message": f"{len(keypoints)}명의 포즈 감지됨"
        })
        
    except ImportError as e:
        return JSONResponse({
            "success": False,
            "error": "mmpose 라이브러리 미설치",
            "message": "mmpose를 설치하세요: pip install mmpose>=0.31.0"
        }, status_code=500)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"처리 중 오류 발생: {str(e)}"
        }, status_code=500)

@app.post("/api/hr-viton-compose", tags=["가상 피팅"])
async def hr_viton_compose(
    person_image: UploadFile = File(..., description="사람 이미지 파일"),
    dress_image: UploadFile = File(..., description="드레스 이미지 파일")
):
    """
    HR-VITON 가상 피팅 - 옷 교체/워핑/합성
    
    사람 이미지에 드레스를 자연스럽게 합성합니다.
    
    Args:
        person_image: 사람 이미지 파일
        dress_image: 드레스 이미지 파일
    
    Returns:
        JSONResponse: 합성된 이미지
    """
    try:
        # 이미지 읽기
        person_contents = await person_image.read()
        dress_contents = await dress_image.read()
        
        person_img = Image.open(io.BytesIO(person_contents)).convert("RGB")
        dress_img = Image.open(io.BytesIO(dress_contents)).convert("RGB")
        
        # 원본 이미지들을 base64로 인코딩
        person_buffered = io.BytesIO()
        person_img.save(person_buffered, format="PNG")
        person_base64 = base64.b64encode(person_buffered.getvalue()).decode()
        
        dress_buffered = io.BytesIO()
        dress_img.save(dress_buffered, format="PNG")
        dress_base64 = base64.b64encode(dress_buffered.getvalue()).decode()
        
        # HR-VITON 구현 (간단한 버전)
        # 실제로는 HR-VITON 저장소의 코드를 사용해야 함
        # 여기서는 기본적인 합성 로직으로 대체
        
        # 이미지 크기 맞추기
        person_array = np.array(person_img)
        dress_array = np.array(dress_img)
        
        # 간단한 합성 (드레스 영역을 사람 이미지에 합성)
        # 실제 HR-VITON은 복잡한 워핑 및 합성 알고리즘 사용
        result_array = person_array.copy()
        
        # 드레스 이미지를 사람 이미지 크기에 맞춰 리사이즈
        dress_resized = dress_img.resize(person_img.size, Image.Resampling.LANCZOS)
        dress_array_resized = np.array(dress_resized)
        
        # 간단한 알파 블렌딩 (실제로는 정교한 워핑 필요)
        alpha = 0.7
        result_array = (alpha * dress_array_resized + (1 - alpha) * result_array).astype(np.uint8)
        
        result_img = Image.fromarray(result_array)
        
        # 결과 이미지를 base64로 인코딩
        result_buffered = io.BytesIO()
        result_img.save(result_buffered, format="PNG")
        result_base64 = base64.b64encode(result_buffered.getvalue()).decode()
        
        return JSONResponse({
            "success": True,
            "person_image": f"data:image/png;base64,{person_base64}",
            "dress_image": f"data:image/png;base64,{dress_base64}",
            "result_image": f"data:image/png;base64,{result_base64}",
            "message": "HR-VITON 가상 피팅 완료 (참고: 실제 HR-VITON 모델 구현 필요)"
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"처리 중 오류 발생: {str(e)}"
        }, status_code=500)

@app.post("/api/generate-shoes", tags=["구두 생성"])
async def generate_shoes(
    prompt: str = Form(..., description="구두 생성 프롬프트"),
    model_type: str = Form("gemini", description="사용할 모델 (gemini 또는 sdxl)")
):
    """
    구두 이미지 생성 (SDXL-LoRA 또는 Gemini 2.5 Image)
    
    프롬프트를 기반으로 구두 이미지를 생성합니다.
    
    Args:
        prompt: 구두 생성 프롬프트
        model_type: 사용할 모델 (gemini 또는 sdxl)
    
    Returns:
        JSONResponse: 생성된 구두 이미지
    """
    try:
        if model_type == "gemini":
            # Gemini 2.5 Image 사용
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return JSONResponse({
                    "success": False,
                    "error": "API key not found",
                    "message": ".env 파일에 GEMINI_API_KEY가 설정되지 않았습니다."
                }, status_code=500)
            
            from google import genai
            client = genai.Client(api_key=api_key)
            
            # 텍스트 프롬프트로 이미지 생성
            response = client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=[f"Generate a high-quality image of {prompt}. The image should be photorealistic and detailed."]
            )
            
            # 응답에서 이미지 추출
            image_parts = [
                part.inline_data.data
                for part in response.candidates[0].content.parts
                if hasattr(part, 'inline_data') and part.inline_data
            ]
            
            if image_parts:
                result_image_base64 = base64.b64encode(image_parts[0]).decode()
                return JSONResponse({
                    "success": True,
                    "result_image": f"data:image/png;base64,{result_image_base64}",
                    "model": "gemini-2.5-flash-image",
                    "message": "Gemini로 구두 이미지 생성 완료"
                })
            else:
                return JSONResponse({
                    "success": False,
                    "error": "No image generated",
                    "message": "Gemini API가 이미지를 생성하지 못했습니다."
                }, status_code=500)
        
        else:
            # SDXL-LoRA 사용
            global sdxl_pipeline
            
            if sdxl_pipeline is None:
                try:
                    from diffusers import StableDiffusionXLPipeline
                    import torch
                    
                    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                    sdxl_pipeline = StableDiffusionXLPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-xl-base-1.0",
                        torch_dtype=torch.float16 if device.startswith('cuda') else torch.float32
                    )
                    sdxl_pipeline = sdxl_pipeline.to(device)
                    print("SDXL 파이프라인 로딩 완료!")
                except Exception as e:
                    return JSONResponse({
                        "success": False,
                        "error": f"모델 로딩 실패: {str(e)}",
                        "message": "SDXL 모델을 로드할 수 없습니다."
                    }, status_code=500)
            
            # 이미지 생성
            image = sdxl_pipeline(prompt=prompt, num_inference_steps=50).images[0]
            
            # 결과 이미지를 base64로 인코딩
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            result_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            return JSONResponse({
                "success": True,
                "result_image": f"data:image/png;base64,{result_base64}",
                "model": "sdxl-base-1.0",
                "message": "SDXL로 구두 이미지 생성 완료"
            })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"처리 중 오류 발생: {str(e)}"
        }, status_code=500)

@app.post("/api/tps-warp", tags=["TPS Warp"])
async def tps_warp(
    shoes_image: UploadFile = File(..., description="구두 이미지 파일"),
    person_image: UploadFile = File(..., description="사람 이미지 파일")
):
    """
    TPS Warp - 구두 워핑 및 착용 삽입
    
    구두 이미지를 사람 이미지의 발 위치에 맞게 워핑하여 합성합니다.
    
    Args:
        shoes_image: 구두 이미지 파일
        person_image: 사람 이미지 파일
    
    Returns:
        JSONResponse: 워핑된 구두가 합성된 이미지
    """
    try:
        import cv2
        from scipy.spatial import distance
        
        # 이미지 읽기
        shoes_contents = await shoes_image.read()
        person_contents = await person_image.read()
        
        shoes_img = Image.open(io.BytesIO(shoes_contents)).convert("RGB")
        person_img = Image.open(io.BytesIO(person_contents)).convert("RGB")
        
        # 원본 이미지들을 base64로 인코딩
        shoes_buffered = io.BytesIO()
        shoes_img.save(shoes_buffered, format="PNG")
        shoes_base64 = base64.b64encode(shoes_buffered.getvalue()).decode()
        
        person_buffered = io.BytesIO()
        person_img.save(person_buffered, format="PNG")
        person_base64 = base64.b64encode(person_buffered.getvalue()).decode()
        
        # OpenCV 형식으로 변환
        shoes_cv = cv2.cvtColor(np.array(shoes_img), cv2.COLOR_RGB2BGR)
        person_cv = cv2.cvtColor(np.array(person_img), cv2.COLOR_RGB2BGR)
        
        # 간단한 TPS Warp 구현
        # 실제로는 발 위치를 감지하고 정교한 워핑 필요
        h, w = person_cv.shape[:2]
        
        # 구두 이미지 리사이즈
        shoes_resized = cv2.resize(shoes_cv, (w // 4, h // 4))
        
        # 사람 이미지의 하단에 구두 합성 (간단한 버전)
        result_cv = person_cv.copy()
        y_offset = h - shoes_resized.shape[0] - 50
        x_offset = w // 2 - shoes_resized.shape[1] // 2
        
        # ROI 추출
        roi = result_cv[y_offset:y_offset+shoes_resized.shape[0], 
                       x_offset:x_offset+shoes_resized.shape[1]]
        
        # 알파 블렌딩
        mask = np.ones(shoes_resized.shape, dtype=shoes_resized.dtype) * 255
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_inv = cv2.bitwise_not(mask)
        
        # 배경과 전경 분리
        img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        img_fg = cv2.bitwise_and(shoes_resized, shoes_resized, mask=mask)
        
        # 합성
        dst = cv2.add(img_bg, img_fg)
        result_cv[y_offset:y_offset+shoes_resized.shape[0], 
                 x_offset:x_offset+shoes_resized.shape[1]] = dst
        
        # 결과 이미지 변환
        result_img = Image.fromarray(cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB))
        
        # 결과 이미지를 base64로 인코딩
        result_buffered = io.BytesIO()
        result_img.save(result_buffered, format="PNG")
        result_base64 = base64.b64encode(result_buffered.getvalue()).decode()
        
        return JSONResponse({
            "success": True,
            "shoes_image": f"data:image/png;base64,{shoes_base64}",
            "person_image": f"data:image/png;base64,{person_base64}",
            "result_image": f"data:image/png;base64,{result_base64}",
            "message": "TPS Warp 구두 합성 완료 (참고: 정교한 워핑 알고리즘 필요)"
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"처리 중 오류 발생: {str(e)}"
        }, status_code=500)

@app.post("/api/upscale", tags=["해상도 향상"])
async def upscale_image(
    file: UploadFile = File(..., description="업스케일할 이미지 파일"),
    scale: int = Form(4, description="업스케일 배율 (2, 4)")
):
    """
    Real-ESRGAN 해상도 향상
    
    이미지의 해상도와 질감을 향상시킵니다.
    
    Args:
        file: 업스케일할 이미지 파일
        scale: 업스케일 배율 (2 또는 4)
    
    Returns:
        JSONResponse: 향상된 해상도의 이미지
    """
    global realesrgan_model
    
    try:
        # 모델 lazy loading
        if realesrgan_model is None:
            try:
                from realesrgan import RealESRGANer
                from realesrgan.archs.srvgg_arch import SRVGGNetCompact
                
                device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                
                # Real-ESRGAN 모델 로드
                model_path = f'weights/RealESRGAN_x{scale}plus.pth'
                model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, 
                                       num_conv=32, upscale=scale, act_type='prelu')
                realesrgan_model = RealESRGANer(scale=scale, model_path=model_path, 
                                               model=model, tile=0, tile_pad=10, 
                                               pre_pad=0, half=False, device=device)
                print("Real-ESRGAN 모델 로딩 완료!")
            except Exception as e:
                # 모델 파일이 없으면 간단한 업스케일링 사용
                print(f"Real-ESRGAN 모델 로딩 실패, 간단한 업스케일링 사용: {e}")
                realesrgan_model = None
        
        # 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 원본 이미지를 base64로 인코딩
        buffered_original = io.BytesIO()
        image.save(buffered_original, format="PNG")
        original_base64 = base64.b64encode(buffered_original.getvalue()).decode()
        
        if realesrgan_model is not None:
            # Real-ESRGAN 사용
            import cv2
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            output, _ = realesrgan_model.enhance(img_bgr, outscale=scale)
            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            result_img = Image.fromarray(output_rgb)
        else:
            # 간단한 업스케일링 (Lanczos 리샘플링)
            new_size = (image.size[0] * scale, image.size[1] * scale)
            result_img = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # 결과 이미지를 base64로 인코딩
        buffered_result = io.BytesIO()
        result_img.save(buffered_result, format="PNG")
        result_base64 = base64.b64encode(buffered_result.getvalue()).decode()
        
        return JSONResponse({
            "success": True,
            "original_image": f"data:image/png;base64,{original_base64}",
            "result_image": f"data:image/png;base64,{result_base64}",
            "scale": scale,
            "original_size": image.size,
            "result_size": result_img.size,
            "message": f"해상도 향상 완료 ({scale}x 업스케일)"
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"처리 중 오류 발생: {str(e)}"
        }, status_code=500)

@app.post("/api/color-harmonize", tags=["색상 보정"])
async def color_harmonize(
    file: UploadFile = File(..., description="색상 보정할 이미지 파일"),
    reference_file: UploadFile = File(None, description="참조 이미지 (선택사항)")
):
    """
    Color Harmonization - 조명 및 색상 보정
    
    이미지의 조명과 색상을 조정하여 자연스러운 결과를 만듭니다.
    
    Args:
        file: 색상 보정할 이미지 파일
        reference_file: 참조 이미지 (선택사항, 없으면 자동 보정)
    
    Returns:
        JSONResponse: 색상 보정된 이미지
    """
    try:
        import cv2
        
        # 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 원본 이미지를 base64로 인코딩
        buffered_original = io.BytesIO()
        image.save(buffered_original, format="PNG")
        original_base64 = base64.b64encode(buffered_original.getvalue()).decode()
        
        # OpenCV 형식으로 변환
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        if reference_file:
            # 참조 이미지가 있으면 색상 전이
            ref_contents = await reference_file.read()
            ref_image = Image.open(io.BytesIO(ref_contents)).convert("RGB")
            ref_array = np.array(ref_image)
            ref_bgr = cv2.cvtColor(ref_array, cv2.COLOR_RGB2BGR)
            
            # LAB 색공간으로 변환
            img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB)
            
            # 색상 전이 (LAB 색공간에서)
            img_lab[:, :, 1] = ref_lab[:, :, 1]  # a 채널
            img_lab[:, :, 2] = ref_lab[:, :, 2]  # b 채널
            
            # BGR로 변환
            result_bgr = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
        else:
            # 자동 색상 보정 (CLAHE 사용)
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # CLAHE 적용 (대비 향상)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # LAB 합성
            lab = cv2.merge([l, a, b])
            result_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 결과 이미지 변환
        result_img = Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
        
        # 결과 이미지를 base64로 인코딩
        buffered_result = io.BytesIO()
        result_img.save(buffered_result, format="PNG")
        result_base64 = base64.b64encode(buffered_result.getvalue()).decode()
        
        return JSONResponse({
            "success": True,
            "original_image": f"data:image/png;base64,{original_base64}",
            "result_image": f"data:image/png;base64,{result_base64}",
            "message": "Color Harmonization 색상 보정 완료"
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"처리 중 오류 발생: {str(e)}"
        }, status_code=500)

# ===================== 통합 파이프라인 API =====================

def _create_rtmpose_fallback_mask(height, width, waist_y=None):
    """
    RTMPose 키포인트 기반 Fallback 마스크 생성
    
    Args:
        height: 이미지 높이
        width: 이미지 너비
        waist_y: 허리 Y 좌표 (None이면 기본값 사용)
    
    Returns:
        상체 마스크 (numpy array, uint8)
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if waist_y is not None and waist_y > 0:
        # 허리 위치 + 20px부터 하체로 간주
        cutoff_y = min(waist_y + 20, height)
    else:
        # 이미지 하단 60%를 하체로 간주 (상단 40%가 상체)
        cutoff_y = int(height * 0.4)
    
    mask[:cutoff_y, :] = 255
    return mask

@app.post("/api/compose-enhanced", tags=["의상합성 고품화"])
async def compose_enhanced(
    person_image: UploadFile = File(..., description="사람 이미지 파일"),
    dress_image: UploadFile = File(..., description="드레스 이미지 파일"),
    generate_shoes: str = Form("false", description="구두 생성 여부"),
    shoes_prompt: Optional[str] = Form(None, description="구두 생성 프롬프트")
):
    """
    의상합성 개선 통합 파이프라인 (구정 후)
    
    7단계 파이프라인이 순차적으로 실행되어 고품질 의상 합성 이미지를 생성합니다.
    
    파이프라인 순서:
    1. SegFormer B2 Human Parsing: 인물 배경 제거
    2. Dress Preprocessing: 드레스 배경 제거 + 정렬
    3. RTMPose: 키포인트 인식
    4. SegFormer B2 Human Parsing: 의상 영역 마스크 생성 (상의/하의/드레스만 추출)
    5. HR-VITON: 의상 영역만 교체하여 자연스러운 드레스 입히기
    6. Real-ESRGAN: 질감/해상도 업스케일
    7. Color Harmonization: 색상/조명 보정
    
    Args:
        person_image: 사람 이미지 파일
        dress_image: 드레스 이미지 파일
        generate_shoes: 구두 생성 여부 (기본값: False, 현재 파이프라인에서는 미사용)
        shoes_prompt: 구두 생성 프롬프트 (현재 파이프라인에서는 미사용)
    
    Returns:
        JSONResponse: 최종 고품화된 합성 이미지
    """
    start_time = time.time()
    pipeline_steps = []
    
    # 전역 변수 선언 (함수 시작 부분에 한 번만)
    global segformer_b2_processor, segformer_b2_model, rtmpose_model
    
    try:
        # 이미지 읽기
        person_contents = await person_image.read()
        dress_contents = await dress_image.read()
        
        person_img = Image.open(io.BytesIO(person_contents)).convert("RGB")
        dress_img = Image.open(io.BytesIO(dress_contents)).convert("RGB")
        
        # 원본 이미지들을 base64로 인코딩
        person_buffered = io.BytesIO()
        person_img.save(person_buffered, format="PNG")
        person_base64 = base64.b64encode(person_buffered.getvalue()).decode()
        
        dress_buffered = io.BytesIO()
        dress_img.save(dress_buffered, format="PNG")
        dress_base64 = base64.b64encode(dress_buffered.getvalue()).decode()
        
        # 이미지 크기 정규화 (512×768)
        TARGET_WIDTH = 512
        TARGET_HEIGHT = 768
        
        # ========== Step 1: RMBG - 인물 배경 제거 ==========
        person_rgba = None
        person_rgba_img = None
        
        try:
            import cv2
            
            print(f"[Step 1] 시작: 원본 이미지 크기: {person_img.size}, 모드: {person_img.mode}")
            
            # 이미지 크기 정규화 (512×768)
            person_resized = person_img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
            person_array = np.array(person_resized)
            print(f"[Step 1] 정규화 완료: {person_resized.size}, 배열 크기: {person_array.shape}")
            
            # SegFormer B2 on LIP 모델 사용
            if segformer_b2_processor is None or segformer_b2_model is None:
                print(f"[Step 1] SegFormer B2 Human Parsing 모델 로딩 중...")
                from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
                # Human parsing에 특화된 SegFormer B2 모델 사용
                # yolo12138/segformer-b2-human-parse-24: human_parsing_29_mix 데이터셋으로 fine-tuned
                segformer_b2_processor = SegformerImageProcessor.from_pretrained("yolo12138/segformer-b2-human-parse-24")
                segformer_b2_model = AutoModelForSemanticSegmentation.from_pretrained("yolo12138/segformer-b2-human-parse-24")
                segformer_b2_model.eval()
                print(f"[Step 1] SegFormer B2 Human Parsing 모델 로딩 완료")
            
            # SegFormer B2 기반 배경 제거
            person_inputs = segformer_b2_processor(images=person_resized, return_tensors="pt")
            with torch.no_grad():
                person_outputs = segformer_b2_model(**person_inputs)
                person_logits = person_outputs.logits.cpu()
            
            person_upsampled = nn.functional.interpolate(
                person_logits,
                size=(TARGET_HEIGHT, TARGET_WIDTH),
                mode="bilinear",
                align_corners=False,
            )
            person_pred = person_upsampled.argmax(dim=1)[0].numpy()
            
            # 배경 마스크 추출 (배경이 아닌 모든 것)
            bg_mask = (person_pred != 0).astype(np.uint8) * 255
            bg_mask_pixel_count = np.sum(bg_mask > 0)
            bg_mask_ratio = bg_mask_pixel_count / (TARGET_HEIGHT * TARGET_WIDTH)
            print(f"[Step 1] 배경 마스크 생성: 픽셀 수: {bg_mask_pixel_count}, 비율: {bg_mask_ratio:.2%}")
            
            if bg_mask_pixel_count == 0:
                raise ValueError("배경 마스크가 비어있습니다. 인물이 감지되지 않았습니다.")
            
            # OpenCV bitwise AND 적용하여 배경 제거
            person_array_bgr = cv2.cvtColor(person_array, cv2.COLOR_RGB2BGR)
            bg_mask_3d = np.stack([bg_mask] * 3, axis=2)
            result_bgr = cv2.bitwise_and(person_array_bgr, bg_mask_3d)
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            
            # RGBA 이미지 생성
            person_rgba = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 4), dtype=np.uint8)
            person_rgba[:, :, :3] = result_rgb
            person_rgba[:, :, 3] = bg_mask
            
            person_rgba_img = Image.fromarray(person_rgba, mode='RGBA')
            print(f"[Step 1] RGBA 이미지 생성 완료: 크기: {person_rgba_img.size}, 모드: {person_rgba_img.mode}")
            
            pipeline_steps.append({"step": "RMBG", "status": "success", 
                                  "message": f"인물 배경 제거 완료 (마스크 비율: {bg_mask_ratio:.1%})"})
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Step 1] 에러: {str(e)}")
            pipeline_steps.append({"step": "RMBG", "status": "skipped", "message": f"스킵됨: {str(e)}"})
            # Fallback: 원본 이미지 사용 (배경 제거 없이 진행)
            person_rgba_img = person_img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS).convert("RGBA")
            print(f"[Step 1] Fallback: 원본 이미지 사용 (RGBA 변환)")
        
        # ========== Step 2: Dress Preprocessing - 드레스 배경 제거 + 정렬 ==========
        dress_ready = None
        dress_ready_img = None
        
        try:
            import cv2
            
            # SegFormer 모델 초기화 확인 및 device 설정
            print(f"[Step 2] 시작: 원본 드레스 이미지 크기: {dress_img.size}, 모드: {dress_img.mode}")
            
            # 드레스 배경 제거 (SegFormer 사용)
            dress_resized = dress_img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
            dress_array = np.array(dress_resized)
            print(f"[Step 2] 정규화 완료: {dress_resized.size}, 배열 크기: {dress_array.shape}")
            
            # SegFormer B2 Human Parsing 모델이 없으면 초기화
            if segformer_b2_processor is None or segformer_b2_model is None:
                print(f"[Step 2] SegFormer B2 Human Parsing 모델 로딩 중...")
                from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
                # Human parsing에 특화된 SegFormer B2 모델 사용
                # yolo12138/segformer-b2-human-parse-24: human_parsing_29_mix 데이터셋으로 fine-tuned
                segformer_b2_processor = SegformerImageProcessor.from_pretrained("yolo12138/segformer-b2-human-parse-24")
                segformer_b2_model = AutoModelForSemanticSegmentation.from_pretrained("yolo12138/segformer-b2-human-parse-24")
                segformer_b2_model.eval()
                print(f"[Step 2] SegFormer B2 Human Parsing 모델 로딩 완료")
            
            # SegFormer B2로 드레스 배경 제거
            dress_inputs = segformer_b2_processor(images=dress_resized, return_tensors="pt")
            with torch.no_grad():
                dress_outputs = segformer_b2_model(**dress_inputs)
                dress_logits = dress_outputs.logits.cpu()
            
            dress_upsampled = nn.functional.interpolate(
                dress_logits,
                size=(TARGET_HEIGHT, TARGET_WIDTH),
                mode="bilinear",
                align_corners=False,
            )
            dress_pred = dress_upsampled.argmax(dim=1)[0].numpy()
            
            # 드레스 마스크 (배경 제외, 얼굴/머리 제외)
            dress_mask = ((dress_pred != 0) & (dress_pred != 11) & (dress_pred != 2)).astype(np.uint8) * 255
            dress_mask_pixel_count = np.sum(dress_mask > 0)
            dress_mask_ratio = dress_mask_pixel_count / (TARGET_HEIGHT * TARGET_WIDTH)
            print(f"[Step 2] 드레스 마스크 생성: 픽셀 수: {dress_mask_pixel_count}, 비율: {dress_mask_ratio:.2%}")
            
            if dress_mask_pixel_count == 0:
                print(f"[Step 2] 경고: 드레스 마스크가 비어있습니다. 전체 이미지를 사용합니다.")
                dress_mask = np.ones((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8) * 255
                dress_mask_pixel_count = TARGET_HEIGHT * TARGET_WIDTH
            
            # 배경 제거 적용
            dress_array_bgr = cv2.cvtColor(dress_array, cv2.COLOR_RGB2BGR)
            dress_mask_3d = np.stack([dress_mask] * 3, axis=2)
            dress_bg_removed = cv2.bitwise_and(dress_array_bgr, dress_mask_3d)
            dress_bg_removed_rgb = cv2.cvtColor(dress_bg_removed, cv2.COLOR_BGR2RGB)
            
            # 세로 크기 기준 리사이즈 (높이 768px 맞춤 - 이미 맞춰짐)
            # 중심 정렬 (드레스 중심선 = 사람 중심선)
            # 목선 또는 어깨선 기준으로 위쪽 여백 맞추기
            # 간단히 중앙 정렬된 드레스 이미지 생성
            dress_ready = dress_bg_removed_rgb.copy()
            
            # RGBA로 변환
            dress_rgba = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 4), dtype=np.uint8)
            dress_rgba[:, :, :3] = dress_ready
            dress_rgba[:, :, 3] = dress_mask
            
            dress_ready_img = Image.fromarray(dress_rgba, mode='RGBA')
            print(f"[Step 2] RGBA 이미지 생성 완료: 크기: {dress_ready_img.size}, 모드: {dress_ready_img.mode}")
            print(f"[Step 2] 드레스 알파 채널 검증: 픽셀 수: {np.sum(np.array(dress_ready_img)[:, :, 3] > 0)}")
            
            pipeline_steps.append({"step": "Dress Preprocessing", "status": "success", 
                                  "message": f"드레스 정렬 완료 (마스크 비율: {dress_mask_ratio:.1%})"})
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Step 2] 에러: {str(e)}")
            pipeline_steps.append({"step": "Dress Preprocessing", "status": "skipped", "message": f"스킵됨: {str(e)}"})
            # Fallback: 원본 드레스 이미지 사용
            dress_ready_img = dress_img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS).convert("RGBA")
            print(f"[Step 2] Fallback: 원본 드레스 이미지 사용 (RGBA 변환)")
        
        # ========== Step 2.5: RTMPose - 포즈/키포인트 인식 (Step 3 이전에 실행) ==========
        keypoints = None
        keypoints_array = None
        waist_y = None
        
        try:
            if rtmpose_model is None:
                from mmpose.apis import init_model
                import mmcv
                
                config_file = 'configs/rtmpose/rtmpose-s_8xb256-420e_coco-256x192.py'
                checkpoint_file = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth'
                
                device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                rtmpose_model = init_model(config_file, checkpoint_file, device=device)
            
            import mmcv
            from mmpose.apis import inference_top_down_pose_model
            
            # person_rgba에서 RGB 추출
            if person_rgba_img is None:
                raise ValueError("person_rgba_img가 None입니다")
            
            person_rgb = person_rgba_img.convert("RGB")
            img_array = np.array(person_rgb)
            img_bgr = mmcv.imconvert(img_array, 'RGB', 'BGR')
            pose_results, _ = inference_top_down_pose_model(rtmpose_model, img_bgr)
            
            if pose_results and len(pose_results) > 0:
                person_result = pose_results[0]
                if 'keypoints' in person_result:
                    keypoints_array = person_result['keypoints']
                    keypoints = keypoints_array.tolist()
                    
                    # 허리 Y 좌표 계산 (골반 키포인트 11, 12 평균)
                    if len(keypoints_array) > 12:
                        pelvis_points = keypoints_array[11:13]
                        valid_pelvis = pelvis_points[pelvis_points[:, 1] > 0]
                        if len(valid_pelvis) > 0:
                            waist_y = int(np.mean(valid_pelvis[:, 1]))
            
            pipeline_steps.append({"step": "RTMPose", "status": "success", "message": f"포즈 인식 완료 (허리 Y: {waist_y if waist_y else 'N/A'})"})
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Step 2.5] RTMPose 에러: {str(e)}")
            pipeline_steps.append({"step": "RTMPose", "status": "skipped", "message": f"스킵됨: {str(e)}"})
            keypoints = None
            waist_y = None
        
        # ========== Step 3: SegFormer B2 Human Parsing - 의상 영역 마스크 생성 ==========
        human_mask = None
        
        try:
            import cv2
            
            # SegFormer 모델 초기화 확인 및 device 설정
            # person_rgba에서 RGB 추출
            if person_rgba_img is None:
                raise ValueError("person_rgba_img가 None입니다")
            
            person_rgb = person_rgba_img.convert("RGB")
            print(f"[Step 3] 입력 이미지 크기: {person_rgb.size}")
            
            if segformer_b2_processor is None or segformer_b2_model is None:
                print(f"[Step 3] SegFormer B2 Human Parsing 모델 로딩 중...")
                from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
                # Human parsing에 특화된 SegFormer B2 모델 사용
                # yolo12138/segformer-b2-human-parse-24: human_parsing_29_mix 데이터셋으로 fine-tuned
                segformer_b2_processor = SegformerImageProcessor.from_pretrained("yolo12138/segformer-b2-human-parse-24")
                segformer_b2_model = AutoModelForSemanticSegmentation.from_pretrained("yolo12138/segformer-b2-human-parse-24")
                segformer_b2_model.eval()
                print(f"[Step 3] SegFormer B2 Human Parsing 모델 로딩 완료")
            
            # Device 설정
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            segformer_b2_model = segformer_b2_model.to(device)
            print(f"[Step 3] Device: {device}")
            
            # SegFormer B2 on LIP로 human parsing 수행
            person_inputs = segformer_b2_processor(images=person_rgb, return_tensors="pt")
            # 입력 텐서를 device로 이동
            person_inputs = {k: v.to(device) for k, v in person_inputs.items()}
            
            with torch.no_grad():
                person_outputs = segformer_b2_model(**person_inputs)
                person_logits = person_outputs.logits.cpu()
            
            person_upsampled = nn.functional.interpolate(
                person_logits,
                size=(TARGET_HEIGHT, TARGET_WIDTH),
                mode="bilinear",
                align_corners=False,
            )
            person_pred = person_upsampled.argmax(dim=1)[0].numpy()
            
            # 의상 영역 추출 (상의, 하의, 드레스 모두 포함)
            # yolo12138/segformer-b2-human-parse-24 모델 클래스:
            # 5 (upper_only_torso_region), 6 (dresses_only_torso_region), 9 (left_pants), 10 (right_patns), 13 (skirts)
            # 얼굴(14), 머리(2), 팔(15,16), 다리(17,18)는 제외하여 의상 영역만 교체
            clothes_mask = ((person_pred == 5) | (person_pred == 6) | (person_pred == 9) | (person_pred == 10) | (person_pred == 13)).astype(np.uint8) * 255
            
            # 검증: 의상 마스크 픽셀 합계 확인
            clothes_mask_ratio = np.sum(clothes_mask > 0) / (TARGET_HEIGHT * TARGET_WIDTH)
            clothes_mask_pixel_count = np.sum(clothes_mask > 0)
            print(f"[Step 3] 의상 영역 마스크 픽셀 수: {clothes_mask_pixel_count}, 비율: {clothes_mask_ratio:.2%}")
            
            # 의상 영역이 없으면 전체 인물 영역을 의상 영역으로 사용 (fallback)
            if clothes_mask_ratio < 0.05:
                print(f"[Step 3] 경고: 의상 영역이 감지되지 않았습니다. 전체 인물 영역을 의상 영역으로 사용합니다.")
                # 전체 인물 영역 마스크 생성 (배경 제외, 하지만 얼굴/손/다리는 보존)
                human_mask = (person_pred != 0).astype(np.uint8) * 255
                # 얼굴, 머리, 팔, 다리 제외 (의상 영역만 남김)
                # yolo12138/segformer-b2-human-parse-24 모델 클래스 매핑 사용
                face_mask = (person_pred == 14).astype(np.uint8)  # 얼굴 (14)
                hair_mask = (person_pred == 2).astype(np.uint8)  # 머리 (2)
                arms_mask = ((person_pred == 15) | (person_pred == 16)).astype(np.uint8)  # 팔 (15: left_arm, 16: right_arm)
                legs_mask = ((person_pred == 17) | (person_pred == 18)).astype(np.uint8)  # 다리 (17: left_leg, 18: right_leg)
                preserve_mask = (face_mask | hair_mask | arms_mask | legs_mask)
                clothes_mask = (human_mask.astype(np.uint8) - preserve_mask * 255).astype(np.uint8)
                clothes_mask = np.clip(clothes_mask, 0, 255)
                clothes_mask_pixel_count = np.sum(clothes_mask > 0)
                print(f"[Step 3] Fallback: 의상 영역 마스크 픽셀 수: {clothes_mask_pixel_count}")
            
            # 의상 영역 마스크를 human_mask 변수에 저장 (이름은 유지하되 의상 영역만 포함)
            human_mask = clothes_mask
            
            # 최종 검증
            mask_ratio = np.sum(human_mask > 0) / (TARGET_HEIGHT * TARGET_WIDTH)
            mask_pixel_count = np.sum(human_mask > 0)
            print(f"[Step 3] 최종 의상 영역 마스크 픽셀 수: {mask_pixel_count}, 비율: {mask_ratio:.2%}")
            
            if mask_ratio < 0.05:
                raise ValueError(f"의상 영역 마스크 비율이 너무 낮습니다 ({mask_ratio:.2%}). 의상이 감지되지 않았습니다.")
            
            pipeline_steps.append({"step": "SegFormer B2 on LIP", "status": "success", 
                                  "message": f"의상 영역 마스크 생성 완료 (픽셀: {mask_pixel_count}, 비율: {mask_ratio:.1%})"})
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Step 3] 에러: {str(e)}")
            pipeline_steps.append({"step": "SegFormer B2 on LIP", "status": "skipped", "message": f"스킵됨: {str(e)}"})
            # Fallback: 전체 이미지의 중앙 하단 60%를 의상 영역으로 사용
            human_mask = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8)
            human_mask[int(TARGET_HEIGHT * 0.4):, :] = 255
            print(f"[Step 3] Fallback: 이미지 하단 60%를 의상 영역으로 사용")
        
        # ========== Step 4: HR-VITON - 의상 영역만 교체 (드레스 입히기) ==========
        viton_result = None
        viton_result_img = None
        
        try:
            import cv2
            
            # None 체크
            if person_rgba_img is None:
                raise ValueError("person_rgba_img가 None입니다")
            if dress_ready_img is None:
                raise ValueError("dress_ready_img가 None입니다")
            if human_mask is None:
                raise ValueError("human_mask가 None입니다")
            
            print(f"[Step 4] 입력 검증 완료")
            print(f"[Step 4] person_rgba_img 크기: {person_rgba_img.size}, 모드: {person_rgba_img.mode}")
            print(f"[Step 4] dress_ready_img 크기: {dress_ready_img.size}, 모드: {dress_ready_img.mode}")
            print(f"[Step 4] human_mask 크기: {human_mask.shape}, 픽셀 수: {np.sum(human_mask > 0)}")
            
            # 두 이미지를 중앙 정렬
            person_rgb = person_rgba_img.convert("RGB")
            dress_rgb = dress_ready_img.convert("RGB")
            
            person_array = np.array(person_rgb)
            dress_array = np.array(dress_rgb)
            h, w = person_array.shape[:2]
            
            print(f"[Step 4] person_array 크기: {person_array.shape}")
            print(f"[Step 4] dress_array 크기: {dress_array.shape}")
            
            # human_mask 크기 확인 및 조정
            if human_mask.shape != (h, w):
                print(f"[Step 4] 경고: human_mask 크기 불일치. 리사이즈 필요: {human_mask.shape} -> ({h}, {w})")
                human_mask = cv2.resize(human_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # 의상 영역 마스크를 3D로 변환
            clothes_mask_3d = np.stack([human_mask] * 3, axis=2) / 255.0
            clothes_mask_pixel_count = np.sum(human_mask > 0)
            print(f"[Step 4] 의상 영역 마스크 픽셀 수: {clothes_mask_pixel_count}, 비율: {clothes_mask_pixel_count / (h * w):.1%}")
            
            # 드레스 알파 채널 추출
            dress_array_full = np.array(dress_ready_img)
            if dress_array_full.shape[2] == 4:
                dress_alpha = dress_array_full[:, :, 3] / 255.0
            else:
                print(f"[Step 4] 경고: 드레스 이미지에 알파 채널이 없습니다. 전체를 사용합니다.")
                dress_alpha = np.ones((h, w), dtype=np.float32)
            
            dress_alpha_3d = np.stack([dress_alpha] * 3, axis=2)
            dress_alpha_pixel_count = np.sum(dress_alpha > 0.5)
            print(f"[Step 4] 드레스 알파 채널 픽셀 수: {dress_alpha_pixel_count}, 비율: {dress_alpha_pixel_count / (h * w):.1%}")
            
            if dress_alpha_pixel_count == 0:
                raise ValueError("드레스 알파 채널이 비어있습니다. 드레스가 감지되지 않았습니다.")
            
            # 드레스 영역만 추출
            dress_extracted = dress_array * dress_alpha_3d
            dress_extracted_pixel_count = np.sum(np.any(dress_extracted > 0, axis=2))
            print(f"[Step 4] 드레스 추출 영역 픽셀 수: {dress_extracted_pixel_count}")
            
            # 인물의 의상 영역만 교체 (얼굴, 손, 다리 등은 보존)
            # 의상 영역에만 드레스 합성 (의상 마스크 AND 드레스 알파)
            dress_region_mask = clothes_mask_3d * dress_alpha_3d
            dress_region_pixel_count = np.sum(dress_region_mask > 0.5)
            print(f"[Step 4] 드레스 합성 영역 픽셀 수: {dress_region_pixel_count}")
            
            if dress_region_pixel_count == 0:
                print(f"[Step 4] 경고: 드레스 합성 영역이 없습니다. 의상 영역에 드레스 적용 시도")
                # 의상 영역 전체에 드레스 적용
                dress_region_mask = clothes_mask_3d
            
            # 의상 영역만 교체하여 자연스러운 "드레스 입히기" 효과
            # 1. 의상 영역의 원본을 완전히 제거하고 드레스로 교체
            # 2. 나머지 영역(얼굴, 손, 다리 등)은 원본 보존
            
            # 결과 배열 초기화
            result_array = person_array.copy().astype(np.float32)
            
            # 의상 영역 전체에 드레스 적용
            # 의상 영역 = 드레스로 완전 교체, 나머지 영역 = 원본 보존
            
            # 드레스 알파 채널을 활용하여 의상 영역에 드레스 합성
            # 의상 영역 전체에 드레스 적용 (드레스 알파 채널 활용)
            dress_composite_mask = clothes_mask_3d * dress_alpha_3d
            
            # Step 1: 드레스가 있는 영역에 드레스로 교체
            result_array = (dress_extracted.astype(np.float32) * dress_composite_mask + 
                          result_array * (1 - dress_composite_mask))
            
            # Step 2: 드레스 알파가 없는 의상 영역도 드레스로 채우기
            # 의상 영역 전체를 드레스로 채우기
            remaining_clothes_mask = clothes_mask_3d - dress_composite_mask
            remaining_pixel_count = np.sum(remaining_clothes_mask > 0.1)
            
            if remaining_pixel_count > 0:
                print(f"[Step 4] 의상 영역 중 드레스 알파가 없는 부분: {remaining_pixel_count} 픽셀")
                # 드레스가 없는 의상 영역도 드레스로 채우기
                # 드레스 이미지를 의상 영역에 맞춰 확장하여 채우기
                dress_for_fill = dress_extracted.copy()
                
                # 드레스 알파가 없는 의상 영역에 드레스 채우기
                result_array = (dress_for_fill.astype(np.float32) * remaining_clothes_mask + 
                              result_array * (1 - remaining_clothes_mask))
            
            # Step 3: 최종 블렌딩으로 자연스러운 경계 만들기
            # 의상 영역과 나머지 영역(얼굴, 손, 다리)의 경계만 부드럽게 처리
            import cv2
            # 가우시안 블러를 사용하여 마스크 경계를 부드럽게 (경계만)
            clothes_mask_smooth = cv2.GaussianBlur(human_mask.astype(np.float32), (5, 5), 1.0) / 255.0
            clothes_mask_smooth_3d = np.stack([clothes_mask_smooth] * 3, axis=2)
            
            # 부드러운 마스크로 최종 합성
            # 의상 영역 내부: 드레스로 완전 교체 (원본과 블렌딩하지 않음)
            # 의상 영역 경계: 부드럽게 블렌딩
            # 나머지 영역: 원본 보존
            # 의상 영역 내부는 원본과 블렌딩하지 않고 드레스만 사용
            result_array = (result_array.astype(np.float32) * clothes_mask_smooth_3d + 
                          person_array.astype(np.float32) * (1 - clothes_mask_smooth_3d))
            
            result_array = np.clip(result_array, 0, 255).astype(np.uint8)
            
            # 합성 결과 검증: 원본과 다른지 확인
            diff = np.abs(result_array.astype(np.int16) - person_array.astype(np.int16))
            diff_pixel_count = np.sum(np.any(diff > 10, axis=2))  # 10 이상 차이나는 픽셀
            diff_ratio = diff_pixel_count / (h * w)
            print(f"[Step 4] 합성 결과 검증: 변경된 픽셀 수: {diff_pixel_count}, 비율: {diff_ratio:.1%}")
            
            if diff_ratio < 0.01:
                print(f"[Step 4] 경고: 합성 결과가 원본과 거의 동일합니다. 드레스가 합성되지 않았을 수 있습니다.")
                pipeline_steps.append({"step": "HR-VITON", "status": "warning", 
                                      "message": f"합성 결과가 원본과 유사함 (변경 비율: {diff_ratio:.1%})"})
            else:
                pipeline_steps.append({"step": "HR-VITON", "status": "success", 
                                      "message": f"의상 영역 교체 완료 - 드레스 입히기 (변경 비율: {diff_ratio:.1%})"})
            
            viton_result = result_array
            viton_result_img = Image.fromarray(viton_result)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Step 4] 에러: {str(e)}")
            pipeline_steps.append({"step": "HR-VITON", "status": "error", "message": f"오류: {str(e)}"})
            # Fallback: 단순 addWeighted 블렌딩
            try:
                print(f"[Step 4] Fallback 시도: addWeighted 블렌딩")
                if person_rgba_img is None or dress_ready_img is None:
                    raise ValueError("필수 이미지가 None입니다")
                
                person_rgb = person_rgba_img.convert("RGB")
                dress_rgb = dress_ready_img.convert("RGB")
                person_array = np.array(person_rgb)
                dress_array = np.array(dress_rgb)
                
                # 이미지 크기 맞추기
                if person_array.shape != dress_array.shape:
                    print(f"[Step 4] Fallback: 이미지 크기 불일치. 리사이즈: {dress_array.shape} -> {person_array.shape}")
                    dress_array = cv2.resize(dress_array, (person_array.shape[1], person_array.shape[0]))
                
                result_array = cv2.addWeighted(person_array, 0.3, dress_array, 0.7, 0)
                viton_result = result_array
                viton_result_img = Image.fromarray(viton_result)
                print(f"[Step 4] Fallback 성공")
            except Exception as fallback_error:
                print(f"[Step 4] Fallback 실패: {str(fallback_error)}")
                viton_result_img = person_rgba_img.convert("RGB") if person_rgba_img else person_img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
        
        # Step 5 (RTMPose)는 이미 Step 2.5로 이동됨
        
        # 현재 이미지를 viton_result로 설정
        if viton_result_img is None:
            print(f"[Step 4] 경고: viton_result_img가 None입니다. person_rgba_img 사용")
            current_image = person_rgba_img.convert("RGB") if person_rgba_img else person_img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
        else:
            print(f"[Step 4] 합성 완료: 결과 이미지 크기: {viton_result_img.size}, 모드: {viton_result_img.mode}")
            current_image = viton_result_img
        
        # ========== Step 6: Real-ESRGAN - 질감/해상도 업스케일 ==========
        upscaled_img = None
        
        try:
            import cv2
            
            print(f"[Step 6] 시작: 입력 이미지 크기: {current_image.size}, 모드: {current_image.mode}")
            
            # 512×768 → 1024×1536 으로 업스케일
            global realesrgan_model
            
            if realesrgan_model is None:
                try:
                    from realesrgan import RealESRGANer
                    from realesrgan.archs.srvgg_arch import SRVGGNetCompact
                    
                    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                    scale = 4
                    model_path = f'weights/RealESRGAN_x{scale}plus.pth'
                    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, 
                                           num_conv=32, upscale=scale, act_type='prelu')
                    realesrgan_model = RealESRGANer(scale=scale, model_path=model_path, 
                                                   model=model, tile=0, tile_pad=10, 
                                                   pre_pad=0, half=False, device=device)
                except:
                    realesrgan_model = None
            
            if realesrgan_model is not None:
                img_array = np.array(current_image)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                # 2배 업스케일 (512×768 → 1024×1536)
                output, _ = realesrgan_model.enhance(img_bgr, outscale=2)
                output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                
                # 픽셀 샤프닝 적용
                kernel = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
                sharpened = cv2.filter2D(output_rgb, -1, kernel)
                upscaled_img = Image.fromarray(sharpened)
                print(f"[Step 6] Real-ESRGAN 업스케일 완료: 결과 크기: {upscaled_img.size}")
                
                pipeline_steps.append({"step": "Real-ESRGAN", "status": "success", "message": "업스케일 완료"})
            else:
                # Fallback: OpenCV resize 대체
                new_size = (TARGET_WIDTH * 2, TARGET_HEIGHT * 2)  # 1024×1536
                upscaled_img = current_image.resize(new_size, Image.Resampling.LANCZOS)
                print(f"[Step 6] Fallback: OpenCV resize 사용, 결과 크기: {upscaled_img.size}")
                pipeline_steps.append({"step": "Real-ESRGAN", "status": "fallback", "message": "OpenCV resize 사용"})
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Step 6] 에러: {str(e)}")
            pipeline_steps.append({"step": "Real-ESRGAN", "status": "skipped", "message": f"스킵됨: {str(e)}"})
            # Fallback: OpenCV resize
            new_size = (TARGET_WIDTH * 2, TARGET_HEIGHT * 2)
            upscaled_img = current_image.resize(new_size, Image.Resampling.LANCZOS)
            print(f"[Step 6] Fallback: OpenCV resize 사용, 결과 크기: {upscaled_img.size}")
        
        # 현재 이미지를 upscaled로 설정
        if upscaled_img is None:
            print(f"[Step 6] 경고: upscaled_img가 None입니다. current_image 사용")
        else:
            current_image = upscaled_img
            print(f"[Step 6] 업스케일 완료: 현재 이미지 크기: {current_image.size}")
        
        # ========== Step 7: Color Harmonization - 색상/조명 보정 ==========
        final_result_img = None
        
        try:
            import cv2
            
            print(f"[Step 7] 시작: 입력 이미지 크기: {current_image.size}, 모드: {current_image.mode}")
            
            # HSV 또는 LAB 색공간으로 변환
            img_array = np.array(current_image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # LAB 색공간으로 변환
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # 전체 인물 영역 색상 보정 (human_mask 사용)
            # 인물 영역 평균 밝기 계산
            if human_mask is not None:
                # human_mask를 현재 이미지 크기로 리사이즈
                h, w = img_array.shape[:2]
                human_mask_resized = cv2.resize(human_mask, (w, h), interpolation=cv2.INTER_NEAREST) / 255.0
                human_mask_3d = np.stack([human_mask_resized] * 3, axis=2)
                
                # 전체 인물 영역 평균 밝기
                person_region = l[human_mask_resized > 0.5]
                person_brightness = np.mean(person_region) if len(person_region) > 0 else np.mean(l)
                
                # 전체 이미지 평균 밝기
                overall_brightness = np.mean(l)
                
                # 전체 인물 영역 밝기 보정
                brightness_diff = overall_brightness - person_brightness
                
                # 채도·명도 조정
                l_adjusted = l.copy()
                if abs(brightness_diff) > 5:  # 밝기 차이가 5 이상일 때만 조정
                    # 전체 인물 영역 밝기 조정
                    l_adjusted = np.clip(l + brightness_diff * 0.3, 0, 255).astype(np.uint8)
                
                # blend ratio = 0.3(인물) + 0.7(드레스)
                l_final = (l_adjusted * 0.3 + l * 0.7).astype(np.uint8)
                
                lab = cv2.merge([l_final, a, b])
                result_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                final_result_img = Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
            else:
                # mask가 없으면 감마 보정만 적용
                result_bgr = cv2.convertScaleAbs(img_bgr, alpha=1.1, beta=5)
                final_result_img = Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
            
            print(f"[Step 7] 색상 보정 완료: 결과 이미지 크기: {final_result_img.size}")
            pipeline_steps.append({"step": "Color Harmonization", "status": "success", "message": "색상 보정 완료"})
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Step 7] 에러: {str(e)}")
            pipeline_steps.append({"step": "Color Harmonization", "status": "skipped", "message": f"스킵됨: {str(e)}"})
            # Fallback: 감마 보정 (1.1× + beta 5)
            try:
                import cv2
                img_array = np.array(current_image)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                result_bgr = cv2.convertScaleAbs(img_bgr, alpha=1.1, beta=5)
                final_result_img = Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
                print(f"[Step 7] Fallback: 감마 보정 완료")
            except Exception as fallback_error:
                print(f"[Step 7] Fallback 실패: {str(fallback_error)}")
                final_result_img = current_image
        
        # 최종 결과 이미지 설정
        if final_result_img is None:
            print(f"[Step 7] 경고: final_result_img가 None입니다. current_image 사용")
        else:
            current_image = final_result_img
            print(f"[Step 7] 최종 결과 이미지 설정 완료: 크기: {current_image.size}, 모드: {current_image.mode}")
        
        # 파이프라인 요약 로그
        success_count = len([s for s in pipeline_steps if s['status'] == 'success'])
        warning_count = len([s for s in pipeline_steps if s['status'] == 'warning'])
        error_count = len([s for s in pipeline_steps if s['status'] == 'error'])
        skipped_count = len([s for s in pipeline_steps if s['status'] == 'skipped'])
        print(f"[파이프라인 요약] 성공: {success_count}, 경고: {warning_count}, 에러: {error_count}, 스킵: {skipped_count}")
        print(f"[파이프라인 요약] 최종 이미지 크기: {current_image.size}, 모드: {current_image.mode}")
        
        # 최종 결과 이미지를 base64로 인코딩
        result_buffered = io.BytesIO()
        current_image.save(result_buffered, format="PNG")
        result_base64 = base64.b64encode(result_buffered.getvalue()).decode()
        
        run_time = time.time() - start_time
        print(f"[파이프라인 요약] 총 실행 시간: {run_time:.2f}초")
        
        # 로그 저장
        model_id = "enhanced-compose-pipeline"
        person_s3_url = upload_log_to_s3(person_buffered.getvalue(), model_id, "person") or ""
        dress_s3_url = upload_log_to_s3(dress_buffered.getvalue(), model_id, "dress") or ""
        
        result_buffered_for_s3 = io.BytesIO()
        current_image.save(result_buffered_for_s3, format="PNG")
        result_s3_url = upload_log_to_s3(result_buffered_for_s3.getvalue(), model_id, "result") or ""
        
        save_test_log(
            person_url=person_s3_url or "",
            dress_url=dress_s3_url or None,
            result_url=result_s3_url or "",
            model=model_id,
            prompt=f"Enhanced pipeline with {len([s for s in pipeline_steps if s['status'] == 'success'])}/7 steps",
            success=True,
            run_time=run_time
        )
        
        return JSONResponse({
            "success": True,
            "person_image": f"data:image/png;base64,{person_base64}",
            "dress_image": f"data:image/png;base64,{dress_base64}",
            "result_image": f"data:image/png;base64,{result_base64}",
            "pipeline_steps": pipeline_steps,
            "run_time": round(run_time, 2),
            "message": f"의상합성 개선 파이프라인 완료 ({len([s for s in pipeline_steps if s['status'] == 'success'])}/7 단계 성공)"
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        run_time = time.time() - start_time
        
        return JSONResponse({
            "success": False,
            "error": str(e),
            "pipeline_steps": pipeline_steps,
            "run_time": round(run_time, 2),
            "message": f"파이프라인 실행 중 오류 발생: {str(e)}"
        }, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

