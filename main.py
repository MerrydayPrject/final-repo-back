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

# 정적 파일 및 템플릿 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images", StaticFiles(directory="images"), name="images")
templates = Jinja2Templates(directory="templates")

# 전역 변수로 모델 저장
processor = None
model = None

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
    global processor, model
    print("SegFormer 모델 로딩 중...")
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model.eval()
    print("모델 로딩 완료!")
    
    # DB 초기화
    print("데이터베이스 초기화 중...")
    init_database()

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

@app.post("/api/compose-dress", tags=["Gemini 이미지 합성"])
async def compose_dress(
    person_image: UploadFile = File(..., description="사람 이미지 파일"),
    dress_image: UploadFile = File(..., description="드레스 이미지 파일"),
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

Task: Apply ONLY the dress from the first image onto the person from the second image.

STRICT REQUIREMENTS:
1. PRESERVE EXACTLY: The person's face, facial features, skin tone, hair, and body proportions
2. PRESERVE EXACTLY: The person's pose, stance, and body position
3. PRESERVE EXACTLY: The background and lighting from the person's image
4. CHANGE ONLY: Replace the person's clothing with the dress from the first image
5. The dress should fit naturally on the person's body shape
6. Maintain realistic shadows and fabric draping on the dress
7. Keep the person's hands, arms, legs exactly as they are in the original

DO NOT change the person's appearance, face, body type, or any physical features.
ONLY apply the dress design, color, and style onto the existing person."""
    
    text_input = prompt or default_prompt
    used_prompt = prompt or default_prompt
    success = False
    person_url = ""
    dress_url = ""
    result_url = ""
    
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
        
        # 이미지 읽기
        person_contents = await person_image.read()
        dress_contents = await dress_image.read()
        
        person_img = Image.open(io.BytesIO(person_contents))
        dress_img = Image.open(io.BytesIO(dress_contents))
        
        # 입력 이미지들을 파일 시스템에 저장
        person_image_path = save_uploaded_image(person_img, "person")
        dress_image_path = save_uploaded_image(dress_img, "dress")
        
        # S3에 입력 이미지 업로드
        person_buffered = io.BytesIO()
        person_img.save(person_buffered, format="PNG")
        person_url = upload_log_to_s3(person_buffered.getvalue(), model_id, "person") or ""
        
        dress_buffered = io.BytesIO()
        dress_img.save(dress_buffered, format="PNG")
        dress_url = upload_log_to_s3(dress_buffered.getvalue(), model_id, "dress") or ""
        
        # 원본 이미지들을 base64로 변환
        person_base64 = base64.b64encode(person_buffered.getvalue()).decode()
        dress_base64 = base64.b64encode(dress_buffered.getvalue()).decode()
        
        # Gemini Client 생성 (공식 문서와 동일한 방식)
        client = genai.Client(api_key=api_key)
        
        # Gemini API 호출 (공식 문서 방식: dress, model, text 순서)
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[dress_img, person_img, text_input]
        )
        
        # 응답 확인
        if not response.candidates or len(response.candidates) == 0:
            error_msg = "Gemini API가 응답을 생성하지 못했습니다. 이미지가 안전 정책에 위배되거나 모델이 이미지를 생성할 수 없습니다."
            run_time = time.time() - start_time
            
            # 실패 로그 저장
            save_test_log(
                person_url=person_url or "",
                dress_url=dress_url or None,
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
            result_url = upload_log_to_s3(result_buffered.getvalue(), model_id, "result") or ""
            
            success = True
            run_time = time.time() - start_time
            
            # 성공 로그 저장
            save_test_log(
                person_url=person_url or "",
                dress_url=dress_url or None,
                result_url=result_url or "",
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
                person_url=person_url or "",
                dress_url=dress_url or None,
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
                person_url=person_url or "",
                dress_url=dress_url or None,
                result_url=result_url or "",
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

@app.get("/api/images/{file_name:path}", tags=["이미지 프록시"])
async def proxy_s3_image(file_name: str):
    """
    S3 이미지를 프록시로 제공
    
    CORS 문제를 우회하기 위해 백엔드에서 S3 이미지를 다운로드하여 제공합니다.
    
    Args:
        file_name: 이미지 파일명 (예: "Adress1.JPG")
    
    Returns:
        이미지 파일 또는 404 에러
    """
    try:
        image_data = get_s3_image(file_name)
        
        if not image_data:
            return Response(
                content="Image not found",
                status_code=404,
                media_type="text/plain"
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
                "Cache-Control": "public, max-age=3600"
            }
        )
    except Exception as e:
        return Response(
            content=f"Error: {str(e)}",
            status_code=500,
            media_type="text/plain"
        )

# ===================== 드레스 관리 API =====================

@app.get("/api/admin/dresses", tags=["드레스 관리"])
async def get_dresses():
    """
    드레스 목록 조회
    
    데이터베이스에 저장된 모든 드레스 정보를 반환합니다.
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
                    SELECT idx as id, file_name as image_name, style, url
                    FROM dresses
                    ORDER BY idx DESC
                """)
                dresses = cursor.fetchall()
                
                return JSONResponse({
                    "success": True,
                    "data": dresses,
                    "total": len(dresses),
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
    드레스 추가 (이미지명만 입력)
    
    이미지명과 스타일을 받아서 데이터베이스에 추가합니다.
    이미지는 images 폴더에 이미 존재한다고 가정합니다.
    """
    try:
        body = await request.json()
        image_name = body.get("image_name")
        style = body.get("style")
        
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
                
                # 이미지 파일 존재 확인
                image_path = Path("images") / image_name
                if not image_path.exists():
                    return JSONResponse({
                        "success": False,
                        "error": "Image file not found",
                        "message": f"이미지 파일 '{image_name}'을 찾을 수 없습니다."
                    }, status_code=404)
                
                # URL 생성 (로컬 이미지 경로)
                image_url = f"/images/{image_name}"
                
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
                else:
                    # 로컬 파일인 경우 삭제 시도
                    local_image_path = Path("images") / file_name
                    if local_image_path.exists():
                        try:
                            local_image_path.unlink()
                            s3_deleted = True
                            print(f"로컬 이미지 삭제 완료: {file_name}")
                        except Exception as e:
                            print(f"로컬 이미지 삭제 오류: {e}")
                
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
                        
                        # URL이 없으면 기본값 생성
                        if not url:
                            url = f"/images/{file_name}"
                        
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

