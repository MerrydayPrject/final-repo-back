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
    dress_image: UploadFile = File(..., description="드레스 이미지 파일")
):
    """
    Gemini API를 사용한 사람과 드레스 이미지 합성
    
    사람 이미지와 드레스 이미지를 받아서 Gemini API를 통해
    사람이 드레스를 입은 것처럼 합성된 이미지를 생성합니다.
    
    Args:
        person_image: 사람 이미지 파일
        dress_image: 드레스 이미지 파일
    
    Returns:
        JSONResponse: 합성된 이미지 (base64)
    """
    person_image_path = None
    dress_image_path = None
    result_image_path = None
    
    try:
        # .env에서 API 키 가져오기
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return JSONResponse({
                "success": False,
                "error": "API key not found",
                "message": ".env 파일에 GEMINI_API_KEY가 설정되지 않았습니다."
            }, status_code=500)
        
        # 이미지 읽기
        person_contents = await person_image.read()
        dress_contents = await dress_image.read()
        
        person_img = Image.open(io.BytesIO(person_contents))
        dress_img = Image.open(io.BytesIO(dress_contents))
        
        # 입력 이미지들을 파일 시스템에 저장
        person_image_path = save_uploaded_image(person_img, "person")
        dress_image_path = save_uploaded_image(dress_img, "dress")
        
        # 원본 이미지들을 base64로 변환
        person_buffered = io.BytesIO()
        person_img.save(person_buffered, format="PNG")
        person_base64 = base64.b64encode(person_buffered.getvalue()).decode()
        
        dress_buffered = io.BytesIO()
        dress_img.save(dress_buffered, format="PNG")
        dress_base64 = base64.b64encode(dress_buffered.getvalue()).decode()
        
        # Gemini Client 생성 (공식 문서와 동일한 방식)
        client = genai.Client(api_key=api_key)
        
        # 프롬프트 생성 (얼굴과 체형 유지 강조)
        text_input = """IMPORTANT: You must preserve the person's identity completely.

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
        
        # Gemini API 호출 (공식 문서 방식: dress, model, text 순서)
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[dress_img, person_img, text_input]
        )
        
        # 응답 확인
        if not response.candidates or len(response.candidates) == 0:
            return JSONResponse({
                "success": False,
                "error": error_message,
                "message": "Gemini API가 응답을 생성하지 못했습니다. 이미지가 안전 정책에 위배되거나 모델이 이미지를 생성할 수 없습니다."
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
            
            return JSONResponse({
                "success": True,
                "person_image": f"data:image/png;base64,{person_base64}",
                "dress_image": f"data:image/png;base64,{dress_base64}",
                "result_image": f"data:image/png;base64,{result_image_base64}",
                "message": "이미지 합성이 완료되었습니다.",
                "gemini_response": result_text
            })
        else:
            return JSONResponse({
                "success": False,
                "error": "No image generated",
                "message": "Gemini API가 이미지를 생성하지 못했습니다. 응답: " + result_text,
                "gemini_response": result_text
            }, status_code=500)
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        
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

def upload_to_s3(file_content: bytes, file_name: str, content_type: str = "image/png") -> Optional[str]:
    """
    S3에 파일 업로드
    
    Args:
        file_content: 파일 내용 (bytes)
        file_name: 파일명
        content_type: MIME 타입
    
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
        s3_key = f"dresses/{file_name}"
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

