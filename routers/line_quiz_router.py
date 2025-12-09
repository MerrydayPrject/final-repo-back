"""라인 퀴즈 테스트 라우터"""
import os
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import List, Dict, Optional
import json

from services.body_service import load_body_line_definitions
from services.database import get_db_connection

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# test_images 디렉토리 경로
BASE_DIR = Path(__file__).parent.parent
TEST_IMAGES_DIR = BASE_DIR / "test_images"
LINE_TYPES = ["A라인", "H라인", "O라인", "X라인"]


@router.get("/line-quiz", response_class=HTMLResponse, tags=["라인 퀴즈"])
async def line_quiz_page(request: Request):
    """라인 퀴즈 테스트 페이지"""
    return templates.TemplateResponse("line_quiz.html", {"request": request})


@router.get("/line-quiz-stats", response_class=HTMLResponse, tags=["라인 퀴즈"])
async def line_quiz_stats_page(request: Request):
    """라인 퀴즈 통계 페이지"""
    return templates.TemplateResponse("line_quiz_stats.html", {"request": request})


@router.get("/api/line-quiz/images", tags=["라인 퀴즈"])
async def get_line_images():
    """
    라인별 사진 리스트 조회
    
    Returns:
        List[Dict]: 라인별 이미지 리스트
        [
            {
                "line_type": "A라인",
                "image_path": "test_images/A라인/image1.jpg",
                "image_url": "/static/test_images/A라인/image1.jpg",
                "filename": "image1.jpg"
            },
            ...
        ]
    """
    try:
        images = []
        
        for line_type in LINE_TYPES:
            line_dir = TEST_IMAGES_DIR / line_type
            
            if not line_dir.exists():
                continue
            
            # 이미지 파일만 필터링
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif'}
            for image_file in line_dir.iterdir():
                if image_file.is_file() and image_file.suffix.lower() in image_extensions:
                    # 상대 경로로 변환
                    relative_path = image_file.relative_to(BASE_DIR)
                    images.append({
                        "line_type": line_type,
                        "image_path": str(relative_path).replace("\\", "/"),
                        "image_url": f"/test_images/{line_type}/{image_file.name}",
                        "filename": image_file.name
                    })
        
        return JSONResponse({
            "success": True,
            "images": images,
            "total": len(images)
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@router.get("/api/line-quiz/definitions", tags=["라인 퀴즈"])
async def get_line_definitions():
    """
    라인별 정의 조회
    
    Returns:
        Dict: 라인별 정의
        {
            "A라인": "정의 텍스트...",
            "H라인": "정의 텍스트...",
            ...
        }
    """
    try:
        definitions = load_body_line_definitions()
        
        # 라인별로 정리
        result = {}
        for line_type in LINE_TYPES:
            if line_type in definitions:
                result[line_type] = definitions[line_type]
            else:
                result[line_type] = "정의가 없습니다."
        
        return JSONResponse({
            "success": True,
            "definitions": result
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@router.post("/api/line-quiz/submit", tags=["라인 퀴즈"])
async def submit_quiz_answer(data: dict):
    """
    퀴즈 정답 제출 및 저장
    
    Request Body:
        {
            "image_path": "test_images/A라인/image1.jpg",
            "user_answer": "A라인",
            "correct_answer": "A라인",  # 선택사항 (없으면 자동 판별)
            "user_name": "사용자명"  # 선택사항
        }
    
    Returns:
        Dict: 제출 결과
    """
    try:
        image_path = data.get("image_path")
        user_answer = data.get("user_answer")
        correct_answer = data.get("correct_answer")
        user_name = data.get("user_name", "익명")
        
        if not image_path or not user_answer:
            return JSONResponse({
                "success": False,
                "error": "image_path와 user_answer는 필수입니다."
            }, status_code=400)
        
        # 정답이 없으면 이미지 경로에서 추출
        if not correct_answer:
            for line_type in LINE_TYPES:
                if line_type in image_path:
                    correct_answer = line_type
                    break
        
        if not correct_answer:
            return JSONResponse({
                "success": False,
                "error": "정답을 확인할 수 없습니다."
            }, status_code=400)
        
        # 정답 여부 확인
        is_correct = user_answer == correct_answer
        
        # DB에 저장
        connection = get_db_connection()
        if connection:
            try:
                with connection.cursor() as cursor:
                    # line_quiz_results 테이블이 없으면 생성
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS line_quiz_results (
                            idx INT AUTO_INCREMENT PRIMARY KEY,
                            image_path VARCHAR(500) NOT NULL,
                            user_answer VARCHAR(50) NOT NULL,
                            correct_answer VARCHAR(50) NOT NULL,
                            is_correct BOOLEAN NOT NULL,
                            user_name VARCHAR(100),
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            INDEX idx_image_path (image_path),
                            INDEX idx_created_at (created_at)
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """)
                    
                    # 결과 저장
                    cursor.execute("""
                        INSERT INTO line_quiz_results 
                        (image_path, user_answer, correct_answer, is_correct, user_name)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (image_path, user_answer, correct_answer, is_correct, user_name))
                    
                    connection.commit()
                    result_id = cursor.lastrowid
                    
            except Exception as e:
                print(f"[WARN] 퀴즈 결과 저장 오류: {e}")
                connection.rollback()
            finally:
                connection.close()
        
        return JSONResponse({
            "success": True,
            "is_correct": is_correct,
            "user_answer": user_answer,
            "correct_answer": correct_answer,
            "message": "정답입니다!" if is_correct else f"틀렸습니다. 정답은 {correct_answer}입니다."
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@router.get("/api/line-quiz/results", tags=["라인 퀴즈"])
async def get_quiz_results(limit: int = 50, offset: int = 0):
    """
    퀴즈 결과 조회
    
    Args:
        limit: 조회할 레코드 수 (기본값: 50)
        offset: 시작 위치 (기본값: 0)
    
    Returns:
        Dict: 퀴즈 결과 리스트 및 통계
    """
    try:
        connection = get_db_connection()
        if not connection:
            return JSONResponse({
                "success": False,
                "error": "데이터베이스 연결 실패"
            }, status_code=500)
        
        try:
            with connection.cursor() as cursor:
                # 결과 조회
                cursor.execute("""
                    SELECT 
                        idx,
                        image_path,
                        user_answer,
                        correct_answer,
                        is_correct,
                        user_name,
                        created_at
                    FROM line_quiz_results
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """, (limit, offset))
                
                results = cursor.fetchall()
                
                # 통계 조회
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct_count,
                        SUM(CASE WHEN is_correct = 0 THEN 1 ELSE 0 END) as incorrect_count
                    FROM line_quiz_results
                """)
                
                stats = cursor.fetchone()
                
                # 결과 포맷팅
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "idx": result.get("idx"),
                        "image_path": result.get("image_path"),
                        "user_answer": result.get("user_answer"),
                        "correct_answer": result.get("correct_answer"),
                        "is_correct": bool(result.get("is_correct")),
                        "user_name": result.get("user_name"),
                        "created_at": result.get("created_at").isoformat() if result.get("created_at") else None
                    })
                
                return JSONResponse({
                    "success": True,
                    "results": formatted_results,
                    "statistics": {
                        "total": stats.get("total", 0) if stats else 0,
                        "correct": stats.get("correct_count", 0) if stats else 0,
                        "incorrect": stats.get("incorrect_count", 0) if stats else 0,
                        "accuracy": round(stats.get("correct_count", 0) / stats.get("total", 1) * 100, 2) if stats and stats.get("total", 0) > 0 else 0
                    }
                })
                
        except Exception as e:
            print(f"[WARN] 퀴즈 결과 조회 오류: {e}")
            return JSONResponse({
                "success": False,
                "error": str(e)
            }, status_code=500)
        finally:
            connection.close()
            
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)
