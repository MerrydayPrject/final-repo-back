"""라인 퀴즈 테스트 라우터"""
import os
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import List, Dict, Optional
import json

from services.body_service import load_body_line_definitions
from services.database import get_db_connection
from core.s3_client import list_files_in_s3_folder
from config.settings import AWS_S3_BUCKET_NAME, AWS_REGION

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
async def get_line_images(
    folder: str = Query("line-quiz-images", description="S3 폴더명 (기본값: line-quiz-images)")
):
    """
    라인별 사진 리스트 조회 (S3 우선, 없으면 로컬)
    
    Args:
        folder: S3 폴더명 (기본값: "line-quiz-images")
    
    Returns:
        List[Dict]: 라인별 이미지 리스트
    """
    try:
        images = []
        bucket_name = os.getenv("AWS_S3_BUCKET_NAME") or AWS_S3_BUCKET_NAME or "marryday1"
        region = os.getenv("AWS_REGION") or AWS_REGION or "ap-northeast-2"
        
        # S3에서 파일 목록 가져오기
        s3_files = list_files_in_s3_folder(folder=folder, max_keys=10000)
        
        print(f"[DEBUG] S3 파일 개수: {len(s3_files) if s3_files else 0}, 폴더: {folder}")
        if s3_files:
            print(f"[DEBUG] 첫 번째 파일 예시: {s3_files[0] if len(s3_files) > 0 else 'None'}")
        
        if not s3_files:
            # S3에 파일이 없으면 로컬 폴더에서 가져오기 (대체)
            for line_type in LINE_TYPES:
                line_dir = TEST_IMAGES_DIR / line_type
                
                if not line_dir.exists():
                    continue
                
                image_extensions = {'.jpg', '.jpeg', '.png', '.gif'}
                for image_file in line_dir.iterdir():
                    if image_file.is_file() and image_file.suffix.lower() in image_extensions:
                        relative_path = image_file.relative_to(BASE_DIR)
                        images.append({
                            "line_type": line_type,
                            "image_path": str(relative_path).replace("\\", "/"),
                            "image_url": f"/test_images/{line_type}/{image_file.name}",
                            "filename": image_file.name,
                            "source": "local"
                        })
        else:
            # S3 파일들을 라인별로 분류
            for s3_file in s3_files:
                s3_key = s3_file.get("key", "")
                s3_url = s3_file.get("url", "")
                
                if not s3_key:
                    continue
                
                # S3 키에서 실제 파일명 추출
                actual_file_name = s3_key.split("/")[-1] if "/" in s3_key else s3_key
                
                # 이미지 파일 확장자 체크
                image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
                if not any(actual_file_name.lower().endswith(ext) for ext in image_extensions):
                    continue
                
                # 라인 타입 추출 (S3 경로 또는 파일명에서)
                line_type = None
                
                # 방법 1: S3 키 경로에서 라인 타입 추출 (하위 폴더 구조)
                # 예: "line-quiz-images/A라인/image.jpg" 또는 "line-quiz-images/A/.../image.jpg"
                for lt in LINE_TYPES:
                    if f"/{lt}/" in s3_key:
                        line_type = lt
                        break
                
                # 방법 1-2: 단일 문자 경로에서 라인 타입 추출 (A/, H/, O/, X/)
                if not line_type:
                    line_char_map = {"A": "A라인", "H": "H라인", "O": "O라인", "X": "X라인"}
                    for char, lt in line_char_map.items():
                        # "line-quiz-images/A/" 또는 "line-quiz-images/A/" 패턴 확인
                        if f"/{char}/" in s3_key or s3_key.startswith(f"line-quiz-images/{char}/"):
                            line_type = lt
                            break
                
                # 방법 2: 파일명에서 라인 타입 추출
                if not line_type:
                    for lt in LINE_TYPES:
                        if lt in actual_file_name:
                            line_type = lt
                            break
                
                # 방법 2-2: 파일명에서 단일 문자로 라인 타입 추출
                if not line_type:
                    line_char_map = {"A": "A라인", "H": "H라인", "O": "O라인", "X": "X라인"}
                    for char, lt in line_char_map.items():
                        if actual_file_name.startswith(f"{char}/") or f"/{char}/" in actual_file_name:
                            line_type = lt
                            break
                
                # 방법 3: 상대 경로에서 라인 타입 추출
                if not line_type:
                    file_name_path = s3_file.get("file_name", "")
                    for lt in LINE_TYPES:
                        if file_name_path.startswith(f"{lt}/"):
                            line_type = lt
                            break
                    
                    # 방법 3-2: 상대 경로에서 단일 문자로 라인 타입 추출
                    if not line_type:
                        line_char_map = {"A": "A라인", "H": "H라인", "O": "O라인", "X": "X라인"}
                        for char, lt in line_char_map.items():
                            if file_name_path.startswith(f"{char}/"):
                                line_type = lt
                                break
                
                # 라인 타입을 찾지 못한 경우 스킵
                if not line_type:
                    print(f"[DEBUG] 라인 타입을 찾을 수 없음: s3_key={s3_key}, file_name={actual_file_name}, file_name_path={s3_file.get('file_name', '')}")
                    continue
                
                images.append({
                    "line_type": line_type,
                    "image_path": s3_key,
                    "image_url": s3_url,
                    "filename": actual_file_name,
                    "source": "s3"
                })
        
        print(f"[DEBUG] 최종 이미지 개수: {len(images)}")
        
        return JSONResponse({
            "success": True,
            "images": images,
            "total": len(images),
            "folder": folder,
            "source": "s3" if s3_files else "local",
            "images_count": len(images)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
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
        image_url = data.get("image_url")
        user_answer = data.get("user_answer")
        correct_answer = data.get("correct_answer")
        user_id = data.get("user_id")
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
                            image_url VARCHAR(1000),
                            user_answer VARCHAR(50) NOT NULL,
                            correct_answer VARCHAR(50) NOT NULL,
                            is_correct BOOLEAN NOT NULL,
                            user_id VARCHAR(100),
                            user_name VARCHAR(100),
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            INDEX idx_image_path (image_path),
                            INDEX idx_user_id (user_id),
                            INDEX idx_created_at (created_at),
                            INDEX idx_is_correct (is_correct)
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """)
                    
                    # 결과 저장
                    cursor.execute("""
                        INSERT INTO line_quiz_results 
                        (image_path, image_url, user_answer, correct_answer, is_correct, user_id, user_name)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (image_path, image_url, user_answer, correct_answer, is_correct, user_id, user_name))
                    
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
                        image_url,
                        user_answer,
                        correct_answer,
                        is_correct,
                        user_id,
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
