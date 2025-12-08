"""드레스 관리 라우터"""
import base64
import os
import json
import csv
import io
import pymysql
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, File, UploadFile, Form, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response
from PIL import Image

from services.database import get_db_connection
from services.category_service import detect_style_from_filename
from services.dress_check_service import get_dress_check_service
from core.s3_client import upload_to_s3, delete_from_s3
from config.settings import AWS_S3_BUCKET_NAME, AWS_REGION

router = APIRouter()


@router.get("/api/admin/dresses", tags=["드레스 관리"])
async def get_dresses(
    page: int = Query(1, ge=1, description="페이지 번호"),
    limit: int = Query(20, ge=1, le=10000, description="페이지당 항목 수")
):
    """
    드레스 목록 조회 (페이징 지원)
    
    데이터베이스에 저장된 드레스 정보를 페이지별로 반환합니다.
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


@router.post("/api/admin/dresses", tags=["드레스 관리"])
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
                    bucket_name = AWS_S3_BUCKET_NAME or "marryday1"
                    region = AWS_REGION or "ap-northeast-2"
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


@router.post("/api/admin/dresses/upload", tags=["드레스 관리"])
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
                            
                            results.append({
                                "file_name": file_name,
                                "success": True,
                                "style": style,
                                "url": s3_url
                            })
                            success_count += 1
                        except pymysql.IntegrityError as e:
                            results.append({
                                "file_name": file_name,
                                "success": False,
                                "error": f"데이터베이스 오류: {str(e)}"
                            })
                            fail_count += 1
                except Exception as e:
                    results.append({
                        "file_name": file.filename,
                        "success": False,
                        "error": str(e)
                    })
                    fail_count += 1
        finally:
            connection.close()
        
        return JSONResponse({
            "success": True,
            "message": f"{success_count}개 파일 업로드 성공, {fail_count}개 파일 실패",
            "results": results,
            "success_count": success_count,
            "fail_count": fail_count
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"드레스 업로드 중 오류 발생: {str(e)}"
        }, status_code=500)


@router.delete("/api/admin/dresses/{dress_id}", tags=["드레스 관리"])
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


@router.get("/api/admin/dresses/export", tags=["드레스 관리"])
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
                        content=csv_content.encode('utf-8-sig'),
                        media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=dresses.csv"}
                    )
                else:
                    # JSON 형식으로 내보내기
                    return JSONResponse({
                        "success": True,
                        "data": dresses,
                        "total": len(dresses)
                    })
        finally:
            connection.close()
            
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"드레스 내보내기 중 오류 발생: {str(e)}"
        }, status_code=500)


@router.post("/api/admin/dresses/import", tags=["드레스 관리"])
async def import_dresses(file: UploadFile = File(...)):
    """
    드레스 목록 가져오기
    
    JSON 또는 CSV 파일로 드레스 정보를 일괄 가져옵니다.
    """
    try:
        import json
        import csv
        import io
        import os
        import pymysql
        
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
        
        # JSON 파일이 {"success": true, "data": [...]} 형식인 경우 처리
        if isinstance(data, dict) and 'data' in data:
            data = data['data']
        
        # 배열이 아닌 경우 에러 반환
        if not isinstance(data, list):
            return JSONResponse({
                "success": False,
                "error": "Invalid data format",
                "message": "JSON 파일은 배열 형식이거나 {'data': [...]} 형식이어야 합니다."
            }, status_code=400)
        
        if len(data) == 0:
            return JSONResponse({
                "success": False,
                "error": "Empty data",
                "message": "가져올 데이터가 없습니다."
            }, status_code=400)
        
        connection = get_db_connection()
        if not connection:
            return JSONResponse({
                "success": False,
                "error": "Database connection failed",
                "message": "데이터베이스 연결에 실패했습니다. 서버 로그를 확인하거나 .env 파일의 데이터베이스 설정을 확인하세요."
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
                    "fail": fail_count
                },
                "results": results,
                "message": f"{success_count}개 항목 가져오기 성공, {fail_count}개 항목 실패"
            })
        finally:
            connection.close()
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"데이터 가져오기 중 오류 발생: {str(e)}"
        }, status_code=500)


@router.post("/api/dress/check", tags=["드레스 관리"])
async def check_single_dress(
    file: UploadFile = File(...),
    model: str = Form("gpt-4o-mini"),
    mode: str = Form("fast")
):
    """
    단일 이미지로 드레스 여부 판별
    """
    try:
        # 모델 검증
        if model not in ["gpt-4o-mini", "gpt-4o"]:
            return JSONResponse({
                "success": False,
                "error": "Invalid model",
                "message": "모델은 gpt-4o-mini 또는 gpt-4o만 선택해주세요."
            }, status_code=400)

        # 모드 검증
        if mode not in ["fast", "accurate"]:
            return JSONResponse({
                "success": False,
                "error": "Invalid mode",
                "message": "모드는 fast 또는 accurate만 선택해주세요."
            }, status_code=400)

        try:
            dress_check_service = get_dress_check_service()
        except Exception as e:
            return JSONResponse({
                "success": False,
                "error": "Service initialization failed",
                "message": f"서비스 초기화 실패: {str(e)}"
            }, status_code=500)

        connection = get_db_connection()
        if not connection:
            return JSONResponse({
                "success": False,
                "error": "Database connection failed",
                "message": "데이터베이스 연결에 실패했습니다."
            }, status_code=500)

        try:
            file_content = await file.read()
            if not file_content:
                return JSONResponse({
                    "success": False,
                    "error": "Empty file",
                    "message": "업로드된 파일이 비어있습니다."
                }, status_code=400)

            # 5MB 용량 제한
            if len(file_content) > 5 * 1024 * 1024:
                return JSONResponse({
                    "success": False,
                    "error": "File too large",
                    "message": "파일 크기는 5MB 이하로 업로드해주세요."
                }, status_code=400)

            import hashlib

            try:
                image = Image.open(io.BytesIO(file_content)).convert("RGB")
            except Exception as e:
                return JSONResponse({
                    "success": False,
                    "error": "Invalid image",
                    "message": f"이미지 로드 실패: {str(e)}"
                }, status_code=400)

            # 이미지 해시 생성
            image_hash = hashlib.md5(file_content).hexdigest()

            # 웨딩드레스 여부 판별 (업로드 차단 전용)
            evaluation = dress_check_service.check_wedding_dress(
                image=image,
                model=model,
                mode=mode
            )
            raw_result = evaluation.get("raw", {})
            predicted_dress = bool(raw_result.get("dress", False))
            confidence = evaluation.get("confidence", 0.0)
            category = evaluation.get("category", "")

            # 썸네일 생성
            thumbnail = None
            try:
                preview = image.copy()
                preview.thumbnail((300, 300))
                buffered = io.BytesIO()
                preview.save(buffered, format="PNG")
                thumb_b64 = base64.b64encode(buffered.getvalue()).decode()
                thumbnail = f"data:image/png;base64,{thumb_b64}"
            except Exception:
                pass

            # 로그 저장 헬퍼
            def log_check(predicted_flag: bool) -> Optional[int]:
                try:
                    with connection.cursor() as cursor:
                        cursor.execute("""
                            INSERT INTO dress_check_logs
                            (filename, image_hash, model, mode, predicted_dress, confidence, category)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """, (
                            file.filename or "uploaded_image",
                            image_hash,
                            model,
                            mode,
                            predicted_flag,
                            confidence,
                            category
                        ))
                        connection.commit()
                        return cursor.lastrowid
                except Exception as db_error:
                    print(f"DB 기록 오류: {db_error}")
                    return None

            # 웨딩드레스 여부 판별
            record_id = None  # 변수 초기화
            if not evaluation.get("is_wedding_dress", False):
                record_id = log_check(predicted_dress)
                return JSONResponse({
                    "success": False,
                    "error": "NotWeddingDress",
                    "message": "웨딩드레스 이미지가 아니어서 업로드를 차단했습니다.",
                    "result": {
                        "filename": file.filename or "uploaded_image",
                        "dress": False,
                        "is_wedding_dress": False,
                        "confidence": confidence,
                        "category": category,
                        "thumbnail": thumbnail,
                        "record_id": record_id
                    }
                }, status_code=400)

            # 정상 처리
            record_id = log_check(predicted_dress)
            return JSONResponse({
                "success": True,
                "result": {
                    "filename": file.filename or "uploaded_image",
                    "dress": predicted_dress,
                    "is_wedding_dress": True,
                    "confidence": confidence,
                    "category": category,
                    "thumbnail": thumbnail,
                    "record_id": record_id
                },
                "message": "판별이 완료되었습니다."
            })

        finally:
            connection.close()

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"단일 판별 처리 중 오류 발생: {str(e)}"
        }, status_code=500)


@router.post("/api/dress/batch-check", tags=["드레스 관리"])
async def batch_check_dresses(
    files: List[UploadFile] = File(...),
    model: str = Form(...),
    mode: str = Form(...)
):
    """
    여러 이미지를 배치로 업로드하여 드레스 여부를 판별
    
    Args:
        files: 업로드할 이미지 파일 리스트 (최대 100장)
        model: 사용할 모델 (gpt-4o-mini 또는 gpt-4o)
        mode: 모드 (fast 또는 accurate)
    
    Returns:
        {
            "success": bool,
            "results": [
                {
                    "index": int,
                    "filename": str,
                    "dress": bool,
                    "confidence": float,
                    "category": str
                }
            ]
        }
    """
    try:
        # 파일 수 제한
        if len(files) > 100:
            return JSONResponse({
                "success": False,
                "error": "Too many files",
                "message": "최대 100장까지만 업로드할 수 있습니다."
            }, status_code=400)
        
        # 모델 검증
        if model not in ["gpt-4o-mini", "gpt-4o"]:
            return JSONResponse({
                "success": False,
                "error": "Invalid model",
                "message": "모델은 gpt-4o-mini 또는 gpt-4o만 선택할 수 있습니다."
            }, status_code=400)
        
        # 모드 검증
        if mode not in ["fast", "accurate"]:
            return JSONResponse({
                "success": False,
                "error": "Invalid mode",
                "message": "모드는 fast 또는 accurate만 선택할 수 있습니다."
            }, status_code=400)
        
        # 드레스 판별 서비스 가져오기
        try:
            dress_check_service = get_dress_check_service()
        except Exception as e:
            return JSONResponse({
                "success": False,
                "error": "Service initialization failed",
                "message": f"서비스 초기화 실패: {str(e)}"
            }, status_code=500)
        
        # DB 연결
        connection = get_db_connection()
        if not connection:
            return JSONResponse({
                "success": False,
                "error": "Database connection failed",
                "message": "데이터베이스 연결에 실패했습니다."
            }, status_code=500)
        
        results = []
        import hashlib
        
        try:
            import asyncio
            
            # 각 파일 처리
            for index, file in enumerate(files):
                try:
                    # 파일 읽기
                    file_content = await file.read()
                    
                    # 이미지 해시 생성
                    image_hash = hashlib.md5(file_content).hexdigest()
                    
                    # 이미지로 변환
                    try:
                        image = Image.open(io.BytesIO(file_content)).convert("RGB")
                    except Exception as e:
                        results.append({
                            "index": index,
                            "filename": file.filename or f"file_{index}",
                            "dress": False,
                            "confidence": 0.0,
                            "category": f"이미지 로드 실패: {str(e)}",
                            "record_id": None
                        })
                        continue
                    
                    # Rate limit 방지를 위한 지연 (각 요청 사이에 0.1초 대기)
                    if index > 0:
                        await asyncio.sleep(0.1)
                    
                    # 드레스 판별
                    check_result = dress_check_service.check_dress(
                        image=image,
                        model=model,
                        mode=mode
                    )
                    
                    # 썸네일 생성 (base64)
                    thumbnail = None
                    try:
                        # 썸네일 크기로 리사이즈
                        image.thumbnail((200, 200))
                        import base64
                        buffered = io.BytesIO()
                        image.save(buffered, format="PNG")
                        thumbnail = base64.b64encode(buffered.getvalue()).decode()
                        thumbnail = f"data:image/png;base64,{thumbnail}"
                    except Exception:
                        pass  # 썸네일 생성 실패해도 계속 진행
                    
                    # DB에 저장
                    record_id = None
                    try:
                        with connection.cursor() as cursor:
                            cursor.execute("""
                                INSERT INTO dress_check_logs 
                                (filename, image_hash, model, mode, predicted_dress, confidence, category)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """, (
                                file.filename or f"file_{index}",
                                image_hash,
                                model,
                                mode,
                                check_result["dress"],
                                check_result["confidence"],
                                check_result["category"]
                            ))
                            connection.commit()
                            record_id = cursor.lastrowid
                    except Exception as db_error:
                        print(f"DB 저장 오류: {db_error}")
                        # DB 저장 실패해도 결과는 반환
                    
                    results.append({
                        "index": index,
                        "filename": file.filename or f"file_{index}",
                        "dress": check_result["dress"],
                        "confidence": check_result["confidence"],
                        "category": check_result["category"],
                        "thumbnail": thumbnail,
                        "record_id": record_id
                    })
                    
                except Exception as e:
                    # 개별 파일 처리 실패 시에도 계속 진행
                    results.append({
                        "index": index,
                        "filename": file.filename or f"file_{index}",
                        "dress": False,
                        "confidence": 0.0,
                        "category": f"처리 오류: {str(e)}",
                        "record_id": None
                    })
                    continue
        finally:
            connection.close()
        
        return JSONResponse({
            "success": True,
            "results": results,
            "message": f"{len(results)}개 이미지 처리 완료"
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"배치 처리 중 오류 발생: {str(e)}"
        }, status_code=500)


@router.post("/api/dress/verify", tags=["드레스 관리"])
async def verify_dress_check(request: Request):
    """
    드레스 판별 결과 검수 저장
    
    Args:
        Body: {
            "record_id": int,
            "verified_dress": bool
        }
    
    Returns:
        {
            "success": bool,
            "message": str
        }
    """
    try:
        body = await request.json()
        record_id = body.get("record_id")
        verified_dress = body.get("verified_dress")
        
        if record_id is None:
            return JSONResponse({
                "success": False,
                "error": "Missing record_id",
                "message": "record_id는 필수입니다."
            }, status_code=400)
        
        if verified_dress is None:
            return JSONResponse({
                "success": False,
                "error": "Missing verified_dress",
                "message": "verified_dress는 필수입니다."
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
                # 레코드 존재 확인
                cursor.execute("SELECT id FROM dress_check_logs WHERE id = %s", (record_id,))
                if not cursor.fetchone():
                    return JSONResponse({
                        "success": False,
                        "error": "Record not found",
                        "message": f"레코드 ID {record_id}를 찾을 수 없습니다."
                    }, status_code=404)
                
                # 검수 결과 업데이트
                cursor.execute("""
                    UPDATE dress_check_logs 
                    SET verified_dress = %s, 
                        is_verified = TRUE, 
                        verified_at = NOW()
                    WHERE id = %s
                """, (bool(verified_dress), record_id))
                connection.commit()
                
                return JSONResponse({
                    "success": True,
                    "message": "검수 결과가 저장되었습니다."
                })
        finally:
            connection.close()
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"검수 저장 중 오류 발생: {str(e)}"
        }, status_code=500)


@router.get("/api/dress/metrics", tags=["드레스 관리"])
async def get_dress_metrics(
    days: Optional[int] = Query(None, description="최근 N일 데이터만 사용"),
    limit: Optional[int] = Query(None, description="최근 N건 데이터만 사용")
):
    """
    드레스 판별 성능 지표 조회
    
    Args:
        days: 최근 N일 데이터만 사용 (선택)
        limit: 최근 N건 데이터만 사용 (선택)
    
    Returns:
        {
            "success": bool,
            "confusion_matrix": {
                "TP": int,
                "FP": int,
                "FN": int,
                "TN": int
            },
            "metrics": {
                "precision": float,
                "recall": float,
                "f1": float,
                "accuracy": float
            },
            "sample_count": int
        }
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
                # 검수 완료된 데이터만 조회
                query = """
                    SELECT predicted_dress, verified_dress
                    FROM dress_check_logs
                    WHERE is_verified = TRUE AND verified_dress IS NOT NULL
                """
                params = []
                
                if days:
                    query += " AND created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)"
                    params.append(days)
                
                query += " ORDER BY created_at DESC"
                
                if limit:
                    query += " LIMIT %s"
                    params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Confusion Matrix 계산
                TP = 0  # True Positive: 예측 드레스, 실제 드레스
                FP = 0  # False Positive: 예측 드레스, 실제 일반옷
                FN = 0  # False Negative: 예측 일반옷, 실제 드레스
                TN = 0  # True Negative: 예측 일반옷, 실제 일반옷
                
                for row in rows:
                    predicted = bool(row['predicted_dress'])
                    verified = bool(row['verified_dress'])
                    
                    if predicted and verified:
                        TP += 1
                    elif predicted and not verified:
                        FP += 1
                    elif not predicted and verified:
                        FN += 1
                    else:  # not predicted and not verified
                        TN += 1
                
                # 성능 지표 계산
                sample_count = len(rows)
                
                # Precision = TP / (TP + FP)
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                
                # Recall = TP / (TP + FN)
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                
                # F1 = 2 * (Precision * Recall) / (Precision + Recall)
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                # Accuracy = (TP + TN) / (TP + FP + FN + TN)
                accuracy = (TP + TN) / sample_count if sample_count > 0 else 0.0
                
                return JSONResponse({
                    "success": True,
                    "confusion_matrix": {
                        "TP": TP,
                        "FP": FP,
                        "FN": FN,
                        "TN": TN
                    },
                    "metrics": {
                        "precision": round(precision, 4),
                        "recall": round(recall, 4),
                        "f1": round(f1, 4),
                        "accuracy": round(accuracy, 4)
                    },
                    "sample_count": sample_count
                })
        finally:
            connection.close()
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"성능 지표 조회 중 오류 발생: {str(e)}"
        }, status_code=500)
