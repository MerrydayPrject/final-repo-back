"""관리자 라우터"""
# 이 파일은 원본 main.py의 관리자 관련 엔드포인트를 포함합니다.
# 원본 파일이 매우 크므로, 실제 구현은 원본 main.py에서 복사하여 추가해야 합니다.
#
# 포함할 엔드포인트:
# - GET /api/admin/stats
# - GET /api/admin/logs
# - GET /api/admin/logs/{log_id}
# - GET /api/admin/category-rules
# - POST /api/admin/category-rules
# - DELETE /api/admin/category-rules

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from typing import Optional
from services.database import get_db_connection
from services.category_service import load_category_rules, save_category_rules
from config.auth_middleware import require_admin

router = APIRouter()


@router.get("/api/admin/stats", tags=["관리자"])
async def get_admin_stats(request: Request):
    """
    관리자 통계 정보 조회
    
    result_logs 테이블에서 통계 정보를 조회합니다.
    """
    # 인증 확인
    await require_admin(request)
    
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


@router.get("/api/admin/logs", tags=["관리자"])
async def get_admin_logs(
    request: Request,
    page: int = Query(1, ge=1, description="페이지 번호"),
    limit: int = Query(20, ge=1, le=100, description="페이지당 항목 수"),
    model: Optional[str] = Query(None, description="모델명으로 검색")
):
    """
    관리자 로그 목록 조회
    
    result_logs 테이블에서 로그 목록을 조회합니다.
    """
    # 인증 확인
    await require_admin(request)
    
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
                        result_url,
                        person_url,
                        dress_url
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


@router.get("/api/admin/logs/{log_id}", tags=["관리자"])
async def get_admin_log_detail(request: Request, log_id: int):
    """
    관리자 로그 상세 정보 조회
    
    특정 로그의 상세 정보를 조회합니다.
    """
    # 인증 확인
    await require_admin(request)
    
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
                # 먼저 테이블 구조 확인 (created_at 컬럼 존재 여부)
                cursor.execute("SHOW COLUMNS FROM result_logs LIKE 'created_at'")
                has_created_at = cursor.fetchone() is not None
                
                # created_at 컬럼이 있으면 포함, 없으면 제외
                if has_created_at:
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
                            created_at
                        FROM result_logs
                        WHERE idx = %s
                    """, (log_id,))
                else:
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
                
                # 안전하게 필드 접근
                created_at = log.get('created_at')
                if created_at and hasattr(created_at, 'isoformat'):
                    created_at = created_at.isoformat()
                elif created_at:
                    created_at = str(created_at)
                else:
                    created_at = None
                
                # run_time 안전하게 처리
                run_time = log.get('run_time')
                if run_time is not None:
                    try:
                        run_time_float = float(run_time)
                        processing_time = f"{run_time_float:.2f}초"
                    except (ValueError, TypeError):
                        processing_time = str(run_time) if run_time else "-"
                else:
                    processing_time = "-"
                
                return JSONResponse({
                    "success": True,
                    "data": {
                        "id": log.get('id') or log.get('idx'),
                        "person_url": log.get('person_url'),
                        "dress_url": log.get('dress_url'),
                        "result_url": log.get('result_url'),
                        "model": log.get('model'),
                        "prompt": log.get('prompt'),
                        "success": log.get('success'),
                        "processing_time": processing_time,
                        "created_at": created_at
                    }
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


@router.get("/api/admin/category-rules", tags=["카테고리 규칙"])
async def get_category_rules(request: Request):
    """
    카테고리 규칙 목록 조회
    """
    # 인증 확인
    await require_admin(request)
    
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


@router.post("/api/admin/category-rules", tags=["카테고리 규칙"])
async def add_category_rule(request: Request):
    """
    새 카테고리 규칙 추가
    """
    # 인증 확인
    await require_admin(request)
    
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


@router.delete("/api/admin/category-rules", tags=["카테고리 규칙"])
async def delete_category_rule(request: Request):
    """
    카테고리 규칙 삭제
    """
    # 인증 확인
    await require_admin(request)
    
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


@router.get("/api/admin/daily-synthesis-stats", tags=["관리자"])
async def get_daily_synthesis_stats(
    request: Request,
    page: int = Query(1, ge=1, description="페이지 번호"),
    limit: int = Query(20, ge=1, le=100, description="페이지당 항목 수"),
    date: Optional[str] = Query(None, description="날짜 검색 (YYYY-MM-DD)")
):
    """
    날짜별 합성 통계 조회
    
    daily_synthesis_count 테이블에서 날짜별 합성 통계 목록을 조회합니다.
    """
    # 인증 확인
    await require_admin(request)
    
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
                # 날짜 필터 조건
                where_clause = ""
                params = []
                
                if date:
                    where_clause = "WHERE synthesis_date = %s"
                    params.append(date)
                
                try:
                    # 전체 건수 조회 (테이블이 없으면 예외 발생)
                    count_query = f"SELECT COUNT(*) as total FROM daily_synthesis_count {where_clause}"
                    if params:
                        cursor.execute(count_query, tuple(params))
                    else:
                        cursor.execute(count_query)
                    count_result = cursor.fetchone()
                    total = count_result['total'] if count_result else 0
                except Exception as table_error:
                    # 테이블이 없으면 빈 결과 반환
                    error_str = str(table_error).lower()
                    if any(keyword in error_str for keyword in ["table", "doesn't exist", "unknown table", "1146"]):
                        return JSONResponse({
                            "success": True,
                            "data": [],
                            "pagination": {
                                "page": page,
                                "limit": limit,
                                "total": 0,
                                "total_pages": 0
                            },
                            "message": "테이블이 아직 생성되지 않았습니다. 합성을 실행하면 자동으로 생성됩니다."
                        })
                    raise  # 다른 오류는 다시 발생시킴
                
                # 총 페이지 수 계산
                total_pages = (total + limit - 1) // limit if total > 0 else 0
                
                # 오프셋 계산
                offset = (page - 1) * limit
                
                # 날짜별 합성 통계 목록 조회
                try:
                    query = f"""
                        SELECT 
                            id,
                            synthesis_date,
                            count
                        FROM daily_synthesis_count
                        {where_clause}
                        ORDER BY synthesis_date DESC
                        LIMIT %s OFFSET %s
                    """
                    query_params = params + [limit, offset]
                    cursor.execute(query, tuple(query_params))
                    stats = cursor.fetchall() or []
                except Exception as query_error:
                    # 쿼리 실행 실패 시 빈 결과 반환
                    error_str = str(query_error).lower()
                    if any(keyword in error_str for keyword in ["table", "doesn't exist", "unknown table", "1146"]):
                        stats = []
                    else:
                        raise  # 다른 오류는 다시 발생시킴
                
                # 날짜 형식 변환 (JSON 직렬화 가능하도록)
                formatted_stats = []
                for stat in stats:
                    synthesis_date = stat.get('synthesis_date')
                    # date 객체를 문자열로 변환
                    date_str = None
                    if synthesis_date:
                        if hasattr(synthesis_date, 'isoformat'):
                            date_str = synthesis_date.isoformat()
                        else:
                            date_str = str(synthesis_date)
                    
                    formatted_stat = {
                        'id': stat.get('id'),
                        'date': date_str,
                        'count': stat.get('count', 0)
                    }
                    formatted_stats.append(formatted_stat)
                
                return JSONResponse({
                    "success": True,
                    "data": formatted_stats,
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
        import traceback
        error_detail = traceback.format_exc()
        print(f"날짜별 합성 통계 조회 오류: {error_detail}")
        
        # 테이블이 없거나 SQL 오류인 경우 빈 결과 반환
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in ["table", "doesn't exist", "unknown table", "1146"]):
            return JSONResponse({
                "success": True,
                "data": [],
                "pagination": {
                    "page": page,
                    "limit": limit,
                    "total": 0,
                    "total_pages": 0
                },
                "message": "테이블이 아직 생성되지 않았습니다. 합성을 한 번 실행하면 테이블이 생성됩니다."
            })
        
        return JSONResponse({
            "success": False,
            "error": str(e),
            "error_detail": error_detail,
            "message": f"날짜별 합성 통계 조회 중 오류 발생: {str(e)}"
        }, status_code=500)


@router.get("/api/admin/daily-visitor-stats", tags=["관리자"])
async def get_daily_visitor_stats(
    request: Request,
    page: int = Query(1, ge=1, description="페이지 번호"),
    limit: int = Query(20, ge=1, le=100, description="페이지당 항목 수"),
    date: Optional[str] = Query(None, description="날짜 검색 (YYYY-MM-DD)")
):
    """
    날짜별 조회수 통계 조회
    
    daily_visitors 테이블에서 날짜별 방문자 수 목록을 조회합니다.
    """
    # 인증 확인
    await require_admin(request)
    
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
                # 날짜 필터 조건
                where_clause = ""
                params = []
                
                if date:
                    where_clause = "WHERE visit_date = %s"
                    params.append(date)
                
                try:
                    # 전체 건수 조회 (테이블이 없으면 예외 발생)
                    count_query = f"SELECT COUNT(*) as total FROM daily_visitors {where_clause}"
                    if params:
                        cursor.execute(count_query, tuple(params))
                    else:
                        cursor.execute(count_query)
                    count_result = cursor.fetchone()
                    total = count_result['total'] if count_result else 0
                except Exception as table_error:
                    # 테이블이 없으면 빈 결과 반환
                    error_str = str(table_error).lower()
                    if any(keyword in error_str for keyword in ["table", "doesn't exist", "unknown table", "1146"]):
                        return JSONResponse({
                            "success": True,
                            "data": [],
                            "pagination": {
                                "page": page,
                                "limit": limit,
                                "total": 0,
                                "total_pages": 0
                            },
                            "message": "테이블이 아직 생성되지 않았습니다."
                        })
                    raise  # 다른 오류는 다시 발생시킴
                
                # 총 페이지 수 계산
                total_pages = (total + limit - 1) // limit if total > 0 else 0
                
                # 오프셋 계산
                offset = (page - 1) * limit
                
                # 날짜별 방문자 통계 목록 조회
                try:
                    query = f"""
                        SELECT 
                            id,
                            visit_date,
                            count
                        FROM daily_visitors
                        {where_clause}
                        ORDER BY visit_date DESC
                        LIMIT %s OFFSET %s
                    """
                    query_params = params + [limit, offset]
                    cursor.execute(query, tuple(query_params))
                    stats = cursor.fetchall() or []
                except Exception as query_error:
                    # 쿼리 실행 실패 시 빈 결과 반환
                    error_str = str(query_error).lower()
                    if any(keyword in error_str for keyword in ["table", "doesn't exist", "unknown table", "1146"]):
                        stats = []
                    else:
                        raise  # 다른 오류는 다시 발생시킴
                
                # 날짜 형식 변환 (JSON 직렬화 가능하도록)
                formatted_stats = []
                for stat in stats:
                    visit_date = stat.get('visit_date')
                    # date 객체를 문자열로 변환
                    date_str = None
                    if visit_date:
                        if hasattr(visit_date, 'isoformat'):
                            date_str = visit_date.isoformat()
                        else:
                            date_str = str(visit_date)
                    
                    formatted_stat = {
                        'id': stat.get('id'),
                        'date': date_str,
                        'count': stat.get('count', 0)
                    }
                    formatted_stats.append(formatted_stat)
                
                return JSONResponse({
                    "success": True,
                    "data": formatted_stats,
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
        import traceback
        error_detail = traceback.format_exc()
        print(f"날짜별 조회수 통계 조회 오류: {error_detail}")
        
        # 테이블이 없거나 SQL 오류인 경우 빈 결과 반환
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in ["table", "doesn't exist", "unknown table", "1146"]):
            return JSONResponse({
                "success": True,
                "data": [],
                "pagination": {
                    "page": page,
                    "limit": limit,
                    "total": 0,
                    "total_pages": 0
                },
                "message": "테이블이 아직 생성되지 않았습니다."
            })
        
        return JSONResponse({
            "success": False,
            "error": str(e),
            "error_detail": error_detail,
            "message": f"날짜별 조회수 통계 조회 중 오류 발생: {str(e)}"
        }, status_code=500)


@router.get("/api/admin/custom-fitting-logs", tags=["관리자"])
async def get_custom_fitting_logs(
    request: Request,
    page: int = Query(1, ge=1, description="페이지 번호"),
    limit: int = Query(20, ge=1, le=100, description="페이지당 항목 수")
):
    """
    커스텀 피팅 로그 목록 조회
    
    result_logs 테이블에서 dress_url이 있는 항목만 조회합니다.
    created_at, run_time, dress_url 필드를 반환합니다.
    """
    # 인증 확인
    await require_admin(request)
    
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
                # dress_url이 NULL이 아닌 항목만 필터링
                where_clause = "WHERE dress_url IS NOT NULL"
                
                # 전체 건수 조회
                count_query = f"SELECT COUNT(*) as total FROM result_logs {where_clause}"
                cursor.execute(count_query)
                total = cursor.fetchone()['total']
                
                # 총 페이지 수 계산
                total_pages = (total + limit - 1) // limit if total > 0 else 0
                
                # 오프셋 계산
                offset = (page - 1) * limit
                
                # 로그 목록 조회
                query = f"""
                    SELECT 
                        idx as id,
                        created_at,
                        run_time,
                        dress_url
                    FROM result_logs
                    {where_clause}
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """
                cursor.execute(query, (limit, offset))
                
                logs = cursor.fetchall()
                
                # 데이터 형식 변환 (created_at을 문자열로 변환)
                from datetime import datetime
                formatted_logs = []
                for log in logs:
                    created_at = log.get('created_at')
                    # created_at을 ISO 형식 문자열로 변환
                    if created_at:
                        try:
                            # datetime 객체인 경우
                            if isinstance(created_at, datetime):
                                created_at = created_at.isoformat()
                            elif hasattr(created_at, 'isoformat'):
                                created_at = created_at.isoformat()
                            elif hasattr(created_at, 'strftime'):
                                # date 객체인 경우
                                created_at = created_at.strftime('%Y-%m-%dT%H:%M:%S')
                            elif isinstance(created_at, str):
                                # 이미 문자열인 경우, MySQL datetime 형식인지 확인
                                # YYYY-MM-DD HH:MM:SS 형식을 ISO 형식으로 변환
                                if ' ' in created_at and 'T' not in created_at:
                                    created_at = created_at.replace(' ', 'T')
                                # 타임존 정보가 없으면 추가하지 않음 (프론트엔드에서 처리)
                            else:
                                # 다른 형식인 경우 문자열로 변환
                                created_at = str(created_at)
                        except Exception as e:
                            # 변환 실패 시 문자열로 변환
                            print(f"created_at 변환 오류: {e}, 값: {created_at}")
                            created_at = str(created_at) if created_at else None
                    else:
                        created_at = None
                    
                    formatted_log = {
                        'id': log.get('id'),
                        'created_at': created_at,
                        'run_time': log.get('run_time'),
                        'dress_url': log.get('dress_url')
                    }
                    formatted_logs.append(formatted_log)
                
                return JSONResponse({
                    "success": True,
                    "data": formatted_logs,
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
        import traceback
        error_detail = traceback.format_exc()
        print(f"커스텀 피팅 로그 조회 오류: {error_detail}")
        return JSONResponse({
            "success": False,
            "error": str(e),
            "error_detail": error_detail,
            "message": f"커스텀 피팅 로그 조회 중 오류 발생: {str(e)}"
        }, status_code=500)


@router.get("/api/admin/tryon-profile-logs", tags=["관리자"])
async def get_tryon_profile_logs(
    request: Request,
    page: int = Query(1, ge=1, description="페이지 번호"),
    limit: int = Query(20, ge=1, le=100, description="페이지당 항목 수"),
    endpoint: Optional[str] = Query(None, description="엔드포인트 필터 (/tryon/compare 또는 /tryon/compare/custom)")
):
    """
    피팅 프로파일링 로그 목록 조회
    
    tryon_profile_summary 테이블에서 프로파일링 로그 목록을 조회합니다.
    endpoint 파라미터로 일반 피팅(/tryon/compare) 또는 커스텀 피팅(/tryon/compare/custom)을 필터링할 수 있습니다.
    """
    # 인증 확인
    await require_admin(request)
    
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
                # 엔드포인트 필터 조건
                where_clause = ""
                params = []
                
                if endpoint:
                    where_clause = "WHERE endpoint = %s"
                    params.append(endpoint)
                
                # 전체 건수 조회
                try:
                    count_query = f"SELECT COUNT(*) as total FROM tryon_profile_summary {where_clause}"
                    if params:
                        cursor.execute(count_query, tuple(params))
                    else:
                        cursor.execute(count_query)
                    count_result = cursor.fetchone()
                    total = count_result['total'] if count_result else 0
                except Exception as table_error:
                    # 테이블이 없으면 빈 결과 반환
                    error_str = str(table_error).lower()
                    if any(keyword in error_str for keyword in ["table", "doesn't exist", "unknown table", "1146"]):
                        return JSONResponse({
                            "success": True,
                            "data": [],
                            "pagination": {
                                "page": page,
                                "limit": limit,
                                "total": 0,
                                "total_pages": 0
                            },
                            "message": "테이블이 아직 생성되지 않았습니다. 피팅을 실행하면 자동으로 생성됩니다."
                        })
                    raise
                
                # 총 페이지 수 계산
                total_pages = (total + limit - 1) // limit if total > 0 else 0
                
                # 오프셋 계산
                offset = (page - 1) * limit
                
                # 프로파일링 로그 목록 조회
                try:
                    query = f"""
                        SELECT 
                            idx,
                            trace_id,
                            endpoint,
                            front_profile_json,
                            server_total_ms,
                            gemini_call_ms,
                            cutout_ms,
                            status,
                            error_stage,
                            created_at
                        FROM tryon_profile_summary
                        {where_clause}
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s
                    """
                    query_params = params + [limit, offset]
                    cursor.execute(query, tuple(query_params))
                    logs = cursor.fetchall() or []
                except Exception as query_error:
                    error_str = str(query_error).lower()
                    if any(keyword in error_str for keyword in ["table", "doesn't exist", "unknown table", "1146"]):
                        logs = []
                    else:
                        raise
                
                # 데이터 형식 변환
                from datetime import datetime
                formatted_logs = []
                for log in logs:
                    created_at = log.get('created_at')
                    if created_at:
                        try:
                            if isinstance(created_at, datetime):
                                created_at = created_at.isoformat()
                            elif hasattr(created_at, 'isoformat'):
                                created_at = created_at.isoformat()
                            elif hasattr(created_at, 'strftime'):
                                created_at = created_at.strftime('%Y-%m-%dT%H:%M:%S')
                            elif isinstance(created_at, str) and ' ' in created_at and 'T' not in created_at:
                                created_at = created_at.replace(' ', 'T')
                            else:
                                created_at = str(created_at)
                        except Exception as e:
                            created_at = str(created_at) if created_at else None
                    else:
                        created_at = None
                    
                    # front_profile_json 파싱
                    front_profile = None
                    front_profile_str = log.get('front_profile_json')
                    if front_profile_str:
                        try:
                            import json
                            front_profile = json.loads(front_profile_str) if isinstance(front_profile_str, str) else front_profile_str
                        except:
                            front_profile = None
                    
                    formatted_log = {
                        'id': log.get('idx'),
                        'trace_id': log.get('trace_id'),
                        'endpoint': log.get('endpoint'),
                        'front_profile': front_profile,
                        'server_total_ms': log.get('server_total_ms'),
                        'gemini_call_ms': log.get('gemini_call_ms'),
                        'cutout_ms': log.get('cutout_ms'),
                        'status': log.get('status'),
                        'error_stage': log.get('error_stage'),
                        'created_at': created_at
                    }
                    formatted_logs.append(formatted_log)
                
                return JSONResponse({
                    "success": True,
                    "data": formatted_logs,
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
        import traceback
        error_detail = traceback.format_exc()
        print(f"프로파일링 로그 조회 오류: {error_detail}")
        
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in ["table", "doesn't exist", "unknown table", "1146"]):
            return JSONResponse({
                "success": True,
                "data": [],
                "pagination": {
                    "page": page,
                    "limit": limit,
                    "total": 0,
                    "total_pages": 0
                },
                "message": "테이블이 아직 생성되지 않았습니다. 피팅을 실행하면 자동으로 생성됩니다."
            })
        
        return JSONResponse({
            "success": False,
            "error": str(e),
            "error_detail": error_detail,
            "message": f"프로파일링 로그 조회 중 오류 발생: {str(e)}"
        }, status_code=500)


@router.get("/api/admin/tryon-profile-logs/{log_id}", tags=["관리자"])
async def get_tryon_profile_log_detail(request: Request, log_id: int):
    """
    피팅 프로파일링 로그 상세 정보 조회
    
    특정 프로파일링 로그의 상세 정보를 조회합니다.
    """
    # 인증 확인
    await require_admin(request)
    
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
                    SELECT 
                        idx,
                        trace_id,
                        endpoint,
                        front_profile_json,
                        server_total_ms,
                        gemini_call_ms,
                        cutout_ms,
                        status,
                        error_stage,
                        created_at
                    FROM tryon_profile_summary
                    WHERE idx = %s
                """, (log_id,))
                
                log = cursor.fetchone()
                
                if not log:
                    return JSONResponse({
                        "success": False,
                        "error": "Log not found",
                        "message": f"로그 ID {log_id}를 찾을 수 없습니다."
                    }, status_code=404)
                
                # created_at 변환
                created_at = log.get('created_at')
                if created_at:
                    try:
                        from datetime import datetime
                        if isinstance(created_at, datetime):
                            created_at = created_at.isoformat()
                        elif hasattr(created_at, 'isoformat'):
                            created_at = created_at.isoformat()
                        elif hasattr(created_at, 'strftime'):
                            created_at = created_at.strftime('%Y-%m-%dT%H:%M:%S')
                        elif isinstance(created_at, str) and ' ' in created_at and 'T' not in created_at:
                            created_at = created_at.replace(' ', 'T')
                        else:
                            created_at = str(created_at)
                    except:
                        created_at = str(created_at) if created_at else None
                else:
                    created_at = None
                
                # front_profile_json 파싱
                front_profile = None
                front_profile_str = log.get('front_profile_json')
                if front_profile_str:
                    try:
                        import json
                        front_profile = json.loads(front_profile_str) if isinstance(front_profile_str, str) else front_profile_str
                    except:
                        front_profile = None
                
                return JSONResponse({
                    "success": True,
                    "data": {
                        "id": log.get('idx'),
                        "trace_id": log.get('trace_id'),
                        "endpoint": log.get('endpoint'),
                        "front_profile": front_profile,
                        "server_total_ms": log.get('server_total_ms'),
                        "gemini_call_ms": log.get('gemini_call_ms'),
                        "cutout_ms": log.get('cutout_ms'),
                        "status": log.get('status'),
                        "error_stage": log.get('error_stage'),
                        "created_at": created_at
                    }
                })
        finally:
            connection.close()
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"프로파일링 로그 상세 조회 오류: {error_detail}")
        return JSONResponse({
            "success": False,
            "error": str(e),
            "error_detail": error_detail,
            "message": f"프로파일링 로그 상세 조회 중 오류 발생: {str(e)}"
        }, status_code=500)

