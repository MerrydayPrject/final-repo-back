"""방문자 수 관리 라우터"""
from fastapi import APIRouter, HTTPException
from datetime import date, datetime
from zoneinfo import ZoneInfo
from services.database import get_db_connection

# 한국 시간대(KST, UTC+9) 기준으로 날짜 계산
KST = ZoneInfo("Asia/Seoul")

def get_today_kst() -> date:
    """한국 시간 기준 오늘 날짜 반환"""
    return datetime.now(KST).date()

router = APIRouter(prefix="/visitor", tags=["Visitor"])


@router.post("/visit")
async def increment_visitor():
    """오늘 방문자 수 증가"""
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="DB 연결 실패")
    
    try:
        with connection.cursor() as cursor:
            today = get_today_kst()
            # UPSERT: 있으면 증가, 없으면 삽입
            cursor.execute("""
                INSERT INTO daily_visitors (visit_date, count) 
                VALUES (%s, 1)
                ON DUPLICATE KEY UPDATE count = count + 1
            """, (today,))
            connection.commit()
            
            cursor.execute("SELECT count FROM daily_visitors WHERE visit_date = %s", (today,))
            result = cursor.fetchone()
            return {"date": str(today), "count": result['count']}
    finally:
        connection.close()


@router.get("/today")
async def get_today_visitors():
    """오늘 방문자 수 조회"""
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="DB 연결 실패")
    
    try:
        with connection.cursor() as cursor:
            today = get_today_kst()
            cursor.execute("SELECT count FROM daily_visitors WHERE visit_date = %s", (today,))
            result = cursor.fetchone()
            return {"date": str(today), "count": result['count'] if result else 0}
    finally:
        connection.close()


@router.get("/total")
async def get_total_visitors():
    """전체 방문자 수 조회"""
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="DB 연결 실패")
    
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT SUM(count) as total FROM daily_visitors")
            result = cursor.fetchone()
            return {"total": result['total'] or 0}
    finally:
        connection.close()

