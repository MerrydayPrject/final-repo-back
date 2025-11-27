"""방문자 수 관리 라우터"""
from fastapi import APIRouter, HTTPException
from datetime import date
from services.database import get_db_connection

router = APIRouter(prefix="/visitor", tags=["Visitor"])


@router.post("/visit")
async def increment_visitor():
    """오늘 방문자 수 증가"""
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="DB 연결 실패")
    
    try:
        with connection.cursor() as cursor:
            today = date.today()
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
            today = date.today()
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

