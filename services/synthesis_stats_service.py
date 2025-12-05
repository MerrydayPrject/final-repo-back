"""합성 통계 서비스"""
from datetime import date, datetime
from zoneinfo import ZoneInfo
from services.database import get_db_connection

# 한국 시간대(KST, UTC+9) 기준으로 날짜 계산
KST = ZoneInfo("Asia/Seoul")

def get_today_kst() -> date:
    """한국 시간 기준 오늘 날짜 반환"""
    return datetime.now(KST).date()


def increment_synthesis_count() -> bool:
    """
    오늘 날짜의 합성 카운트를 1 증가시킵니다.
    
    오늘 날짜(자정 기준)의 row가 없으면 자동으로 INSERT하고,
    있으면 UPDATE하여 카운트를 증가시킵니다.
    
    Returns:
        성공 여부 (True/False)
    """
    connection = get_db_connection()
    if not connection:
        print("DB 연결 실패 - 합성 카운트 증가 건너뜀")
        return False
    
    try:
        with connection.cursor() as cursor:
            today = get_today_kst()
            
            # UPSERT: 오늘 날짜의 row가 없으면 INSERT, 있으면 UPDATE
            insert_query = """
            INSERT INTO daily_synthesis_count (synthesis_date, count) 
            VALUES (%s, 1)
            ON DUPLICATE KEY UPDATE count = count + 1
            """
            cursor.execute(insert_query, (today,))
            connection.commit()
            print(f"합성 카운트 증가 완료: {today}")
            return True
    except Exception as e:
        print(f"합성 카운트 증가 오류: {e}")
        connection.rollback()
        return False
    finally:
        connection.close()

