"""드레스 피팅 로그 서비스"""
from services.database import get_db_connection


def log_dress_fitting(dress_id: int) -> bool:
    """
    드레스 피팅 로그를 기록합니다.
    
    Args:
        dress_id: 드레스 ID (dresses.idx)
    
    Returns:
        성공 여부 (True/False)
    """
    connection = get_db_connection()
    if not connection:
        print("DB 연결 실패 - 드레스 피팅 로그 기록 건너뜀")
        return False
    
    try:
        with connection.cursor() as cursor:
            # dress_fitting_logs 테이블에 로그 추가
            insert_query = """
            INSERT INTO dress_fitting_logs (dress_id)
            VALUES (%s)
            """
            cursor.execute(insert_query, (dress_id,))
            connection.commit()
            
            print(f"[드레스 피팅 로그] dress_id: {dress_id} 기록 완료")
            return True
    except Exception as e:
        print(f"드레스 피팅 로그 기록 오류: {e}")
        connection.rollback()
        return False
    finally:
        connection.close()
