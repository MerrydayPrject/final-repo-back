"""로그 저장 서비스"""
from typing import Optional
from services.database import get_db_connection


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

