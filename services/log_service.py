"""로그 저장 서비스"""
from typing import Optional
from services.database import get_db_connection


def save_custom_fitting_log(
    dress_url: str,
    run_time: float
) -> bool:
    """
    커스텀 피팅 로그를 MySQL에 저장 (created_at, run_time, dress_url만 저장)
    
    Args:
        dress_url: 의상 이미지 S3 URL
        run_time: 실행 시간 (초)
    
    Returns:
        저장 성공 여부 (True/False)
    """
    connection = get_db_connection()
    if not connection:
        print("DB 연결 실패 - 커스텀 피팅 로그 저장 건너뜀")
        return False
    
    try:
        with connection.cursor() as cursor:
            # 필수 필드에는 최소한의 더미 값을 넣고, 실제로 필요한 3가지만 저장
            # created_at은 데이터베이스에서 자동으로 생성됨 (DEFAULT CURRENT_TIMESTAMP)
            insert_query = """
            INSERT INTO result_logs (person_url, dress_url, result_url, model, prompt, success, run_time)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (
                "",  # person_url (필수 필드이므로 빈 문자열)
                dress_url,  # dress_url (실제 저장할 값)
                "",  # result_url (필수 필드이므로 빈 문자열)
                "custom-fitting",  # model (필수 필드이므로 기본값)
                "",  # prompt (필수 필드이므로 빈 문자열)
                True,  # success (필수 필드이므로 기본값)
                run_time  # run_time (실제 저장할 값)
            ))
            connection.commit()
            print(f"커스텀 피팅 로그 저장 완료: dress_url={dress_url}, run_time={run_time:.2f}초")
            return True
    except Exception as e:
        print(f"커스텀 피팅 로그 저장 오류: {e}")
        connection.rollback()
        return False
    finally:
        connection.close()


def save_general_fitting_log(
    run_time: float,
    dress_url: Optional[str] = None
) -> bool:
    """
    일반 피팅 로그를 MySQL에 저장 (created_at, run_time만 저장, person_url과 result_url은 저장하지 않음)
    
    Args:
        run_time: 실행 시간 (초)
        dress_url: 의상 이미지 S3 URL (선택사항)
    
    Returns:
        저장 성공 여부 (True/False)
    """
    connection = get_db_connection()
    if not connection:
        print("DB 연결 실패 - 일반 피팅 로그 저장 건너뜀")
        return False
    
    try:
        with connection.cursor() as cursor:
            # 필수 필드에는 최소한의 더미 값을 넣고, 실제로 필요한 값만 저장
            # created_at은 데이터베이스에서 자동으로 생성됨 (DEFAULT CURRENT_TIMESTAMP)
            insert_query = """
            INSERT INTO result_logs (person_url, dress_url, result_url, model, prompt, success, run_time)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (
                "",  # person_url (필수 필드이므로 빈 문자열, 저장하지 않음)
                dress_url if dress_url else None,  # dress_url (선택사항)
                "",  # result_url (필수 필드이므로 빈 문자열, 저장하지 않음)
                "general-fitting",  # model (일반피팅 구분용)
                "",  # prompt (필수 필드이므로 빈 문자열)
                True,  # success (성공한 경우만 저장)
                run_time  # run_time (실제 저장할 값)
            ))
            connection.commit()
            print(f"일반 피팅 로그 저장 완료: run_time={run_time:.2f}초")
            return True
    except Exception as e:
        print(f"일반 피팅 로그 저장 오류: {e}")
        connection.rollback()
        return False
    finally:
        connection.close()


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

