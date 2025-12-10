"""기능별 사용횟수 통계 서비스"""
from datetime import date, datetime
from zoneinfo import ZoneInfo
from typing import Optional
from services.database import get_db_connection

# 한국 시간대(KST, UTC+9) 기준으로 날짜 계산
KST = ZoneInfo("Asia/Seoul")

# 카운팅 기준일 (이 날짜 이후부터만 카운팅)
# 일반피팅과 커스텀피팅 로그 저장 기능이 추가된 날짜를 기준으로 설정
# 오늘 날짜부터 카운팅하도록 설정 (필요시 변경 가능)
COUNTING_START_DATE = datetime.now(KST).date()


def get_general_fitting_count(start_date: Optional[date] = None) -> int:
    """
    일반피팅 사용횟수 조회
    
    result_logs 테이블에서 model="general-fitting"이고 success=TRUE인 레코드 수를 반환합니다.
    start_date 이후의 데이터만 카운팅합니다.
    
    Args:
        start_date: 카운팅 시작 날짜 (기본값: COUNTING_START_DATE)
    
    Returns:
        int: 일반피팅 사용횟수
    """
    if start_date is None:
        start_date = COUNTING_START_DATE
    
    connection = get_db_connection()
    if not connection:
        print("DB 연결 실패 - 일반피팅 카운트 조회 건너뜀")
        return 0
    
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) as count FROM result_logs WHERE model = %s AND success = TRUE AND DATE(created_at) >= %s",
                ("general-fitting", start_date)
            )
            result = cursor.fetchone()
            return result.get('count', 0) if result else 0
    except Exception as e:
        print(f"일반피팅 카운트 조회 오류: {e}")
        return 0
    finally:
        connection.close()


def get_custom_fitting_count(start_date: Optional[date] = None) -> int:
    """
    커스텀피팅 사용횟수 조회
    
    result_logs 테이블에서 model="custom-fitting"이고 success=TRUE인 레코드 수를 반환합니다.
    start_date 이후의 데이터만 카운팅합니다.
    
    Args:
        start_date: 카운팅 시작 날짜 (기본값: COUNTING_START_DATE)
    
    Returns:
        int: 커스텀피팅 사용횟수
    """
    if start_date is None:
        start_date = COUNTING_START_DATE
    
    connection = get_db_connection()
    if not connection:
        print("DB 연결 실패 - 커스텀피팅 카운트 조회 건너뜀")
        return 0
    
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) as count FROM result_logs WHERE model = %s AND success = TRUE AND DATE(created_at) >= %s",
                ("custom-fitting", start_date)
            )
            result = cursor.fetchone()
            return result.get('count', 0) if result else 0
    except Exception as e:
        print(f"커스텀피팅 카운트 조회 오류: {e}")
        return 0
    finally:
        connection.close()


def get_body_analysis_count(start_date: Optional[date] = None) -> int:
    """
    체형분석 사용횟수 조회
    
    body_logs 테이블의 레코드 수를 반환합니다.
    start_date 이후의 데이터만 카운팅합니다.
    (body_logs는 이미 성공한 것만 저장되므로 별도 success 조건 불필요)
    
    Args:
        start_date: 카운팅 시작 날짜 (기본값: COUNTING_START_DATE)
    
    Returns:
        int: 체형분석 사용횟수
    """
    if start_date is None:
        start_date = COUNTING_START_DATE
    
    connection = get_db_connection()
    if not connection:
        print("DB 연결 실패 - 체형분석 카운트 조회 건너뜀")
        return 0
    
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) as count FROM body_logs WHERE DATE(created_at) >= %s",
                (start_date,)
            )
            result = cursor.fetchone()
            return result.get('count', 0) if result else 0
    except Exception as e:
        print(f"체형분석 카운트 조회 오류: {e}")
        return 0
    finally:
        connection.close()


def get_all_usage_counts(start_date: Optional[date] = None) -> dict:
    """
    모든 기능의 사용횟수를 조회하여 반환
    
    Args:
        start_date: 카운팅 시작 날짜 (기본값: COUNTING_START_DATE)
    
    Returns:
        dict: {
            "general_fitting": int,
            "custom_fitting": int,
            "body_analysis": int,
            "total": int,
            "start_date": str (카운팅 시작 날짜)
        }
    """
    if start_date is None:
        start_date = COUNTING_START_DATE
    
    general_count = get_general_fitting_count(start_date)
    custom_count = get_custom_fitting_count(start_date)
    body_count = get_body_analysis_count(start_date)
    
    return {
        "general_fitting": general_count,
        "custom_fitting": custom_count,
        "body_analysis": body_count,
        "total": general_count + custom_count + body_count,
        "start_date": str(start_date)
    }

