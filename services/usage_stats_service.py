"""기능별 사용횟수 통계 서비스"""
from services.database import get_db_connection


def get_general_fitting_count() -> int:
    """
    일반피팅 사용횟수 조회
    
    result_logs 테이블에서 model="general-fitting"이고 success=TRUE인 레코드 수를 반환합니다.
    
    Returns:
        int: 일반피팅 사용횟수
    """
    connection = get_db_connection()
    if not connection:
        print("DB 연결 실패 - 일반피팅 카운트 조회 건너뜀")
        return 0
    
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) as count FROM result_logs WHERE model = %s AND success = TRUE",
                ("general-fitting",)
            )
            result = cursor.fetchone()
            return result.get('count', 0) if result else 0
    except Exception as e:
        print(f"일반피팅 카운트 조회 오류: {e}")
        return 0
    finally:
        connection.close()


def get_custom_fitting_count() -> int:
    """
    커스텀피팅 사용횟수 조회
    
    result_logs 테이블에서 model="custom-fitting"이고 success=TRUE인 레코드 수를 반환합니다.
    
    Returns:
        int: 커스텀피팅 사용횟수
    """
    connection = get_db_connection()
    if not connection:
        print("DB 연결 실패 - 커스텀피팅 카운트 조회 건너뜀")
        return 0
    
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) as count FROM result_logs WHERE model = %s AND success = TRUE",
                ("custom-fitting",)
            )
            result = cursor.fetchone()
            return result.get('count', 0) if result else 0
    except Exception as e:
        print(f"커스텀피팅 카운트 조회 오류: {e}")
        return 0
    finally:
        connection.close()


def get_body_analysis_count() -> int:
    """
    체형분석 사용횟수 조회
    
    body_logs 테이블의 전체 레코드 수를 반환합니다.
    (body_logs는 이미 성공한 것만 저장되므로 별도 success 조건 불필요)
    
    Returns:
        int: 체형분석 사용횟수
    """
    connection = get_db_connection()
    if not connection:
        print("DB 연결 실패 - 체형분석 카운트 조회 건너뜀")
        return 0
    
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM body_logs")
            result = cursor.fetchone()
            return result.get('count', 0) if result else 0
    except Exception as e:
        print(f"체형분석 카운트 조회 오류: {e}")
        return 0
    finally:
        connection.close()


def get_all_usage_counts() -> dict:
    """
    모든 기능의 사용횟수를 조회하여 반환
    
    Returns:
        dict: {
            "general_fitting": int,
            "custom_fitting": int,
            "body_analysis": int,
            "total": int
        }
    """
    general_count = get_general_fitting_count()
    custom_count = get_custom_fitting_count()
    body_count = get_body_analysis_count()
    
    return {
        "general_fitting": general_count,
        "custom_fitting": custom_count,
        "body_analysis": body_count,
        "total": general_count + custom_count + body_count
    }

