"""피팅 프로파일링 저장 서비스"""
import json
from typing import Dict, Optional
from services.database import get_db_connection


def ensure_table_exists():
    """테이블이 없으면 자동 생성 (안전장치)"""
    connection = get_db_connection()
    if not connection:
        return False
    
    try:
        with connection.cursor() as cursor:
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS tryon_profile_summary (
                idx INT AUTO_INCREMENT PRIMARY KEY,
                trace_id VARCHAR(255) NOT NULL UNIQUE COMMENT '프론트엔드에서 생성한 추적 ID',
                endpoint VARCHAR(255) NOT NULL COMMENT '엔드포인트 경로 (/tryon/compare 또는 /tryon/compare/custom)',
                front_profile_json JSON COMMENT '프론트엔드 측정 구간 시간 (ms)',
                server_total_ms FLOAT COMMENT '서버 전체 처리 시간 (ms)',
                gemini_call_ms FLOAT COMMENT 'Gemini API 호출 시간 (ms)',
                cutout_ms FLOAT DEFAULT NULL COMMENT '의상 누끼 처리 시간 (ms, 커스텀만)',
                status VARCHAR(50) NOT NULL COMMENT '성공 여부 (success/fail)',
                error_stage VARCHAR(255) DEFAULT NULL COMMENT '에러 발생 단계 (선택사항)',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_trace_id (trace_id),
                INDEX idx_endpoint (endpoint),
                INDEX idx_created_at (created_at),
                INDEX idx_status (status)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='피팅 프로파일링 요약 테이블';
            """
            cursor.execute(create_table_sql)
            connection.commit()
            return True
    except Exception as e:
        print(f"[프로파일링] 테이블 생성 실패: {e}")
        return False
    finally:
        connection.close()


def save_tryon_profile(
    trace_id: str,
    endpoint: str,
    front_profile_json: Optional[Dict] = None,
    server_total_ms: Optional[float] = None,
    gemini_call_ms: Optional[float] = None,
    cutout_ms: Optional[float] = None,
    status: str = "success",
    error_stage: Optional[str] = None
) -> bool:
    """
    피팅 프로파일링 데이터 저장
    
    Args:
        trace_id: 추적 ID
        endpoint: 엔드포인트 경로
        front_profile_json: 프론트엔드 측정 구간 시간 (dict)
        server_total_ms: 서버 전체 처리 시간 (ms)
        gemini_call_ms: Gemini API 호출 시간 (ms)
        cutout_ms: 의상 누끼 처리 시간 (ms, 커스텀만)
        status: 성공 여부 (success/fail)
        error_stage: 에러 발생 단계 (선택사항)
    
    Returns:
        bool: 저장 성공 여부
    """
    # 테이블이 없으면 자동 생성 (안전장치)
    if not ensure_table_exists():
        print(f"[프로파일링] 테이블 생성 실패 - 저장 건너뜀")
        return False
    
    connection = get_db_connection()
    if not connection:
        print(f"[프로파일링] DB 연결 실패 - 저장 건너뜀")
        return False
    
    try:
        with connection.cursor() as cursor:
            # front_profile_json을 JSON 문자열로 변환
            front_profile_str = None
            if front_profile_json:
                try:
                    front_profile_str = json.dumps(front_profile_json, ensure_ascii=False)
                except Exception as e:
                    print(f"[프로파일링] JSON 변환 실패: {e}")
                    front_profile_str = None
            
            insert_sql = """
            INSERT INTO tryon_profile_summary 
            (trace_id, endpoint, front_profile_json, server_total_ms, gemini_call_ms, cutout_ms, status, error_stage)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                endpoint = VALUES(endpoint),
                front_profile_json = VALUES(front_profile_json),
                server_total_ms = VALUES(server_total_ms),
                gemini_call_ms = VALUES(gemini_call_ms),
                cutout_ms = VALUES(cutout_ms),
                status = VALUES(status),
                error_stage = VALUES(error_stage)
            """
            
            cursor.execute(
                insert_sql,
                (
                    trace_id,
                    endpoint,
                    front_profile_str,
                    server_total_ms,
                    gemini_call_ms,
                    cutout_ms,
                    status,
                    error_stage
                )
            )
            connection.commit()
            print(f"[프로파일링] 저장 완료: trace_id={trace_id}, endpoint={endpoint}, status={status}")
            return True
    except Exception as e:
        print(f"[프로파일링] 저장 실패: {e}")
        connection.rollback()
        return False
    finally:
        connection.close()

