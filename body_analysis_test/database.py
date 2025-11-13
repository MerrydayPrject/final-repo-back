"""
체형 분석용 데이터베이스 연결 모듈
체형별 정의 데이터를 조회하는 함수 제공
"""
import pymysql
import os
import json
from typing import Optional, Dict, List
from dotenv import load_dotenv
from pathlib import Path

# .env 파일 로드 (상위 디렉토리에서도 찾기)
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


def get_db_connection():
    """
    데이터베이스 연결 생성
    
    Returns:
        pymysql.Connection: 데이터베이스 연결 객체
    """
    host = os.getenv("MYSQL_HOST", "localhost")
    port = int(os.getenv("MYSQL_PORT", 3306))
    user = os.getenv("MYSQL_USER", "devuser")
    password = os.getenv("MYSQL_PASSWORD", "")
    database = os.getenv("MYSQL_DATABASE", "marryday")
    
    try:
        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        return connection
    except Exception as e:
        print(f"[WARN] 데이터베이스 연결 실패: {e}")
        return None


def get_body_type_definition(body_feature: str) -> Optional[Dict]:
    """
    체형별 정의 데이터 조회
    
    Args:
        body_feature: 체형 특징 ('키가 작은 체형', '글래머러스한 체형', '어깨가 넓은 체형' 등)
    
    Returns:
        Dict: 체형별 정의 데이터 (없으면 None)
        {
            'body_feature': str,
            'strengths': str,
            'style_tips': str,
            'recommended_dresses': str,
            'avoid_dresses': str
        }
    """
    connection = get_db_connection()
    if connection is None:
        return None
    
    try:
        with connection.cursor() as cursor:
            # body_type_definitions 테이블에서 조회
            sql = """
                SELECT 
                    body_feature,
                    strengths,
                    style_tips,
                    recommended_dresses,
                    avoid_dresses
                FROM body_type_definitions
                WHERE body_feature = %s
                LIMIT 1
            """
            cursor.execute(sql, (body_feature,))
            result = cursor.fetchone()
            
            if result:
                return {
                    'body_feature': result.get('body_feature'),
                    'strengths': result.get('strengths', ''),
                    'style_tips': result.get('style_tips', ''),
                    'recommended_dresses': result.get('recommended_dresses', ''),
                    'avoid_dresses': result.get('avoid_dresses', '')
                }
            else:
                print(f"[INFO] 체형 특징 '{body_feature}'에 대한 정의가 DB에 없습니다.")
                return None
                
    except pymysql.Error as e:
        print(f"[WARN] DB 조회 오류: {e}")
        return None
    finally:
        connection.close()


def get_multiple_body_definitions(body_features: List[str]) -> List[Dict]:
    """
    여러 체형 특징에 대한 정의 데이터 조회
    
    Args:
        body_features: 체형 특징 리스트
    
    Returns:
        List[Dict]: 체형별 정의 데이터 리스트
    """
    if not body_features:
        return []
    
    connection = get_db_connection()
    if connection is None:
        return []
    
    try:
        with connection.cursor() as cursor:
            # 여러 체형 특징 조회
            placeholders = ','.join(['%s'] * len(body_features))
            sql = f"""
                SELECT 
                    body_feature,
                    strengths,
                    style_tips,
                    recommended_dresses,
                    avoid_dresses
                FROM body_type_definitions
                WHERE body_feature IN ({placeholders})
            """
            cursor.execute(sql, body_features)
            results = cursor.fetchall()
            
            definitions = []
            for result in results:
                definitions.append({
                    'body_feature': result.get('body_feature'),
                    'strengths': result.get('strengths', ''),
                    'style_tips': result.get('style_tips', ''),
                    'recommended_dresses': result.get('recommended_dresses', ''),
                    'avoid_dresses': result.get('avoid_dresses', '')
                })
            
            return definitions
                
    except pymysql.Error as e:
        print(f"[WARN] DB 조회 오류: {e}")
        return []
    finally:
        connection.close()


def format_body_type_info_for_prompt(definitions: List[Dict]) -> str:
    """
    체형별 정의 데이터를 Gemini 프롬프트에 포함할 형식으로 변환
    
    Args:
        definitions: get_multiple_body_definitions()의 반환값 (리스트)
    
    Returns:
        str: 프롬프트에 포함할 텍스트 (없으면 빈 문자열)
    """
    if not definitions:
        return ""
    
    parts = []
    
    for definition in definitions:
        body_feature = definition.get('body_feature', '')
        
        if definition.get('strengths'):
            parts.append(f"**{body_feature}**: {definition['strengths']}")
        
        if definition.get('style_tips'):
            parts.append(f"  - 스타일 팁: {definition['style_tips']}")
        
        if definition.get('recommended_dresses'):
            parts.append(f"  - 추천 드레스: {definition['recommended_dresses']}")
        
        if definition.get('avoid_dresses'):
            parts.append(f"  - 피해야 할 드레스: {definition['avoid_dresses']}")
        
        parts.append("")  # 빈 줄 추가
    
    if parts:
        return "\n\n**체형별 정의 정보**:\n" + "\n".join(parts) + "\n"
    else:
        return ""


def save_body_analysis_result(
    model: str = 'body_analysis',
    run_time: float = 0.0,
    height: Optional[float] = None,
    weight: Optional[float] = None,
    prompt: str = '체형 분석',
    bmi: Optional[float] = None,
    characteristic: Optional[str] = None,
    analysis_results: Optional[str] = None
) -> Optional[int]:
    """
    체형 분석 결과를 body_logs 테이블에 저장
    
    Args:
        model: 모델명 (기본값: 'body_analysis')
        run_time: 처리 시간 (초)
        height: 키 (cm)
        weight: 몸무게 (kg)
        prompt: AI 명령어 (프롬프트)
        bmi: BMI 수치
        characteristic: 체형 특징 (문자열, 쉼표로 구분 가능)
        analysis_results: 분석 결과 (텍스트)
    
    Returns:
        int: 저장된 레코드의 idx (실패 시 None)
    """
    connection = get_db_connection()
    if connection is None:
        return None
    
    try:
        with connection.cursor() as cursor:
            sql = """
                INSERT INTO body_logs (
                    model,
                    run_time,
                    height,
                    weight,
                    prompt,
                    bmi,
                    characteristic,
                    analysis_results
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s
                )
            """
            
            cursor.execute(sql, (
                model,
                run_time,
                height,
                weight,
                prompt,
                bmi,
                characteristic,
                analysis_results
            ))
            
            connection.commit()
            return cursor.lastrowid
            
    except pymysql.Error as e:
        print(f"[WARN] 체형 분석 결과 저장 오류: {e}")
        connection.rollback()
        return None
    finally:
        connection.close()


def get_body_logs(limit: int = 50, offset: int = 0) -> List[Dict]:
    """
    body_logs 테이블에서 체형 분석 로그 조회
    
    Args:
        limit: 조회할 레코드 수 (기본값: 50)
        offset: 시작 위치 (기본값: 0)
    
    Returns:
        List[Dict]: 체형 분석 로그 리스트
    """
    connection = get_db_connection()
    if connection is None:
        return []
    
    try:
        with connection.cursor() as cursor:
            sql = """
                SELECT 
                    idx,
                    model,
                    run_time,
                    height,
                    weight,
                    prompt,
                    bmi,
                    characteristic,
                    analysis_results,
                    created_at
                FROM body_logs
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """
            cursor.execute(sql, (limit, offset))
            results = cursor.fetchall()
            
            logs = []
            for result in results:
                logs.append({
                    'idx': result.get('idx'),
                    'model': result.get('model'),
                    'run_time': result.get('run_time'),
                    'height': result.get('height'),
                    'weight': result.get('weight'),
                    'prompt': result.get('prompt'),
                    'bmi': result.get('bmi'),
                    'characteristic': result.get('characteristic'),
                    'analysis_results': result.get('analysis_results'),
                    'created_at': result.get('created_at').isoformat() if result.get('created_at') else None
                })
            
            return logs
                
    except pymysql.Error as e:
        print(f"[WARN] DB 조회 오류: {e}")
        return []
    finally:
        connection.close()


def get_body_logs_count() -> int:
    """
    body_logs 테이블의 전체 레코드 수 조회
    
    Returns:
        int: 전체 레코드 수
    """
    connection = get_db_connection()
    if connection is None:
        return 0
    
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM body_logs")
            result = cursor.fetchone()
            return result.get('count', 0) if result else 0
                
    except pymysql.Error as e:
        print(f"[WARN] DB 조회 오류: {e}")
        return 0
    finally:
        connection.close()

