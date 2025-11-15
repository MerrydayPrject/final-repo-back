"""데이터베이스 연결 및 초기화"""
import pymysql
from config.database import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, 
    MYSQL_PASSWORD, MYSQL_DATABASE
)


def get_db_connection():
    """MySQL 데이터베이스 연결 반환"""
    try:
        connection = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        return connection
    except pymysql.Error as e:
        error_msg = str(e)
        print(f"DB 연결 오류: {error_msg}")
        # 에러 타입에 따른 상세 메시지
        if "Access denied" in error_msg or "1045" in error_msg:
            print("⚠️  데이터베이스 인증 실패. .env 파일의 MYSQL_USER와 MYSQL_PASSWORD를 확인하세요.")
        elif "Unknown database" in error_msg or "1049" in error_msg:
            print("⚠️  데이터베이스가 존재하지 않습니다. 'marryday' 데이터베이스를 생성하세요.")
        elif "Can't connect" in error_msg or "2003" in error_msg:
            print("⚠️  MySQL 서버에 연결할 수 없습니다. MySQL 서비스가 실행 중인지 확인하세요.")
        else:
            print(f"⚠️  데이터베이스 연결 오류: {error_msg}")
        return None
    except Exception as e:
        print(f"DB 연결 오류 (예상치 못한 오류): {e}")
        return None


def init_database():
    """데이터베이스 테이블 생성"""
    connection = get_db_connection()
    if not connection:
        print("DB 연결 실패 - 테이블 생성 건너뜀")
        return
    
    try:
        with connection.cursor() as cursor:
            # dresses 테이블 생성
            create_dresses_table = """
            CREATE TABLE IF NOT EXISTS dresses (
                idx INT AUTO_INCREMENT PRIMARY KEY,
                dress_name VARCHAR(255) NOT NULL UNIQUE,
                file_name VARCHAR(255) NOT NULL,
                style VARCHAR(255) NOT NULL,
                url TEXT NOT NULL,
                INDEX idx_file_name (file_name),
                INDEX idx_style (style)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """
            cursor.execute(create_dresses_table)
            
            # 기존 테이블에 UNIQUE 제약 조건 추가 (테이블이 이미 존재하는 경우)
            try:
                cursor.execute("ALTER TABLE dresses ADD UNIQUE KEY uk_dress_name (dress_name)")
                print("UNIQUE 제약 조건 추가 완료: dress_name")
            except Exception as e:
                # 이미 제약 조건이 존재하거나 테이블이 없는 경우는 무시
                if "Duplicate key name" not in str(e) and "Unknown column" not in str(e):
                    print(f"UNIQUE 제약 조건 추가 시도: {e}")
            
            connection.commit()
            print("DB 테이블 생성 완료: dresses")
            
            # result_logs 테이블 생성
            create_result_logs_table = """
            CREATE TABLE IF NOT EXISTS result_logs (
                idx INT AUTO_INCREMENT PRIMARY KEY,
                person_url TEXT NOT NULL COMMENT '인물 이미지 (Input 1)',
                dress_url TEXT COMMENT '의상 이미지 (Input 2)',
                result_url TEXT NOT NULL COMMENT '결과 이미지',
                model VARCHAR(255) NOT NULL COMMENT '사용된 AI 모델명',
                prompt TEXT NOT NULL COMMENT '사용된 AI 명령어',
                success BOOLEAN NOT NULL COMMENT '실행 성공 (TRUE/FALSE)',
                run_time DOUBLE NOT NULL COMMENT '실행 시간 (초)',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_model (model),
                INDEX idx_success (success),
                INDEX idx_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """
            cursor.execute(create_result_logs_table)
            connection.commit()
            print("DB 테이블 생성 완료: result_logs")
    except Exception as e:
        print(f"테이블 생성 오류: {e}")
    finally:
        connection.close()

