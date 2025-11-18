"""
마이그레이션 스크립트 실행 도구
"""
import pymysql
import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

print("=" * 50)
print("마이그레이션 실행")
print("=" * 50)

# 환경 변수 확인
host = os.getenv("MYSQL_HOST", "localhost")
port = int(os.getenv("MYSQL_PORT", 3306))
user = os.getenv("MYSQL_USER", "devuser")
password = os.getenv("MYSQL_PASSWORD", "")
database = os.getenv("MYSQL_DATABASE", "marryday")

print(f"\n[연결 정보]")
print(f"  Host: {host}")
print(f"  Port: {port}")
print(f"  User: {user}")
print(f"  Database: {database}")

# SQL 파일 읽기
sql_file = Path(__file__).parent / "001_body_type_definitions.sql"
if not sql_file.exists():
    print(f"\n❌ SQL 파일을 찾을 수 없습니다: {sql_file}")
    exit(1)

print(f"\n[SQL 파일 읽기]")
print(f"  파일: {sql_file}")

try:
    with open(sql_file, 'r', encoding='utf-8') as f:
        sql_content = f.read()
    
    # 데이터베이스 연결
    print(f"\n[데이터베이스 연결 시도]")
    connection = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    print("✅ 연결 성공!")
    
    # SQL 실행
    print(f"\n[SQL 실행 중...]")
    with connection.cursor() as cursor:
        # 세미콜론으로 구분된 여러 쿼리 실행
        statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
        
        for i, statement in enumerate(statements, 1):
            try:
                cursor.execute(statement)
                print(f"  [{i}/{len(statements)}] 쿼리 실행 완료")
            except Exception as e:
                # 일부 오류는 무시 (예: 테이블이 이미 존재하는 경우)
                if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                    print(f"  [{i}/{len(statements)}] 경고: {str(e)[:50]}... (무시)")
                else:
                    print(f"  [{i}/{len(statements)}] 오류: {e}")
                    raise
    
    connection.commit()
    print("\n✅ 마이그레이션 완료!")
    
    # 결과 확인
    with connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) as count FROM body_type_definitions")
        count = cursor.fetchone()['count']
        print(f"\n[결과 확인]")
        print(f"  체형별 정의 개수: {count}개")
        
        if count > 0:
            cursor.execute("SELECT body_feature FROM body_type_definitions ORDER BY body_feature")
            features = cursor.fetchall()
            print(f"\n[체형 특징 목록]")
            for feature in features:
                print(f"  - {feature['body_feature']}")
    
    connection.close()
    
except pymysql.Error as e:
    error_code, error_msg = e.args
    print(f"\n❌ 오류 발생!")
    print(f"  에러 코드: {error_code}")
    print(f"  에러 메시지: {error_msg}")
    
    if error_code == 1045:
        print("\n💡 해결 방법:")
        print("  1. .env 파일의 MYSQL_PASSWORD가 올바른지 확인")
        print("  2. MySQL 사용자 권한 확인")
    elif error_code == 1049:
        print(f"\n💡 해결 방법:")
        print(f"  '{database}' 데이터베이스가 존재하는지 확인")
    else:
        print(f"\n💡 에러 코드 {error_code}에 대한 해결 방법을 검색해보세요.")
    
    exit(1)
    
except Exception as e:
    print(f"\n❌ 예상치 못한 오류: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 50)




