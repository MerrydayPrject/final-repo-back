"""
데이터베이스 연결 테스트 스크립트
"""
import pymysql
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

print("=" * 50)
print("데이터베이스 연결 테스트")
print("=" * 50)

# 환경 변수 확인
print("\n[환경 변수 확인]")
host = os.getenv("MYSQL_HOST", "localhost")
port = int(os.getenv("MYSQL_PORT", 3306))
user = os.getenv("MYSQL_USER", "devuser")
password = os.getenv("MYSQL_PASSWORD", "")
database = os.getenv("MYSQL_DATABASE", "marryday")

print(f"  Host: {host}")
print(f"  Port: {port}")
print(f"  User: {user}")
print(f"  Password: {'*' * len(password) if password else '(비어있음)'}")
print(f"  Database: {database}")

# .env 파일 확인
if not os.path.exists(".env"):
    print("\n⚠️  경고: .env 파일이 없습니다!")
    print("  final-repo-back 폴더에 .env 파일을 생성하고 다음 내용을 추가하세요:")
    print("""
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=devuser
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=marryday
""")

# 데이터베이스 연결 시도
print("\n[연결 시도]")
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
    print("✅ 데이터베이스 연결 성공!")
    
    # 테이블 확인
    with connection.cursor() as cursor:
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        print(f"\n[테이블 목록] ({len(tables)}개)")
        for table in tables:
            table_name = list(table.values())[0]
            print(f"  - {table_name}")
        
        # dresses 테이블 확인
        cursor.execute("SHOW TABLES LIKE 'dresses'")
        if cursor.fetchone():
            print("\n✅ dresses 테이블이 존재합니다.")
            cursor.execute("SELECT COUNT(*) as count FROM dresses")
            count = cursor.fetchone()['count']
            print(f"  현재 드레스 개수: {count}개")
        else:
            print("\n⚠️  dresses 테이블이 없습니다.")
            print("  서버를 실행하면 자동으로 생성됩니다.")
    
    connection.close()
    
except pymysql.Error as e:
    error_code, error_msg = e.args
    print(f"❌ 데이터베이스 연결 실패!")
    print(f"  에러 코드: {error_code}")
    print(f"  에러 메시지: {error_msg}")
    
    # 에러 타입별 해결 방법 제시
    if error_code == 1045:
        print("\n💡 해결 방법:")
        print("  1. .env 파일의 MYSQL_USER와 MYSQL_PASSWORD가 올바른지 확인")
        print("  2. MySQL 사용자 권한 확인")
    elif error_code == 1049:
        print("\n💡 해결 방법:")
        print(f"  1. '{database}' 데이터베이스를 생성하세요:")
        print(f"     CREATE DATABASE {database} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
    elif error_code == 2003:
        print("\n💡 해결 방법:")
        print("  1. MySQL 서비스가 실행 중인지 확인")
        print("  2. Windows: net start MySQL")
        print("  3. .env 파일의 MYSQL_HOST와 MYSQL_PORT 확인")
    else:
        print(f"\n💡 에러 코드 {error_code}에 대한 해결 방법을 검색해보세요.")
        
except Exception as e:
    print(f"❌ 예상치 못한 오류: {e}")

print("\n" + "=" * 50)




