"""
002번 마이그레이션 실행 스크립트
body_logs 테이블 생성 (체형 분석 결과 저장용)
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
print("002번 마이그레이션 실행")
print("body_logs 테이블 생성 (체형 분석 결과 저장용)")
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
sql_file = Path(__file__).parent / "002_add_body_analysis_to_result_logs.sql"
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
        # 주석 제거 및 세미콜론으로 구분된 여러 쿼리 실행
        # 주석 라인 제거
        lines = sql_content.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # 빈 줄이나 주석만 있는 줄 제거
            if stripped and not stripped.startswith('--'):
                cleaned_lines.append(line)
        
        cleaned_sql = '\n'.join(cleaned_lines)
        statements = [stmt.strip() for stmt in cleaned_sql.split(';') if stmt.strip()]
        
        print(f"  발견된 쿼리 개수: {len(statements)}개")
        
        for i, statement in enumerate(statements, 1):
            try:
                if statement:
                    cursor.execute(statement)
                    print(f"  [{i}/{len(statements)}] 쿼리 실행 완료")
            except Exception as e:
                # 이미 테이블이 존재하는 경우는 무시
                if "already exists" in str(e).lower() or "Duplicate" in str(e):
                    print(f"  [{i}/{len(statements)}] 경고: {str(e)[:50]}... (이미 존재함, 무시)")
                else:
                    print(f"  [{i}/{len(statements)}] 오류: {e}")
                    raise
    
    connection.commit()
    print("\n✅ 마이그레이션 완료!")
    
    # 결과 확인
    with connection.cursor() as cursor:
        cursor.execute("SHOW TABLES LIKE 'body_logs'")
        table_exists = cursor.fetchone()
        
        print(f"\n[결과 확인]")
        if table_exists:
            print("  ✅ body_logs 테이블이 생성되었습니다.")
            
            cursor.execute("DESCRIBE body_logs")
            columns = cursor.fetchall()
            print(f"  컬럼 개수: {len(columns)}개")
            
            print(f"\n[테이블 구조]")
            for col in columns:
                null_info = "NOT NULL" if col['Null'] == 'NO' else "NULL"
                print(f"  - {col['Field']} ({col['Type']}) {null_info}")
            
            cursor.execute("SELECT COUNT(*) as count FROM body_logs")
            count = cursor.fetchone()['count']
            print(f"\n  현재 저장된 분석 결과: {count}개")
            
            if count > 0:
                cursor.execute("SELECT idx, model, height, weight, bmi, characteristic FROM body_logs ORDER BY created_at DESC LIMIT 3")
                recent = cursor.fetchall()
                print(f"\n[최근 분석 결과 샘플]")
                for r in recent:
                    print(f"  - ID: {r['idx']}, 모델: {r['model']}, 키: {r['height']}cm, 몸무게: {r['weight']}kg, BMI: {r['bmi']:.1f}")
        else:
            print("  ⚠️  body_logs 테이블이 생성되지 않았습니다.")
    
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

