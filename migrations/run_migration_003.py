"""
003번 마이그레이션 실행 스크립트
체형별 정의 데이터 업데이트 (사용 불가능한 드레스 스타일 수정)
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
print("003번 마이그레이션 실행")
print("체형별 정의 데이터 업데이트")
print("(드롭 웨이스트, 하이웨이스트, 엠파이어 라인 → 보유 드레스로 수정)")
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

try:
    # 데이터베이스 연결
    connection = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    
    print("\n[데이터베이스 연결 성공]")
    
    # SQL 파일 읽기
    sql_file = Path(__file__).parent / '003_update_body_type_definitions.sql'
    
    if not sql_file.exists():
        print(f"\n[오류] SQL 파일을 찾을 수 없습니다: {sql_file}")
        exit(1)
    
    with open(sql_file, 'r', encoding='utf-8') as f:
        sql_content = f.read()
    
    # SQL 문장들을 세미콜론으로 분리 (주석 제거)
    sql_statements = []
    current_statement = ""
    
    for line in sql_content.split('\n'):
        line = line.strip()
        # 주석 제거
        if line.startswith('--') or not line:
            continue
        
        current_statement += line + ' '
        
        if line.endswith(';'):
            sql_statements.append(current_statement.strip())
            current_statement = ""
    
    # 각 SQL 문장 실행
    with connection.cursor() as cursor:
        print(f"\n[업데이트 시작] 총 {len(sql_statements)}개의 SQL 문장 실행")
        
        for i, sql in enumerate(sql_statements, 1):
            if not sql.strip() or sql.strip() == ';':
                continue
                
            try:
                print(f"\n[{i}/{len(sql_statements)}] 실행 중...")
                cursor.execute(sql)
                affected_rows = cursor.rowcount
                
                if affected_rows > 0:
                    print(f"  ✓ {affected_rows}개 행 업데이트됨")
                else:
                    print(f"  ⚠ 업데이트된 행이 없습니다 (이미 업데이트되었거나 조건에 맞는 행이 없음)")
                    
            except Exception as e:
                print(f"  ✗ 오류 발생: {e}")
                print(f"  SQL: {sql[:100]}...")
                # SELECT 문은 오류가 나도 계속 진행
                if sql.strip().upper().startswith('SELECT'):
                    continue
                else:
                    raise
    
    # 변경사항 커밋
    connection.commit()
    print("\n[업데이트 완료] 모든 변경사항이 커밋되었습니다.")
    
    # 업데이트 결과 확인
    print("\n[업데이트 결과 확인]")
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT body_feature, recommended_dresses, avoid_dresses 
            FROM body_type_definitions 
            WHERE body_feature IN ('키가 작은 체형', '허리가 짧은 체형', '키가 큰 체형')
            ORDER BY body_feature
        """)
        
        results = cursor.fetchall()
        
        if results:
            print("\n업데이트된 체형 정의:")
            for row in results:
                print(f"\n  체형 특징: {row['body_feature']}")
                print(f"    추천 드레스: {row['recommended_dresses']}")
                print(f"    피해야 할 드레스: {row['avoid_dresses']}")
        else:
            print("  업데이트된 데이터가 없습니다.")
    
    print("\n" + "=" * 50)
    print("마이그레이션 완료!")
    print("=" * 50)
    
except pymysql.Error as e:
    print(f"\n[데이터베이스 오류] {e}")
    if connection:
        connection.rollback()
    exit(1)
    
except Exception as e:
    print(f"\n[오류 발생] {e}")
    import traceback
    traceback.print_exc()
    if connection:
        connection.rollback()
    exit(1)
    
finally:
    if 'connection' in locals() and connection:
        connection.close()
        print("\n[데이터베이스 연결 종료]")


