"""
체형 분석 결과 조회 스크립트
"""
import pymysql
import os
import json
from dotenv import load_dotenv
from datetime import datetime

# .env 파일 로드
load_dotenv()

def get_db_connection():
    """데이터베이스 연결"""
    try:
        connection = pymysql.connect(
            host=os.getenv("MYSQL_HOST", "localhost"),
            port=int(os.getenv("MYSQL_PORT", 3306)),
            user=os.getenv("MYSQL_USER", "devuser"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=os.getenv("MYSQL_DATABASE", "marryday"),
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        return connection
    except Exception as e:
        print(f"❌ 데이터베이스 연결 실패: {e}")
        return None

def view_results(limit=10, format='table'):
    """
    체형 분석 결과 조회
    
    Args:
        limit: 조회할 레코드 수 (기본값: 10)
        format: 출력 형식 ('table', 'json', 'simple')
    """
    connection = get_db_connection()
    if connection is None:
        return
    
    try:
        with connection.cursor() as cursor:
            sql = """
                SELECT 
                    idx,
                    model,
                    run_time,
                    height,
                    weight,
                    bmi,
                    characteristic,
                    analysis_results,
                    created_at
                FROM body_logs
                ORDER BY created_at DESC
                LIMIT %s
            """
            cursor.execute(sql, (limit,))
            results = cursor.fetchall()
            
            # 전체 개수 조회
            cursor.execute("SELECT COUNT(*) as count FROM body_logs")
            total_count = cursor.fetchone()['count']
            
            print("=" * 80)
            print(f"체형 분석 결과 조회 (전체: {total_count}개, 표시: {len(results)}개)")
            print("=" * 80)
            
            if format == 'json':
                print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
            elif format == 'simple':
                for i, result in enumerate(results, 1):
                    char = result.get('characteristic', 'N/A')[:50] if result.get('characteristic') else 'N/A'
                    print(f"{i}. ID: {result['idx']} | 키: {result['height']}cm | 몸무게: {result['weight']}kg | BMI: {result['bmi']:.1f} | 특징: {char} | 날짜: {result['created_at']}")
            else:  # table format
                for i, result in enumerate(results, 1):
                    print(f"\n[{i}] ID: {result['idx']}")
                    print(f"  모델: {result['model']}")
                    print(f"  키: {result['height']}cm")
                    print(f"  몸무게: {result['weight']}kg")
                    print(f"  BMI: {result['bmi']:.1f}")
                    print(f"  처리 시간: {result['run_time']:.2f}초")
                    if result.get('characteristic'):
                        print(f"  체형 특징: {result['characteristic']}")
                    if result.get('analysis_results'):
                        analysis = result['analysis_results']
                        # 긴 텍스트를 여러 줄로 표시 (80자 기준으로 줄바꿈)
                        print(f"  분석 결과:")
                        # 문장 단위로 나누어 표시
                        lines = analysis.split('\n')
                        for line in lines:
                            if line.strip():
                                # 너무 긴 줄은 적절히 나눔
                                if len(line) > 100:
                                    words = line.split()
                                    current_line = ""
                                    for word in words:
                                        if len(current_line + word) > 100:
                                            print(f"    {current_line.strip()}")
                                            current_line = word + " "
                                        else:
                                            current_line += word + " "
                                    if current_line.strip():
                                        print(f"    {current_line.strip()}")
                                else:
                                    print(f"    {line.strip()}")
                    print(f"  생성일시: {result['created_at']}")
                    print("-" * 80)
            
    except Exception as e:
        print(f"❌ 조회 오류: {e}")
    finally:
        connection.close()

def view_detail(idx):
    """특정 결과 상세 조회"""
    connection = get_db_connection()
    if connection is None:
        return
    
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
                WHERE idx = %s
            """
            cursor.execute(sql, (idx,))
            result = cursor.fetchone()
            
            if not result:
                print(f"❌ ID {idx}에 해당하는 결과를 찾을 수 없습니다.")
                return
            
            print("=" * 80)
            print(f"체형 분석 결과 상세 (ID: {idx})")
            print("=" * 80)
            print(f"모델: {result['model']}")
            print(f"키: {result['height']}cm")
            print(f"몸무게: {result['weight']}kg")
            print(f"BMI: {result['bmi']:.1f}")
            print(f"처리 시간: {result['run_time']:.2f}초")
            print(f"생성일시: {result['created_at']}")
            if result.get('characteristic'):
                print(f"\n체형 특징:")
                print(f"  {result['characteristic']}")
            if result.get('prompt'):
                print(f"\n프롬프트:")
                print(f"  {result['prompt'][:500]}...")
            if result.get('analysis_results'):
                print(f"\n분석 결과:")
                print(f"  {result['analysis_results']}")
            print("=" * 80)
            
    except Exception as e:
        print(f"❌ 조회 오류: {e}")
    finally:
        connection.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--detail" or sys.argv[1] == "-d":
            if len(sys.argv) > 2:
                try:
                    idx = int(sys.argv[2])
                    view_detail(idx)
                except ValueError:
                    print("❌ ID는 숫자여야 합니다.")
            else:
                print("❌ 사용법: python view_results.py --detail <ID>")
        elif sys.argv[1] == "--json" or sys.argv[1] == "-j":
            limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            view_results(limit=limit, format='json')
        elif sys.argv[1] == "--simple" or sys.argv[1] == "-s":
            limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            view_results(limit=limit, format='simple')
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("""
체형 분석 결과 조회 스크립트

사용법:
  python view_results.py                    # 최근 10개 결과 조회 (테이블 형식)
  python view_results.py 20                  # 최근 20개 결과 조회
  python view_results.py --detail <ID>       # 특정 ID의 상세 결과 조회
  python view_results.py --json [개수]       # JSON 형식으로 출력
  python view_results.py --simple [개수]     # 간단한 형식으로 출력
  python view_results.py --help              # 도움말 표시

예제:
  python view_results.py 5                   # 최근 5개 조회
  python view_results.py --detail 1           # ID 1번 상세 조회
  python view_results.py --json 3             # 최근 3개를 JSON 형식으로
            """)
        else:
            try:
                limit = int(sys.argv[1])
                view_results(limit=limit)
            except ValueError:
                print("❌ 잘못된 인자입니다. --help로 사용법을 확인하세요.")
    else:
        view_results(limit=10)

