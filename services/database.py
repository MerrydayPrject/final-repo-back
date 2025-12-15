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
        # 한국시간(Asia/Seoul, UTC+9)으로 타임존 설정
        with connection.cursor() as cursor:
            cursor.execute("SET time_zone = '+09:00'")
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
            
            # body_type_definitions 테이블 생성
            create_body_type_definitions_table = """
            CREATE TABLE IF NOT EXISTS body_type_definitions (
                id INT PRIMARY KEY AUTO_INCREMENT,
                body_feature VARCHAR(50) NOT NULL UNIQUE COMMENT '체형 특징 (키가 작은 체형, 글래머러스한 체형 등)',
                strengths TEXT COMMENT '체형의 강점 설명',
                style_tips TEXT COMMENT '스타일 팁',
                recommended_dresses TEXT COMMENT '추천 드레스 스타일',
                avoid_dresses TEXT COMMENT '피해야 할 드레스 스타일',
                definition_text TEXT COMMENT '라인별 체형 정의 텍스트 (A라인, H라인, X라인, O라인용)',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_body_feature (body_feature)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='체형별 정의 및 드레스 추천 정보';
            """
            cursor.execute(create_body_type_definitions_table)
            connection.commit()
            print("DB 테이블 생성 완료: body_type_definitions")
            
            # 기존 테이블에 definition_text 컬럼 추가 (테이블이 이미 존재하는 경우)
            try:
                cursor.execute("ALTER TABLE body_type_definitions ADD COLUMN definition_text TEXT COMMENT '라인별 체형 정의 텍스트 (A라인, H라인, X라인, O라인용)'")
                connection.commit()
                print("definition_text 컬럼 추가 완료")
            except Exception as e:
                # 이미 컬럼이 존재하거나 다른 오류인 경우 무시
                if "Duplicate column name" not in str(e) and "Unknown column" not in str(e):
                    print(f"definition_text 컬럼 추가 시도: {e}")
            
            # body_type_definitions 초기 데이터 삽입 (데이터가 없을 때만)
            cursor.execute("SELECT COUNT(*) as count FROM body_type_definitions")
            count = cursor.fetchone()['count']
            
            if count == 0:
                print("body_type_definitions 초기 데이터 삽입 중...")
                initial_data = [
                    ('키가 작은 체형', '작은 키의 신부님은 허리선이 높게 올라간 프린세스 드레스가 체형을 길어 보이게 만들어줍니다.', '심플한 디자인을 선택하면 키가 더 커 보이는 효과를 볼 수 있습니다.', '프린세스', '슬림 (키가 작아 보일 수 있음)'),
                    ('글래머러스한 체형', '곡선미가 뚜렷한 체형으로, 몸매 라인을 강조하면서도 우아한 분위기를 연출할 수 있습니다.', '머메이드 라인 드레스가 제격입니다.', '머메이드', '슬림 (곡선미를 제대로 살리기 어려움)'),
                    ('어깨가 넓은 체형', '어깨가 넓다면 상체는 비교적 심플하게 정리되고, 스커트가 자연스럽게 퍼지는 A라인이나 프린세스 라인이 균형 잡힌 실루엣을 만들어줍니다.', '상체를 심플하게 정리하고 하체 볼륨을 주는 스타일이 적합합니다.', 'A라인, 프린세스', '슬림 (상체가 더 넓어 보일 수 있음)'),
                    ('마른 체형', '슬림한 체형에는 프린세스 라인이 잘 어울립니다.', '풍성한 스커트가 체형을 보완해주고, 전체적으로 사랑스러운 이미지를 만들어줍니다.', '프린세스', '슬림 (마른 체형이 더 마르게 보일 수 있음)'),
                    ('팔 라인이 신경 쓰이는 체형', '팔 라인이 고민된다면 상체를 너무 타이트하게 드러내는 슬림 라인보다는, 상체를 적당히 감싸주고 스커트가 퍼지는 A라인이나 벨라인이 안정감 있게 연출해 줍니다.', '상체를 적당히 감싸주는 디자인이 적합합니다.', 'A라인, 벨라인', '슬림 (팔 라인이 노출될 수 있음)'),
                    ('허리가 짧은 체형', '허리선을 강조하는 벨라인 드레스는 하체를 길어 보이게 만들어 전체적인 비율을 맞춰줍니다.', '허리 라인을 강조하여 비율을 조화롭게 연출하는 스타일이 적합합니다.', '벨라인', '슬림 (허리가 더 짧아 보일 수 있음)'),
                    ('복부가 신경 쓰이는 체형', '복부를 자연스럽게 커버하려면 A라인 드레스가 최적입니다.', '허리에서 자연스럽게 퍼지는 라인이 체형 커버에 탁월합니다.', 'A라인', '슬림 (복부가 노출될 수 있음), 머메이드 (복부 라인이 강조될 수 있음)'),
                    ('키가 큰 체형', '키가 큰 신부님은 긴 기장의 슬림 드레스가 우아함을 더해줍니다.', '특히 미니멀한 디자인은 세련된 이미지를 강조해줍니다.', '슬림, 미니드레스', '프린세스 (키가 더 커 보일 수 있음)'),
                    ('어깨가 좁은 체형', '어깨가 좁다면 상체에 볼륨이 살아나는 프린세스 라인이나 벨라인이 균형감을 잡아줍니다.', '상체에 레이스나 셔링 같은 디테일이 들어간 A라인 드레스도 어깨와 상체 라인을 보완해 주는 데 도움이 됩니다.', '프린세스, 벨라인, A라인', '슬림 (어깨가 더 좁아 보일 수 있음)'),
                    ('체형 전체를 커버하고 싶은 경우', '체형 고민이 많을 때는 클래식한 벨라인 드레스가 가장 안전한 선택입니다.', '로맨틱하면서도 웅장한 분위기를 연출할 수 있습니다.', '벨라인', '특별히 피해야 할 스타일은 없으나, 체형의 특성에 따라 선택적으로 피하는 것이 좋습니다.')
                ]
                
                insert_query = """
                INSERT INTO body_type_definitions (body_feature, strengths, style_tips, recommended_dresses, avoid_dresses, definition_text)
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                # definition_text는 None으로 설정
                initial_data_with_definition = [(row + (None,)) for row in initial_data]
                cursor.executemany(insert_query, initial_data_with_definition)
                connection.commit()
                print(f"body_type_definitions 초기 데이터 삽입 완료: {len(initial_data)}개")
            else:
                print(f"body_type_definitions 데이터 이미 존재: {count}개")
            
            # 라인별 체형 정의 데이터 삽입/업데이트 (항상 실행)
            from pathlib import Path
            base_dir = Path(__file__).parent.parent
            definition_file = base_dir / "body_line_definitions.md"
            
            if definition_file.exists():
                try:
                    with open(definition_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 마크다운 파싱하여 라인별로 분리
                    line_definitions = []
                    current_line = None
                    current_text = []
                    
                    for line in content.split('\n'):
                        # 라인 헤더 감지 (## X 라인, ## H 라인 등)
                        if line.startswith('## '):
                            # 이전 라인 저장
                            if current_line and current_text:
                                line_definitions.append((current_line, '\n'.join(current_text).strip()))
                            
                            # 새 라인 시작
                            line_name = line.replace('## ', '').strip()
                            # "X 라인 (X-Line) – 모래시계형" -> "X라인"
                            if '라인' in line_name:
                                parts = line_name.split('라인')
                                if parts:
                                    line_char = parts[0].strip()
                                    current_line = line_char + '라인'
                                else:
                                    current_line = line_name
                            else:
                                current_line = line_name
                            current_text = []
                        elif line.startswith('---'):
                            # 구분선 무시
                            continue
                        elif current_line:
                            # 빈 줄이 아닌 경우 추가
                            current_text.append(line)
                    
                    # 마지막 라인 저장
                    if current_line and current_text:
                        line_definitions.append((current_line, '\n'.join(current_text).strip()))
                    
                    # DB에 삽입/업데이트
                    for line_name, definition_text in line_definitions:
                        cursor.execute("""
                            INSERT INTO body_type_definitions (body_feature, strengths, style_tips, recommended_dresses, avoid_dresses, definition_text)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON DUPLICATE KEY UPDATE definition_text = VALUES(definition_text)
                        """, (line_name, None, None, None, None, definition_text))
                        print(f"라인별 정의 삽입/업데이트: {line_name}")
                    
                    connection.commit()
                    print("라인별 체형 정의 데이터 삽입/업데이트 완료")
                except Exception as e:
                    print(f"라인별 체형 정의 삽입/업데이트 오류: {e}")
            else:
                print(f"라인별 체형 정의 파일을 찾을 수 없습니다: {definition_file}")
            
            # body_logs 테이블 생성
            create_body_logs_table = """
            CREATE TABLE IF NOT EXISTS body_logs (
                idx INT AUTO_INCREMENT PRIMARY KEY,
                model VARCHAR(255) NOT NULL COMMENT '모델명',
                run_time FLOAT NOT NULL COMMENT '처리 시간',
                height FLOAT NOT NULL COMMENT '키',
                weight FLOAT NOT NULL COMMENT '몸무게',
                prompt TEXT NOT NULL COMMENT 'AI 명령어',
                bmi FLOAT NOT NULL COMMENT '비만도',
                characteristic TEXT COMMENT '체형 특징',
                analysis_results TEXT COMMENT '분석 결과',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_model (model),
                INDEX idx_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='체형 분석 로그 테이블';
            """
            cursor.execute(create_body_logs_table)
            connection.commit()
            print("DB 테이블 생성 완료: body_logs")
            
            # reviews 테이블 생성
            create_reviews_table = """
            CREATE TABLE IF NOT EXISTS reviews (
                idx INT AUTO_INCREMENT PRIMARY KEY,
                rating TINYINT NOT NULL CHECK (rating BETWEEN 1 AND 5),
                content TEXT,
                category ENUM('general', 'custom', 'analysis') NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_category (category),
                INDEX idx_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """
            cursor.execute(create_reviews_table)
            connection.commit()
            print("DB 테이블 생성 완료: reviews")
            
            # daily_visitors 테이블 생성
            create_daily_visitors_table = """
            CREATE TABLE IF NOT EXISTS daily_visitors (
                id INT AUTO_INCREMENT PRIMARY KEY,
                visit_date DATE NOT NULL UNIQUE,
                count INT DEFAULT 0,
                INDEX idx_visit_date (visit_date)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='일별 방문자 수';
            """
            cursor.execute(create_daily_visitors_table)
            connection.commit()
            print("DB 테이블 생성 완료: daily_visitors")
            
            # dress_check_logs 테이블 생성
            create_dress_check_logs_table = """
            CREATE TABLE IF NOT EXISTS dress_check_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                filename VARCHAR(255),
                image_hash VARCHAR(64),
                model VARCHAR(50),
                mode VARCHAR(20),
                predicted_dress BOOLEAN,
                confidence FLOAT,
                category VARCHAR(100),
                verified_dress BOOLEAN DEFAULT NULL,
                is_verified BOOLEAN DEFAULT FALSE,
                verified_at TIMESTAMP NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_created_at (created_at),
                INDEX idx_is_verified (is_verified),
                INDEX idx_predicted (predicted_dress),
                INDEX idx_verified (verified_dress)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='드레스 판별 로그 및 검수 결과';
            """
            cursor.execute(create_dress_check_logs_table)
            connection.commit()
            print("DB 테이블 생성 완료: dress_check_logs")
            
            # daily_synthesis_count 테이블 생성
            create_daily_synthesis_count_table = """
            CREATE TABLE IF NOT EXISTS daily_synthesis_count (
                id INT AUTO_INCREMENT PRIMARY KEY,
                synthesis_date DATE NOT NULL UNIQUE,
                count INT DEFAULT 0,
                INDEX idx_synthesis_date (synthesis_date)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='일별 합성 카운트';
            """
            cursor.execute(create_daily_synthesis_count_table)
            connection.commit()
            print("DB 테이블 생성 완료: daily_synthesis_count")
            
            # tryon_profile_summary 테이블 생성
            create_tryon_profile_summary_table = """
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
            cursor.execute(create_tryon_profile_summary_table)
            connection.commit()
            print("DB 테이블 생성 완료: tryon_profile_summary")
            
            # synthesis_quality_evaluations 테이블 생성
            create_synthesis_quality_evaluations_table = """
            CREATE TABLE IF NOT EXISTS synthesis_quality_evaluations (
                idx INT AUTO_INCREMENT PRIMARY KEY,
                result_log_idx INT NOT NULL COMMENT 'result_logs.idx 참조',
                quality_score INT NOT NULL COMMENT '평가 점수 (0-100)',
                quality_comment TEXT COMMENT '평가 상세 내용',
                is_success BOOLEAN NOT NULL COMMENT '성공 여부 (점수 70점 이상)',
                evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '평가 시간',
                INDEX idx_result_log_idx (result_log_idx),
                INDEX idx_is_success (is_success),
                INDEX idx_evaluated_at (evaluated_at),
                FOREIGN KEY (result_log_idx) REFERENCES result_logs(idx) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='합성 품질 평가 결과';
            """
            cursor.execute(create_synthesis_quality_evaluations_table)
            connection.commit()
            print("DB 테이블 생성 완료: synthesis_quality_evaluations")
            
            # dress_fitting_logs 테이블 생성
            create_dress_fitting_logs_table = """
            CREATE TABLE IF NOT EXISTS dress_fitting_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                dress_id INT NOT NULL COMMENT '드레스 ID (dresses.idx 참조)',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '피팅 시각',
                INDEX idx_dress_id (dress_id),
                INDEX idx_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='드레스 피팅 로그';
            """
            cursor.execute(create_dress_fitting_logs_table)
            connection.commit()
            print("DB 테이블 생성 완료: dress_fitting_logs")
    except Exception as e:
        print(f"테이블 생성 오류: {e}")
    finally:
        connection.close()

