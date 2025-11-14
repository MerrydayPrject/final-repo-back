-- body_logs 테이블 생성 (체형 분석 결과 저장용)
-- 실행 방법: python migrations/run_migration_002.py (또는 직접 SQL 실행)

-- 기존 테이블 삭제 (있다면)
DROP TABLE IF EXISTS body_logs;

-- 테이블 생성
CREATE TABLE body_logs (
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
