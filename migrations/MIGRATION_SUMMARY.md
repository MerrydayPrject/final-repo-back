# 체형 분석 DB 마이그레이션 요약

## 📌 마이그레이션 파일

### 001_body_type_definitions.sql
- **목적**: 체형별 정의 데이터 테이블 생성 및 초기 데이터 삽입
- **테이블**: `body_type_definitions`
- **데이터**: 10가지 체형 특징에 대한 정의 (강점, 스타일 팁, 추천 드레스, 피해야 할 드레스)

### 002_add_body_analysis_to_result_logs.sql
- **목적**: 체형 분석 결과 저장 테이블 생성
- **테이블**: `body_logs` (신규 테이블 생성)
- **컬럼**: 9개 (idx, model, run_time, height, weight, prompt, bmi, characteristic, analysis_results, created_at)

---

## 🚀 빠른 실행

```bash
cd final-repo-back
python migrations/run_migration.py
```

또는 개별 실행:
```bash
# 001번
python -c "exec(open('migrations/001_body_type_definitions.sql').read())"

# 002번
python -c "exec(open('migrations/002_add_body_analysis_to_result_logs.sql').read())"
```

---

## ✅ 확인 방법

```bash
python check_db.py
```

또는 MySQL에서:
```sql
-- 001번 확인
SELECT COUNT(*) FROM body_type_definitions;  -- 10개여야 함

-- 002번 확인
SHOW TABLES LIKE 'body_logs';  -- 테이블 존재 확인
DESCRIBE body_logs;  -- 9개 컬럼 확인
```

---

## 📊 추가된 기능

### 001번 마이그레이션
- 체형 분석 시 DB에서 체형별 정의를 자동으로 조회
- Gemini 프롬프트에 체형별 정의 정보 포함
- 더 정확한 분석 결과 도출

### 002번 마이그레이션
- `body_logs` 테이블 생성 및 체형 분석 결과 자동 저장
- 저장되는 정보:
  - 모델명, 처리 시간
  - 키, 몸무게, BMI
  - AI 명령어 (프롬프트)
  - 체형 특징 (characteristic)
  - 분석 결과 (analysis_results)

---

## 📝 상세 문서

- **README.md**: 전체 가이드 및 상세 설명
- **QUICK_START.md**: 빠른 시작 가이드
- **BODY_TYPE_DEFINITIONS.md**: 체형별 정의 데이터 상세 내용

---

## ⚠️ 주의사항

- 운영 환경에서는 **반드시 백업 후 실행**
- `.env` 파일의 DB 연결 정보 확인
- 마이그레이션 실행 후 `check_db.py`로 결과 확인

