# 체형 분석 DB 마이그레이션 - 빠른 시작 가이드

## 🚀 빠른 실행 (3단계)

### 1단계: 환경 확인
```bash
cd final-repo-back
python check_db.py
```

### 2단계: 마이그레이션 실행
```bash
# 모든 마이그레이션 실행
python migrations/run_migration.py
```

또는 개별 실행:
```bash
# 001번: 체형별 정의 데이터 추가
python -c "import pymysql; from pathlib import Path; import os; from dotenv import load_dotenv; load_dotenv(); conn = pymysql.connect(host=os.getenv('MYSQL_HOST', 'localhost'), port=int(os.getenv('MYSQL_PORT', 3306)), user=os.getenv('MYSQL_USER', 'devuser'), password=os.getenv('MYSQL_PASSWORD', ''), database=os.getenv('MYSQL_DATABASE', 'marryday'), charset='utf8mb4'); sql = Path('migrations/001_body_type_definitions.sql').read_text(encoding='utf-8'); [conn.cursor().execute(stmt) for stmt in sql.split(';') if stmt.strip()]; conn.commit(); conn.close(); print('✅ 001번 완료!')"

# 002번: 분석 결과 저장 컬럼 추가
python -c "import pymysql; from pathlib import Path; import os; from dotenv import load_dotenv; load_dotenv(); conn = pymysql.connect(host=os.getenv('MYSQL_HOST', 'localhost'), port=int(os.getenv('MYSQL_PORT', 3306)), user=os.getenv('MYSQL_USER', 'devuser'), password=os.getenv('MYSQL_PASSWORD', ''), database=os.getenv('MYSQL_DATABASE', 'marryday'), charset='utf8mb4'); sql = Path('migrations/002_add_body_analysis_to_result_logs.sql').read_text(encoding='utf-8'); [conn.cursor().execute(stmt) for stmt in sql.split(';') if stmt.strip()]; conn.commit(); conn.close(); print('✅ 002번 완료!')"
```

### 3단계: 결과 확인
```bash
python check_db.py
```

**예상 출력:**
```
✅ body_type_definitions 테이블이 존재합니다.
  현재 체형별 정의 개수: 10개

✅ body_logs 테이블이 생성되었습니다.
```

---

## 📋 마이그레이션 파일

1. **001_body_type_definitions.sql**
   - 체형별 정의 테이블 생성
   - 10가지 체형 특징 데이터 삽입

2. **002_add_body_analysis_to_result_logs.sql**
   - `body_logs` 테이블 생성 (체형 분석 결과 저장용)
   - 분석 결과 자동 저장 기능 활성화

---

## 📋 포함된 데이터

### 체형별 정의 (10가지)

1. 키가 작은 체형 → 엠파이어 라인
2. 글래머러스한 체형 → 머메이드
3. 어깨가 넓은 체형 → A라인, 프린세스
4. 마른 체형 → 프린세스
5. 팔 라인이 신경 쓰이는 체형 → A라인, 벨라인
6. 허리가 짧은 체형 → 드롭 웨이스트
7. 복부가 신경 쓰이는 체형 → A라인
8. 키가 큰 체형 → 슬림
9. 어깨가 좁은 체형 → 프린세스, 벨라인, A라인
10. 체형 전체를 커버하고 싶은 경우 → 벨라인

---

## 💡 시스템 동작

체형 분석 시:
1. 랜드마크로 체형 라인 판별
2. BMI로 체형 특징 판별
3. **DB에서 체형별 정의 조회** ← 001번 마이그레이션으로 추가됨
4. Gemini가 모든 정보를 종합하여 분석
5. **분석 결과를 DB에 자동 저장** ← 002번 마이그레이션으로 추가됨

**저장되는 정보:**
- 모델명, 처리 시간
- 키, 몸무게, BMI
- AI 명령어 (프롬프트)
- 체형 특징 (characteristic)
- 분석 결과 (analysis_results)

---

## ❓ 문제 발생 시

**연결 오류:**
- `.env` 파일의 DB 정보 확인

**데이터가 안 보임:**
- `python check_db.py` 실행하여 확인
- MySQL에서 직접 확인: `SELECT * FROM body_type_definitions;`

**자세한 내용은 `README.md` 참고**

