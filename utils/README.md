# 유틸리티 스크립트

이 폴더에는 프로젝트 관리 및 유지보수를 위한 유틸리티 스크립트들이 포함되어 있습니다.

## 파일 목록

### `check_db.py`
데이터베이스 연결 및 테이블 구조 확인 스크립트

**사용법:**
```bash
python utils/check_db.py
```

### `download_inswapper.py`
INSwapper 모델 다운로드 스크립트

**사용법:**
```bash
python utils/download_inswapper.py
```

### `verify_inswapper.py`
INSwapper 모델 파일 검증 스크립트

**사용법:**
```bash
python utils/verify_inswapper.py [파일경로]
```

### `view_results.py`
체형 분석 결과 조회 스크립트

**사용법:**
```bash
python utils/view_results.py [옵션]
python utils/view_results.py --help  # 도움말
```

### `download_model.py`
MediaPipe Pose Landmarker 모델 다운로드 스크립트

**사용법:**
```bash
python utils/download_model.py
```

**기능:**
- MediaPipe Pose Landmarker 모델 자동 다운로드
- 저장 위치: `models/body_analysis/pose_landmarker_lite.task`

## 참고사항

- 이 스크립트들은 프로젝트 실행에 필수적이지 않습니다.
- 필요할 때만 수동으로 실행하는 유틸리티입니다.
- 각 스크립트는 독립적으로 실행 가능합니다.
