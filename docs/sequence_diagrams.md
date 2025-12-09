알겠습니다! `v4` 관련된 부분을 제거하고, `v5`를 중심으로 수정된 다이어그램을 한글로 정리한 내용입니다. 아래는 수정된 내용입니다.

---

## 1. 인증 (Authentication)

**엔드포인트**: `/api/auth/login`, `/api/auth/logout`, `/api/auth/verify`

```mermaid
sequenceDiagram
    actor User
    participant FE as Frontend (App.jsx)
    participant API as API Utility (api.js)
    participant BE as Backend (auth_router)
    participant DB as Database

    User->>Frontend: 페이지 로드
    Frontend->>API: getDresses()
    API->>Backend: GET /api/admin/dresses?limit=1000
    Backend->>DB: SELECT * FROM dresses LIMIT 1000
    DB-->>Backend: 드레스 목록 반환
    Backend-->>API: {success: true, data: [...]}
    API-->>Frontend: 드레스 목록
    Frontend->>Frontend: 드레스 데이터 변환<br/>(S3 URL → 프록시 URL)
    Frontend-->>User: 드레스 목록 표시
```

### 1.2 일반 피팅 전체 플로우 (로깅 포함)

```mermaid
sequenceDiagram
    participant User as 사용자
    participant Frontend as 프론트엔드<br/>(GeneralFitting.jsx)
    participant API as API 유틸<br/>(api.js)
    participant Backend as 백엔드<br/>(tryon_router.py)
    participant BodyService as 체형 분석 서비스
    participant TryonService as 트라이온 서비스<br/>(tryon_compare_service.py)
    participant ProfileService as 프로파일링 서비스<br/>(profile_service.py)
    participant DressLogService as 드레스 피팅 로그 서비스
    participant Gemini as Gemini API
    participant DB as 데이터베이스

    User->>Frontend: 배경 선택
    Frontend->>Frontend: traceId 생성<br/>(trace_${Date.now()}_${random})
    Frontend->>Frontend: bg_select_start = Date.now()
    Frontend->>Frontend: bg_select_ms 측정 완료
    
    User->>Frontend: 전신 이미지 업로드
    Frontend->>Frontend: person_upload_start = Date.now()
    Frontend->>API: validatePerson(image)
    API->>Backend: POST /api/validate-person
    Backend->>BodyService: MediaPipe로 사람 감지
    BodyService-->>Backend: {success: true, is_person: true}
    Backend-->>API: 검증 결과
    API-->>Frontend: 검증 성공
    Frontend->>Frontend: person_validate_ms 측정 완료
    Frontend->>Frontend: person_upload_ms 측정 완료
    
    User->>Frontend: 드레스 드래그 앤 드롭
    Frontend->>Frontend: dress_drop_ms 측정 시작
    Frontend->>API: autoMatchImageV5V5(person, dress, background, traceId, profile)
    Note over Frontend,API: profile에 타이밍 정보 포함<br/>(bg_select_ms, person_upload_ms,<br/>person_validate_ms, dress_drop_ms)
    
    API->>Backend: POST /tryon/compare<br/>(FormData: person_image, garment_image,<br/>background_image, profile_front,<br/>dress_id (선택사항),<br/>Headers: X-Trace-Id)
    Backend->>Backend: traceId 추출 또는 생성<br/>server_start_time 기록
    Backend->>Backend: front_profile_json 파싱<br/>dress_id 추출 (Form 파라미터)
    
    alt 입력 검증 실패
        Backend->>Backend: server_total_ms 계산
        Backend->>ProfileService: save_tryon_profile(<br/>  status="fail",<br/>  error_stage="input_validation"<br/>)
        Backend-->>API: 400 에러 응답
        API-->>Frontend: 에러 메시지
    else 입력 검증 성공
        Backend->>TryonService: run_v4v5_compare(person, garment, background, enable_logging=True)
    TryonService->>TryonService: V5 파이프라인 병렬 실행 (2회)
    
    par V5-1 실행
        TryonService->>TryonService: load_v5_unified_prompt()<br/>(prompts/v5/prompt_unified.txt<br/>에서 정적 프롬프트 로드)
        TryonService->>Gemini: 이미지 합성 요청<br/>(person_img, garment_img, background_img,<br/>정적 프롬프트 포함,<br/>temperature=0.0, safety_settings=BLOCK_NONE)
        Gemini-->>TryonService: 합성 이미지 (base64)<br/>gemini_call_ms 포함
    and V5-2 실행
        TryonService->>TryonService: load_v5_unified_prompt()<br/>(prompts/v5/prompt_unified.txt<br/>에서 정적 프롬프트 로드)
        TryonService->>Gemini: 이미지 합성 요청<br/>(person_img, garment_img, background_img,<br/>정적 프롬프트 포함,<br/>temperature=0.0, safety_settings=BLOCK_NONE)
        Gemini-->>TryonService: 합성 이미지 (base64)<br/>gemini_call_ms 포함
    end
    
    TryonService->>TryonService: resize_ms, gemini_call_ms 수집
    TryonService-->>Backend: {v4_result: {...}, v5_result: {...},<br/>gemini_call_ms, resize_ms}
    
    Backend->>Backend: server_total_ms 계산<br/>(time.time() - server_start_time) * 1000
    Backend->>Backend: status 판단<br/>(result.success 여부)
    
    opt v5_result가 성공한 경우
        Backend->>Backend: increment_synthesis_count()<br/>(날짜별 합성 카운트 증가)
        Backend->>DB: INSERT INTO daily_synthesis_count<br/>(synthesis_date, count)<br/>VALUES (오늘날짜, 1)<br/>ON DUPLICATE KEY UPDATE count = count + 1
        DB-->>Backend: 카운트 증가 완료
        
        opt dress_id가 있는 경우
            Backend->>DressLogService: log_dress_fitting(dress_id)
            DressLogService->>DB: INSERT INTO dress_fitting_logs<br/>(dress_id)
            DB-->>DressLogService: 저장 완료
            DressLogService-->>Backend: 성공
        end
    end
    
    Backend->>ProfileService: save_tryon_profile(<br/>  trace_id,<br/>  endpoint="/tryon/compare",<br/>  front_profile_json,<br/>  server_total_ms,<br/>  resize_ms,<br/>  gemini_call_ms,<br/>  cutout_ms=None,<br/>  status,<br/>  error_stage<br/>)
    ProfileService->>ProfileService: ensure_table_exists()<br/>(테이블 자동 생성)
    ProfileService->>DB: INSERT INTO tryon_profile_summary<br/>(trace_id, endpoint, front_profile_json,<br/>server_total_ms, resize_ms, gemini_call_ms,<br/>cutout_ms, status, error_stage)<br/>ON DUPLICATE KEY UPDATE
    DB-->>ProfileService: 저장 완료
    ProfileService-->>Backend: 성공
    
    alt 예외 발생
        Backend->>Backend: server_total_ms 계산
        Backend->>ProfileService: save_tryon_profile(<br/>  status="fail",<br/>  error_stage="exception"<br/>)
        Backend-->>API: 500 에러 응답
        API-->>Frontend: 에러 메시지
    else 정상 처리
        Backend-->>API: V4V5CompareResponse<br/>(Headers: X-Trace-Id)
        API-->>Frontend: 결과 이미지 (v4_result, v5_result)
    end
    
    Frontend->>Frontend: result_image_load_start = Date.now()
    Frontend->>Frontend: 이미지 선택 모달 표시<br/>(2개 이미지가 있는 경우)
    User->>Frontend: 결과 이미지 선택
    Frontend->>Frontend: result_image_load_ms 측정 완료
    Frontend-->>User: 선택한 이미지 표시
    
    opt 리뷰 모달 표시 (쿠키 확인 후)
        Frontend->>Frontend: 3초 후 리뷰 모달 표시
        User->>Frontend: 리뷰 제출
        Frontend->>API: submitReview({category: 'general', rating, content})
        API->>Backend: POST /api/reviews
        Backend->>DB: INSERT INTO reviews<br/>(rating, content, category)
        DB-->>Backend: review_id 반환
        Backend-->>API: {success: true, review_id}
        API-->>Frontend: 리뷰 제출 완료
        Frontend->>Frontend: 쿠키에 리뷰 제출 기록 저장
    end
```

### 1.3 이미지 필터 적용

```mermaid
sequenceDiagram
    participant User as 사용자
    participant Frontend as 프론트엔드<br/>(GeneralFitting.jsx)
    participant API as API 유틸<br/>(api.js)
    participant Backend as 백엔드<br/>(image_processing.py)
    participant ImageService as 이미지 처리 서비스

    User->>Frontend: 필터 선택 (grayscale, vintage, warm 등)
    Frontend->>API: applyImageFilter(resultImage, filterPreset)
    API->>Backend: POST /api/apply-image-filters<br/>(FormData: file, filter_preset)
    Backend->>ImageService: 필터 적용 처리
    ImageService->>ImageService: PIL/Pillow로 필터 적용
    ImageService-->>Backend: 필터 적용된 이미지 (base64)
    Backend-->>API: {success: true, result_image: "data:image/..."}
    API-->>Frontend: 필터 적용된 이미지
    Frontend-->>User: 필터 적용된 결과 표시
```

## 2. 커스텀 피팅 (Custom Fitting)

### 2.1 커스텀 피팅 전체 플로우 (로깅 포함)

```mermaid
sequenceDiagram
    participant User as 사용자
    participant Frontend as 프론트엔드<br/>(CustomFitting.jsx)
    participant API as API 유틸<br/>(api.js)
    participant Backend as 백엔드<br/>(custom_v4v5_router.py)
    participant BodyService as 체형 분석 서비스
    participant DressService as 드레스 체크 서비스
    participant CustomTryonService as 커스텀 트라이온 서비스<br/>(custom_v4v5_compare_service.py)
    participant ProfileService as 프로파일링 서비스<br/>(profile_service.py)
    participant LogService as 로그 서비스<br/>(log_service.py)
    participant SegformerService as SegFormer 서비스
    participant Gemini as Gemini API
    participant S3 as S3 Storage
    participant DB as 데이터베이스

    User->>Frontend: 배경 선택
    Frontend->>Frontend: traceId 생성<br/>(trace_${Date.now()}_${random})
    Frontend->>Frontend: bg_select_start = Date.now()
    Frontend->>Frontend: bg_select_ms 측정 완료
    
    User->>Frontend: 전신 이미지 업로드
    Frontend->>Frontend: person_upload_start = Date.now()
    Frontend->>API: validatePerson(image)
    API->>Backend: POST /api/validate-person
    Backend->>BodyService: MediaPipe로 사람 감지
    BodyService-->>Backend: {success: true, is_person: true}
    Backend-->>API: 검증 결과
    API-->>Frontend: 검증 성공
    Frontend->>Frontend: person_validate_ms 측정 완료
    Frontend->>Frontend: person_upload_ms 측정 완료
    
    User->>Frontend: 드레스 이미지 업로드
    Frontend->>Frontend: dress_upload_start = Date.now()
    Frontend->>API: checkDress(dressImage)
    API->>Backend: POST /api/dress/check<br/>(FormData: file, model, mode)
    Backend->>DressService: GPT-4o-mini로 드레스 체크
    DressService->>DressService: 이미지 분석 및 드레스 여부 판단
    DressService-->>Backend: {success: true, result: {dress: true}}
    Backend-->>API: 드레스 체크 결과
    API-->>Frontend: {success: true, result: {dress: true}}
    Frontend->>Frontend: dress_validate_ms 측정 완료
    Frontend->>Frontend: dress_upload_ms 측정 완료
    
    alt 드레스가 아닌 경우
        Frontend->>Frontend: 에러 모달 표시
        Frontend-->>User: "드레스 사진을 넣어주세요"
    end
    
    User->>Frontend: 매칭하기 버튼 클릭
    Frontend->>Frontend: compose_click_start = Date.now()
    Frontend->>API: customV5V5MatchImage(fullBody, dress, background, traceId, profile)
    Note over Frontend,API: profile에 타이밍 정보 포함<br/>(bg_select_ms, person_upload_ms,<br/>person_validate_ms, dress_upload_ms,<br/>dress_validate_ms)
    
    API->>Backend: POST /tryon/compare/custom<br/>(FormData: person_image, garment_image,<br/>background_image, profile_front,<br/>Headers: X-Trace-Id)
    Backend->>Backend: traceId 추출 또는 생성<br/>server_start_time 기록
    Backend->>Backend: front_profile_json 파싱
    
    alt 입력 검증 실패
        Backend->>Backend: server_total_ms 계산
        Backend->>ProfileService: save_tryon_profile(<br/>  status="fail",<br/>  error_stage="input_validation"<br/>)
        Backend-->>API: 400 에러 응답
        API-->>Frontend: 에러 메시지
    else 입력 검증 성공
        Backend->>CustomTryonService: run_v4v5_custom_compare(person, garment, background, enable_logging=True)
    CustomTryonService->>CustomTryonService: 의상 이미지 S3 업로드<br/>(로깅용)
    CustomTryonService->>S3: upload_log_to_s3(garment_image, "custom-fitting", "garment")
    S3-->>CustomTryonService: garment_s3_url
    
    CustomTryonService->>CustomTryonService: V5 파이프라인 병렬 실행 (2회)
    
    par CustomV5-1 실행
        CustomTryonService->>SegformerService: 의상 누끼 처리 (배경 제거)
        SegformerService-->>CustomTryonService: 누끼 처리된 의상 이미지<br/>cutout_ms 포함
        CustomTryonService->>CustomTryonService: load_v5_unified_prompt()<br/>(prompts/v5/prompt_unified.txt<br/>에서 정적 프롬프트 로드)
        CustomTryonService->>Gemini: 이미지 합성 요청<br/>(person_img, garment_nukki_img,<br/>background_img, 정적 프롬프트 포함,<br/>temperature=0.0, safety_settings=BLOCK_NONE)
        Gemini-->>CustomTryonService: 합성 이미지 (base64)<br/>gemini_call_ms 포함
    and CustomV5-2 실행
        CustomTryonService->>SegformerService: 의상 누끼 처리 (배경 제거)
        SegformerService-->>CustomTryonService: 누끼 처리된 의상 이미지<br/>cutout_ms 포함
        CustomTryonService->>CustomTryonService: load_v5_unified_prompt()<br/>(prompts/v5/prompt_unified.txt<br/>에서 정적 프롬프트 로드)
        CustomTryonService->>Gemini: 이미지 합성 요청<br/>(person_img, garment_nukki_img,<br/>background_img, 정적 프롬프트 포함,<br/>temperature=0.0, safety_settings=BLOCK_NONE)
        Gemini-->>CustomTryonService: 합성 이미지 (base64)<br/>gemini_call_ms 포함
    end
    
    CustomTryonService->>CustomTryonService: total_time 계산<br/>cutout_ms, gemini_call_ms, resize_ms 수집
    
    opt enable_logging == True && garment_s3_url 존재
        CustomTryonService->>LogService: save_custom_fitting_log(<br/>  dress_url=garment_s3_url,<br/>  run_time=total_time<br/>)
        LogService->>DB: INSERT INTO result_logs<br/>(person_url="", dress_url, result_url="",<br/>model="custom-fitting", prompt="",<br/>success=True, run_time)
        DB-->>LogService: 저장 완료
        LogService-->>CustomTryonService: 성공
    end
    
    CustomTryonService-->>Backend: {v4_result: {...}, v5_result: {...},<br/>cutout_ms, gemini_call_ms, resize_ms}
    
    Backend->>Backend: server_total_ms 계산<br/>(time.time() - server_start_time) * 1000
    Backend->>Backend: cutout_ms, gemini_call_ms, resize_ms 추출<br/>status 판단
    
    opt v5_result가 성공한 경우
        Backend->>Backend: increment_synthesis_count()<br/>(날짜별 합성 카운트 증가)
        Backend->>DB: INSERT INTO daily_synthesis_count<br/>(synthesis_date, count)<br/>VALUES (오늘날짜, 1)<br/>ON DUPLICATE KEY UPDATE count = count + 1
        DB-->>Backend: 카운트 증가 완료
    end
    
    Backend->>ProfileService: save_tryon_profile(<br/>  trace_id,<br/>  endpoint="/tryon/compare/custom",<br/>  front_profile_json,<br/>  server_total_ms,<br/>  resize_ms,<br/>  gemini_call_ms,<br/>  cutout_ms,<br/>  status,<br/>  error_stage<br/>)
    ProfileService->>ProfileService: ensure_table_exists()<br/>(테이블 자동 생성)
    ProfileService->>DB: INSERT INTO tryon_profile_summary<br/>(trace_id, endpoint, front_profile_json,<br/>server_total_ms, resize_ms, gemini_call_ms,<br/>cutout_ms, status, error_stage)<br/>ON DUPLICATE KEY UPDATE
    DB-->>ProfileService: 저장 완료
    ProfileService-->>Backend: 성공
    
    alt 예외 발생
        Backend->>Backend: server_total_ms 계산
        Backend->>ProfileService: save_tryon_profile(<br/>  status="fail",<br/>  error_stage="exception"<br/>)
        Backend-->>API: 500 에러 응답
        API-->>Frontend: 에러 메시지
    else 정상 처리
        Backend-->>API: V4V5CustomCompareResponse<br/>(Headers: X-Trace-Id)
        API-->>Frontend: 결과 이미지 (v4_result, v5_result)
    end
    Frontend->>Frontend: compose_click_to_response_ms 측정 완료
    
    Frontend->>Frontend: result_image_load_start = Date.now()
    Frontend->>Frontend: 이미지 선택 모달 표시<br/>(2개 이미지가 있는 경우)
    User->>Frontend: 결과 이미지 선택
    Frontend->>Frontend: result_image_load_ms 측정 완료
    Frontend-->>User: 선택한 이미지 표시
    
    opt 리뷰 모달 표시
        Frontend->>API: submitReview({category: 'custom', rating, content})
        API->>Backend: POST /api/reviews
        Backend->>DB: INSERT INTO reviews<br/>(rating, content, category)
        DB-->>Backend: review_id 반환
        Backend-->>API: {success: true, review_id}
        API-->>Frontend: 리뷰 제출 완료
        Frontend->>Frontend: 쿠키에 리뷰 제출 기록 저장
    end
```

---

### 2.2 Custom Match V5V5

### 3.1 체형 분석 전체 플로우 (로깅 포함)

```mermaid
sequenceDiagram
    participant User as 사용자
    participant Frontend as 프론트엔드<br/>(BodyAnalysis.jsx)
    participant API as API 유틸<br/>(api.js)
    participant Backend as 백엔드<br/>(body_analysis.py)
    participant BodyService as 체형 분석 서비스
    participant BodyLogService as 체형 분석 로그 서비스<br/>(body_analysis_database.py)
    participant MediaPipe as MediaPipe Pose
    participant Gemini as Gemini API
    participant DB as 데이터베이스

    User->>Frontend: 전신 이미지 업로드
    Frontend->>API: validatePerson(image)
    API->>Backend: POST /api/validate-person
    Backend->>BodyService: MediaPipe로 사람 감지
    BodyService->>MediaPipe: 포즈 랜드마크 추출
    MediaPipe-->>BodyService: 랜드마크 데이터
    
    alt 동물 감지
        BodyService-->>Backend: {is_animal: true}
        Backend-->>API: 에러 응답
        API-->>Frontend: "인물사진을 업로드해주세요"
    else 얼굴만 감지 (전신 랜드마크 없음)
        BodyService-->>Backend: {is_face_only: true}
        Backend-->>API: 에러 응답
        API-->>Frontend: "전신 사진을 넣어주세요"
    else 사람 감지 성공
        BodyService-->>Backend: {success: true, is_person: true}
        Backend-->>API: 검증 성공
        API-->>Frontend: 검증 성공
    end
    
    User->>Frontend: 키와 몸무게 입력
    User->>Frontend: 분석하기 버튼 클릭
    Frontend->>API: analyzeBody(image, height, weight)
    API->>Backend: POST /api/analyze-body<br/>(FormData: file, height, weight)
    
    Backend->>Backend: 분석 시작 시간 기록<br/>(start_time = time.time())
    Backend->>BodyService: 체형 분석 시작
    BodyService->>MediaPipe: 포즈 랜드마크 추출
    MediaPipe-->>BodyService: 33개 랜드마크 좌표
    
    BodyService->>BodyService: 체형 특징 계산<br/>(어깨, 허리, 엉덩이 비율 등)
    BodyService->>BodyService: 체형 타입 분류<br/>(키, 체형 라인 등)
    BodyService->>BodyService: BMI 계산<br/>(height, weight)
    
    BodyService->>Gemini: 체형 분석 상세 설명 요청<br/>(랜드마크 데이터, 체형 특징, 키/몸무게)
    Gemini-->>BodyService: 상세 분석 텍스트<br/>(추천 드레스 스타일 포함)
    
    BodyService->>Backend: {body_analysis: {...}, gemini_analysis: {...}}
    Backend->>Backend: run_time 계산<br/>(time.time() - start_time)
    Backend->>Backend: 체형 특징을 문자열로 변환<br/>(쉼표로 구분)
    
    Backend->>BodyLogService: save_body_analysis_result(<br/>  model='body_analysis',<br/>  run_time,<br/>  height,<br/>  weight,<br/>  prompt='체형 분석 (MediaPipe + Gemini)',<br/>  bmi,<br/>  characteristic,<br/>  analysis_results=gemini_analysis_text<br/>)
    BodyLogService->>DB: INSERT INTO body_logs<br/>(model, run_time, height, weight,<br/>prompt, bmi, characteristic,<br/>analysis_results)
    DB-->>BodyLogService: result_id 반환
    BodyLogService-->>Backend: result_id
    
    Backend-->>API: {success: true, body_analysis: {...},<br/>gemini_analysis: {...}, run_time}
    API-->>Frontend: 분석 결과
    Frontend->>Frontend: 분석 결과 파싱<br/>(추천 드레스 카테고리 추출)
    Frontend-->>User: 체형 분석 결과 표시<br/>(체형 타입, 특징, 추천 드레스)
    
    opt 추천 드레스 카테고리 클릭
        User->>Frontend: 추천 카테고리 클릭
        Frontend->>Frontend: 일반 피팅 페이지로 이동<br/>(해당 카테고리로 필터링)
    end
    
    opt 리뷰 모달 표시
        Frontend->>API: submitReview({category: 'analysis', rating, content})
        API->>Backend: POST /api/reviews
        Backend->>DB: INSERT INTO reviews<br/>(rating, content, category)
        DB-->>Backend: review_id 반환
        Backend-->>API: {success: true, review_id}
        API-->>Frontend: 리뷰 제출 완료
        Frontend->>Frontend: 쿠키에 리뷰 제출 기록 저장
    end
```

## 4. 드레스 관리

### 4.1 드레스 목록 조회 및 프록시

```mermaid
sequenceDiagram
    participant User as 사용자
    participant Frontend as 프론트엔드
    participant API as API 유틸<br/>(api.js)
    participant Backend as 백엔드<br/>(dress_management.py)
    participant ProxyRouter as 프록시 라우터<br/>(proxy.py)
    participant DB as 데이터베이스
    participant S3 as S3 Storage

    User->>Frontend: 드레스 목록 조회
    Frontend->>API: getDresses()
    API->>Backend: GET /api/admin/dresses?limit=1000
    Backend->>DB: SELECT * FROM dresses LIMIT 1000
    DB-->>Backend: 드레스 목록 (S3 URL 포함)
    Backend-->>API: {success: true, data: [{url: "s3://..."}, ...]}
    
    API-->>Frontend: 드레스 목록
    Frontend->>Frontend: S3 URL을 프록시 URL로 변환<br/>(/api/proxy-image?url=...)
    
    User->>Frontend: 드레스 이미지 표시 요청
    Frontend->>Backend: GET /api/proxy-image?url={s3_url}
    Backend->>ProxyRouter: 프록시 요청 처리
    ProxyRouter->>S3: S3 이미지 다운로드
    S3-->>ProxyRouter: 이미지 바이너리
    ProxyRouter->>ProxyRouter: 이미지 반환
    ProxyRouter-->>Frontend: 이미지 응답
    Frontend-->>User: 드레스 이미지 표시
```

### 4.2 드레스 체크 (커스텀 피팅용)

```mermaid
sequenceDiagram
    actor User
    participant Page as GeneralFitting.jsx
    participant API as api.js
    participant BE as Backend (body_analysis)
    participant S as BodyAnalysisService
    participant MP as MediaPipe
    participant AI as Gemini

    User->>Page: 이미지 업로드 (유효성 검사)
    Page->>Page: setIsValidatingPerson(true)
    Page->>API: validatePerson(image)
    API->>BE: POST /api/validate-person
    BE->>S: extract_landmarks()
    S->>MP: 포즈 랜드마크 추출
    MP-->>S: 랜드마크
    BE->>BE: 전체 몸 확인
    
    alt 유효한 인물
        BE-->>API: { success: true, is_person: true }
        API-->>Page: 유효성 검사 성공
        Page->>Page: handleImageUpload(image)
    else 동물 / 잘못된 인물
        BE-->>API: { is_animal: true, message: "동물 감지됨" }
        API-->>Page: 유효성 검사 실패
        Page->>Page: setValidationMessage(msg)
        Page->>Page: setValidationModalOpen(true)
    else 오류
        BE-->>API: 500 오류
        API-->>Page: 오류 발생
        Page->>Page: setValidationMessage("오류")
        Page->>Page: setValidationModalOpen(true)
    end
    Page->>Page: setIsValidatingPerson(false)

    User->>Page: 전체 몸 + 키/체중 업로드 (분석 페이지)
    Page->>API: analyzeBody(image, h, w)
    API->>BE: POST /api/analyze-body
    BE->>S: extract_landmarks()
    S->>MP: 포즈 랜드마크 추출
    MP-->>S: 랜드마크
    S->>S: 측정값 계산
    S->>AI: classify_body_line_with_gemini()
    AI-->>S: 체형 유형
    S->>AI: analyze_body_with_gemini()
    AI-->>S: 상세 분석
    S-->>BE: 분석 결과
    BE-->>API: JSON 응답
    API-->>Page: 분석 데이터
    Page-->>User: 분석 결과 표시
```

---

## 4. 관리자 및 관리 (Admin & Management)

**엔드포인트**: `/api/admin/dresses`, `/api/admin/stats`, `/api/admin/logs`
**프론트엔드**: `AdminPage.jsx` (예시)

```mermaid
sequenceDiagram
    actor Admin
    participant Page as AdminPage.jsx
    participant API as api.js
    participant BE as Backend (admin_router)
    participant DB as Database

    Admin->>Page: 대시보드 보기
    Page->>API: getAdminStats()
    API->>BE: GET /api/admin/stats
    BE->>DB: 방문자/합성 통계 조회
    DB-->>BE: 통계 데이터
    BE-->>API: 통계 JSON
    API-->>Page: 대시보드 업데이트

    Admin->>Page: 드레스 업로드
    Page->>API: uploadDress(file, meta)
    API->>BE: POST /api/admin/dresses (Multipart)
    BE->>BE: 이미지 저장 (Static/S3)
    BE->>DB: 드레스 메타 데이터
```


삽입
DB-->>BE: 드레스 ID
BE-->>API: 성공 응답
API-->>Page: 성공 토스트 표시

```
Admin->>Page: 체형 분석 로그 보기
Page->>API: getBodyAnalysisLogs()
API->>BE: GET /api/admin/body-logs
BE->>DB: 로그 조회
DB-->>BE: 로그 목록
BE-->>API: 로그 JSON
API-->>Page: 로그 테이블 렌더링
```

````

---

## 5. 드레스 관리 (Dress Management)
**엔드포인트**: `/api/dress/check`
**프론트엔드**: `CustomFitting.jsx`, `api.js`

```mermaid
sequenceDiagram
    actor User
    participant Page as CustomFitting.jsx
    participant API as api.js
    participant BE as Backend (dress_management)
    participant AI as External AI (GPT-4o-mini)

    User->>Page: 드레스 이미지 업로드
    Page->>Page: handleDressFile()
    Page->>Page: setIsCheckingDress(true)
    Page->>API: checkDress(image)
    API->>BE: POST /api/dress/check
    BE->>AI: 이미지 분석 (드레스 여부 확인)
    AI-->>BE: 분석 결과
    BE-->>API: JSON 응답 (is_dress)
    API-->>Page: 결과 확인
    
    alt 드레스
        Page->>Page: setDressCheckResult(true)
    else 드레스 아님
        Page->>Page: setErrorMessage("드레스 아님")
        Page->>Page: setErrorModalOpen(true)
    end
    Page->>Page: setIsCheckingDress(false)
````

---

## 6. 리뷰 시스템 (Review System)

**엔드포인트**: `/api/reviews`
**프론트엔드**: `ReviewModal.jsx`

```mermaid
sequenceDiagram
    actor User
    participant Modal as ReviewModal.jsx
    participant API as api.js
    participant BE as Backend (review_router)
    participant DB as Database

    User->>Modal: 리뷰 제출 (평점, 내용)
    Modal->>API: submitReview(data)
    API->>BE: POST /api/reviews
    BE->>DB: 리뷰 삽입
    DB-->>BE: 성공
    BE-->>API: 200 OK
    API-->>Modal: 성공
    Modal->>Modal: 모달 닫고 쿠키 설정

    User->>Modal: 리뷰 보기
    Modal->>API: getReviews()
    API->>BE: GET /api/reviews
    BE->>DB: 리뷰 조회
    DB-->>BE: 리뷰 목록
    BE-->>API: 리뷰 JSON
    API-->>Modal: 리뷰 렌더링
```

---

## 7. 방문자 추적 (Visitor Tracking)

**엔드포인트**: `/visitor/visit`, `/visitor/today`
**프론트엔드**: `App.jsx`, `api.js`

```mermaid
sequenceDiagram
    actor User
    participant App as App.jsx
    participant API as api.js
    participant BE as Backend (visitor_router)
    participant DB as Database

    User->>Frontend: 메인 페이지 접속
    Frontend->>Frontend: useEffect에서 자동 호출
    Frontend->>API: countVisitor()
    API->>Backend: POST /visitor/visit<br/>(빈 body)
    
    Backend->>Backend: IP 주소 추출<br/>(Request 헤더에서)
    Backend->>Backend: 현재 날짜/시간 기록
    Backend->>DB: INSERT INTO visitors<br/>(ip_address, visit_date, visit_time)
    DB-->>Backend: 저장 완료
    
    Backend-->>API: {success: true}
    API-->>Frontend: 성공 응답 (조용히 처리)
    
    Note over Frontend: 접속자 카운팅 실패는<br/>사용자 경험에 영향 없도록<br/>에러를 표시하지 않음
```

## 9. 프로파일링 시스템

### 9.1 트라이온 프로파일링 데이터 수집 (로깅 포함)

```mermaid
sequenceDiagram
    participant Frontend as 프론트엔드<br/>(GeneralFitting.jsx)
    participant Backend as 백엔드<br/>(tryon_router.py)
    participant ProfileService as 프로파일링 서비스<br/>(profile_service.py)
    participant DB as 데이터베이스

    Note over Frontend: 사용자 액션별로<br/>타이밍 측정
    
    Frontend->>Frontend: 배경 선택 시작
    Frontend->>Frontend: bg_select_start = Date.now()
    Frontend->>Frontend: 배경 선택 완료
    Frontend->>Frontend: bg_select_ms = Date.now() - bg_select_start
    
    Frontend->>Frontend: 이미지 업로드 시작
    Frontend->>Frontend: person_upload_start = Date.now()
    Frontend->>Backend: POST /api/validate-person
    Backend-->>Frontend: 검증 완료
    Frontend->>Frontend: person_validate_start = Date.now()
    Frontend->>Frontend: person_validate_ms = Date.now() - person_validate_start
    Frontend->>Frontend: person_upload_ms = Date.now() - person_upload_start
    
    Frontend->>Frontend: 드레스 드롭
    Frontend->>Frontend: dress_drop_start = Date.now()
    Frontend->>Frontend: dress_drop_ms = Date.now() - dress_drop_start
    
    Frontend->>Frontend: compose_click_start = Date.now()
    Frontend->>Backend: POST /tryon/compare<br/>(profile_front: JSON.stringify({<br/>  bg_select_ms,<br/>  person_upload_ms,<br/>  person_validate_ms,<br/>  dress_drop_ms<br/>}),<br/>Headers: X-Trace-Id)
    
    Backend->>Backend: traceId 추출 또는 생성<br/>server_start_time = time.time()<br/>front_profile_json 파싱
    
    Backend->>Backend: 트라이온 처리 실행<br/>(run_v4v5_compare)
    Backend->>Backend: 서버 처리 완료 시간 기록<br/>server_total_ms = (time.time() - server_start_time) * 1000<br/>gemini_call_ms, resize_ms 수집
    
    Backend->>ProfileService: save_tryon_profile(<br/>  trace_id,<br/>  endpoint="/tryon/compare",<br/>  front_profile_json,<br/>  server_total_ms,<br/>  resize_ms,<br/>  gemini_call_ms,<br/>  cutout_ms=None,<br/>  status,<br/>  error_stage<br/>)
    
    ProfileService->>ProfileService: ensure_table_exists()<br/>(테이블 자동 생성)
    ProfileService->>DB: INSERT INTO tryon_profile_summary<br/>(trace_id, endpoint, front_profile_json,<br/>server_total_ms, resize_ms, gemini_call_ms,<br/>cutout_ms, status, error_stage)<br/>ON DUPLICATE KEY UPDATE
    DB-->>ProfileService: 저장 완료
    ProfileService-->>Backend: 성공
    
    Backend-->>Frontend: 응답 (X-Trace-Id 헤더 포함)
    Frontend->>Frontend: compose_click_to_response_ms = Date.now() - compose_click_start
    Frontend->>Frontend: result_image_load_start = Date.now()
    Frontend->>Frontend: result_image_load_ms = Date.now() - result_image_load_start
```

## 10. 통합 플로우: 일반 피팅 완전한 시퀀스 (로깅 포함)

```mermaid
sequenceDiagram
    participant User as 사용자
    participant Frontend as 프론트엔드
    participant API as API Layer
    participant Router as 라우터<br/>(tryon_router.py)
    participant TryonService as 트라이온 서비스<br/>(tryon_compare_service.py)
    participant ProfileService as 프로파일링 서비스
    participant DressLogService as 드레스 피팅 로그 서비스
    participant External as 외부 서비스<br/>(Gemini, S3, DB)

    User->>Frontend: 페이지 로드
    Frontend->>API: getDresses()
    API->>Router: GET /api/admin/dresses
    Router->>External: DB 쿼리
    External-->>Router: 드레스 데이터
    Router-->>API: 드레스 목록
    API-->>Frontend: 드레스 목록
    
    User->>Frontend: 배경 선택
    Frontend->>Frontend: traceId 생성<br/>bg_select_ms 측정
    
    User->>Frontend: 이미지 업로드
    Frontend->>Frontend: person_upload_start 기록
    Frontend->>API: validatePerson(image)
    API->>Router: POST /api/validate-person
    Router->>External: MediaPipe 처리
    External-->>Router: 감지 결과
    Router-->>API: 검증 성공
    API-->>Frontend: 검증 완료
    Frontend->>Frontend: person_validate_ms, person_upload_ms 측정
    
    User->>Frontend: 드레스 드래그
    Frontend->>Frontend: dress_drop_ms 측정
    Frontend->>Frontend: compose_click_start 기록
    Frontend->>API: autoMatchImageV5V5(..., traceId, profile)
    API->>Router: POST /tryon/compare<br/>(Headers: X-Trace-Id, FormData: profile_front)
    
    Router->>Router: traceId 추출 또는 생성<br/>server_start_time 기록<br/>dress_id 추출 (Form 파라미터)
    Router->>TryonService: run_v4v5_compare(..., enable_logging=True)
    TryonService->>TryonService: load_v5_unified_prompt()<br/>(정적 프롬프트 로드)
    TryonService->>External: Gemini API 호출 (병렬 2회)<br/>(이미지 + 정적 프롬프트 전달)
    External-->>TryonService: 합성 이미지<br/>(gemini_call_ms, resize_ms 포함)
    TryonService-->>Router: {v4_result, v5_result, gemini_call_ms, resize_ms}
    
    Router->>Router: server_total_ms 계산<br/>status 판단
    
    opt v5_result 성공
        Router->>External: increment_synthesis_count()<br/>(날짜별 합성 카운트 증가)
        External->>External: INSERT INTO daily_synthesis_count<br/>(synthesis_date, count)<br/>VALUES (오늘날짜, 1)<br/>ON DUPLICATE KEY UPDATE count = count + 1
        External-->>Router: 카운트 증가 완료
        
        opt dress_id 존재
            Router->>DressLogService: log_dress_fitting(dress_id)
            DressLogService->>External: INSERT INTO dress_fitting_logs
            External-->>DressLogService: 저장 완료
        end
    end
    
    Router->>ProfileService: save_tryon_profile(<br/>  trace_id, endpoint, front_profile_json,<br/>  server_total_ms, resize_ms,<br/>  gemini_call_ms, cutout_ms=None,<br/>  status, error_stage)
    ProfileService->>External: INSERT INTO tryon_profile_summary<br/>ON DUPLICATE KEY UPDATE
    External-->>ProfileService: 저장 완료
    ProfileService-->>Router: 성공
    
    Router-->>API: 응답 (X-Trace-Id 헤더)
    API-->>Frontend: 결과 이미지
    Frontend->>Frontend: compose_click_to_response_ms 측정<br/>result_image_load_ms 측정
    Frontend-->>User: 결과 표시
    
    opt 필터 적용
        User->>Frontend: 필터 선택
        Frontend->>API: applyImageFilter(...)
        API->>Router: POST /api/apply-image-filters
        Router->>External: 필터 처리
        External-->>Router: 필터 적용 이미지
        Router-->>API: 결과
        API-->>Frontend: 필터 적용 이미지
        Frontend-->>User: 필터 적용 결과
    end
    
    opt 리뷰 제출
        User->>Frontend: 리뷰 입력
        Frontend->>API: submitReview(...)
        API->>Router: POST /api/reviews
        Router->>External: INSERT INTO reviews
        External-->>Router: review_id
        Router-->>API: 응답
        API-->>Frontend: 리뷰 제출 완료
        Frontend->>Frontend: 쿠키에 리뷰 제출 기록 저장
    end
```

