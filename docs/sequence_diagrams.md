# 시스템 시퀀스 다이어그램

이 문서는 백엔드와 프론트엔드 간의 주요 기능별 상세 시퀀스 다이어그램을 포함합니다.

## 1. 일반 피팅 (General Fitting)

### 1.1 드레스 목록 조회

```mermaid
sequenceDiagram
    participant User as 사용자
    participant Frontend as 프론트엔드<br/>(GeneralFitting.jsx)
    participant API as API 유틸<br/>(api.js)
    participant Backend as 백엔드<br/>(dress_management.py)
    participant DB as 데이터베이스

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

### 1.2 일반 피팅 전체 플로우

```mermaid
sequenceDiagram
    participant User as 사용자
    participant Frontend as 프론트엔드<br/>(GeneralFitting.jsx)
    participant API as API 유틸<br/>(api.js)
    participant Backend as 백엔드
    participant BodyService as 체형 분석 서비스
    participant TryonService as 트라이온 서비스
    participant Gemini as Gemini API
    participant S3 as S3 Storage

    User->>Frontend: 배경 선택
    Frontend->>Frontend: traceId 생성
    
    User->>Frontend: 전신 이미지 업로드
    Frontend->>API: validatePerson(image)
    API->>Backend: POST /api/validate-person
    Backend->>BodyService: MediaPipe로 사람 감지
    BodyService-->>Backend: {success: true, is_person: true}
    Backend-->>API: 검증 결과
    API-->>Frontend: 검증 성공
    
    User->>Frontend: 드레스 드래그 앤 드롭
    Frontend->>API: autoMatchImageV5V5(person, dress, background, traceId, profile)
    Note over Frontend,API: profile에 타이밍 정보 포함<br/>(bg_select_ms, person_upload_ms 등)
    
    API->>Backend: POST /tryon/compare<br/>(FormData: person_image, garment_image, background_image, profile_front)
    Backend->>Backend: traceId 추출 및 프로파일링 시작
    
    Backend->>TryonService: run_v4v5_compare(person, garment, background)
    TryonService->>TryonService: V5 파이프라인 병렬 실행 (2회)
    
    par V5-1 실행
        TryonService->>Gemini: 프롬프트 생성 요청
        Gemini-->>TryonService: 프롬프트 반환
        TryonService->>Gemini: 이미지 합성 요청
        Gemini-->>TryonService: 합성 이미지 (base64)
    and V5-2 실행
        TryonService->>Gemini: 프롬프트 생성 요청
        Gemini-->>TryonService: 프롬프트 반환
        TryonService->>Gemini: 이미지 합성 요청
        Gemini-->>TryonService: 합성 이미지 (base64)
    end
    
    TryonService->>Backend: {v4_result: {...}, v5_result: {...}}
    Backend->>Backend: 프로파일링 데이터 저장<br/>(save_tryon_profile)
    Backend-->>API: V4V5CompareResponse
    API-->>Frontend: 결과 이미지 (v4_result, v5_result)
    
    Frontend->>Frontend: 이미지 선택 모달 표시<br/>(2개 이미지가 있는 경우)
    User->>Frontend: 결과 이미지 선택
    Frontend-->>User: 선택한 이미지 표시
    
    opt 리뷰 모달 표시 (쿠키 확인 후)
        Frontend->>Frontend: 3초 후 리뷰 모달 표시
        User->>Frontend: 리뷰 제출
        Frontend->>API: submitReview({category: 'general', rating, content})
        API->>Backend: POST /api/reviews
        Backend->>DB: INSERT INTO reviews
        DB-->>Backend: 성공
        Backend-->>API: {success: true}
        API-->>Frontend: 리뷰 제출 완료
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

### 2.1 커스텀 피팅 전체 플로우

```mermaid
sequenceDiagram
    participant User as 사용자
    participant Frontend as 프론트엔드<br/>(CustomFitting.jsx)
    participant API as API 유틸<br/>(api.js)
    participant Backend as 백엔드
    participant BodyService as 체형 분석 서비스
    participant DressService as 드레스 체크 서비스
    participant CustomTryonService as 커스텀 트라이온 서비스
    participant SegformerService as SegFormer 서비스
    participant Gemini as Gemini API

    User->>Frontend: 배경 선택
    Frontend->>Frontend: traceId 생성
    
    User->>Frontend: 전신 이미지 업로드
    Frontend->>API: validatePerson(image)
    API->>Backend: POST /api/validate-person
    Backend->>BodyService: MediaPipe로 사람 감지
    BodyService-->>Backend: {success: true, is_person: true}
    Backend-->>API: 검증 결과
    API-->>Frontend: 검증 성공
    
    User->>Frontend: 드레스 이미지 업로드
    Frontend->>API: checkDress(dressImage)
    API->>Backend: POST /api/dress/check<br/>(FormData: file, model, mode)
    Backend->>DressService: GPT-4o-mini로 드레스 체크
    DressService->>DressService: 이미지 분석 및 드레스 여부 판단
    DressService-->>Backend: {success: true, result: {dress: true}}
    Backend-->>API: 드레스 체크 결과
    API-->>Frontend: {success: true, result: {dress: true}}
    
    alt 드레스가 아닌 경우
        Frontend->>Frontend: 에러 모달 표시
        Frontend-->>User: "드레스 사진을 넣어주세요"
    end
    
    User->>Frontend: 매칭하기 버튼 클릭
    Frontend->>API: customV5V5MatchImage(fullBody, dress, background, traceId, profile)
    
    API->>Backend: POST /tryon/compare/custom<br/>(FormData: person_image, garment_image, background_image, profile_front)
    Backend->>Backend: traceId 추출 및 프로파일링 시작
    
    Backend->>CustomTryonService: run_v4v5_custom_compare(person, garment, background)
    
    par CustomV5-1 실행
        CustomTryonService->>SegformerService: 의상 누끼 처리 (배경 제거)
        SegformerService-->>CustomTryonService: 누끼 처리된 의상 이미지
        CustomTryonService->>Gemini: 프롬프트 생성 요청
        Gemini-->>CustomTryonService: 프롬프트 반환
        CustomTryonService->>Gemini: 이미지 합성 요청
        Gemini-->>CustomTryonService: 합성 이미지 (base64)
    and CustomV5-2 실행
        CustomTryonService->>SegformerService: 의상 누끼 처리 (배경 제거)
        SegformerService-->>CustomTryonService: 누끼 처리된 의상 이미지
        CustomTryonService->>Gemini: 프롬프트 생성 요청
        Gemini-->>CustomTryonService: 프롬프트 반환
        CustomTryonService->>Gemini: 이미지 합성 요청
        Gemini-->>CustomTryonService: 합성 이미지 (base64)
    end
    
    CustomTryonService->>Backend: {v4_result: {...}, v5_result: {...}}
    Backend->>Backend: 프로파일링 데이터 저장
    Backend-->>API: V4V5CustomCompareResponse
    API-->>Frontend: 결과 이미지 (v4_result, v5_result)
    
    Frontend->>Frontend: 이미지 선택 모달 표시<br/>(2개 이미지가 있는 경우)
    User->>Frontend: 결과 이미지 선택
    Frontend-->>User: 선택한 이미지 표시
    
    opt 리뷰 모달 표시
        Frontend->>API: submitReview({category: 'custom', rating, content})
        API->>Backend: POST /api/reviews
        Backend->>DB: INSERT INTO reviews
        Backend-->>API: {success: true}
        API-->>Frontend: 리뷰 제출 완료
    end
```

## 3. 체형 분석 (Body Analysis)

### 3.1 체형 분석 전체 플로우

```mermaid
sequenceDiagram
    participant User as 사용자
    participant Frontend as 프론트엔드<br/>(BodyAnalysis.jsx)
    participant API as API 유틸<br/>(api.js)
    participant Backend as 백엔드<br/>(body_analysis.py)
    participant BodyService as 체형 분석 서비스
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
    
    Backend->>BodyService: 체형 분석 시작
    BodyService->>MediaPipe: 포즈 랜드마크 추출
    MediaPipe-->>BodyService: 33개 랜드마크 좌표
    
    BodyService->>BodyService: 체형 특징 계산<br/>(어깨, 허리, 엉덩이 비율 등)
    BodyService->>BodyService: 체형 타입 분류<br/>(키, 체형 라인 등)
    
    BodyService->>Gemini: 체형 분석 상세 설명 요청<br/>(랜드마크 데이터, 체형 특징, 키/몸무게)
    Gemini-->>BodyService: 상세 분석 텍스트<br/>(추천 드레스 스타일 포함)
    
    BodyService->>Backend: {body_analysis: {...}, gemini_analysis: {...}}
    Backend->>DB: 체형 분석 결과 저장<br/>(save_body_analysis_result)
    DB-->>Backend: 저장 완료
    Backend-->>API: 분석 결과
    API-->>Frontend: {success: true, body_analysis: {...}, gemini_analysis: {...}}
    
    Frontend->>Frontend: 분석 결과 파싱<br/>(추천 드레스 카테고리 추출)
    Frontend-->>User: 체형 분석 결과 표시<br/>(체형 타입, 특징, 추천 드레스)
    
    opt 추천 드레스 카테고리 클릭
        User->>Frontend: 추천 카테고리 클릭
        Frontend->>Frontend: 일반 피팅 페이지로 이동<br/>(해당 카테고리로 필터링)
    end
    
    opt 리뷰 모달 표시
        Frontend->>API: submitReview({category: 'analysis', rating, content})
        API->>Backend: POST /api/reviews
        Backend->>DB: INSERT INTO reviews
        Backend-->>API: {success: true}
        API-->>Frontend: 리뷰 제출 완료
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
    participant User as 사용자
    participant Frontend as 프론트엔드<br/>(CustomFitting.jsx)
    participant API as API 유틸<br/>(api.js)
    participant Backend as 백엔드<br/>(dress_management.py)
    participant DressService as 드레스 체크 서비스
    participant OpenAI as OpenAI API<br/>(GPT-4o-mini)

    User->>Frontend: 드레스 이미지 업로드
    Frontend->>API: checkDress(imageFile, model='gpt-4o-mini', mode='fast')
    API->>Backend: POST /api/dress/check<br/>(FormData: file, model, mode)
    
    Backend->>DressService: 드레스 체크 시작
    DressService->>DressService: 이미지를 base64로 인코딩
    
    alt mode == 'fast'
        DressService->>OpenAI: Vision API 호출<br/>(GPT-4o-mini, 간단한 프롬프트)
    else mode == 'accurate'
        DressService->>OpenAI: Vision API 호출<br/>(GPT-4o, 상세 프롬프트)
    end
    
    OpenAI-->>DressService: 분석 결과 (드레스 여부 판단)
    DressService->>Backend: {success: true, result: {dress: true/false}}
    Backend-->>API: 드레스 체크 결과
    API-->>Frontend: {success: true, result: {dress: true/false}}
    
    alt 드레스가 아닌 경우
        Frontend->>Frontend: 에러 모달 표시
        Frontend-->>User: "드레스 사진을 넣어주세요"
    else 드레스인 경우
        Frontend->>Frontend: 드레스 이미지 미리보기 표시
        Frontend-->>User: 드레스 이미지 표시
    end
```

## 5. 리뷰 시스템

### 5.1 리뷰 제출

```mermaid
sequenceDiagram
    participant User as 사용자
    participant Frontend as 프론트엔드<br/>(ReviewModal.jsx)
    participant API as API 유틸<br/>(api.js)
    participant Backend as 백엔드<br/>(review.py)
    participant DB as 데이터베이스

    User->>Frontend: 리뷰 모달에서 별점 및 내용 입력
    User->>Frontend: 리뷰 제출 버튼 클릭
    
    Frontend->>Frontend: 쿠키 확인<br/>(isReviewCompleted)
    
    alt 이미 리뷰를 제출한 경우
        Frontend-->>User: 리뷰 모달 표시 안 함
    else 리뷰를 제출하지 않은 경우
        Frontend->>API: submitReview({category, rating, content})
        API->>Backend: POST /api/reviews<br/>(JSON: {category, rating, content})
        
        Backend->>Backend: 카테고리 유효성 검사<br/>(general, custom, analysis)
        
        Backend->>DB: INSERT INTO reviews<br/>(rating, content, category)
        DB-->>Backend: review_id 반환
        
        Backend->>Frontend: {success: true, review_id}
        Frontend->>Frontend: 쿠키에 리뷰 제출 기록 저장
        Frontend-->>User: 리뷰 제출 완료 메시지
    end
```

### 5.2 리뷰 조회

```mermaid
sequenceDiagram
    participant Admin as 관리자
    participant Frontend as 프론트엔드<br/>(관리자 페이지)
    participant API as API 유틸
    participant Backend as 백엔드<br/>(review.py)
    participant DB as 데이터베이스

    Admin->>Frontend: 리뷰 목록 조회
    Frontend->>API: GET /api/reviews?category=general&limit=100&offset=0
    API->>Backend: GET /api/reviews?category=general&limit=100&offset=0
    
    Backend->>DB: SELECT COUNT(*) FROM reviews WHERE category = 'general'
    DB-->>Backend: total count
    
    Backend->>DB: SELECT * FROM reviews<br/>WHERE category = 'general'<br/>ORDER BY created_at DESC<br/>LIMIT 100 OFFSET 0
    DB-->>Backend: 리뷰 목록
    
    Backend->>DB: SELECT AVG(rating) FROM reviews
    DB-->>Backend: 평균 별점
    
    Backend-->>API: {success: true, total, average_rating, reviews: [...]}
    API-->>Frontend: 리뷰 목록 및 통계
    Frontend-->>Admin: 리뷰 목록 표시
```

## 6. 인증 시스템

### 6.1 관리자 로그인

```mermaid
sequenceDiagram
    participant Admin as 관리자
    participant Frontend as 프론트엔드<br/>(관리자 페이지)
    participant API as API 유틸
    participant Backend as 백엔드<br/>(auth.py)
    participant SessionService as 세션 서비스
    participant DB as 데이터베이스

    Admin->>Frontend: 로그인 페이지 접속
    Admin->>Frontend: 아이디/비밀번호 입력
    Admin->>Frontend: 로그인 버튼 클릭
    
    Frontend->>API: POST /api/auth/login<br/>(JSON: {username, password})
    API->>Backend: POST /api/auth/login
    
    Backend->>Backend: 비밀번호 해시 검증
    Backend->>DB: 사용자 정보 조회
    
    alt 인증 실패
        DB-->>Backend: 사용자 없음 또는 비밀번호 불일치
        Backend-->>API: {success: false, message: "인증 실패"}
        API-->>Frontend: 에러 응답
        Frontend-->>Admin: 로그인 실패 메시지
    else 인증 성공
        DB-->>Backend: 사용자 정보
        Backend->>SessionService: 세션 생성
        SessionService-->>Backend: session_id
        Backend->>Backend: 세션 쿠키 설정
        Backend-->>API: {success: true, session_id}
        API-->>Frontend: 로그인 성공
        Frontend->>Frontend: 세션 쿠키 저장
        Frontend-->>Admin: 관리자 페이지로 리다이렉트
    end
```

### 6.2 인증 검증

```mermaid
sequenceDiagram
    participant Admin as 관리자
    participant Frontend as 프론트엔드<br/>(관리자 페이지)
    participant API as API 유틸
    participant Backend as 백엔드<br/>(auth.py)
    participant SessionService as 세션 서비스
    participant AuthMiddleware as 인증 미들웨어

    Admin->>Frontend: 관리자 페이지 접속
    Frontend->>API: GET /api/auth/verify<br/>(Cookie: session_id)
    API->>Backend: GET /api/auth/verify
    
    Backend->>AuthMiddleware: 세션 검증
    AuthMiddleware->>SessionService: 세션 유효성 확인
    SessionService-->>AuthMiddleware: 세션 유효 여부
    
    alt 세션 만료 또는 무효
        AuthMiddleware-->>Backend: 인증 실패
        Backend-->>API: {success: false, authenticated: false}
        API-->>Frontend: 인증 실패
        Frontend-->>Admin: 로그인 페이지로 리다이렉트
    else 세션 유효
        AuthMiddleware-->>Backend: 인증 성공
        Backend-->>API: {success: true, authenticated: true, user: {...}}
        API-->>Frontend: 인증 성공
        Frontend-->>Admin: 관리자 페이지 표시
    end
```

## 7. 이미지 처리

### 7.1 이미지 프록시 (CORS 해결)

```mermaid
sequenceDiagram
    participant Frontend as 프론트엔드
    participant Backend as 백엔드<br/>(proxy.py)
    participant S3 as S3 Storage
    participant ExternalAPI as 외부 API

    Frontend->>Backend: GET /api/proxy-image?url={s3_url}
    Backend->>Backend: URL 파싱 및 검증
    
    alt S3 URL인 경우
        Backend->>S3: 이미지 다운로드 요청
        S3-->>Backend: 이미지 바이너리
        Backend->>Backend: 이미지 반환<br/>(Content-Type: image/jpeg)
        Backend-->>Frontend: 이미지 응답
    else 외부 URL인 경우
        Backend->>ExternalAPI: HTTP GET 요청
        ExternalAPI-->>Backend: 이미지 바이너리
        Backend->>Backend: 이미지 반환
        Backend-->>Frontend: 이미지 응답
    end
    
    Frontend->>Frontend: 이미지 표시
```

### 7.2 이미지 필터 적용

```mermaid
sequenceDiagram
    participant User as 사용자
    participant Frontend as 프론트엔드
    participant API as API 유틸<br/>(api.js)
    participant Backend as 백엔드<br/>(image_processing.py)
    participant ImageService as 이미지 처리 서비스
    participant PIL as PIL/Pillow

    User->>Frontend: 필터 선택 (grayscale, vintage, warm, cool, high_contrast)
    Frontend->>API: applyImageFilter(image, filterPreset)
    
    alt 이미지가 File 객체인 경우
        API->>Backend: POST /api/apply-image-filters<br/>(FormData: file, filter_preset)
    else 이미지가 Data URL인 경우
        API->>API: Data URL을 File 객체로 변환
        API->>Backend: POST /api/apply-image-filters<br/>(FormData: file, filter_preset)
    else 이미지가 URL인 경우
        API->>Backend: GET /api/proxy-image?url={image_url}
        Backend-->>API: 이미지 바이너리
        API->>API: 이미지를 File 객체로 변환
        API->>Backend: POST /api/apply-image-filters<br/>(FormData: file, filter_preset)
    end
    
    Backend->>ImageService: 필터 적용 처리
    ImageService->>PIL: 이미지 로드
    PIL-->>ImageService: PIL Image 객체
    
    alt filterPreset == 'grayscale'
        ImageService->>PIL: convert('L') - 그레이스케일 변환
    else filterPreset == 'vintage'
        ImageService->>PIL: 색상 조정 (세피아 톤)
    else filterPreset == 'warm'
        ImageService->>PIL: 색상 조정 (따뜻한 톤)
    else filterPreset == 'cool'
        ImageService->>PIL: 색상 조정 (차가운 톤)
    else filterPreset == 'high_contrast'
        ImageService->>PIL: 대비 조정
    end
    
    PIL-->>ImageService: 필터 적용된 이미지
    ImageService->>ImageService: base64 인코딩
    ImageService-->>Backend: {result_image: "data:image/..."}
    Backend-->>API: {success: true, result_image: "data:image/..."}
    API-->>Frontend: 필터 적용된 이미지
    Frontend-->>User: 필터 적용된 결과 표시
```

## 8. 접속자 통계

### 8.1 접속자 카운팅

```mermaid
sequenceDiagram
    participant User as 사용자
    participant Frontend as 프론트엔드<br/>(App.jsx)
    participant API as API 유틸<br/>(api.js)
    participant Backend as 백엔드<br/>(visitor_router.py)
    participant DB as 데이터베이스

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

### 9.1 트라이온 프로파일링 데이터 수집

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
    Frontend->>Frontend: person_validate_ms = Date.now() - person_validate_start
    Frontend->>Frontend: person_upload_ms = Date.now() - person_upload_start
    
    Frontend->>Frontend: 드레스 드롭
    Frontend->>Frontend: dress_drop_ms = Date.now() - dress_drop_start
    
    Frontend->>Backend: POST /tryon/compare<br/>(profile_front: {bg_select_ms, person_upload_ms, ...})
    Backend->>Backend: 서버 처리 시작 시간 기록
    
    Backend->>Backend: 트라이온 처리 실행
    Backend->>Backend: 서버 처리 완료 시간 기록
    Backend->>Backend: server_total_ms 계산
    
    Backend->>ProfileService: save_tryon_profile(<br/>  trace_id,<br/>  endpoint,<br/>  front_profile_json,<br/>  server_total_ms,<br/>  gemini_call_ms,<br/>  status<br/>)
    
    ProfileService->>DB: INSERT INTO tryon_profiles<br/>(trace_id, endpoint, front_timings, server_timings, ...)
    DB-->>ProfileService: 저장 완료
    ProfileService-->>Backend: 성공
    
    Backend-->>Frontend: 응답 (X-Trace-Id 헤더 포함)
```

## 10. 통합 플로우: 일반 피팅 완전한 시퀀스

```mermaid
sequenceDiagram
    participant User as 사용자
    participant Frontend as 프론트엔드
    participant API as API Layer
    participant Router as 라우터
    participant Service as 서비스 레이어
    participant External as 외부 서비스<br/>(Gemini, S3, DB)

    User->>Frontend: 페이지 로드
    Frontend->>API: getDresses()
    API->>Router: GET /api/admin/dresses
    Router->>Service: 드레스 목록 조회
    Service->>External: DB 쿼리
    External-->>Service: 드레스 데이터
    Service-->>Router: 응답
    Router-->>API: 드레스 목록
    API-->>Frontend: 드레스 목록
    
    User->>Frontend: 배경 선택
    Frontend->>Frontend: traceId 생성, 타이밍 시작
    
    User->>Frontend: 이미지 업로드
    Frontend->>API: validatePerson(image)
    API->>Router: POST /api/validate-person
    Router->>Service: 사람 감지
    Service->>External: MediaPipe 처리
    External-->>Service: 감지 결과
    Service-->>Router: 검증 결과
    Router-->>API: 검증 성공
    API-->>Frontend: 검증 완료
    
    User->>Frontend: 드레스 드래그
    Frontend->>API: autoMatchImageV5V5(...)
    API->>Router: POST /tryon/compare
    Router->>Service: V4V5 비교 실행
    Service->>External: Gemini API 호출 (병렬 2회)
    External-->>Service: 합성 이미지
    Service->>External: 프로파일링 데이터 저장
    Service-->>Router: 결과
    Router-->>API: 응답
    API-->>Frontend: 결과 이미지
    Frontend-->>User: 결과 표시
    
    opt 필터 적용
        User->>Frontend: 필터 선택
        Frontend->>API: applyImageFilter(...)
        API->>Router: POST /api/apply-image-filters
        Router->>Service: 필터 처리
        Service-->>Router: 필터 적용 이미지
        Router-->>API: 결과
        API-->>Frontend: 필터 적용 이미지
        Frontend-->>User: 필터 적용 결과
    end
    
    opt 리뷰 제출
        User->>Frontend: 리뷰 입력
        Frontend->>API: submitReview(...)
        API->>Router: POST /api/reviews
        Router->>Service: 리뷰 저장
        Service->>External: DB 저장
        External-->>Service: 저장 완료
        Service-->>Router: 성공
        Router-->>API: 응답
        API-->>Frontend: 리뷰 제출 완료
    end
```

