"""통합 트라이온 라우터"""
import io
import time
import uuid
import json
from fastapi import APIRouter, File, UploadFile, Request, Form
from fastapi.responses import JSONResponse
from PIL import Image
from typing import Optional

from services.tryon_service import generate_unified_tryon, generate_unified_tryon_v2, generate_unified_tryon_v5
from services.tryon_compare_service import run_v4v5_compare
from services.face_swap_service import FaceSwapService
from services.profile_service import save_tryon_profile
from schemas.tryon_schema import UnifiedTryonResponse, V4V5CompareResponse

router = APIRouter()


@router.post("/api/tryon/unified", tags=["통합 트라이온"], response_model=UnifiedTryonResponse)
async def unified_tryon(
    person_image: UploadFile = File(..., description="사람 이미지 파일"),
    dress_image: UploadFile = File(..., description="드레스 이미지 파일"),
    background_image: UploadFile = File(..., description="배경 이미지 파일"),
):
    """
    통합 트라이온 파이프라인: X.AI 프롬프트 생성 + Gemini 2.5 Flash 이미지 합성 (배경 포함)
    
    이 엔드포인트는 다음 단계를 수행합니다:
    1. 이미지 타입 감지 (전신 사진인지 확인)
    2. 전신 사진이면 합성 불가 메시지 반환
    3. 상체/얼굴 사진이면 X.AI를 사용하여 프롬프트 생성 후 Gemini 2.5 Flash로 합성 (배경 포함)
    
    Returns:
        UnifiedTryonResponse: 생성된 프롬프트와 합성 이미지 (base64)
    """
    try:
        # 이미지 읽기
        person_bytes = await person_image.read()
        dress_bytes = await dress_image.read()
        background_bytes = await background_image.read()
        
        if not person_bytes or not dress_bytes or not background_bytes:
            return JSONResponse(
                {
                    "success": False,
                    "prompt": "",
                    "result_image": "",
                    "message": "사람 이미지, 드레스 이미지, 배경 이미지를 모두 업로드해주세요.",
                    "llm": None
                },
                status_code=400,
            )
        
        # PIL Image로 변환
        person_img = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        dress_img = Image.open(io.BytesIO(dress_bytes)).convert("RGB")
        background_img = Image.open(io.BytesIO(background_bytes)).convert("RGB")
        
        # 이미지 타입 감지 (전신 vs 상체/얼굴)
        face_swap_service = FaceSwapService()
        image_type_info = face_swap_service.detect_image_type(person_img)
        image_type = image_type_info.get("type", "unknown")
        confidence = image_type_info.get("confidence", 0.0)
        
        # 전신 사진이면 합성 불가 메시지 반환
        if image_type == "full_body":
            return JSONResponse(
                {
                    "success": False,
                    "prompt": "",
                    "result_image": "",
                    "message": "지금 올려주신 사진은 전신사진입니다. 상체만 나온 사진이나 얼굴만 나온 사진을 업로드해주세요.",
                    "llm": None,
                    "image_type": image_type,
                    "image_type_confidence": round(confidence, 2)
                },
                status_code=400,
            )
        
        # 통합 트라이온 서비스 호출 (상체/얼굴 사진인 경우, 배경 포함)
        result = await generate_unified_tryon(person_img, dress_img, background_img)
        
        # 결과에 이미지 타입 정보 추가
        if isinstance(result, dict):
            result["image_type"] = image_type
            result["image_type_confidence"] = round(confidence, 2)
        
        if result["success"]:
            return JSONResponse(result)
        else:
            status_code = 500 if "error" in result else 400
            return JSONResponse(result, status_code=status_code)
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"통합 트라이온 엔드포인트 오류: {e}")
        print(error_detail)
        
        return JSONResponse(
            {
                "success": False,
                "prompt": "",
                "result_image": "",
                "message": f"통합 트라이온 처리 중 오류가 발생했습니다: {str(e)}",
                "llm": None
            },
            status_code=500,
        )


@router.post("/api/compose_xai_gemini_v2", tags=["통합 트라이온 V2"], response_model=UnifiedTryonResponse)
async def compose_xai_gemini_v2(
    person_image: UploadFile = File(..., description="사람 이미지 파일"),
    garment_image: UploadFile = File(..., description="의상 이미지 파일"),
    background_image: UploadFile = File(..., description="배경 이미지 파일"),
):
    """
    통합 트라이온 파이프라인 V2: SegFormer B2 Garment Parsing + X.AI 프롬프트 생성 + Gemini 2.5 Flash 이미지 합성 (배경 포함)
    
    V2는 다음 단계를 수행합니다:
    1. SegFormer B2 Human Parsing을 사용하여 garment_image에서 garment_only 이미지 추출
    2. X.AI를 사용하여 person_image와 garment_only 이미지로부터 프롬프트 생성
    3. 생성된 프롬프트와 이미지들(인물, garment_only, 배경)을 사용하여 Gemini 2.5 Flash로 최종 합성 이미지 생성
    
    V2는 배경 이미지를 포함하여 합성합니다.
    
    Returns:
        UnifiedTryonResponse: 생성된 프롬프트와 합성 이미지 (base64)
    """
    try:
        # 이미지 읽기
        person_bytes = await person_image.read()
        garment_bytes = await garment_image.read()
        background_bytes = await background_image.read()
        
        if not person_bytes or not garment_bytes or not background_bytes:
            return JSONResponse(
                {
                    "success": False,
                    "prompt": "",
                    "result_image": "",
                    "message": "사람 이미지, 의상 이미지, 배경 이미지를 모두 업로드해주세요.",
                    "llm": None
                },
                status_code=400,
            )
        
        # PIL Image로 변환
        person_img = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        garment_img = Image.open(io.BytesIO(garment_bytes)).convert("RGB")
        background_img = Image.open(io.BytesIO(background_bytes)).convert("RGB")
        
        # V2 통합 트라이온 서비스 호출
        result = await generate_unified_tryon_v2(person_img, garment_img, background_img)
        
        if result["success"]:
            return JSONResponse(result)
        else:
            status_code = 500 if "error" in result else 400
            return JSONResponse(result, status_code=status_code)
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"통합 트라이온 V2 엔드포인트 오류: {e}")
        print(error_detail)
        
        return JSONResponse(
            {
                "success": False,
                "prompt": "",
                "result_image": "",
                "message": f"통합 트라이온 V2 처리 중 오류가 발생했습니다: {str(e)}",
                "llm": None
            },
            status_code=500,
        )


@router.post("/fit/v5/compose", tags=["통합 트라이온 V5"], response_model=UnifiedTryonResponse)
async def compose_v5(
    person_image: UploadFile = File(..., description="인물 이미지 파일"),
    garment_image: UploadFile = File(..., description="의상 이미지 파일"),
    background_image: UploadFile = File(..., description="배경 이미지 파일"),
):
    """
    통합 트라이온 파이프라인 V5: Gemini 3 Flash 직접 처리 (X.AI 제거)
    
    V5는 X.AI 이미지 분석 단계를 제거하고 Gemini 3 Flash가 직접 처리합니다:
    1. Gemini 3 Flash가 인물/의상/배경 이미지를 직접 분석
    2. 의상 교체 + 배경 합성을 한 번에 수행
    
    V4와의 차이점:
    - X.AI 이미지 분석 및 프롬프트 생성 단계 제거
    - 정적 통합 프롬프트 사용으로 응답 속도 향상
    
    Returns:
        UnifiedTryonResponse: 생성된 프롬프트와 합성 이미지 (base64)
    """
    try:
        # 이미지 읽기
        person_bytes = await person_image.read()
        garment_bytes = await garment_image.read()
        background_bytes = await background_image.read()
        
        if not person_bytes or not garment_bytes or not background_bytes:
            return JSONResponse(
                {
                    "success": False,
                    "prompt": "",
                    "result_image": "",
                    "message": "인물 이미지, 의상 이미지, 배경 이미지를 모두 업로드해주세요.",
                    "llm": None
                },
                status_code=400,
            )
        
        # PIL Image로 변환
        person_img = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        garment_img = Image.open(io.BytesIO(garment_bytes)).convert("RGB")
        background_img = Image.open(io.BytesIO(background_bytes)).convert("RGB")
        
        # V5 통합 트라이온 서비스 호출
        result = await generate_unified_tryon_v5(person_img, garment_img, background_img)
        
        if result["success"]:
            return JSONResponse(result)
        else:
            status_code = 500 if "error" in result else 400
            return JSONResponse(result, status_code=status_code)
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"통합 트라이온 V5 엔드포인트 오류: {e}")
        print(error_detail)
        
        return JSONResponse(
            {
                "success": False,
                "prompt": "",
                "result_image": "",
                "message": f"통합 트라이온 V5 처리 중 오류가 발생했습니다: {str(e)}",
                "llm": None
            },
            status_code=500,
        )


@router.post("/fit/v5v5/compose", tags=["통합 트라이온 V5V5일반"], response_model=UnifiedTryonResponse)
async def compose_v5v5(
    person_image: UploadFile = File(..., description="인물 이미지 파일"),
    garment_image: UploadFile = File(..., description="의상 이미지 파일"),
    background_image: UploadFile = File(..., description="배경 이미지 파일"),
):
    """
    V5V5일반 통합 트라이온 파이프라인: V5 파이프라인을 두 번 병렬 실행하고 v5_result 반환
    
    - V5-1: X.AI 없이 Gemini 3 Flash 직접 처리
    - V5-2: X.AI 없이 Gemini 3 Flash 직접 처리
    
    같은 V5 파이프라인을 두 번 병렬로 실행하고 v5_result만 반환합니다.
    /fit/ 경로로 배포 가능하도록 설계되었습니다.
    
    Returns:
        UnifiedTryonResponse: v5_result를 직접 반환 (단일 결과)
    """
    try:
        # 이미지 읽기
        person_bytes = await person_image.read()
        garment_bytes = await garment_image.read()
        background_bytes = await background_image.read()
        
        if not person_bytes or not garment_bytes or not background_bytes:
            return JSONResponse(
                {
                    "success": False,
                    "prompt": "",
                    "result_image": "",
                    "message": "인물 이미지, 의상 이미지, 배경 이미지를 모두 업로드해주세요.",
                    "llm": None
                },
                status_code=400,
            )
        
        # PIL Image로 변환
        person_img = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        garment_img = Image.open(io.BytesIO(garment_bytes)).convert("RGB")
        background_img = Image.open(io.BytesIO(background_bytes)).convert("RGB")
        
        # V4V5일반 비교 실행 (로깅 활성화 - 문서용 로그에 기록)
        # enable_logging=True: S3 업로드 및 DB 로그 저장 활성화
        print("[DEBUG 라우터] /fit/v5v5/compose - enable_logging=True로 호출 (문서용 로그 활성화)")
        result = await run_v4v5_compare(person_img, garment_img, background_img, enable_logging=True)
        
        # v5_result만 반환
        if result.get("success") and result.get("v5_result"):
            v5_result = result["v5_result"]
            
            # 날짜별 합성 카운트 증가 (v5_result가 성공한 경우에만)
            if v5_result.get("success", False):
                from services.synthesis_stats_service import increment_synthesis_count
                print("[일반 피팅] 합성 성공 - 카운팅 시작")
                try:
                    count_success = increment_synthesis_count()
                    if count_success:
                        print("[일반 피팅] ✅ 합성 카운트 증가 성공")
                    else:
                        print("[일반 피팅] ⚠️ 합성 카운트 증가 실패 (DB 연결 또는 쿼리 오류)")
                except Exception as e:
                    print(f"[일반 피팅] ❌ 합성 카운트 증가 예외 발생: {e}")
            
            return JSONResponse({
                "success": v5_result.get("success", False),
                "prompt": v5_result.get("prompt", ""),
                "result_image": v5_result.get("result_image", ""),
                "message": v5_result.get("message") or result.get("message", "V5V5일반 파이프라인이 완료되었습니다."),
                "llm": v5_result.get("llm")
            })
        else:
            return JSONResponse(
                {
                    "success": False,
                    "prompt": "",
                    "result_image": "",
                    "message": result.get("message", "V5V5일반 파이프라인 처리 중 오류가 발생했습니다."),
                    "llm": None
                },
                status_code=500,
            )
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"V5V5일반 통합 트라이온 엔드포인트 오류: {e}")
        print(error_detail)
        
        return JSONResponse(
            {
                "success": False,
                "prompt": "",
                "result_image": "",
                "message": f"V5V5일반 통합 트라이온 처리 중 오류가 발생했습니다: {str(e)}",
                "llm": None
            },
            status_code=500,
        )


@router.post("/tryon/compare", tags=["V4V5일반"], response_model=V4V5CompareResponse)
async def compare_v4v5(
    request: Request,
    person_image: UploadFile = File(..., description="인물 이미지 파일"),
    garment_image: UploadFile = File(..., description="의상 이미지 파일"),
    background_image: UploadFile = File(..., description="배경 이미지 파일"),
    profile_front: Optional[str] = Form(None, description="프론트엔드 프로파일링 데이터 (JSON 문자열)"),
    dress_id: Optional[int] = Form(None, description="드레스 ID (피팅 로그 기록용)"),
):
    """
    V4V5일반 비교 엔드포인트: V5 파이프라인을 두 번 병렬 실행하고 두 결과를 반환
    
    - V5-1: X.AI 없이 Gemini 3 Flash 직접 처리
    - V5-2: X.AI 없이 Gemini 3 Flash 직접 처리
    
    같은 V5 파이프라인을 두 번 병렬로 실행하여 결과를 비교할 수 있습니다.
    
    Returns:
        V4V5CompareResponse: V5-1과 V5-2 결과를 모두 포함한 비교 응답
    """
    # trace_id 추출 또는 생성
    trace_id = request.headers.get("X-Trace-Id")
    if not trace_id:
        trace_id = str(uuid.uuid4())
    
    # 프로파일링 시작 시간
    server_start_time = time.time()
    
    # front_profile_json 추출
    front_profile_json = None
    if profile_front:
        try:
            front_profile_json = json.loads(profile_front)
        except Exception as e:
            print(f"[프로파일링] front_profile_json 파싱 실패: {e}")
    
    try:
        # 이미지 읽기
        person_bytes = await person_image.read()
        garment_bytes = await garment_image.read()
        background_bytes = await background_image.read()
        
        if not person_bytes or not garment_bytes or not background_bytes:
            server_total_ms = round((time.time() - server_start_time) * 1000, 2)
            # 프로파일링 저장 (실패)
            save_tryon_profile(
                trace_id=trace_id,
                endpoint="/tryon/compare",
                front_profile_json=front_profile_json,
                server_total_ms=server_total_ms,
                resize_ms=None,
                gemini_call_ms=None,
                status="fail",
                error_stage="input_validation"
            )
            return JSONResponse(
                {
                    "success": False,
                    "v4_result": {"success": False, "prompt": "", "result_image": "", "message": "입력 오류", "llm": None},
                    "v5_result": {"success": False, "prompt": "", "result_image": "", "message": "입력 오류", "llm": None},
                    "total_time": 0.0,
                    "message": "인물 이미지, 의상 이미지, 배경 이미지를 모두 업로드해주세요."
                },
                status_code=400,
                headers={"X-Trace-Id": trace_id}
            )
        
        # PIL Image로 변환
        person_img = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        garment_img = Image.open(io.BytesIO(garment_bytes)).convert("RGB")
        background_img = Image.open(io.BytesIO(background_bytes)).convert("RGB")
        
        # V4V5일반 비교 실행
        result = await run_v4v5_compare(person_img, garment_img, background_img)
        
        # 서버 전체 처리 시간 측정
        server_total_ms = round((time.time() - server_start_time) * 1000, 2)
        
        # gemini_call_ms, resize_ms 추출
        gemini_call_ms = result.get("gemini_call_ms")
        resize_ms = result.get("resize_ms")
        
        # 성공 여부 판단
        status = "success" if result.get("success") else "fail"
        error_stage = None
        if not result.get("success"):
            error_stage = "pipeline_execution"
        
        # 날짜별 합성 카운트 증가 (v5_result가 성공한 경우에만)
        if result.get("success") and result.get("v5_result"):
            v5_result = result["v5_result"]
            if v5_result.get("success", False):
                from services.synthesis_stats_service import increment_synthesis_count
                print("[일반 피팅] 합성 성공 - 카운팅 시작")
                try:
                    count_success = increment_synthesis_count()
                    if count_success:
                        print("[일반 피팅] ✅ 합성 카운트 증가 성공")
                    else:
                        print("[일반 피팅] ⚠️ 합성 카운트 증가 실패 (DB 연결 또는 쿼리 오류)")
                except Exception as e:
                    print(f"[일반 피팅] ❌ 합성 카운트 증가 예외 발생: {e}")
                
                # 일반 피팅 로그 기록 (result_logs에 저장)
                from services.log_service import save_general_fitting_log
                print("[일반 피팅] 일반 피팅 로그 기록 시작")
                try:
                    # run_time을 초 단위로 변환 (server_total_ms는 밀리초)
                    run_time_seconds = server_total_ms / 1000.0
                    log_success = save_general_fitting_log(run_time=run_time_seconds)
                    if log_success:
                        print("[일반 피팅] ✅ 일반 피팅 로그 기록 성공")
                    else:
                        print("[일반 피팅] ⚠️ 일반 피팅 로그 기록 실패")
                except Exception as e:
                    print(f"[일반 피팅] ❌ 일반 피팅 로그 기록 예외 발생: {e}")
                
                # 드레스 피팅 로그 기록 (dress_id가 있는 경우)
                if dress_id:
                    from services.dress_fitting_log_service import log_dress_fitting
                    print(f"[일반 피팅] 드레스 피팅 로그 기록 시작 - dress_id: {dress_id}")
                    try:
                        log_success = log_dress_fitting(dress_id)
                        if log_success:
                            print(f"[일반 피팅] ✅ 드레스 피팅 로그 기록 성공 - dress_id: {dress_id}")
                        else:
                            print(f"[일반 피팅] ⚠️ 드레스 피팅 로그 기록 실패 - dress_id: {dress_id}")
                    except Exception as e:
                        print(f"[일반 피팅] ❌ 드레스 피팅 로그 기록 예외 발생: {e}")
        
        # 프로파일링 저장
        save_tryon_profile(
            trace_id=trace_id,
            endpoint="/tryon/compare",
            front_profile_json=front_profile_json,
            server_total_ms=server_total_ms,
            resize_ms=resize_ms,
            gemini_call_ms=gemini_call_ms,
            status=status,
            error_stage=error_stage
        )
        
        # 비교 작업은 완료되었으므로 성공 여부와 관계없이 200 반환
        # (실제 성공 여부는 result["success"]와 각 result의 success 필드로 확인)
        return JSONResponse(result, headers={"X-Trace-Id": trace_id})
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"V4V5일반 비교 엔드포인트 오류: {e}")
        print(error_detail)
        
        # 서버 전체 처리 시간 측정
        server_total_ms = round((time.time() - server_start_time) * 1000, 2)
        
        # 프로파일링 저장 (에러)
        save_tryon_profile(
            trace_id=trace_id,
            endpoint="/tryon/compare",
            front_profile_json=front_profile_json,
            server_total_ms=server_total_ms,
            resize_ms=None,
            gemini_call_ms=None,
            status="fail",
            error_stage="exception"
        )
        
        return JSONResponse(
            {
                "success": False,
                "v4_result": {"success": False, "prompt": "", "result_image": "", "message": str(e), "llm": None},
                "v5_result": {"success": False, "prompt": "", "result_image": "", "message": str(e), "llm": None},
                "total_time": 0.0,
                "message": f"V4V5일반 비교 처리 중 오류가 발생했습니다: {str(e)}"
            },
            status_code=500,
            headers={"X-Trace-Id": trace_id}
        )

