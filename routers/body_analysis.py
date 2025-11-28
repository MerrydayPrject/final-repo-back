"""체형 분석 라우터"""
import time
import io
import traceback
from fastapi import APIRouter, File, UploadFile, Form, Query
from fastapi.responses import JSONResponse
from PIL import Image

from core.model_loader import get_body_analysis_service, get_image_classifier_service
from services.body_service import determine_body_features, analyze_body_with_gemini
from services.database import get_db_connection
from services.body_analysis_database import save_body_analysis_result, get_body_logs, get_body_logs_count
import numpy as np
from typing import Optional

router = APIRouter()


@router.post("/api/validate-person", tags=["사람 감지"])
async def validate_person(
    file: UploadFile = File(..., description="이미지 파일")
):
    """
    이미지에서 사람이 감지되는지 확인 (체형분석용 - 전신 랜드마크 필수)
    
    체형분석에서는 전신 랜드마크가 필수이므로, 전신 랜드마크가 없으면 차단합니다.
    """
    try:
        # 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        classification_result = None
        is_animal_detected = False
        
        # 1. 동물 감지 검증 (우선 검증)
        try:
            image_classifier_service = get_image_classifier_service()
            if image_classifier_service and image_classifier_service.is_initialized:
                classification_result = image_classifier_service.classify_image(image)
                
                if classification_result:
                    # 동물 키워드 확인 (즉시 차단)
                    animal_keywords = [
                        "animal", "dog", "cat", "bear", "monkey", "ape", "gorilla",
                        "orangutan", "chimpanzee", "elephant", "lion", "tiger",
                        "bird", "fish", "horse", "cow", "pig", "sheep", "goat",
                        "rabbit", "mouse", "rat", "hamster", "squirrel", "deer",
                        "wolf", "fox", "panda", "koala", "kangaroo", "zebra",
                        "giraffe", "camel", "donkey", "mule", "llama", "alpaca"
                    ]
                    
                    for category in classification_result:
                        category_name_lower = category["category_name"].lower()
                        if any(keyword in category_name_lower for keyword in animal_keywords):
                            is_animal_detected = True
                            print(f"❌ 동물 감지: {category['category_name']} (신뢰도: {category['score']:.2f}) - 즉시 차단")
                            break
                    
                    # 동물이 감지되면 즉시 차단
                    if is_animal_detected:
                        print(f"❌ 동물 감지로 인해 즉시 차단")
                        return JSONResponse({
                            "success": True,
                            "is_person": False,
                            "is_face_only": False,
                            "face_detected": False,
                            "landmarks_count": 0,
                            "detection_type": None,
                            "classification_result": classification_result[:3],
                            "message": "인물사진을 업로드해주세요."
                        })
        except Exception as e:
            print(f"동물 감지 검증 오류 (무시): {e}")
        
        # 2. 전신 랜드마크 확인 (가장 중요 - 체형분석에서는 전신 랜드마크가 필수)
        body_analysis_service = get_body_analysis_service()
        if body_analysis_service and body_analysis_service.is_initialized:
            landmarks = body_analysis_service.extract_landmarks(image)
            if landmarks is None or len(landmarks) == 0:
                # 전신 랜드마크가 없으면 무조건 차단
                print(f"❌ 전신 랜드마크 없음 - 차단")
                return JSONResponse({
                    "success": True,
                    "is_person": False,
                    "is_face_only": True,
                    "face_detected": False,
                    "landmarks_count": 0,
                    "detection_type": None,
                    "classification_result": classification_result[:3] if classification_result else None,
                    "message": "전신 사진을 넣어주세요."
                })
            
            # 전신 랜드마크가 제대로 감지되었는지 확인
            # MediaPipe Pose는 33개의 랜드마크를 사용
            # 전신 사진인지 확인하려면 하체 랜드마크가 감지되었는지 확인
            # 랜드마크 ID: 23(왼쪽 엉덩이), 24(오른쪽 엉덩이), 25(왼쪽 무릎), 26(오른쪽 무릎), 27(왼쪽 발목), 28(오른쪽 발목)
            # 또는 어깨와 발목의 y 좌표 차이로 전신 여부 판단
            
            # 하체 랜드마크 ID (엉덩이, 무릎, 발목, 발뒤꿈치, 발가락)
            # 23: 왼쪽 엉덩이, 24: 오른쪽 엉덩이
            # 25: 왼쪽 무릎, 26: 오른쪽 무릎
            # 27: 왼쪽 발목, 28: 오른쪽 발목
            # 29: 왼쪽 발뒤꿈치, 30: 오른쪽 발뒤꿈치
            # 31: 왼쪽 발가락, 32: 오른쪽 발가락
            lower_body_ids = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            upper_body_ids = [11, 12]  # 어깨
            
            # 하체 랜드마크가 감지되었는지 확인
            lower_body_detected = False
            upper_body_y = None
            lower_body_y = None
            
            for landmark in landmarks:
                landmark_id = landmark.get("id")
                visibility = landmark.get("visibility", 0)
                y = landmark.get("y", 0)
                
                # visibility가 0.3 이상이면 감지된 것으로 판단
                if visibility >= 0.3:
                    if landmark_id in upper_body_ids:
                        # 어깨의 y 좌표 (상체)
                        if upper_body_y is None or y < upper_body_y:
                            upper_body_y = y
                    
                    if landmark_id in lower_body_ids:
                        # 하체 랜드마크 감지
                        lower_body_detected = True
                        # 하체의 y 좌표 (하체)
                        if lower_body_y is None or y > lower_body_y:
                            lower_body_y = y
            
            # 하체 랜드마크가 감지되지 않으면 차단
            if not lower_body_detected:
                print(f"❌ 하체 랜드마크 없음 (얼굴/상체만 감지됨) - 차단")
                return JSONResponse({
                    "success": True,
                    "is_person": False,
                    "is_face_only": True,
                    "face_detected": False,
                    "landmarks_count": len(landmarks) if landmarks else 0,
                    "detection_type": None,
                    "classification_result": classification_result[:3] if classification_result else None,
                    "message": "전신 사진을 넣어주세요."
                })
            
            # 어깨와 하체의 y 좌표 차이로 전신 여부 추가 확인
            # 전신 사진이라면 하체가 어깨보다 아래에 있어야 함
            if upper_body_y is not None and lower_body_y is not None:
                body_height_ratio = lower_body_y - upper_body_y
                # y 좌표 차이가 0.2 미만이면 전신이 아닌 것으로 판단 (너무 작은 차이)
                if body_height_ratio < 0.2:
                    print(f"❌ 전신 비율 부족 (상체-하체 차이: {body_height_ratio:.3f}) - 차단")
                    return JSONResponse({
                        "success": True,
                        "is_person": False,
                        "is_face_only": True,
                        "face_detected": False,
                        "landmarks_count": len(landmarks) if landmarks else 0,
                        "detection_type": None,
                        "classification_result": classification_result[:3] if classification_result else None,
                        "message": "전신 사진을 넣어주세요."
                    })
            
            # 전신 랜드마크가 제대로 감지되었으면 통과
            print(f"✅ 전신 랜드마크 확인됨 (하체 포함) - 통과")
            return JSONResponse({
                "success": True,
                "is_person": True,
                "is_face_only": False,
                "face_detected": False,
                "landmarks_count": len(landmarks) if landmarks else 0,
                "detection_type": "full_body",
                "classification_result": classification_result[:3] if classification_result else None,
                "message": "전신 사진이 확인되었습니다."
            })
        else:
            # 체형 분석 서비스가 없으면 차단
            print(f"❌ 체형 분석 서비스 없음 - 차단")
            return JSONResponse({
                "success": True,
                "is_person": False,
                "is_face_only": True,
                "face_detected": False,
                "landmarks_count": 0,
                "detection_type": None,
                "classification_result": classification_result[:3] if classification_result else None,
                "message": "전신 사진을 넣어주세요."
            })
        
    except Exception as e:
        import traceback
        print(f"사람 감지 오류: {e}")
        print(traceback.format_exc())
        return JSONResponse({
            "success": False,
            "is_person": False,
            "message": "얼굴사진을 넣어주세요."
        }, status_code=500)


@router.post("/api/analyze-body", tags=["체형 분석"])
async def analyze_body(
    file: UploadFile = File(..., description="전신 이미지 파일"),
    height: float = Form(..., description="키 (cm)"),
    weight: float = Form(..., description="몸무게 (kg)")
):
    """
    전신 이미지 체형 분석
    
    MediaPipe Pose Landmarker로 포즈 랜드마크를 추출하고,
    체형 비율을 계산한 후 Gemini API로 상세 분석을 수행합니다.
    """
    start_time = time.time()
    
    try:
        # 체형 분석 서비스 확인
        body_analysis_service = get_body_analysis_service()
        if not body_analysis_service or not body_analysis_service.is_initialized:
            return JSONResponse({
                "success": False,
                "error": "Body analysis service not initialized",
                "message": "체형 분석 서비스가 초기화되지 않았습니다. 모델 파일을 확인해주세요."
            }, status_code=500)
        
        # 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 0. 동물 감지 검증 (validatePerson과 동일한 방식)
        is_animal_detected = False
        image_classifier_service = get_image_classifier_service()
        if image_classifier_service and image_classifier_service.is_initialized:
            try:
                classification_result = image_classifier_service.classify_image(image)
                
                if classification_result:
                    # 동물 키워드 확인 (즉시 차단)
                    animal_keywords = [
                        "animal", "dog", "cat", "bear", "monkey", "ape", "gorilla",
                        "orangutan", "chimpanzee", "elephant", "lion", "tiger",
                        "bird", "fish", "horse", "cow", "pig", "sheep", "goat",
                        "rabbit", "mouse", "rat", "hamster", "squirrel", "deer",
                        "wolf", "fox", "panda", "koala", "kangaroo", "zebra",
                        "giraffe", "camel", "donkey", "mule", "llama", "alpaca"
                    ]
                    
                    for category in classification_result:
                        category_name_lower = category["category_name"].lower()
                        if any(keyword in category_name_lower for keyword in animal_keywords):
                            is_animal_detected = True
                            print(f"❌ 동물 감지: {category['category_name']} (신뢰도: {category['score']:.2f}) - 즉시 차단")
                            break
                    
                    # 동물이 감지되면 즉시 차단
                    if is_animal_detected:
                        return JSONResponse({
                            "success": False,
                            "error": "Animal detected",
                            "is_animal": True,
                            "message": "인물사진을 업로드해주세요."
                        }, status_code=400)
            except Exception as e:
                print(f"동물 감지 검증 오류 (무시): {e}")
        
        # 1. 포즈 랜드마크 추출 (전신 감지)
        landmarks = body_analysis_service.extract_landmarks(image)
        
        if landmarks is None:
            return JSONResponse({
                "success": False,
                "error": "No pose detected",
                "message": "전신 사진을 넣어주세요."
            }, status_code=400)
        
        # 2. 체형 측정값 계산
        measurements = body_analysis_service.calculate_measurements(landmarks)
        
        # 3. 체형 타입 분류 (랜드마크 기반)
        body_type = body_analysis_service.classify_body_type(measurements)
        
        # 4. BMI 계산 및 체형 특징 판단
        bmi = None
        body_features = []
        if height and weight:
            # BMI 계산: kg / (m^2)
            height_m = height / 100.0
            bmi = weight / (height_m ** 2)
            body_features = determine_body_features(body_type, bmi, height, measurements)
        
        # 5. Gemini API로 상세 분석
        gemini_analysis = None
        gemini_analysis_text = None
        try:
            gemini_analysis = await analyze_body_with_gemini(
                image, measurements, body_type, bmi, height, body_features
            )
            if gemini_analysis and gemini_analysis.get('detailed_analysis'):
                gemini_analysis_text = gemini_analysis['detailed_analysis']
        except Exception as e:
            print(f"Gemini 분석 실패: {e}")
        
        # 6. 처리 시간 계산
        run_time = time.time() - start_time
        
        # 7. 분석 결과를 DB에 저장
        try:
            # 체형 특징을 문자열로 변환 (쉼표로 구분)
            characteristic_str = ', '.join(body_features) if body_features else None
            
            # 프롬프트는 간단히 저장 (필요시 상세 프롬프트 저장 가능)
            prompt_text = '체형 분석 (MediaPipe + Gemini)'
            
            # 키/몸무게가 없으면 0으로 저장 (NOT NULL 제약 조건)
            result_id = save_body_analysis_result(
                model='body_analysis',
                run_time=run_time,
                height=height if height else 0.0,
                weight=weight if weight else 0.0,
                prompt=prompt_text,
                bmi=bmi if bmi else 0.0,
                characteristic=characteristic_str,
                analysis_results=gemini_analysis_text
            )
            if result_id:
                print(f"✅ 체형 분석 결과 저장 완료 (ID: {result_id}, 처리시간: {run_time:.2f}초)")
            else:
                print("⚠️  체형 분석 결과 저장 실패")
        except Exception as e:
            print(f"⚠️  체형 분석 결과 저장 중 오류: {e}")
        
        return JSONResponse({
            "success": True,
            "body_analysis": {
                "body_type": body_type.get('type', 'unknown'),
                "body_features": body_features,
                "measurements": measurements
            },
            "gemini_analysis": gemini_analysis,
            "run_time": run_time,
            "message": "체형 분석이 완료되었습니다."
        })
        
    except Exception as e:
        print(f"체형 분석 오류: {traceback.format_exc()}")
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"체형 분석 중 오류 발생: {str(e)}"
        }, status_code=500)


@router.get("/api/admin/body-logs", tags=["관리자"])
async def get_body_analysis_logs(
    page: int = Query(1, ge=1, description="페이지 번호"),
    limit: int = Query(20, ge=1, le=100, description="페이지당 항목 수")
):
    """
    체형 분석 로그 목록 조회
    
    body_logs 테이블에서 체형 분석 로그 목록을 조회합니다.
    """
    try:
        # 전체 개수 조회
        total_count = get_body_logs_count()
        
        # 총 페이지 수 계산
        total_pages = (total_count + limit - 1) // limit if total_count > 0 else 0
        
        # 오프셋 계산
        offset = (page - 1) * limit
        
        # 로그 목록 조회
        logs = get_body_logs(limit=limit, offset=offset)
        
        # 데이터 형식 변환
        formatted_logs = []
        for log in logs:
            formatted_logs.append({
                'id': log.get('idx'),
                'model': log.get('model', 'body_analysis'),
                'processing_time': f"{log.get('run_time', 0):.2f}초",
                'height': log.get('height'),
                'weight': log.get('weight'),
                'bmi': log.get('bmi'),
                'characteristic': log.get('characteristic'),
                'created_at': log.get('created_at')
            })
        
        return JSONResponse({
            "success": True,
            "data": formatted_logs,
            "pagination": {
                "page": page,
                "limit": limit,
                "total": total_count,
                "total_pages": total_pages
            }
        })
    except Exception as e:
        print(f"체형 분석 로그 조회 오류: {traceback.format_exc()}")
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"로그 조회 중 오류 발생: {str(e)}"
        }, status_code=500)


@router.get("/api/admin/body-logs/{log_id}", tags=["관리자"])
async def get_body_analysis_log_detail(log_id: int):
    """
    체형 분석 로그 상세 정보 조회
    
    특정 체형 분석 로그의 상세 정보를 조회합니다.
    """
    try:
        connection = get_db_connection()
        if not connection:
            return JSONResponse({
                "success": False,
                "error": "Database connection failed",
                "message": "데이터베이스 연결에 실패했습니다."
            }, status_code=500)
        
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
                cursor.execute(sql, (log_id,))
                log = cursor.fetchone()
                
                if not log:
                    return JSONResponse({
                        "success": False,
                        "error": "Log not found",
                        "message": f"로그 ID {log_id}를 찾을 수 없습니다."
                    }, status_code=404)
                
                # 안전하게 필드 접근
                created_at = log.get('created_at')
                if created_at and hasattr(created_at, 'isoformat'):
                    created_at = created_at.isoformat()
                elif created_at:
                    created_at = str(created_at)
                
                return JSONResponse({
                    "success": True,
                    "data": {
                        "id": log.get('idx'),
                        "model": log.get('model', 'body_analysis'),
                        "processing_time": f"{log.get('run_time', 0):.2f}초",
                        "height": log.get('height'),
                        "weight": log.get('weight'),
                        "bmi": log.get('bmi'),
                        "characteristic": log.get('characteristic'),
                        "analysis_results": log.get('analysis_results'),
                        "prompt": log.get('prompt'),
                        "created_at": created_at
                    }
                })
        finally:
            connection.close()
            
    except Exception as e:
        print(f"체형 분석 로그 상세 조회 오류: {traceback.format_exc()}")
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"로그 상세 조회 중 오류 발생: {str(e)}"
        }, status_code=500)


@router.post("/api/pose-landmarks", tags=["포즈 랜드마크"])
async def get_pose_landmarks(
    file: UploadFile = File(..., description="이미지 파일")
):
    """
    이미지에서 포즈 랜드마크 추출 (시각화용)
    
    이미지를 업로드하면 포즈 랜드마크 좌표를 반환합니다.
    기존 랜드마크 추출 방식과 동일하게 body_analysis_service.extract_landmarks()를 사용합니다.
    """
    try:
        # 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 포즈 랜드마크 추출 (기존과 동일한 방식)
        body_analysis_service = get_body_analysis_service()
        if not body_analysis_service or not body_analysis_service.is_initialized:
            return JSONResponse({
                "success": False,
                "error": "Body analysis service not initialized",
                "message": "체형 분석 서비스가 초기화되지 않았습니다."
            }, status_code=500)
        
        landmarks = body_analysis_service.extract_landmarks(image)
        
        # 이미지 크기 정보도 함께 반환
        return JSONResponse({
            "success": True,
            "landmarks": landmarks,
            "image_width": image.width,
            "image_height": image.height,
            "landmarks_count": len(landmarks) if landmarks else 0,
            "message": "포즈 랜드마크 추출 완료"
        })
        
    except Exception as e:
        import traceback
        print(f"포즈 랜드마크 추출 오류: {e}")
        print(traceback.format_exc())
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"포즈 랜드마크 추출 중 오류 발생: {str(e)}"
        }, status_code=500)

