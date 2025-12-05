"""V4V5일반 비교 서비스 - Adapter + Orchestrator 패턴"""
import asyncio
import io
import time
import base64
from typing import Dict
from PIL import Image

from services.tryon_service import generate_unified_tryon_v5
from services.log_service import save_test_log
from core.s3_client import upload_log_to_s3


# ============================================================
# Adapter 클래스: 기존 서비스 함수를 동일한 인터페이스로 래핑
# ============================================================

class V5Adapter:
    """V5 파이프라인 어댑터"""
    
    @staticmethod
    async def run(
        person_img: Image.Image,
        garment_img: Image.Image,
        background_img: Image.Image
    ) -> Dict:
        """V5 파이프라인 실행"""
        return await generate_unified_tryon_v5(person_img, garment_img, background_img)


# ============================================================
# Orchestrator: V4/V5 파이프라인 병렬 실행
# ============================================================

class V4V5Orchestrator:
    """V4V5일반 파이프라인 오케스트레이터"""
    
    def __init__(self):
        self.v5_adapter_1 = V5Adapter()
        self.v5_adapter_2 = V5Adapter()
    
    async def run_parallel(
        self,
        person_img: Image.Image,
        garment_img: Image.Image,
        background_img: Image.Image
    ) -> Dict:
        """V4/V5 파이프라인 병렬 실행"""
        start_time = time.time()
        model_id = "v4v5-compare"
        
        print("\n" + "="*80)
        print("[V4V5일반] 병렬 파이프라인 시작")
        print("="*80)
        
        # 입력 이미지 S3 업로드
        person_s3_url = ""
        garment_s3_url = ""
        background_s3_url = ""
        
        try:
            person_buffered = io.BytesIO()
            person_img.save(person_buffered, format="PNG")
            person_s3_url = upload_log_to_s3(person_buffered.getvalue(), model_id, "person") or ""
            
            garment_buffered = io.BytesIO()
            garment_img.save(garment_buffered, format="PNG")
            garment_s3_url = upload_log_to_s3(garment_buffered.getvalue(), model_id, "garment") or ""
            
            background_buffered = io.BytesIO()
            background_img.save(background_buffered, format="PNG")
            background_s3_url = upload_log_to_s3(background_buffered.getvalue(), model_id, "background") or ""
        except Exception as e:
            print(f"[V4V5일반] 입력 이미지 S3 업로드 실패: {e}")
        
        # asyncio.gather로 병렬 실행
        v4_result, v5_result = await asyncio.gather(
            self.v5_adapter_1.run(person_img, garment_img, background_img),
            self.v5_adapter_2.run(person_img, garment_img, background_img),
            return_exceptions=True
        )
        
        total_time = time.time() - start_time
        
        # 예외 처리
        if isinstance(v4_result, Exception):
            v4_result = {
                "success": False,
                "prompt": "",
                "result_image": "",
                "message": f"V5-1 파이프라인 오류: {str(v4_result)}",
                "llm": None
            }
        
        if isinstance(v5_result, Exception):
            v5_result = {
                "success": False,
                "prompt": "",
                "result_image": "",
                "message": f"V5-2 파이프라인 오류: {str(v5_result)}",
                "llm": None
            }
        
        # V4 결과 로깅
        v4_result_s3_url = ""
        if v4_result.get("success") and v4_result.get("result_image"):
            try:
                # base64 이미지 디코딩
                result_image_data = v4_result["result_image"]
                if result_image_data.startswith("data:image"):
                    # data:image/png;base64, 제거
                    base64_data = result_image_data.split(",")[1]
                    image_bytes = base64.b64decode(base64_data)
                    
                    # S3 업로드
                    v4_result_s3_url = upload_log_to_s3(image_bytes, f"{model_id}-v5-1", "result") or ""
                    
                    # 로그 저장
                    save_test_log(
                        person_url=person_s3_url or "",
                        dress_url=garment_s3_url or None,
                        result_url=v4_result_s3_url or "",
                        model=f"{model_id}-v5-1",
                        prompt=v4_result.get("prompt", ""),
                        success=True,
                        run_time=total_time
                    )
            except Exception as e:
                print(f"[V4V5일반] V5-1 결과 로깅 실패: {e}")
                try:
                    save_test_log(
                        person_url=person_s3_url or "",
                        dress_url=garment_s3_url or None,
                        result_url="",
                        model=f"{model_id}-v5-1",
                        prompt=v4_result.get("prompt", ""),
                        success=False,
                        run_time=total_time
                    )
                except:
                    pass
        else:
            # V5-1 실패 로깅
            try:
                save_test_log(
                    person_url=person_s3_url or "",
                    dress_url=garment_s3_url or None,
                    result_url="",
                    model=f"{model_id}-v5-1",
                    prompt=v4_result.get("prompt", ""),
                    success=False,
                    run_time=total_time
                )
            except:
                pass
        
        # V5 결과 로깅
        v5_result_s3_url = ""
        if v5_result.get("success") and v5_result.get("result_image"):
            try:
                # base64 이미지 디코딩
                result_image_data = v5_result["result_image"]
                if result_image_data.startswith("data:image"):
                    # data:image/png;base64, 제거
                    base64_data = result_image_data.split(",")[1]
                    image_bytes = base64.b64decode(base64_data)
                    
                    # S3 업로드
                    v5_result_s3_url = upload_log_to_s3(image_bytes, f"{model_id}-v5-2", "result") or ""
                    
                    # 로그 저장
                    save_test_log(
                        person_url=person_s3_url or "",
                        dress_url=garment_s3_url or None,
                        result_url=v5_result_s3_url or "",
                        model=f"{model_id}-v5-2",
                        prompt=v5_result.get("prompt", ""),
                        success=True,
                        run_time=total_time
                    )
            except Exception as e:
                print(f"[V4V5일반] V5-2 결과 로깅 실패: {e}")
                try:
                    save_test_log(
                        person_url=person_s3_url or "",
                        dress_url=garment_s3_url or None,
                        result_url="",
                        model=f"{model_id}-v5-2",
                        prompt=v5_result.get("prompt", ""),
                        success=False,
                        run_time=total_time
                    )
                except:
                    pass
        else:
            # V5-2 실패 로깅
            try:
                save_test_log(
                    person_url=person_s3_url or "",
                    dress_url=garment_s3_url or None,
                    result_url="",
                    model=f"{model_id}-v5-2",
                    prompt=v5_result.get("prompt", ""),
                    success=False,
                    run_time=total_time
                )
            except:
                pass
        
        # 전체 성공 여부 판단
        overall_success = v4_result.get("success", False) or v5_result.get("success", False)
        
        print("\n" + "="*80)
        print("[V4V5일반] 병렬 파이프라인 완료")
        print(f"전체 실행 시간: {total_time:.2f}초")
        print(f"V5-1 성공: {v4_result.get('success', False)}")
        print(f"V5-2 성공: {v5_result.get('success', False)}")
        print("="*80 + "\n")
        
        return {
            "success": overall_success,
            "v4_result": v4_result,
            "v5_result": v5_result,
            "total_time": round(total_time, 2),
            "message": "V4V5일반 비교가 완료되었습니다."
        }


# ============================================================
# 편의 함수
# ============================================================

async def run_v4v5_compare(
    person_img: Image.Image,
    garment_img: Image.Image,
    background_img: Image.Image
) -> Dict:
    """
    V4V5일반 비교 실행 편의 함수
    
    Args:
        person_img: 사람 이미지 (PIL Image)
        garment_img: 의상 이미지 (PIL Image)
        background_img: 배경 이미지 (PIL Image)
    
    Returns:
        dict: V4V5 비교 결과
    """
    orchestrator = V4V5Orchestrator()
    return await orchestrator.run_parallel(person_img, garment_img, background_img)

