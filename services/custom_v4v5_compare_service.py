"""V4V5커스텀 비교 서비스 - Adapter + Orchestrator 패턴"""
import asyncio
import io
import time
import base64
from typing import Dict
from PIL import Image

from services.custom_v5_service import generate_unified_tryon_custom_v5
from services.log_service import save_test_log, save_custom_fitting_log
from core.s3_client import upload_log_to_s3


# ============================================================
# Adapter 클래스: 커스텀 서비스 함수를 동일한 인터페이스로 래핑
# ============================================================

class CustomV5Adapter:
    """CustomV5 파이프라인 어댑터"""
    
    @staticmethod
    async def run(
        person_img: Image.Image,
        garment_img: Image.Image,
        background_img: Image.Image
    ) -> Dict:
        """CustomV5 파이프라인 실행"""
        return await generate_unified_tryon_custom_v5(person_img, garment_img, background_img)


# ============================================================
# Orchestrator: CustomV4/CustomV5 파이프라인 병렬 실행
# ============================================================

class V4V5CustomOrchestrator:
    """V4V5커스텀 파이프라인 오케스트레이터"""
    
    def __init__(self):
        self.v5_adapter_1 = CustomV5Adapter()
        self.v5_adapter_2 = CustomV5Adapter()
    
    async def run_parallel(
        self,
        person_img: Image.Image,
        garment_img: Image.Image,
        background_img: Image.Image,
        enable_logging: bool = True
    ) -> Dict:
        """CustomV4/CustomV5 파이프라인 병렬 실행"""
        # 디버그: enable_logging 값 확인
        print(f"[DEBUG V4V5커스텀] enable_logging={enable_logging}")
        
        start_time = time.time()
        model_id = "v4v5-custom-compare"
        
        if enable_logging:
            print("\n" + "="*80)
            print("[V4V5커스텀] 병렬 파이프라인 시작")
            print("="*80)
        
        # 입력 이미지 S3 업로드 (의상 이미지만 로깅용으로 업로드)
        garment_s3_url = ""
        
        if enable_logging:
            # 의상 이미지 S3 업로드 (커스텀 피팅 로그용)
            try:
                garment_buffered = io.BytesIO()
                garment_img.save(garment_buffered, format="PNG")
                garment_s3_url = upload_log_to_s3(garment_buffered.getvalue(), "custom-fitting", "garment") or ""
                if garment_s3_url:
                    print(f"[V4V5커스텀] 의상 이미지 S3 업로드 완료: {garment_s3_url}")
            except Exception as e:
                print(f"[V4V5커스텀] 의상 이미지 S3 업로드 실패: {e}")
                garment_s3_url = ""
        
        # asyncio.gather로 병렬 실행
        v4_result, v5_result = await asyncio.gather(
            self.v5_adapter_1.run(person_img, garment_img, background_img),
            self.v5_adapter_2.run(person_img, garment_img, background_img),
            return_exceptions=True
        )
        
        total_time = time.time() - start_time
        
        # cutout_ms, gemini_call_ms, resize_ms 수집 (v5_result에서 가져오기, 없으면 v4_result에서)
        cutout_ms = None
        gemini_call_ms = None
        resize_ms = None
        if isinstance(v5_result, dict) and not isinstance(v5_result, Exception):
            cutout_ms = v5_result.get("cutout_ms")
            gemini_call_ms = v5_result.get("gemini_call_ms")
            resize_ms = v5_result.get("resize_ms")
        elif isinstance(v4_result, dict) and not isinstance(v4_result, Exception):
            cutout_ms = v4_result.get("cutout_ms")
            gemini_call_ms = v4_result.get("gemini_call_ms")
            resize_ms = v4_result.get("resize_ms")
        
        # 커스텀 피팅 로그 저장 (created_at, run_time, dress_url만 저장)
        if enable_logging and garment_s3_url:
            try:
                save_custom_fitting_log(
                    dress_url=garment_s3_url,
                    run_time=total_time
                )
                print(f"[V4V5커스텀] 커스텀 피팅 로그 저장 완료: dress_url={garment_s3_url}, run_time={total_time:.2f}초")
            except Exception as e:
                print(f"[V4V5커스텀] 커스텀 피팅 로그 저장 실패: {e}")
        
        # 예외 처리
        if isinstance(v4_result, Exception):
            v4_result = {
                "success": False,
                "prompt": "",
                "result_image": "",
                "message": f"CustomV5-1 파이프라인 오류: {str(v4_result)}",
                "llm": None
            }
        
        if isinstance(v5_result, Exception):
            v5_result = {
                "success": False,
                "prompt": "",
                "result_image": "",
                "message": f"CustomV5-2 파이프라인 오류: {str(v5_result)}",
                "llm": None
            }
        
        # V4 결과 로깅 (로깅 활성화 시에만)
        v4_result_s3_url = ""
        if enable_logging:
            if v4_result.get("success") and v4_result.get("result_image"):
                try:
                    # base64 이미지 디코딩
                    result_image_data = v4_result["result_image"]
                    if result_image_data.startswith("data:image"):
                        # data:image/png;base64, 제거
                        base64_data = result_image_data.split(",")[1]
                        image_bytes = base64.b64decode(base64_data)
                        
                        # S3 업로드 (비활성화)
                        # v4_result_s3_url = upload_log_to_s3(image_bytes, f"{model_id}-v5-1", "result") or ""
                        v4_result_s3_url = ""
                        
                        # 로그 저장 (비활성화)
                        # save_test_log(
                        #     person_url=person_s3_url or "",
                        #     dress_url=garment_s3_url or None,
                        #     result_url=v4_result_s3_url or "",
                        #     model=f"{model_id}-v5-1",
                        #     prompt=v4_result.get("prompt", ""),
                        #     success=True,
                        #     run_time=total_time
                        # )
                except Exception as e:
                    print(f"[V4V5커스텀] CustomV5-1 결과 로깅 실패: {e}")
                    # try:
                    #     save_test_log(
                    #         person_url=person_s3_url or "",
                    #         dress_url=garment_s3_url or None,
                    #         result_url="",
                    #         model=f"{model_id}-v5-1",
                    #         prompt=v4_result.get("prompt", ""),
                    #         success=False,
                    #         run_time=total_time
                    #     )
                    # except:
                    #     pass
            else:
                # CustomV5-1 실패 로깅 (비활성화)
                # try:
                #     save_test_log(
                #         person_url=person_s3_url or "",
                #         dress_url=garment_s3_url or None,
                #         result_url="",
                #         model=f"{model_id}-v5-1",
                #         prompt=v4_result.get("prompt", ""),
                #         success=False,
                #         run_time=total_time
                #     )
                # except:
                #     pass
                pass
        
        # V5 결과 로깅 (로깅 활성화 시에만)
        v5_result_s3_url = ""
        if enable_logging:
            if v5_result.get("success") and v5_result.get("result_image"):
                try:
                    # base64 이미지 디코딩
                    result_image_data = v5_result["result_image"]
                    if result_image_data.startswith("data:image"):
                        # data:image/png;base64, 제거
                        base64_data = result_image_data.split(",")[1]
                        image_bytes = base64.b64decode(base64_data)
                        
                        # S3 업로드 (비활성화)
                        # v5_result_s3_url = upload_log_to_s3(image_bytes, f"{model_id}-v5-2", "result") or ""
                        v5_result_s3_url = ""
                        
                        # 로그 저장 (비활성화)
                        # save_test_log(
                        #     person_url=person_s3_url or "",
                        #     dress_url=garment_s3_url or None,
                        #     result_url=v5_result_s3_url or "",
                        #     model=f"{model_id}-v5-2",
                        #     prompt=v5_result.get("prompt", ""),
                        #     success=True,
                        #     run_time=total_time
                        # )
                except Exception as e:
                    print(f"[V4V5커스텀] CustomV5-2 결과 로깅 실패: {e}")
                    # try:
                    #     save_test_log(
                    #         person_url=person_s3_url or "",
                    #         dress_url=garment_s3_url or None,
                    #         result_url="",
                    #         model=f"{model_id}-v5-2",
                    #         prompt=v5_result.get("prompt", ""),
                    #         success=False,
                    #         run_time=total_time
                    #     )
                    # except:
                    #     pass
            else:
                # CustomV5-2 실패 로깅 (비활성화)
                # try:
                #     save_test_log(
                #         person_url=person_s3_url or "",
                #         dress_url=garment_s3_url or None,
                #         result_url="",
                #         model=f"{model_id}-v5-2",
                #         prompt=v5_result.get("prompt", ""),
                #         success=False,
                #         run_time=total_time
                #     )
                # except:
                #     pass
                pass
        
        # 전체 성공 여부 판단
        overall_success = v4_result.get("success", False) or v5_result.get("success", False)
        
        if enable_logging:
            print("\n" + "="*80)
            print("[V4V5커스텀] 병렬 파이프라인 완료")
            print(f"전체 실행 시간: {total_time:.2f}초")
            print(f"CustomV5-1 성공: {v4_result.get('success', False)}")
            print(f"CustomV5-2 성공: {v5_result.get('success', False)}")
            print("="*80 + "\n")
        
        return {
            "success": overall_success,
            "v4_result": v4_result,
            "v5_result": v5_result,
            "total_time": round(total_time, 2),
            "message": "V4V5커스텀 비교가 완료되었습니다.",
            "cutout_ms": cutout_ms,
            "gemini_call_ms": gemini_call_ms,
            "resize_ms": resize_ms
        }


# ============================================================
# 편의 함수
# ============================================================

async def run_v4v5_custom_compare(
    person_img: Image.Image,
    garment_img: Image.Image,
    background_img: Image.Image,
    enable_logging: bool = True
) -> Dict:
    """
    V4V5커스텀 비교 실행 편의 함수
    
    Args:
        person_img: 사람 이미지 (PIL Image)
        garment_img: 의상 이미지 (PIL Image)
        background_img: 배경 이미지 (PIL Image)
        enable_logging: 로깅 활성화 여부 (기본값: True)
    
    Returns:
        dict: V4V5커스텀 비교 결과
    """
    print(f"[DEBUG run_v4v5_custom_compare] enable_logging={enable_logging}")
    orchestrator = V4V5CustomOrchestrator()
    return await orchestrator.run_parallel(person_img, garment_img, background_img, enable_logging)

