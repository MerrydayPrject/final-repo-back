"""SegFormer B2 Garment Parsing (HuggingFace Inference API)"""
import os
import base64
import requests
import traceback
import time
import numpy as np
# import torch  # 주석 처리: torch/transformers 미사용 (HuggingFace API 사용)
# import torch.nn.functional as F  # 주석 처리: torch/transformers 미사용 (HuggingFace API 사용)
from typing import Dict, Optional
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# HuggingFace Inference API 설정
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
SEGFORMER_MODEL_ID = "yolo12138/segformer-b2-human-parse-24"
# 새로운 라우터 엔드포인트 사용 (2025년 업데이트)
HUGGINGFACE_API_BASE_URL = os.getenv(
    "HUGGINGFACE_API_BASE_URL",
    "https://router.huggingface.co/hf-inference/models"
)
SEGFORMER_API_URL = f"{HUGGINGFACE_API_BASE_URL}/{SEGFORMER_MODEL_ID}"
API_TIMEOUT = int(os.getenv("SEGFORMER_API_TIMEOUT", "60"))

# V3 전용 모델 설정
SEGFORMER_MODEL_ID_V3 = "mattmdjaga/segformer_b2_clothes"
SEGFORMER_API_URL_V3 = f"{HUGGINGFACE_API_BASE_URL}/{SEGFORMER_MODEL_ID_V3}"


def parse_garment_image(
    garment_img: Image.Image
) -> Dict[str, Optional[Image.Image]]:
    """
    SegFormer B2 Human Parsing 모델을 HuggingFace Inference API로 호출하여
    의상 이미지에서 garment_only 추출
    
    Args:
        garment_img: 의상 이미지 (PIL Image)
    
    Returns:
        dict: {
            "success": bool,
            "garment_mask": Optional[Image.Image],  # garment_mask.png
            "garment_only": Optional[Image.Image],  # garment_only.png
            "message": str,
            "error": Optional[str]
        }
    """
    if not HUGGINGFACE_API_KEY:
        error_msg = (
            "HUGGINGFACE_API_KEY가 설정되지 않았습니다!\n\n"
            "해결 방법:\n"
            "1. final-repo-back/.env 파일 생성 또는 수정\n"
            "2. 다음 줄 추가: HUGGINGFACE_API_KEY=your_hf_api_key_here\n"
            "3. https://huggingface.co/settings/tokens 에서 API 키 발급\n"
            "4. 서버 재시작"
        )
        print(f"[SegFormer B2 Garment Parser] 오류: {error_msg}")
        return {
            "success": False,
            "garment_mask": None,
            "garment_only": None,
            "message": error_msg,
            "error": "api_key_not_found"
        }
    
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 이미지를 base64로 인코딩
    def image_to_base64(img: Image.Image) -> str:
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode("utf-8")
    
    garment_b64 = image_to_base64(garment_img)
    original_size = garment_img.size
    
    # HuggingFace Inference API 요청 데이터 형식
    payload = {
        "inputs": f"data:image/png;base64,{garment_b64}"
    }
    
    try:
        print(f"[SegFormer B2 Garment Parser] Parsing 요청 시작")
        print(f"[SegFormer B2 Garment Parser] 엔드포인트: {SEGFORMER_API_URL}")
        print(f"[SegFormer B2 Garment Parser] 모델: {SEGFORMER_MODEL_ID}")
        print(f"[SegFormer B2 Garment Parser] 원본 이미지 크기: {original_size[0]}x{original_size[1]}")
        
        # HuggingFace Inference API 호출
        response = requests.post(
            SEGFORMER_API_URL,
            headers=headers,
            json=payload,
            timeout=API_TIMEOUT
        )
        
        print(f"[SegFormer B2 Garment Parser] 응답 상태 코드: {response.status_code}")
        
        # 410 Gone 오류 처리 (구 엔드포인트 사용 시)
        if response.status_code == 410:
            error_msg = "구 HuggingFace Inference API 엔드포인트가 더 이상 지원되지 않습니다. 새로운 라우터 엔드포인트를 사용하세요."
            print(f"[SegFormer B2 Garment Parser] {error_msg}")
            print(f"[SegFormer B2 Garment Parser] 현재 엔드포인트: {SEGFORMER_API_URL}")
            return {
                "success": False,
                "garment_mask": None,
                "garment_only": None,
                "message": error_msg,
                "error": "deprecated_endpoint"
            }
        
        # 401 Unauthorized 오류 처리
        if response.status_code == 401:
            error_msg = "HuggingFace API 키가 유효하지 않습니다. API 키를 확인하세요."
            print(f"[SegFormer B2 Garment Parser] {error_msg}")
            return {
                "success": False,
                "garment_mask": None,
                "garment_only": None,
                "message": error_msg,
                "error": "unauthorized"
            }
        
        # 404 Not Found 오류 처리
        if response.status_code == 404:
            error_msg = f"모델을 찾을 수 없습니다: {SEGFORMER_MODEL_ID}"
            error_detail = response.text
            try:
                error_json = response.json()
                error_detail = error_json.get("error", {}).get("message", response.text)
                print(f"[SegFormer B2 Garment Parser] 오류 응답: {error_json}")
            except:
                print(f"[SegFormer B2 Garment Parser] 원시 오류: {response.text}")
            
            return {
                "success": False,
                "garment_mask": None,
                "garment_only": None,
                "message": f"{error_msg}\n상세: {error_detail}",
                "error": "model_not_found"
            }
        
        # 503 Service Unavailable (모델 로딩 중)
        if response.status_code == 503:
            estimated_time = response.headers.get("estimated_time", 10)
            try:
                estimated_time = int(estimated_time)
            except:
                estimated_time = 10
            print(f"[SegFormer B2 Garment Parser] 모델 로딩 중... 예상 대기 시간: {estimated_time}초")
            time.sleep(min(estimated_time + 2, 30))
            
            # 재시도
            print(f"[SegFormer B2 Garment Parser] 재시도 중...")
            response = requests.post(
                SEGFORMER_API_URL,
                headers=headers,
                json=payload,
                timeout=API_TIMEOUT
            )
            print(f"[SegFormer B2 Garment Parser] 재시도 후 응답 상태 코드: {response.status_code}")
        
        # 성공 응답 처리
        if response.status_code == 200:
            result = response.json()
            
            # API 응답에서 세그멘테이션 결과 추출
            # HuggingFace Inference API는 일반적으로 numpy 배열이나 base64 인코딩된 이미지를 반환
            pred_seg = None
            
            if isinstance(result, dict):
                # 응답에 "label" 또는 "mask" 키가 있는 경우
                if "label" in result:
                    # numpy 배열 형태의 레이블 맵
                    if isinstance(result["label"], str):
                        # base64 인코딩된 numpy 배열
                        label_bytes = base64.b64decode(result["label"])
                        pred_seg = np.frombuffer(label_bytes, dtype=np.uint8)
                        # 이미지 크기에 맞게 reshape
                        pred_seg = pred_seg.reshape((original_size[1], original_size[0]))
                    else:
                        pred_seg = np.array(result["label"], dtype=np.uint8)
                
                elif "mask" in result:
                    mask_data = result["mask"]
                    if isinstance(mask_data, str):
                        # base64 인코딩된 마스크 이미지
                        mask_bytes = base64.b64decode(mask_data.split(",")[1] if "," in mask_data else mask_data)
                        mask_img = Image.open(BytesIO(mask_bytes))
                        pred_seg = np.array(mask_img.convert("L"))
                    else:
                        pred_seg = np.array(mask_data, dtype=np.uint8)
                
                # 리스트 형태의 응답인 경우
                elif isinstance(result.get("output"), list):
                    # 첫 번째 요소가 세그멘테이션 결과
                    output_data = result["output"][0]
                    if isinstance(output_data, str):
                        # base64 인코딩된 이미지
                        mask_bytes = base64.b64decode(output_data.split(",")[1] if "," in output_data else output_data)
                        mask_img = Image.open(BytesIO(mask_bytes))
                        pred_seg = np.array(mask_img.convert("L"))
                    else:
                        pred_seg = np.array(output_data, dtype=np.uint8)
            
            elif isinstance(result, list) and len(result) > 0:
                # 리스트 형태의 직접 응답
                output_data = result[0]
                if isinstance(output_data, dict) and "mask" in output_data:
                    mask_data = output_data["mask"]
                    if isinstance(mask_data, str):
                        mask_bytes = base64.b64decode(mask_data.split(",")[1] if "," in mask_data else mask_data)
                        mask_img = Image.open(BytesIO(mask_bytes))
                        pred_seg = np.array(mask_img.convert("L"))
                    else:
                        pred_seg = np.array(mask_data, dtype=np.uint8)
                else:
                    pred_seg = np.array(output_data, dtype=np.uint8)
            
            # pred_seg가 None인 경우 Fallback: 간단한 배경 제거
            if pred_seg is None:
                print("[SegFormer B2 Garment Parser] API 응답에서 세그멘테이션 결과를 추출할 수 없습니다. Fallback 로직 사용...")
                # 간단한 배경 제거 (중앙 픽셀 기준)
                garment_array = np.array(garment_img.convert("RGB"))
                center_pixel = garment_array[garment_array.shape[0]//2, garment_array.shape[1]//2]
                diff = np.abs(garment_array - center_pixel).sum(axis=2)
                threshold = 100
                pred_seg = (diff > threshold).astype(np.uint8)
                # 전체를 의상 영역으로 간주 (배경이 없는 경우)
                if np.sum(pred_seg) < original_size[0] * original_size[1] * 0.1:
                    pred_seg = np.ones((original_size[1], original_size[0]), dtype=np.uint8)
            
            # pred_seg를 원본 이미지 크기에 맞게 리사이즈 (필요한 경우)
            if pred_seg.shape != (original_size[1], original_size[0]):
                pred_seg_img = Image.fromarray(pred_seg, mode='L')
                pred_seg_img = pred_seg_img.resize(original_size, Image.Resampling.NEAREST)
                pred_seg = np.array(pred_seg_img)
            
            # 의상 관련 레이블 추출
            # yolo12138/segformer-b2-human-parse-24 모델의 24개 클래스 중 의상 관련 레이블
            # 일반적인 Human Parsing 레이블 (확인 필요):
            # - Upper clothes (상의): 레이블 4 또는 5
            # - Lower clothes (하의): 레이블 6
            # - Dress (드레스): 레이블 7
            # - Skirt (스커트): 레이블 5 또는 6
            # - Coat/Jacket (코트/자켓): 레이블 8 또는 9
            # 배경(0) 및 신체 부위(얼굴, 손, 다리 등)는 제외
            
            # 의상 레이블 정의 (routers/composition.py 참고: 레이블 5, 6, 9, 10, 13이 의상 영역)
            # 하지만 이것은 person 이미지 기준이므로, garment 이미지에서는 다를 수 있음
            # 일반적으로 배경(0)이 아닌 모든 영역을 의상으로 간주하는 것이 안전
            
            # 배경(0)이 아닌 모든 영역을 의상으로 간주
            garment_mask_array = (pred_seg != 0).astype(np.uint8) * 255
            
            # 의상 영역이 너무 작으면 전체 이미지를 의상으로 간주 (Fallback)
            mask_ratio = np.sum(garment_mask_array > 0) / (original_size[0] * original_size[1])
            if mask_ratio < 0.05:
                print("[SegFormer B2 Garment Parser] 의상 영역이 감지되지 않았습니다. 전체 이미지를 의상으로 간주합니다.")
                garment_mask_array = np.ones((original_size[1], original_size[0]), dtype=np.uint8) * 255
            
            # PIL 이미지로 변환
            garment_mask = Image.fromarray(garment_mask_array, mode='L')
            
            # garment_only 이미지 생성 (RGBA)
            garment_array = np.array(garment_img.convert("RGB"))
            garment_only_rgba = np.zeros((garment_array.shape[0], garment_array.shape[1], 4), dtype=np.uint8)
            garment_only_rgba[:, :, :3] = garment_array  # RGB 채널
            garment_only_rgba[:, :, 3] = garment_mask_array  # Alpha 채널 (마스크)
            
            garment_only = Image.fromarray(garment_only_rgba, mode='RGBA')
            
            # RGB 모드로도 변환 (호환성)
            garment_only_rgb = garment_only.convert("RGB")
            
            print(f"[SegFormer B2 Garment Parser] 성공! garment_only 이미지 추출 완료")
            print(f"[SegFormer B2 Garment Parser] 의상 영역 비율: {mask_ratio:.2%}")
            
            return {
                "success": True,
                "garment_mask": garment_mask,
                "garment_only": garment_only_rgb,  # RGB 모드로 반환 (호환성)
                "message": "SegFormer B2 Garment Parsing 완료"
            }
        else:
            # 기타 오류
            error_detail = response.text
            try:
                error_json = response.json()
                error_detail = error_json.get("error", {}).get("message", response.text)
                print(f"[SegFormer B2 Garment Parser] 오류 응답: {error_json}")
            except:
                print(f"[SegFormer B2 Garment Parser] 원시 오류: {response.text}")
            
            return {
                "success": False,
                "garment_mask": None,
                "garment_only": None,
                "message": error_detail or f"SegFormer B2 API 호출 실패 (상태 코드: {response.status_code})",
                "error": f"api_error_{response.status_code}"
            }
            
    except requests.exceptions.Timeout:
        print(f"[SegFormer B2 Garment Parser] 타임아웃 오류")
        return {
            "success": False,
            "garment_mask": None,
            "garment_only": None,
            "message": "Parsing 요청이 시간 초과되었습니다. 다시 시도해주세요.",
            "error": "timeout"
        }
    except Exception as e:
        print(f"[SegFormer B2 Garment Parser] 예외 발생: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "garment_mask": None,
            "garment_only": None,
            "message": f"Parsing 중 오류 발생: {str(e)}",
            "error": str(e)
        }


def parse_garment_image_v3(
    garment_img: Image.Image
) -> Dict[str, Optional[Image.Image]]:
    """
    SegFormer B2 Clothes Parsing 모델을 HuggingFace Inference API로 호출하여
    의상 이미지에서 garment_only 추출 (V3 전용)
    
    Args:
        garment_img: 의상 이미지 (PIL Image)
    
    Returns:
        dict: {
            "success": bool,
            "garment_mask": Optional[Image.Image],  # garment_mask.png
            "garment_only": Optional[Image.Image],  # garment_only.png
            "message": str,
            "error": Optional[str]
        }
    """
    if not HUGGINGFACE_API_KEY:
        error_msg = (
            "HUGGINGFACE_API_KEY가 설정되지 않았습니다!\n\n"
            "해결 방법:\n"
            "1. final-repo-back/.env 파일 생성 또는 수정\n"
            "2. 다음 줄 추가: HUGGINGFACE_API_KEY=your_hf_api_key_here\n"
            "3. https://huggingface.co/settings/tokens 에서 API 키 발급\n"
            "4. 서버 재시작"
        )
        print(f"[SegFormer B2 Clothes Parser] 오류: {error_msg}")
        return {
            "success": False,
            "garment_mask": None,
            "garment_only": None,
            "message": error_msg,
            "error": "api_key_not_found"
        }
    
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 이미지를 base64로 인코딩
    def image_to_base64(img: Image.Image) -> str:
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode("utf-8")
    
    garment_b64 = image_to_base64(garment_img)
    original_size = garment_img.size
    
    # HuggingFace Inference API 요청 데이터 형식
    payload = {
        "inputs": f"data:image/png;base64,{garment_b64}"
    }
    
    try:
        print(f"[SegFormer B2 Clothes Parser] Parsing 요청 시작")
        print(f"[SegFormer B2 Clothes Parser] 엔드포인트: {SEGFORMER_API_URL_V3}")
        print(f"[SegFormer B2 Clothes Parser] 모델: {SEGFORMER_MODEL_ID_V3}")
        print(f"[SegFormer B2 Clothes Parser] 원본 이미지 크기: {original_size[0]}x{original_size[1]}")
        
        # HuggingFace Inference API 호출
        response = requests.post(
            SEGFORMER_API_URL_V3,
            headers=headers,
            json=payload,
            timeout=API_TIMEOUT
        )
        
        print(f"[SegFormer B2 Clothes Parser] 응답 상태 코드: {response.status_code}")
        
        # 410 Gone 오류 처리 (구 엔드포인트 사용 시)
        if response.status_code == 410:
            error_msg = "구 HuggingFace Inference API 엔드포인트가 더 이상 지원되지 않습니다. 새로운 라우터 엔드포인트를 사용하세요."
            print(f"[SegFormer B2 Clothes Parser] {error_msg}")
            print(f"[SegFormer B2 Clothes Parser] 현재 엔드포인트: {SEGFORMER_API_URL_V3}")
            return {
                "success": False,
                "garment_mask": None,
                "garment_only": None,
                "message": error_msg,
                "error": "deprecated_endpoint"
            }
        
        # 401 Unauthorized 오류 처리
        if response.status_code == 401:
            error_msg = "HuggingFace API 키가 유효하지 않습니다. API 키를 확인하세요."
            print(f"[SegFormer B2 Clothes Parser] {error_msg}")
            return {
                "success": False,
                "garment_mask": None,
                "garment_only": None,
                "message": error_msg,
                "error": "unauthorized"
            }
        
        # 404 Not Found 오류 처리
        if response.status_code == 404:
            error_msg = f"모델을 찾을 수 없습니다: {SEGFORMER_MODEL_ID_V3}"
            error_detail = response.text
            try:
                error_json = response.json()
                error_detail = error_json.get("error", {}).get("message", response.text)
                print(f"[SegFormer B2 Clothes Parser] 오류 응답: {error_json}")
            except:
                print(f"[SegFormer B2 Clothes Parser] 원시 오류: {response.text}")
            
            return {
                "success": False,
                "garment_mask": None,
                "garment_only": None,
                "message": f"{error_msg}\n상세: {error_detail}",
                "error": "model_not_found"
            }
        
        # 503 Service Unavailable (모델 로딩 중)
        if response.status_code == 503:
            estimated_time = response.headers.get("estimated_time", 10)
            try:
                estimated_time = int(estimated_time)
            except:
                estimated_time = 10
            print(f"[SegFormer B2 Clothes Parser] 모델 로딩 중... 예상 대기 시간: {estimated_time}초")
            time.sleep(min(estimated_time + 2, 30))
            
            # 재시도
            print(f"[SegFormer B2 Clothes Parser] 재시도 중...")
            response = requests.post(
                SEGFORMER_API_URL_V3,
                headers=headers,
                json=payload,
                timeout=API_TIMEOUT
            )
            print(f"[SegFormer B2 Clothes Parser] 재시도 후 응답 상태 코드: {response.status_code}")
        
        # 성공 응답 처리
        if response.status_code == 200:
            result = response.json()
            
            # API 응답에서 세그멘테이션 결과 추출
            # HuggingFace Inference API는 일반적으로 numpy 배열이나 base64 인코딩된 이미지를 반환
            pred_seg = None
            
            if isinstance(result, dict):
                # 응답에 "label" 또는 "mask" 키가 있는 경우
                if "label" in result:
                    # numpy 배열 형태의 레이블 맵
                    if isinstance(result["label"], str):
                        # base64 인코딩된 numpy 배열
                        label_bytes = base64.b64decode(result["label"])
                        pred_seg = np.frombuffer(label_bytes, dtype=np.uint8)
                        # 이미지 크기에 맞게 reshape
                        pred_seg = pred_seg.reshape((original_size[1], original_size[0]))
                    else:
                        pred_seg = np.array(result["label"], dtype=np.uint8)
                
                elif "mask" in result:
                    mask_data = result["mask"]
                    if isinstance(mask_data, str):
                        # base64 인코딩된 마스크 이미지
                        mask_bytes = base64.b64decode(mask_data.split(",")[1] if "," in mask_data else mask_data)
                        mask_img = Image.open(BytesIO(mask_bytes))
                        pred_seg = np.array(mask_img.convert("L"))
                    else:
                        pred_seg = np.array(mask_data, dtype=np.uint8)
                
                # 리스트 형태의 응답인 경우
                elif isinstance(result.get("output"), list):
                    # 첫 번째 요소가 세그멘테이션 결과
                    output_data = result["output"][0]
                    if isinstance(output_data, str):
                        # base64 인코딩된 이미지
                        mask_bytes = base64.b64decode(output_data.split(",")[1] if "," in output_data else output_data)
                        mask_img = Image.open(BytesIO(mask_bytes))
                        pred_seg = np.array(mask_img.convert("L"))
                    else:
                        pred_seg = np.array(output_data, dtype=np.uint8)
            
            elif isinstance(result, list) and len(result) > 0:
                # 리스트 형태의 직접 응답
                output_data = result[0]
                if isinstance(output_data, dict) and "mask" in output_data:
                    mask_data = output_data["mask"]
                    if isinstance(mask_data, str):
                        mask_bytes = base64.b64decode(mask_data.split(",")[1] if "," in mask_data else mask_data)
                        mask_img = Image.open(BytesIO(mask_bytes))
                        pred_seg = np.array(mask_img.convert("L"))
                    else:
                        pred_seg = np.array(mask_data, dtype=np.uint8)
                else:
                    pred_seg = np.array(output_data, dtype=np.uint8)
            
            # pred_seg가 None인 경우 Fallback: 간단한 배경 제거
            if pred_seg is None:
                print("[SegFormer B2 Clothes Parser] API 응답에서 세그멘테이션 결과를 추출할 수 없습니다. Fallback 로직 사용...")
                # 간단한 배경 제거 (중앙 픽셀 기준)
                garment_array = np.array(garment_img.convert("RGB"))
                center_pixel = garment_array[garment_array.shape[0]//2, garment_array.shape[1]//2]
                diff = np.abs(garment_array - center_pixel).sum(axis=2)
                threshold = 100
                pred_seg = (diff > threshold).astype(np.uint8)
                # 전체를 의상 영역으로 간주 (배경이 없는 경우)
                if np.sum(pred_seg) < original_size[0] * original_size[1] * 0.1:
                    pred_seg = np.ones((original_size[1], original_size[0]), dtype=np.uint8)
            
            # pred_seg를 원본 이미지 크기에 맞게 리사이즈 (필요한 경우)
            if pred_seg.shape != (original_size[1], original_size[0]):
                pred_seg_img = Image.fromarray(pred_seg, mode='L')
                pred_seg_img = pred_seg_img.resize(original_size, Image.Resampling.NEAREST)
                pred_seg = np.array(pred_seg_img)
            
            # 의상 관련 레이블 추출
            # mattmdjaga/segformer_b2_clothes 모델의 18개 클래스 중 의상 관련 레이블
            # 레이블 정의 (config/settings.py 참고):
            # - Background: 레이블 0
            # - Upper-clothes: 레이블 4
            # - Skirt: 레이블 5
            # - Pants: 레이블 6
            # - Dress: 레이블 7
            # 기타 신체 부위(얼굴, 손, 다리 등)는 제외
            # 일반적으로 배경(0)이 아닌 모든 영역을 의상으로 간주하는 것이 안전
            
            # 배경(0)이 아닌 모든 영역을 의상으로 간주
            garment_mask_array = (pred_seg != 0).astype(np.uint8) * 255
            
            # 의상 영역이 너무 작으면 전체 이미지를 의상으로 간주 (Fallback)
            mask_ratio = np.sum(garment_mask_array > 0) / (original_size[0] * original_size[1])
            if mask_ratio < 0.05:
                print("[SegFormer B2 Clothes Parser] 의상 영역이 감지되지 않았습니다. 전체 이미지를 의상으로 간주합니다.")
                garment_mask_array = np.ones((original_size[1], original_size[0]), dtype=np.uint8) * 255
            
            # PIL 이미지로 변환
            garment_mask = Image.fromarray(garment_mask_array, mode='L')
            
            # garment_only 이미지 생성 (RGBA)
            garment_array = np.array(garment_img.convert("RGB"))
            garment_only_rgba = np.zeros((garment_array.shape[0], garment_array.shape[1], 4), dtype=np.uint8)
            garment_only_rgba[:, :, :3] = garment_array  # RGB 채널
            garment_only_rgba[:, :, 3] = garment_mask_array  # Alpha 채널 (마스크)
            
            garment_only = Image.fromarray(garment_only_rgba, mode='RGBA')
            
            # RGB 모드로도 변환 (호환성)
            garment_only_rgb = garment_only.convert("RGB")
            
            print(f"[SegFormer B2 Clothes Parser] 성공! garment_only 이미지 추출 완료")
            print(f"[SegFormer B2 Clothes Parser] 의상 영역 비율: {mask_ratio:.2%}")
            
            return {
                "success": True,
                "garment_mask": garment_mask,
                "garment_only": garment_only_rgb,  # RGB 모드로 반환 (호환성)
                "message": "SegFormer B2 Clothes Parsing 완료"
            }
        else:
            # 기타 오류
            error_detail = response.text
            try:
                error_json = response.json()
                error_detail = error_json.get("error", {}).get("message", response.text)
                print(f"[SegFormer B2 Clothes Parser] 오류 응답: {error_json}")
            except:
                print(f"[SegFormer B2 Clothes Parser] 원시 오류: {response.text}")
            
            return {
                "success": False,
                "garment_mask": None,
                "garment_only": None,
                "message": error_detail or f"SegFormer B2 Clothes API 호출 실패 (상태 코드: {response.status_code})",
                "error": f"api_error_{response.status_code}"
            }
            
    except requests.exceptions.Timeout:
        print(f"[SegFormer B2 Clothes Parser] 타임아웃 오류")
        return {
            "success": False,
            "garment_mask": None,
            "garment_only": None,
            "message": "Parsing 요청이 시간 초과되었습니다. 다시 시도해주세요.",
            "error": "timeout"
        }
    except Exception as e:
        print(f"[SegFormer B2 Clothes Parser] 예외 발생: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "garment_mask": None,
            "garment_only": None,
            "message": f"Parsing 중 오류 발생: {str(e)}",
            "error": str(e)
        }


def parse_garment_image_v4(
    garment_img: Image.Image
) -> Dict[str, Optional[Image.Image]]:
    """
    SegFormer B2 Clothes Parsing 모델을 HuggingFace Inference API로 호출하여
    의상 이미지에서 garment_only 추출 (V4 전용)
    
    Args:
        garment_img: 의상 이미지 (PIL Image)
    
    Returns:
        dict: {
            "success": bool,
            "garment_mask": Optional[Image.Image],  # garment_mask.png
            "garment_only": Optional[Image.Image],  # garment_only.png (RGB)
            "message": str,
            "error": Optional[str]
        }
    """
    if not HUGGINGFACE_API_KEY:
        error_msg = (
            "HUGGINGFACE_API_KEY가 설정되지 않았습니다!\n\n"
            "해결 방법:\n"
            "1. final-repo-back/.env 파일 생성 또는 수정\n"
            "2. 다음 줄 추가: HUGGINGFACE_API_KEY=your_hf_api_key_here\n"
            "3. https://huggingface.co/settings/tokens 에서 API 키 발급\n"
            "4. 서버 재시작"
        )
        print(f"[SegFormer B2 Clothes Parser V4] 오류: {error_msg}")
        return {
            "success": False,
            "garment_mask": None,
            "garment_only": None,
            "message": error_msg,
            "error": "api_key_not_found"
        }
    
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 이미지를 base64로 인코딩
    def image_to_base64(img: Image.Image) -> str:
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode("utf-8")
    
    garment_b64 = image_to_base64(garment_img)
    original_size = garment_img.size
    
    # HuggingFace Inference API 요청 데이터 형식
    payload = {
        "inputs": f"data:image/png;base64,{garment_b64}"
    }
    
    try:
        print(f"[SegFormer B2 Clothes Parser V4] Parsing 요청 시작")
        print(f"[SegFormer B2 Clothes Parser V4] 엔드포인트: {SEGFORMER_API_URL_V3}")
        print(f"[SegFormer B2 Clothes Parser V4] 모델: {SEGFORMER_MODEL_ID_V3}")
        print(f"[SegFormer B2 Clothes Parser V4] 원본 이미지 크기: {original_size[0]}x{original_size[1]}")
        
        # HuggingFace Inference API 호출
        response = requests.post(
            SEGFORMER_API_URL_V3,
            headers=headers,
            json=payload,
            timeout=API_TIMEOUT
        )
        
        print(f"[SegFormer B2 Clothes Parser V4] 응답 상태 코드: {response.status_code}")
        
        # 410 Gone 오류 처리 (구 엔드포인트 사용 시)
        if response.status_code == 410:
            error_msg = "구 HuggingFace Inference API 엔드포인트가 더 이상 지원되지 않습니다. 새로운 라우터 엔드포인트를 사용하세요."
            print(f"[SegFormer B2 Clothes Parser V4] {error_msg}")
            print(f"[SegFormer B2 Clothes Parser V4] 현재 엔드포인트: {SEGFORMER_API_URL_V3}")
            return {
                "success": False,
                "garment_mask": None,
                "garment_only": None,
                "message": error_msg,
                "error": "deprecated_endpoint"
            }
        
        # 401 Unauthorized 오류 처리
        if response.status_code == 401:
            error_msg = "HuggingFace API 키가 유효하지 않습니다. API 키를 확인하세요."
            print(f"[SegFormer B2 Clothes Parser V4] {error_msg}")
            return {
                "success": False,
                "garment_mask": None,
                "garment_only": None,
                "message": error_msg,
                "error": "unauthorized"
            }
        
        # 404 Not Found 오류 처리
        if response.status_code == 404:
            error_msg = f"모델을 찾을 수 없습니다: {SEGFORMER_MODEL_ID_V3}"
            error_detail = response.text
            try:
                error_json = response.json()
                error_detail = error_json.get("error", {}).get("message", response.text)
                print(f"[SegFormer B2 Clothes Parser V4] 오류 응답: {error_json}")
            except:
                print(f"[SegFormer B2 Clothes Parser V4] 원시 오류: {response.text}")
            
            return {
                "success": False,
                "garment_mask": None,
                "garment_only": None,
                "message": f"{error_msg}\n상세: {error_detail}",
                "error": "model_not_found"
            }
        
        # 503 Service Unavailable (모델 로딩 중)
        if response.status_code == 503:
            estimated_time = response.headers.get("estimated_time", 10)
            try:
                estimated_time = int(estimated_time)
            except:
                estimated_time = 10
            print(f"[SegFormer B2 Clothes Parser V4] 모델 로딩 중... 예상 대기 시간: {estimated_time}초")
            time.sleep(min(estimated_time + 2, 30))
            
            # 재시도
            print(f"[SegFormer B2 Clothes Parser V4] 재시도 중...")
            response = requests.post(
                SEGFORMER_API_URL_V3,
                headers=headers,
                json=payload,
                timeout=API_TIMEOUT
            )
            print(f"[SegFormer B2 Clothes Parser V4] 재시도 후 응답 상태 코드: {response.status_code}")
        
        # 성공 응답 처리
        if response.status_code == 200:
            result = response.json()
            
            # API 응답에서 세그멘테이션 결과 추출
            # HuggingFace Inference API는 일반적으로 numpy 배열이나 base64 인코딩된 이미지를 반환
            pred_seg = None
            
            if isinstance(result, dict):
                # 응답에 "label" 또는 "mask" 키가 있는 경우
                if "label" in result:
                    # numpy 배열 형태의 레이블 맵
                    if isinstance(result["label"], str):
                        # base64 인코딩된 numpy 배열
                        label_bytes = base64.b64decode(result["label"])
                        pred_seg = np.frombuffer(label_bytes, dtype=np.uint8)
                        # 이미지 크기에 맞게 reshape
                        pred_seg = pred_seg.reshape((original_size[1], original_size[0]))
                    else:
                        pred_seg = np.array(result["label"], dtype=np.uint8)
                
                elif "mask" in result:
                    mask_data = result["mask"]
                    if isinstance(mask_data, str):
                        # base64 인코딩된 마스크 이미지
                        mask_bytes = base64.b64decode(mask_data.split(",")[1] if "," in mask_data else mask_data)
                        mask_img = Image.open(BytesIO(mask_bytes))
                        pred_seg = np.array(mask_img.convert("L"))
                    else:
                        pred_seg = np.array(mask_data, dtype=np.uint8)
                
                # 리스트 형태의 응답인 경우
                elif isinstance(result.get("output"), list):
                    # 첫 번째 요소가 세그멘테이션 결과
                    output_data = result["output"][0]
                    if isinstance(output_data, str):
                        # base64 인코딩된 이미지
                        mask_bytes = base64.b64decode(output_data.split(",")[1] if "," in output_data else output_data)
                        mask_img = Image.open(BytesIO(mask_bytes))
                        pred_seg = np.array(mask_img.convert("L"))
                    else:
                        pred_seg = np.array(output_data, dtype=np.uint8)
            
            elif isinstance(result, list) and len(result) > 0:
                # 리스트 형태의 직접 응답
                output_data = result[0]
                if isinstance(output_data, dict) and "mask" in output_data:
                    mask_data = output_data["mask"]
                    if isinstance(mask_data, str):
                        mask_bytes = base64.b64decode(mask_data.split(",")[1] if "," in mask_data else mask_data)
                        mask_img = Image.open(BytesIO(mask_bytes))
                        pred_seg = np.array(mask_img.convert("L"))
                    else:
                        pred_seg = np.array(mask_data, dtype=np.uint8)
                else:
                    pred_seg = np.array(output_data, dtype=np.uint8)
            
            # pred_seg가 None인 경우 Fallback: 간단한 배경 제거
            if pred_seg is None:
                print("[SegFormer B2 Clothes Parser V4] API 응답에서 세그멘테이션 결과를 추출할 수 없습니다. Fallback 로직 사용...")
                # 간단한 배경 제거 (중앙 픽셀 기준)
                garment_array = np.array(garment_img.convert("RGB"))
                center_pixel = garment_array[garment_array.shape[0]//2, garment_array.shape[1]//2]
                diff = np.abs(garment_array - center_pixel).sum(axis=2)
                threshold = 100
                pred_seg = (diff > threshold).astype(np.uint8)
                # 전체를 의상 영역으로 간주 (배경이 없는 경우)
                if np.sum(pred_seg) < original_size[0] * original_size[1] * 0.1:
                    pred_seg = np.ones((original_size[1], original_size[0]), dtype=np.uint8)
            
            # pred_seg를 원본 이미지 크기에 맞게 리사이즈 (필요한 경우)
            if pred_seg.shape != (original_size[1], original_size[0]):
                pred_seg_img = Image.fromarray(pred_seg, mode='L')
                pred_seg_img = pred_seg_img.resize(original_size, Image.Resampling.NEAREST)
                pred_seg = np.array(pred_seg_img)
            
            # 의상 관련 레이블 추출
            # mattmdjaga/segformer_b2_clothes 모델의 18개 클래스 중 의상 관련 레이블
            # 레이블 정의:
            # - Background: 레이블 0
            # - Upper-clothes: 레이블 4
            # - Skirt: 레이블 5
            # - Pants: 레이블 6
            # - Dress: 레이블 7
            # 기타 신체 부위(얼굴, 손, 다리 등)는 제외
            # 일반적으로 배경(0)이 아닌 모든 영역을 의상으로 간주하는 것이 안전
            
            # 배경(0)이 아닌 모든 영역을 의상으로 간주
            garment_mask_array = (pred_seg != 0).astype(np.uint8) * 255
            
            # 의상 영역이 너무 작으면 전체 이미지를 의상으로 간주 (Fallback)
            mask_ratio = np.sum(garment_mask_array > 0) / (original_size[0] * original_size[1])
            if mask_ratio < 0.05:
                print("[SegFormer B2 Clothes Parser V4] 의상 영역이 감지되지 않았습니다. 전체 이미지를 의상으로 간주합니다.")
                garment_mask_array = np.ones((original_size[1], original_size[0]), dtype=np.uint8) * 255
            
            # PIL 이미지로 변환
            garment_mask = Image.fromarray(garment_mask_array, mode='L')
            
            # garment_only 이미지 생성 (RGBA)
            garment_array = np.array(garment_img.convert("RGB"))
            garment_only_rgba = np.zeros((garment_array.shape[0], garment_array.shape[1], 4), dtype=np.uint8)
            garment_only_rgba[:, :, :3] = garment_array  # RGB 채널
            garment_only_rgba[:, :, 3] = garment_mask_array  # Alpha 채널 (마스크)
            
            garment_only = Image.fromarray(garment_only_rgba, mode='RGBA')
            
            # RGB 모드로 변환 (V4는 RGB로 반환)
            garment_only_rgb = garment_only.convert("RGB")
            
            print(f"[SegFormer B2 Clothes Parser V4] 성공! garment_only 이미지 추출 완료")
            print(f"[SegFormer B2 Clothes Parser V4] 의상 영역 비율: {mask_ratio:.2%}")
            
            return {
                "success": True,
                "garment_mask": garment_mask,
                "garment_only": garment_only_rgb,  # RGB 모드로 반환
                "message": "SegFormer B2 Clothes Parsing 완료 (V4)"
            }
        else:
            # 기타 오류
            error_detail = response.text
            try:
                error_json = response.json()
                error_detail = error_json.get("error", {}).get("message", response.text)
                print(f"[SegFormer B2 Clothes Parser V4] 오류 응답: {error_json}")
            except:
                print(f"[SegFormer B2 Clothes Parser V4] 원시 오류: {response.text}")
            
            return {
                "success": False,
                "garment_mask": None,
                "garment_only": None,
                "message": error_detail or f"SegFormer B2 Clothes API 호출 실패 (상태 코드: {response.status_code})",
                "error": f"api_error_{response.status_code}"
            }
            
    except requests.exceptions.Timeout:
        print(f"[SegFormer B2 Clothes Parser V4] 타임아웃 오류")
        return {
            "success": False,
            "garment_mask": None,
            "garment_only": None,
            "message": "Parsing 요청이 시간 초과되었습니다. 다시 시도해주세요.",
            "error": "timeout"
        }
    except Exception as e:
        print(f"[SegFormer B2 Clothes Parser V4] 예외 발생: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "garment_mask": None,
            "garment_only": None,
            "message": f"Parsing 중 오류 발생: {str(e)}",
            "error": str(e)
        }
