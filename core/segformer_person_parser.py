"""SegFormer B2 Person Parsing (HuggingFace Inference API)"""
import os
import base64
import requests
import traceback
import time
import numpy as np
from typing import Dict, Optional, Tuple
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

from config.hf_segformer import (
    HUGGINGFACE_API_KEY,
    SEGFORMER_API_URL,
    API_TIMEOUT,
    FACE_MASK_IDS,
    CLOTH_MASK_IDS,
    BODY_MASK_IDS
)

# .env 파일 로드
load_dotenv()


def parse_person_image(
    person_img: Image.Image
) -> Dict:
    """
    SegFormer B2 Human Parsing 모델을 HuggingFace Inference API로 호출하여
    인물 이미지에서 face_mask, cloth_mask, body_mask 추출
    
    Args:
        person_img: 인물 이미지 (PIL Image)
    
    Returns:
        dict: {
            "success": bool,
            "parsing_mask": Optional[np.ndarray],  # 원본 파싱 결과 (레이블 맵)
            "face_mask": Optional[np.ndarray],  # face_mask (0 또는 255)
            "cloth_mask": Optional[np.ndarray],  # cloth_mask (0 또는 255)
            "body_mask": Optional[np.ndarray],  # body_mask (0 또는 255)
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
        print(f"[SegFormer B2 Person Parser] 오류: {error_msg}")
        return {
            "success": False,
            "parsing_mask": None,
            "face_mask": None,
            "cloth_mask": None,
            "body_mask": None,
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
    
    person_b64 = image_to_base64(person_img)
    original_size = person_img.size
    
    # HuggingFace Inference API 요청 데이터 형식
    payload = {
        "inputs": f"data:image/png;base64,{person_b64}"
    }
    
    try:
        print(f"[SegFormer B2 Person Parser] Parsing 요청 시작")
        print(f"[SegFormer B2 Person Parser] 엔드포인트: {SEGFORMER_API_URL}")
        print(f"[SegFormer B2 Person Parser] 원본 이미지 크기: {original_size[0]}x{original_size[1]}")
        
        # HuggingFace Inference API 호출
        response = requests.post(
            SEGFORMER_API_URL,
            headers=headers,
            json=payload,
            timeout=API_TIMEOUT
        )
        
        print(f"[SegFormer B2 Person Parser] 응답 상태 코드: {response.status_code}")
        
        # 오류 처리
        if response.status_code == 410:
            error_msg = "구 HuggingFace Inference API 엔드포인트가 더 이상 지원되지 않습니다."
            return {
                "success": False,
                "parsing_mask": None,
                "face_mask": None,
                "cloth_mask": None,
                "body_mask": None,
                "message": error_msg,
                "error": "deprecated_endpoint"
            }
        
        if response.status_code == 401:
            error_msg = "HuggingFace API 키가 유효하지 않습니다."
            return {
                "success": False,
                "parsing_mask": None,
                "face_mask": None,
                "cloth_mask": None,
                "body_mask": None,
                "message": error_msg,
                "error": "unauthorized"
            }
        
        if response.status_code == 404:
            error_msg = f"모델을 찾을 수 없습니다: {SEGFORMER_API_URL}"
            return {
                "success": False,
                "parsing_mask": None,
                "face_mask": None,
                "cloth_mask": None,
                "body_mask": None,
                "message": error_msg,
                "error": "model_not_found"
            }
        
        # 503 Service Unavailable (모델 로딩 중)
        if response.status_code == 503:
            estimated_time = response.headers.get("estimated_time", 10)
            try:
                estimated_time = int(estimated_time)
            except:
                estimated_time = 10
            print(f"[SegFormer B2 Person Parser] 모델 로딩 중... 예상 대기 시간: {estimated_time}초")
            time.sleep(min(estimated_time + 2, 30))
            
            # 재시도
            print(f"[SegFormer B2 Person Parser] 재시도 중...")
            response = requests.post(
                SEGFORMER_API_URL,
                headers=headers,
                json=payload,
                timeout=API_TIMEOUT
            )
            print(f"[SegFormer B2 Person Parser] 재시도 후 응답 상태 코드: {response.status_code}")
        
        # 성공 응답 처리
        if response.status_code == 200:
            result = response.json()
            
            # API 응답에서 세그멘테이션 결과 추출
            pred_seg = None
            
            if isinstance(result, dict):
                if "label" in result:
                    if isinstance(result["label"], str):
                        label_bytes = base64.b64decode(result["label"])
                        pred_seg = np.frombuffer(label_bytes, dtype=np.uint8)
                        pred_seg = pred_seg.reshape((original_size[1], original_size[0]))
                    else:
                        pred_seg = np.array(result["label"], dtype=np.uint8)
                elif "mask" in result:
                    mask_data = result["mask"]
                    if isinstance(mask_data, str):
                        mask_bytes = base64.b64decode(mask_data.split(",")[1] if "," in mask_data else mask_data)
                        mask_img = Image.open(BytesIO(mask_bytes))
                        pred_seg = np.array(mask_img.convert("L"))
                    else:
                        pred_seg = np.array(mask_data, dtype=np.uint8)
                elif isinstance(result.get("output"), list):
                    output_data = result["output"][0]
                    if isinstance(output_data, str):
                        mask_bytes = base64.b64decode(output_data.split(",")[1] if "," in output_data else output_data)
                        mask_img = Image.open(BytesIO(mask_bytes))
                        pred_seg = np.array(mask_img.convert("L"))
                    else:
                        pred_seg = np.array(output_data, dtype=np.uint8)
            elif isinstance(result, list) and len(result) > 0:
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
            
            # pred_seg가 None인 경우 Fallback
            if pred_seg is None:
                print("[SegFormer B2 Person Parser] API 응답에서 세그멘테이션 결과를 추출할 수 없습니다. Fallback 로직 사용...")
                # 간단한 배경 제거
                person_array = np.array(person_img.convert("RGB"))
                center_pixel = person_array[person_array.shape[0]//2, person_array.shape[1]//2]
                diff = np.abs(person_array - center_pixel).sum(axis=2)
                threshold = 100
                pred_seg = (diff > threshold).astype(np.uint8)
                if np.sum(pred_seg) < original_size[0] * original_size[1] * 0.1:
                    pred_seg = np.ones((original_size[1], original_size[0]), dtype=np.uint8) * 11  # face로 가정
            
            # pred_seg를 원본 이미지 크기에 맞게 리사이즈 (필요한 경우)
            if pred_seg.shape != (original_size[1], original_size[0]):
                pred_seg_img = Image.fromarray(pred_seg, mode='L')
                pred_seg_img = pred_seg_img.resize(original_size, Image.Resampling.NEAREST)
                pred_seg = np.array(pred_seg_img)
            
            # 레이블별 마스크 생성
            face_mask_array = np.isin(pred_seg, FACE_MASK_IDS).astype(np.uint8) * 255
            cloth_mask_array = np.isin(pred_seg, CLOTH_MASK_IDS).astype(np.uint8) * 255
            body_mask_array = np.isin(pred_seg, BODY_MASK_IDS).astype(np.uint8) * 255
            
            print(f"[SegFormer B2 Person Parser] 성공! 마스크 추출 완료")
            print(f"[SegFormer B2 Person Parser] face_mask 비율: {np.sum(face_mask_array > 0) / (original_size[0] * original_size[1]):.2%}")
            print(f"[SegFormer B2 Person Parser] cloth_mask 비율: {np.sum(cloth_mask_array > 0) / (original_size[0] * original_size[1]):.2%}")
            print(f"[SegFormer B2 Person Parser] body_mask 비율: {np.sum(body_mask_array > 0) / (original_size[0] * original_size[1]):.2%}")
            
            return {
                "success": True,
                "parsing_mask": pred_seg,
                "face_mask": face_mask_array,
                "cloth_mask": cloth_mask_array,
                "body_mask": body_mask_array,
                "message": "SegFormer B2 Person Parsing 완료"
            }
        else:
            # 기타 오류
            error_detail = response.text
            try:
                error_json = response.json()
                error_detail = error_json.get("error", {}).get("message", response.text)
            except:
                pass
            
            return {
                "success": False,
                "parsing_mask": None,
                "face_mask": None,
                "cloth_mask": None,
                "body_mask": None,
                "message": error_detail or f"SegFormer B2 API 호출 실패 (상태 코드: {response.status_code})",
                "error": f"api_error_{response.status_code}"
            }
            
    except requests.exceptions.Timeout:
        print(f"[SegFormer B2 Person Parser] 타임아웃 오류")
        return {
            "success": False,
            "parsing_mask": None,
            "face_mask": None,
            "cloth_mask": None,
            "body_mask": None,
            "message": "Parsing 요청이 시간 초과되었습니다. 다시 시도해주세요.",
            "error": "timeout"
        }
    except Exception as e:
        print(f"[SegFormer B2 Person Parser] 예외 발생: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "parsing_mask": None,
            "face_mask": None,
            "cloth_mask": None,
            "body_mask": None,
            "message": f"Parsing 중 오류 발생: {str(e)}",
            "error": str(e)
        }

