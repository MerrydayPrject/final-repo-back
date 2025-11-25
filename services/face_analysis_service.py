"""
얼굴 분석 서비스
HuggingFace Inference Endpoint를 사용하여 InsightFace 얼굴 분석
"""
import os
import base64
import io
import requests
import numpy as np
from PIL import Image
from typing import Optional, Dict, List
from config.settings import INSIGHTFACE_ENDPOINT_URL, INSIGHTFACE_API_KEY


class FaceAnalysisService:
    """얼굴 분석 서비스 (HuggingFace Inference Endpoint 기반)"""
    
    def __init__(self, endpoint_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        초기화
        
        Args:
            endpoint_url: HuggingFace Inference Endpoint URL (None이면 설정에서 가져옴)
            api_key: HuggingFace API 키 (None이면 설정에서 가져옴)
        """
        self.endpoint_url = endpoint_url or INSIGHTFACE_ENDPOINT_URL
        self.api_key = api_key or INSIGHTFACE_API_KEY
        self.is_initialized = bool(self.endpoint_url and self.api_key)
        
        if not self.is_initialized:
            print("⚠️  InsightFace Inference Endpoint가 설정되지 않았습니다.")
            print("   INSIGHTFACE_ENDPOINT_URL과 INSIGHTFACE_API_KEY를 .env 파일에 설정하세요.")
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """
        PIL Image를 Base64 문자열로 변환
        
        Args:
            image: PIL Image 객체
            
        Returns:
            Base64 인코딩된 이미지 문자열
        """
        buffered = io.BytesIO()
        # RGB로 변환 (RGBA인 경우)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(buffered, format="JPEG", quality=95)
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return img_base64
    
    def _numpy_to_base64(self, image: np.ndarray) -> str:
        """
        numpy 배열 이미지를 Base64 문자열로 변환
        
        Args:
            image: BGR 형식의 numpy 배열 이미지
            
        Returns:
            Base64 인코딩된 이미지 문자열
        """
        # BGR -> RGB 변환
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = image[:, :, ::-1]  # BGR to RGB
        else:
            image_rgb = image
        
        # PIL Image로 변환
        pil_image = Image.fromarray(image_rgb)
        return self._image_to_base64(pil_image)
    
    def detect_face(self, image: np.ndarray) -> Optional[Dict]:
        """
        이미지에서 얼굴 감지 및 분석
        
        Args:
            image: BGR 형식의 numpy 배열 이미지
            
        Returns:
            얼굴 분석 결과 딕셔너리 또는 None
        """
        if not self.is_initialized:
            print("서비스가 초기화되지 않았습니다.")
            return None
        
        try:
            # 이미지를 Base64로 인코딩
            img_base64 = self._numpy_to_base64(image)
            
            # HuggingFace Inference Endpoint 요청 형식
            payload = {
                "inputs": {
                    "image": f"data:image/jpeg;base64,{img_base64}"
                }
            }
            
            # API 호출
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                self.endpoint_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"얼굴 분석 API 호출 실패: {response.status_code}")
                print(f"응답: {response.text}")
                return None
            
            # 응답 파싱
            result = response.json()
            
            # Inference Endpoint 응답 형식에 따라 파싱
            # 일반적으로 {"faces": [...]} 또는 직접 배열 형식
            if isinstance(result, dict):
                if "faces" in result:
                    faces = result["faces"]
                elif "data" in result:
                    faces = result["data"]
                else:
                    # 직접 얼굴 정보인 경우
                    faces = [result]
            elif isinstance(result, list):
                faces = result
            else:
                print(f"예상하지 못한 응답 형식: {result}")
                return None
            
            # 첫 번째 얼굴 반환 (가장 큰 얼굴)
            if len(faces) > 0:
                # 여러 얼굴이 있는 경우 가장 큰 얼굴 선택
                if len(faces) > 1:
                    largest_face = max(faces, key=lambda f: (
                        f.get("bbox", [0, 0, 0, 0])[2] - f.get("bbox", [0, 0, 0, 0])[0]
                    ) * (
                        f.get("bbox", [0, 0, 0, 0])[3] - f.get("bbox", [0, 0, 0, 0])[1]
                    ))
                else:
                    largest_face = faces[0]
                
                return largest_face
            
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"얼굴 분석 API 요청 오류: {e}")
            return None
        except Exception as e:
            print(f"얼굴 분석 오류: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        이미지에서 모든 얼굴 감지 및 분석
        
        Args:
            image: BGR 형식의 numpy 배열 이미지
            
        Returns:
            얼굴 분석 결과 리스트
        """
        if not self.is_initialized:
            print("서비스가 초기화되지 않았습니다.")
            return []
        
        try:
            # 이미지를 Base64로 인코딩
            img_base64 = self._numpy_to_base64(image)
            
            # HuggingFace Inference Endpoint 요청 형식
            payload = {
                "inputs": {
                    "image": f"data:image/jpeg;base64,{img_base64}"
                }
            }
            
            # API 호출
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                self.endpoint_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"얼굴 분석 API 호출 실패: {response.status_code}")
                print(f"응답: {response.text}")
                return []
            
            # 응답 파싱
            result = response.json()
            
            # Inference Endpoint 응답 형식에 따라 파싱
            if isinstance(result, dict):
                if "faces" in result:
                    faces = result["faces"]
                elif "data" in result:
                    faces = result["data"]
                else:
                    faces = [result]
            elif isinstance(result, list):
                faces = result
            else:
                return []
            
            return faces if isinstance(faces, list) else []
            
        except requests.exceptions.RequestException as e:
            print(f"얼굴 분석 API 요청 오류: {e}")
            return []
        except Exception as e:
            print(f"얼굴 분석 오류: {e}")
            import traceback
            traceback.print_exc()
            return []

