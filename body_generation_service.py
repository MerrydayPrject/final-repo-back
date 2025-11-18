"""
체형 생성 서비스 모듈
Grok-2 image 모델을 사용하여 상반신 사진에서 전신 사진 생성

기능:
1. 얼굴과 목 영역 자동 감지 및 추출
2. Grok API로 전신 이미지 생성 (slim 체형)
3. 얼굴+목과 체형을 자연스럽게 합성
"""
from PIL import Image, ImageDraw, ImageFilter
import io
import os
import base64
import requests
import numpy as np
from typing import Optional, Dict, Tuple
from dotenv import load_dotenv
from pathlib import Path

# Config 파일에서 설정 가져오기
try:
    from config.body_generation_config import (
        GROK_API_KEY as CONFIG_GROK_API_KEY,
        GROK_API_BASE_URL as CONFIG_GROK_API_BASE_URL,
        GROK_IMAGE_MODEL as CONFIG_GROK_IMAGE_MODEL,
        GROK_VISION_MODEL as CONFIG_GROK_VISION_MODEL,
        FACE_DETECTION_MIN_CONFIDENCE,
        FACE_NECK_EXTENSION_RATIO,
        BLEND_FADE_WIDTH,
        BODY_IMAGE_TARGET_HEIGHT_RATIO,
        GROK_API_TIMEOUT
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("⚠️  config/body_generation_config.py를 찾을 수 없습니다. 기본 설정을 사용합니다.")

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️  MediaPipe가 설치되지 않았습니다. OpenCV로 얼굴 감지를 시도합니다.")

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("⚠️  OpenCV가 설치되지 않았습니다. 얼굴 감지 기능이 제한될 수 있습니다.")


class BodyGenerationService:
    """체형 생성 서비스"""
    
    def __init__(self):
        """서비스 초기화"""
        # Config 파일에서 설정 가져오기 (없으면 환경 변수에서)
        if CONFIG_AVAILABLE:
            self.api_key = CONFIG_GROK_API_KEY
            self.api_base_url = CONFIG_GROK_API_BASE_URL
            self.model_name = CONFIG_GROK_IMAGE_MODEL
            self.vision_model_name = CONFIG_GROK_VISION_MODEL
        else:
            load_dotenv()
            self.api_key = os.getenv("GROK_API_KEY")
            self.api_base_url = os.getenv("GROK_API_BASE_URL", "https://api.x.ai/v1")
            self.model_name = os.getenv("GROK_IMAGE_MODEL", "grok-2-image")
            self.vision_model_name = os.getenv("GROK_VISION_MODEL", "grok-2-vision-1212")
        
        # 얼굴 감지 초기화 (MediaPipe 우선, 실패 시 OpenCV)
        self.face_detector = None
        self.opencv_face_cascade = None
        
        # MediaPipe 시도
        if MEDIAPIPE_AVAILABLE:
            try:
                self._init_mediapipe_face_detection()
            except Exception as e:
                print(f"⚠️  MediaPipe 얼굴 감지 초기화 실패: {e}")
                self.face_detector = None
        
        # MediaPipe 실패 시 OpenCV 시도
        if self.face_detector is None and OPENCV_AVAILABLE:
            try:
                self._init_opencv_face_detection()
            except Exception as e:
                print(f"⚠️  OpenCV 얼굴 감지 초기화 실패: {e}")
                self.opencv_face_cascade = None
    
    def _init_mediapipe_face_detection(self):
        """MediaPipe Face Detection 초기화"""
        if not MEDIAPIPE_AVAILABLE:
            return
        
        try:
            # MediaPipe Face Detection 초기화
            # 최신 버전에서는 모델 파일 없이도 작동하지만, 일부 버전에서는 필요할 수 있음
            base_options = python.BaseOptions(
                delegate=python.BaseOptions.Delegate.CPU
            )
            min_confidence = FACE_DETECTION_MIN_CONFIDENCE if CONFIG_AVAILABLE else 0.5
            options = vision.FaceDetectorOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                min_detection_confidence=min_confidence
            )
            self.face_detector = vision.FaceDetector.create_from_options(options)
            print("✅ MediaPipe 얼굴 감지 모델 초기화 완료")
        except Exception as e:
            print(f"⚠️  MediaPipe 얼굴 감지 모델 초기화 실패: {e}")
            self.face_detector = None
    
    def _init_opencv_face_detection(self):
        """OpenCV Haar Cascade 얼굴 감지 초기화"""
        if not OPENCV_AVAILABLE:
            return
        
        try:
            # OpenCV의 기본 얼굴 감지 모델 로드
            # cv2.data.haarcascades에 기본 모델이 포함되어 있음
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.opencv_face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.opencv_face_cascade.empty():
                raise Exception("Haar Cascade 모델을 로드할 수 없습니다.")
            
            print("✅ OpenCV 얼굴 감지 모델 초기화 완료")
        except Exception as e:
            print(f"⚠️  OpenCV 얼굴 감지 모델 초기화 실패: {e}")
            self.opencv_face_cascade = None
    
    def _detect_face_and_neck(self, image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        """
        얼굴과 목 영역 감지 및 추출
        
        MediaPipe 우선 시도, 실패 시 OpenCV 사용, 둘 다 실패 시 상단 40% 사용
        
        Args:
            image: 입력 이미지 (PIL Image)
        
        Returns:
            (x, y, width, height) 또는 None (감지 실패 시)
            얼굴과 목 부분을 포함하는 영역의 좌표
        """
        # PIL Image를 numpy array로 변환
        image_array = np.array(image)
        
        # RGB 형식으로 변환
        if len(image_array.shape) == 3 and image_array.shape[2] == 4:
            image_array = image_array[:, :, :3]
        
        img_width, img_height = image.size
        
        # 1. MediaPipe 시도
        if self.face_detector:
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
                detection_result = self.face_detector.detect(mp_image)
                
                if detection_result.detections:
                    face = detection_result.detections[0]
                    bbox = face.bounding_box
                    
                    x = int(bbox.origin_x)
                    y = int(bbox.origin_y)
                    width = int(bbox.width)
                    height = int(bbox.height)
                    
                    # 목 부분과 어깨를 포함하도록 영역 확장
                    extension_ratio = FACE_NECK_EXTENSION_RATIO if CONFIG_AVAILABLE else 1.2
                    neck_extension = int(height * extension_ratio)
                    extended_height = height + neck_extension
                    
                    # 이미지 경계 내로 제한
                    x = max(0, x - int(width * 0.15))
                    y = max(0, y - int(height * 0.1))
                    width = min(img_width - x, int(width * 1.4))
                    height = min(img_height - y, extended_height)
                    
                    print(f"[DEBUG] MediaPipe 얼굴 감지 성공: ({x}, {y}, {width}, {height})")
                    return (x, y, width, height)
            except Exception as e:
                print(f"⚠️  MediaPipe 얼굴 감지 오류: {e}")
        
        # 2. OpenCV 시도
        if self.opencv_face_cascade:
            try:
                # OpenCV는 BGR 형식 사용
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                faces = self.opencv_face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                if len(faces) > 0:
                    # 첫 번째 얼굴 사용
                    (x, y, w, h) = faces[0]
                    
                    # 목 부분과 어깨를 포함하도록 영역 확장
                    extension_ratio = FACE_NECK_EXTENSION_RATIO if CONFIG_AVAILABLE else 1.2
                    neck_extension = int(h * extension_ratio)
                    extended_height = h + neck_extension
                    
                    # 이미지 경계 내로 제한
                    x = max(0, x - int(w * 0.15))
                    y = max(0, y - int(h * 0.1))
                    width = min(img_width - x, int(w * 1.4))
                    height = min(img_height - y, extended_height)
                    
                    print(f"[DEBUG] OpenCV 얼굴 감지 성공: ({x}, {y}, {width}, {height})")
                    return (x, y, width, height)
            except Exception as e:
                print(f"⚠️  OpenCV 얼굴 감지 오류: {e}")
        
        # 3. Fallback: 이미지 상단 40% 영역 사용
        print("⚠️  얼굴 감지 실패. 이미지 상단 40% 영역을 얼굴+목+어깨로 사용합니다.")
        return (0, 0, img_width, int(img_height * 0.4))
    
    def _extract_face_neck_region(self, image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
        """
        얼굴과 목 영역 추출
        
        Args:
            image: 원본 이미지
            bbox: (x, y, width, height) 얼굴+목 영역 좌표
        
        Returns:
            얼굴+목 영역 이미지
        """
        x, y, width, height = bbox
        # 얼굴+목 영역 크롭
        face_neck = image.crop((x, y, x + width, y + height))
        return face_neck
    
    def _blend_images(self, face_neck: Image.Image, body_image: Image.Image, face_bbox: Tuple[int, int, int, int]) -> Image.Image:
        """
        얼굴+목 영역과 체형 이미지를 자연스럽게 합성
        
        Args:
            face_neck: 얼굴+목 영역 이미지
            body_image: 생성된 체형 이미지
            face_bbox: 얼굴+목 영역의 원본 좌표 (x, y, width, height)
        
        Returns:
            합성된 전신 이미지
        """
        try:
            # RGB 모드로 변환 (확실하게)
            if body_image.mode != 'RGB':
                body_image = body_image.convert('RGB')
            if face_neck.mode != 'RGB':
                face_neck = face_neck.convert('RGB')
            
            # 체형 이미지 크기로 결과 이미지 생성
            result = body_image.copy()
            
            # 얼굴+목 영역 크기 조정 (체형 이미지의 얼굴 영역에 맞춤)
            body_width, body_height = body_image.size
            face_x, face_y, face_width, face_height = face_bbox
            
            print(f"[DEBUG] 합성 파라미터 - 체형 크기: {body_width}x{body_height}, 얼굴+목 크기: {face_neck.size}")
            
            # 체형 이미지에서 얼굴 영역 위치 추정 (상단 10% 위치로 조정 - 더 가깝게)
            target_y = int(body_height * 0.10)  # 15% → 10%로 변경 (더 위로)
            height_ratio = BODY_IMAGE_TARGET_HEIGHT_RATIO if CONFIG_AVAILABLE else 0.30  # 25% → 30%로 증가 (더 큰 영역)
            target_height = int(body_height * height_ratio)  # 얼굴+목 영역 높이
            
            print(f"[DEBUG] 합성 위치 - target_y: {target_y}, target_height: {target_height}")
            
            # 얼굴+목 이미지가 너무 작으면 확대
            if face_neck.height < target_height:
                # 비율 유지하면서 크기 조정
                aspect_ratio = face_neck.width / face_neck.height
                target_width = int(target_height * aspect_ratio)
                face_neck_resized = face_neck.resize(
                    (target_width, target_height),
                    Image.Resampling.LANCZOS
                )
            else:
                # 이미 충분히 크면 그대로 사용
                face_neck_resized = face_neck.resize(
                    (int(face_neck.width * target_height / face_neck.height), target_height),
                    Image.Resampling.LANCZOS
                )
            
            print(f"[DEBUG] 리사이즈 후 얼굴+목 크기: {face_neck_resized.size}")
            
            # 중앙 정렬
            target_x = (body_width - face_neck_resized.width) // 2
            
            print(f"[DEBUG] 합성 위치 - target_x: {target_x}, target_y: {target_y}")
            
            # 블렌딩을 위한 마스크 생성 (부드러운 경계)
            mask = Image.new('L', face_neck_resized.size, 255)
            draw = ImageDraw.Draw(mask)
            
            # 가장자리 페이드 아웃 효과
            fade_width = BLEND_FADE_WIDTH if CONFIG_AVAILABLE else 20
            # 페이드 너비가 이미지 크기보다 크면 조정
            fade_width = min(fade_width, face_neck_resized.width // 4, face_neck_resized.height // 4)
            
            for i in range(fade_width):
                alpha = int(255 * (1 - i / fade_width))
                # 좌우 페이드
                draw.rectangle([(i, 0), (i + 1, face_neck_resized.height)], fill=alpha, outline=None)
                draw.rectangle([(face_neck_resized.width - i - 1, 0), (face_neck_resized.width - i, face_neck_resized.height)], fill=alpha, outline=None)
                # 상하 페이드 (하단만, 상단은 명확하게)
                if i < fade_width // 2:  # 하단 페이드
                    draw.rectangle([(0, face_neck_resized.height - i - 1), (face_neck_resized.width, face_neck_resized.height - i)], fill=alpha, outline=None)
            
            # 합성 영역이 이미지 범위를 벗어나지 않는지 확인
            if target_x < 0:
                target_x = 0
            if target_y < 0:
                target_y = 0
            if target_x + face_neck_resized.width > body_width:
                # 너무 크면 리사이즈
                face_neck_resized = face_neck_resized.resize(
                    (body_width - target_x, face_neck_resized.height),
                    Image.Resampling.LANCZOS
                )
                mask = mask.resize(face_neck_resized.size, Image.Resampling.LANCZOS)
            if target_y + face_neck_resized.height > body_height:
                # 너무 크면 리사이즈
                face_neck_resized = face_neck_resized.resize(
                    (face_neck_resized.width, body_height - target_y),
                    Image.Resampling.LANCZOS
                )
                mask = mask.resize(face_neck_resized.size, Image.Resampling.LANCZOS)
            
            print(f"[DEBUG] 최종 합성 위치 - target_x: {target_x}, target_y: {target_y}, 크기: {face_neck_resized.size}")
            
            # 얼굴+목 영역을 체형 이미지에 합성
            print(f"[DEBUG] 합성 실행 - 위치: ({target_x}, {target_y}), 크기: {face_neck_resized.size}")
            result.paste(face_neck_resized, (target_x, target_y), mask)
            
            # 합성 결과 검증
            if result is None:
                print("❌ 합성 결과가 None입니다.")
                return body_image
            
            print(f"[DEBUG] 합성 완료 - 결과 이미지 크기: {result.size}")
            print(f"[DEBUG] 합성 완료 - 결과 이미지 모드: {result.mode}")
            
            # 합성이 실제로 적용되었는지 간단히 확인 (픽셀 색상 변화 확인)
            # 상단 영역의 평균 색상이 체형 이미지와 다르면 합성된 것으로 간주
            try:
                # 체형 이미지 상단 영역 샘플
                body_sample = np.array(body_image.crop((target_x, target_y, min(target_x + 50, body_width), min(target_y + 50, body_height))))
                # 합성 이미지 상단 영역 샘플
                result_sample = np.array(result.crop((target_x, target_y, min(target_x + 50, body_width), min(target_y + 50, body_height))))
                
                if not np.array_equal(body_sample, result_sample):
                    print("✅ 합성 검증: 픽셀 변화 감지됨 (합성 성공)")
                else:
                    print("⚠️  합성 검증: 픽셀 변화 없음 (합성이 적용되지 않았을 수 있음)")
            except Exception as verify_error:
                print(f"⚠️  합성 검증 중 오류 (무시): {verify_error}")
            
            return result
            
        except Exception as e:
            print(f"❌ _blend_images 오류: {e}")
            import traceback
            traceback.print_exc()
            # 오류 발생 시 None 반환 (호출부에서 처리)
            return None
    
    def _call_grok_image_api(self, prompt: str) -> Optional[Image.Image]:
        """
        Grok Image API를 호출하여 이미지 생성
        
        grok-2-image 모델을 사용하여 텍스트 프롬프트로 이미지 생성
        
        Args:
            prompt: 이미지 생성 프롬프트 텍스트
        
        Returns:
            생성된 이미지 (PIL Image) 또는 None (실패 시)
        """
        if not self.api_key:
            print("❌ GROK_API_KEY가 설정되지 않았습니다.")
            return None
        
        try:
            # API 요청 구성
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Images Generations 엔드포인트 사용 (텍스트로 이미지 생성)
            # 모델명: "grok-2-image" (별칭, 권장) 또는 "grok-2-image-1212" (정식 ID)
            # 엔드포인트: /v1/images/generations (chat/completions 아님)
            # 모델 목록 확인: /v1/image-generation-models
            payload = {
                "model": self.model_name,  # "grok-2-image" (별칭 사용 권장)
                "prompt": prompt
            }
            
            # API 호출 - Images Generations 엔드포인트 사용
            timeout = GROK_API_TIMEOUT if CONFIG_AVAILABLE else 120
            response = requests.post(
                f"{self.api_base_url}/images/generations",
                headers=headers,
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"[DEBUG] Grok API 응답: {result}")
                
                # Images Generations 응답 형식 확인
                # 가능한 형식:
                # 1. {"created": ..., "data": [{"url": "..."}]}
                # 2. {"data": [{"b64_json": "..."}]}
                # 3. {"data": [{"url": "..."}]}
                if "data" in result and len(result["data"]) > 0:
                    image_data_item = result["data"][0]
                    
                    # URL이 있는 경우
                    if "url" in image_data_item:
                        img_url = image_data_item["url"]
                        print(f"[DEBUG] 이미지 URL: {img_url}")
                        img_response = requests.get(img_url, timeout=30)
                        if img_response.status_code == 200:
                            generated_image = Image.open(io.BytesIO(img_response.content))
                            return generated_image
                    
                    # base64 데이터가 있는 경우
                    if "b64_json" in image_data_item:
                        image_data = base64.b64decode(image_data_item["b64_json"])
                        generated_image = Image.open(io.BytesIO(image_data))
                        return generated_image
                    
                    # 다른 형식의 응답
                    print(f"⚠️  API 응답에 이미지 데이터가 없습니다: {result}")
                    print(f"⚠️  응답 구조: {list(image_data_item.keys())}")
                else:
                    print(f"⚠️  API 응답에 data 필드가 없습니다: {result}")
            else:
                print(f"❌ Grok API 호출 실패: {response.status_code}")
                print(f"응답: {response.text}")
                
                # 403 에러인 경우 크레딧 문제일 수 있음
                if response.status_code == 403:
                    try:
                        error_data = response.json()
                        error_msg = str(error_data.get("error", "")).lower()
                        if "credits" in error_msg or "credit" in error_msg:
                            print("⚠️  xAI 계정에 크레딧이 없습니다.")
                            print("⚠️  https://console.x.ai 에서 크레딧을 구매해주세요.")
                    except:
                        pass
                
                # 400 에러인 경우 파라미터 문제
                if response.status_code == 400:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", "")
                        print(f"⚠️  API 요청 파라미터 오류: {error_msg}")
                        print(f"⚠️  사용된 payload: {payload}")
                    except:
                        pass
                
        except Exception as e:
            print(f"❌ Grok API 호출 중 오류: {e}")
            import traceback
            traceback.print_exc()
        
        return None
    
    def _call_grok_vision_api(self, input_image: Image.Image, prompt: str) -> Optional[Image.Image]:
        """
        Grok Vision API를 호출하여 원본 이미지를 입력으로 받고 얼굴 보존 전신 이미지 생성
        
        grok-2-vision-1212 모델을 사용하여 이미지와 텍스트 프롬프트로 이미지 생성
        
        Args:
            input_image: 원본 이미지 (PIL Image)
            prompt: 이미지 생성 프롬프트 텍스트
        
        Returns:
            생성된 이미지 (PIL Image) 또는 None (실패 시)
        """
        if not self.api_key:
            print("❌ GROK_API_KEY가 설정되지 않았습니다.")
            return None
        
        try:
            # 이미지를 base64로 인코딩
            buffered = io.BytesIO()
            input_image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # API 요청 구성
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Chat Completions 엔드포인트 사용 (Vision 모델)
            # grok-2-vision-1212는 이미지와 텍스트를 모두 입력으로 받을 수 있음
            vision_model = self.vision_model_name if hasattr(self, 'vision_model_name') else "grok-2-vision-1212"
            
            payload = {
                "model": vision_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ]
            }
            
            # API 호출 - Chat Completions 엔드포인트 사용
            timeout = GROK_API_TIMEOUT if CONFIG_AVAILABLE else 120
            response = requests.post(
                f"{self.api_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"[DEBUG] Grok Vision API 응답 구조: {list(result.keys())}")
                
                # Chat Completions 응답 형식 확인
                # 가능한 형식:
                # 1. {"choices": [{"message": {"content": "..."}}]} - 텍스트 설명
                # 2. {"choices": [{"message": {"content": [{"type": "image_url", "image_url": {"url": "..."}}]}}]} - 이미지 URL
                # 3. {"data": [{"url": "..."}]} - 직접 이미지 URL
                
                # 먼저 choices 확인
                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        content = choice["message"]["content"]
                        
                        # 텍스트인 경우
                        if isinstance(content, str):
                            print(f"⚠️  Grok Vision API가 텍스트를 반환했습니다: {content[:200]}...")
                            print("⚠️  이미지 생성이 아닌 텍스트 설명이 반환되었습니다.")
                            return None
                        
                        # 리스트인 경우 (멀티모달)
                        if isinstance(content, list):
                            for item in content:
                                if item.get("type") == "image_url":
                                    image_url = item["image_url"].get("url")
                                    if image_url:
                                        if image_url.startswith("data:image"):
                                            # base64 데이터
                                            header, encoded = image_url.split(",", 1)
                                            image_data = base64.b64decode(encoded)
                                            generated_image = Image.open(io.BytesIO(image_data))
                                            return generated_image
                                        else:
                                            # URL
                                            img_response = requests.get(image_url, timeout=30)
                                            if img_response.status_code == 200:
                                                generated_image = Image.open(io.BytesIO(img_response.content))
                                                return generated_image
                
                # data 필드 확인 (직접 이미지 URL)
                if "data" in result and len(result["data"]) > 0:
                    image_data_item = result["data"][0]
                    if "url" in image_data_item:
                        img_url = image_data_item["url"]
                        img_response = requests.get(img_url, timeout=30)
                        if img_response.status_code == 200:
                            generated_image = Image.open(io.BytesIO(img_response.content))
                            return generated_image
                
                print(f"⚠️  API 응답에 이미지 데이터가 없습니다: {result}")
                return None
            else:
                print(f"❌ Grok Vision API 호출 실패: {response.status_code}")
                print(f"응답: {response.text}")
                
                # 403 에러인 경우 크레딧 문제일 수 있음
                if response.status_code == 403:
                    try:
                        error_data = response.json()
                        error_msg = str(error_data.get("error", "")).lower()
                        if "credits" in error_msg or "credit" in error_msg:
                            print("⚠️  xAI 계정에 크레딧이 없습니다.")
                            print("⚠️  https://console.x.ai 에서 크레딧을 구매해주세요.")
                    except:
                        pass
                
                # 400 에러인 경우 파라미터 문제
                if response.status_code == 400:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", "")
                        print(f"⚠️  API 요청 파라미터 오류: {error_msg}")
                    except:
                        pass
        
        except Exception as e:
            print(f"❌ Grok Vision API 호출 중 오류: {e}")
            import traceback
            traceback.print_exc()
        
        return None
    
    def _extract_face_neck_with_segmentation(
        self, 
        input_image: Image.Image,
        segmentation_model,
        segmentation_processor
    ) -> Optional[Image.Image]:
        """
        SegFormer를 사용하여 얼굴+목 영역만 추출 (배경 제거)
        
        Args:
            input_image: 원본 이미지
            segmentation_model: SegFormer 모델
            segmentation_processor: SegFormer 프로세서
        
        Returns:
            얼굴+목 영역만 추출된 이미지 (RGBA, 배경 투명) 또는 None
        """
        try:
            import torch
            import torch.nn as nn
            import numpy as np
            
            print("[1/4] 얼굴+목 영역 감지 중...")
            face_bbox = self._detect_face_and_neck(input_image)
            
            if not face_bbox:
                print("⚠️  얼굴을 감지할 수 없습니다. 이미지 상단 40%를 얼굴+목+어깨로 사용합니다.")
                img_width, img_height = input_image.size
                face_bbox = (0, 0, img_width, int(img_height * 0.4))
            
            print(f"[DEBUG] 얼굴+목 영역 좌표: x={face_bbox[0]}, y={face_bbox[1]}, width={face_bbox[2]}, height={face_bbox[3]}")
            
            # 얼굴+목 영역 확장 (세그멘테이션을 위해 더 넓게)
            x, y, width, height = face_bbox
            img_width, img_height = input_image.size
            
            # 영역 확장 (20% 여유)
            expand_ratio = 0.2
            x = max(0, int(x - width * expand_ratio))
            y = max(0, int(y - height * expand_ratio))
            width = min(img_width - x, int(width * (1 + expand_ratio * 2)))
            height = min(img_height - y, int(height * (1 + expand_ratio * 2)))
            
            # 얼굴+목 영역 크롭
            face_neck_region = input_image.crop((x, y, x + width, y + height))
            
            print("[2/4] SegFormer로 얼굴+목 영역 세그멘테이션 중...")
            
            # SegFormer로 세그멘테이션
            inputs = segmentation_processor(images=face_neck_region, return_tensors="pt")
            
            with torch.no_grad():
                outputs = segmentation_model(**inputs)
                logits = outputs.logits.cpu()
            
            # 업샘플링
            upsampled_logits = nn.functional.interpolate(
                logits,
                size=face_neck_region.size[::-1],
                mode="bilinear",
                align_corners=False,
            )
            
            # 세그멘테이션 마스크 생성
            pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
            
            # 얼굴(11), 머리(2), 목 영역 추출
            # 레이블: 1=모자, 2=머리, 11=얼굴, 4=상의(목 부분 포함 가능)
            face_neck_mask = np.zeros_like(pred_seg, dtype=bool)
            face_neck_mask |= (pred_seg == 11)  # 얼굴
            face_neck_mask |= (pred_seg == 2)   # 머리
            face_neck_mask |= (pred_seg == 1)   # 모자 (선택적)
            
            # 목 영역 추출: 얼굴 아래 부분 + 상의 상단 부분
            face_pixels = np.where(pred_seg == 11)
            if len(face_pixels[0]) > 0:
                face_bottom = np.max(face_pixels[0])
                face_center_x = int(np.mean(face_pixels[1]))
                
                # 얼굴 하단에서 아래로 확장하여 목 포함
                # 얼굴 높이의 50%만큼 아래로 확장
                face_height = np.max(face_pixels[0]) - np.min(face_pixels[0])
                neck_extension = int(face_height * 0.5)
                
                if neck_extension > 0:
                    neck_start = min(face_bottom + 1, face_neck_region.height - 1)
                    neck_end = min(face_bottom + neck_extension, face_neck_region.height)
                    
                    # 목 영역: 얼굴 중심에서 좌우로 확장 (얼굴 폭의 80%)
                    face_width = np.max(face_pixels[1]) - np.min(face_pixels[1])
                    neck_width = int(face_width * 0.8)
                    neck_left = max(0, face_center_x - neck_width // 2)
                    neck_right = min(face_neck_region.width, face_center_x + neck_width // 2)
                    
                    face_neck_mask[neck_start:neck_end, neck_left:neck_right] = True
                
                # 상의(4) 레이블이 얼굴 근처에 있으면 목으로 간주
                upper_clothes_pixels = np.where(pred_seg == 4)
                if len(upper_clothes_pixels[0]) > 0:
                    upper_top = np.min(upper_clothes_pixels[0])
                    # 상의 상단이 얼굴 하단 근처(20% 이내)에 있으면 목 영역으로 포함
                    if abs(upper_top - face_bottom) < face_height * 0.2:
                        # 얼굴 하단부터 상의 상단까지를 목으로 간주
                        neck_area_top = face_bottom
                        neck_area_bottom = min(upper_top + int(face_height * 0.3), face_neck_region.height)
                        face_neck_mask[neck_area_top:neck_area_bottom, :] = True
            
            # 마스크를 uint8로 변환
            mask = face_neck_mask.astype(np.uint8) * 255
            
            # RGBA 이미지 생성 (배경 투명)
            image_array = np.array(face_neck_region)
            result_image = np.zeros((image_array.shape[0], image_array.shape[1], 4), dtype=np.uint8)
            result_image[:, :, :3] = image_array  # RGB 채널
            result_image[:, :, 3] = mask          # 알파 채널 (마스크)
            
            # PIL 이미지로 변환
            face_neck_extracted = Image.fromarray(result_image, mode='RGBA')
            
            print(f"[DEBUG] 얼굴+목 추출 완료 - 크기: {face_neck_extracted.size}, 모드: {face_neck_extracted.mode}")
            
            return face_neck_extracted
            
        except Exception as e:
            print(f"⚠️  얼굴+목 추출 오류: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_full_body(
        self, 
        input_image: Image.Image,
        preserve_face: bool = True,
        segmentation_model=None,
        segmentation_processor=None
    ) -> Optional[Image.Image]:
        """
        얼굴 사진에서 얼굴+목만 추출하고 체형 이미지에 합성
        
        프로세스:
        1. 얼굴+목 영역 감지
        2. SegFormer로 얼굴+목만 추출 (배경 제거, 누끼 처리)
        3. Grok API로 전신 체형 생성
        4. 얼굴+목을 체형 위에 합성
        
        Args:
            input_image: 입력 이미지 (얼굴 사진)
            preserve_face: 얼굴 보존 여부 (기본값: True)
            segmentation_model: SegFormer 모델 (얼굴+목 추출용)
            segmentation_processor: SegFormer 프로세서
        
        Returns:
            합성된 전신 이미지 (PIL Image) 또는 None (실패 시)
        """
        if not preserve_face:
            # 얼굴 보존하지 않는 경우 체형만 생성 (입력 이미지 참조)
            return self._generate_body_only(input_image)
        
        try:
            # 1. 얼굴+목 영역 추출 (배경 제거)
            if segmentation_model is None or segmentation_processor is None:
                print("⚠️  SegFormer 모델이 제공되지 않았습니다. 얼굴+목 추출을 건너뜁니다.")
                return self._generate_body_only(input_image)
            
            print("[1/4] 얼굴+목 영역 추출 중 (배경 제거)...")
            face_neck_extracted = self._extract_face_neck_with_segmentation(
                input_image,
                segmentation_model,
                segmentation_processor
            )
            
            if face_neck_extracted is None:
                print("⚠️  얼굴+목 추출 실패. 체형만 생성합니다 (입력 이미지 참조).")
                return self._generate_body_only(input_image)
            
            # 2. 체형 이미지 생성 (입력 이미지를 참조로 사용)
            print("[3/4] Grok API로 전신 체형 생성 중 (얼굴+목 이미지 참조)...")
            body_image = self._generate_body_only(input_image)
            
            if body_image is None:
                print("❌ 전신 체형 생성 실패")
                return None
            
            # 3. 얼굴+목을 체형 위에 합성
            print("[4/4] 얼굴+목과 체형 합성 중...")
            result = self._composite_face_neck_on_body(face_neck_extracted, body_image)
            
            if result is None:
                print("⚠️  합성 실패. 체형만 반환합니다.")
                return body_image
            
            print("✅ 전신 이미지 생성 완료")
            return result
            
        except Exception as e:
            print(f"❌ 전신 이미지 생성 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_body_only(self, input_image: Optional[Image.Image] = None) -> Optional[Image.Image]:
        """
        체형 생성 (얼굴+목 이미지를 참조로 사용)
        
        프롬프트 내용 (한국어 번역):
        - 제공된 얼굴+목 사진을 상단 참조로 사용
        - 원본 얼굴과 목을 정확히 그대로 유지 (픽셀, 정체성, 조명 변경 없음)
        - 목 하단부터 어깨부터 발까지 슬림하고 현실적인 전신 생성
        - 어깨와 상체가 기존 목 영역에서 자연스럽게 연결되도록
        - 목과 생성된 어깨/가슴 사이에 이음새나 불일치가 없도록 (피부톤, 조명, 관점 일치)
        - 중립적인 정면 서있는 자세
        - 흰색 타이트한 티셔츠와 파란 청바지
        - 깔끔한 자연 조명과 단순한 배경 (체형이 명확히 보이도록)
        
        Args:
            input_image: 입력 이미지 (얼굴+목 사진, None이면 텍스트-투-이미지)
        
        프롬프트 길이 제한: 1024자 이하
        """
        prompt = """Use the provided face-and-neck photo as the upper reference. Keep the original face and neck exactly as they are, without changing their pixels, identity, or lighting. Starting from the bottom of the neck in this image, generate a slim, realistic full body from the shoulders down to the feet. The shoulders and upper torso must grow naturally out of the existing neck area, so that the neck, collarbone, and shoulders form a smooth, continuous connection. There should be no visible seam, cut, or mismatch in skin tone, lighting, or perspective where the neck meets the generated shoulders and chest. The person is standing in a neutral, front-facing pose. Dress the body in a plain white fitted T-shirt and simple blue jeans. Use clean, natural lighting and a simple background so the overall body shape is clearly visible."""
        
        # 입력 이미지가 제공된 경우 이미지-투-이미지 생성
        if input_image is not None:
            return self._call_grok_vision_api(input_image, prompt)
        else:
            # 입력 이미지가 없는 경우 텍스트-투-이미지 생성
            return self._call_grok_image_api(prompt)
    
    def _composite_face_neck_on_body(
        self, 
        face_neck_extracted: Image.Image, 
        body_image: Image.Image
    ) -> Optional[Image.Image]:
        """
        얼굴+목(배경 제거된 RGBA)을 체형 이미지 위에 합성
        
        Args:
            face_neck_extracted: 얼굴+목 영역 (RGBA, 배경 투명)
            body_image: 생성된 체형 이미지 (RGB)
        
        Returns:
            합성된 전신 이미지 (RGB)
        """
        try:
            # RGB 모드로 변환
            if body_image.mode != 'RGB':
                body_image = body_image.convert('RGB')
            
            # RGBA 모드 확인
            if face_neck_extracted.mode != 'RGBA':
                face_neck_extracted = face_neck_extracted.convert('RGBA')
            
            # 체형 이미지 크기로 결과 이미지 생성
            result = body_image.copy()
            body_width, body_height = body_image.size
            
            # 얼굴+목 영역 크기 조정 (체형 이미지의 어깨 영역에 맞춤)
            # 체형 이미지 상단(어깨 위치)에 배치
            target_y = 0  # 상단부터 시작 (목 부분이 없으므로)
            
            # 얼굴+목 크기를 체형 이미지 너비에 맞춤
            # 얼굴+목이 체형 너비의 20-25% 정도가 되도록 조정
            target_width_ratio = 0.22  # 체형 너비의 22%
            target_width = int(body_width * target_width_ratio)
            
            # 비율 유지하며 높이 계산
            face_neck_ratio = face_neck_extracted.width / face_neck_extracted.height
            target_height = int(target_width / face_neck_ratio)
            
            # 리사이즈
            face_neck_resized = face_neck_extracted.resize(
                (target_width, target_height),
                Image.Resampling.LANCZOS
            )
            
            # 중앙 정렬
            target_x = (body_width - target_width) // 2
            
            # 이미지 범위 확인
            if target_x < 0:
                target_x = 0
            if target_y < 0:
                target_y = 0
            if target_x + target_width > body_width:
                target_width = body_width - target_x
                face_neck_resized = face_neck_resized.resize(
                    (target_width, target_height),
                    Image.Resampling.LANCZOS
                )
            if target_y + target_height > body_height:
                target_height = body_height - target_y
                face_neck_resized = face_neck_resized.resize(
                    (target_width, target_height),
                    Image.Resampling.LANCZOS
                )
            
            print(f"[DEBUG] 합성 위치 - x: {target_x}, y: {target_y}, 크기: {face_neck_resized.size}")
            
            # RGBA 이미지를 RGB 배경에 합성 (알파 채널 사용)
            result.paste(face_neck_resized, (target_x, target_y), face_neck_resized)
            
            print(f"[DEBUG] 합성 완료 - 결과 이미지 크기: {result.size}")
            
            return result
            
        except Exception as e:
            print(f"❌ 합성 오류: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def is_available(self) -> bool:
        """서비스 사용 가능 여부 확인"""
        return self.api_key is not None and self.api_key != ""


def generate_full_body_from_upper(
    upper_body_image: Image.Image,
    preserve_face: bool = True
) -> Optional[Image.Image]:
    """
    상반신 이미지에서 전신 이미지 생성 (편의 함수)
    
    Args:
        upper_body_image: 상반신 이미지 (PIL Image)
        preserve_face: 얼굴과 목 부분 보존 여부
    
    Returns:
        생성된 전신 이미지 (PIL Image) 또는 None (실패 시)
    """
    service = BodyGenerationService()
    if not service.is_available():
        print("⚠️  Grok API 서비스를 사용할 수 없습니다.")
        return None
    
    return service.generate_full_body(upper_body_image, preserve_face)

