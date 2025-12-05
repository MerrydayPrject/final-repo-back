"""
포즈 랜드마크 서비스
HuggingFace Spaces에 배포된 MediaPipe Pose API를 사용하여 포즈 랜드마크 추출
"""
import os
import io
import requests
from PIL import Image
from typing import Optional, List, Dict
from config.settings import MEDIAPIPE_SPACE_URL


class PoseLandmarkService:
    """포즈 랜드마크 서비스 (HuggingFace Spaces API 기반)"""
    
    def __init__(self, space_url: Optional[str] = None):
        """
        초기화
        
        Args:
            space_url: HuggingFace Spaces URL (None이면 설정에서 가져옴)
        """
        self.space_url = space_url or MEDIAPIPE_SPACE_URL
        self.is_initialized = True  # API 서비스는 항상 초기화됨
        print(f"[PoseLandmarkService] 초기화 완료 - Space URL: {self.space_url}")
    
    
    def extract_landmarks(self, image: Image.Image) -> Optional[List[Dict]]:
        """
        이미지에서 포즈 랜드마크 추출
        
        Args:
            image: PIL Image 객체
            
        Returns:
            랜드마크 좌표 리스트 (33개 포인트) 또는 None
        """
        if not self.is_initialized:
            print("[PoseLandmarkService] ⚠️ 서비스가 초기화되지 않았습니다.")
            return None
        
        try:
            # 이미지가 너무 크면 리사이즈
            max_size = 1920
            if image.width > max_size or image.height > max_size:
                ratio = min(max_size / image.width, max_size / image.height)
                new_width = int(image.width * ratio)
                new_height = int(image.height * ratio)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 이미지를 바이트로 변환
            buffered = io.BytesIO()
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            image.save(buffered, format="JPEG", quality=95)
            img_bytes = buffered.getvalue()
            
            # FastAPI 엔드포인트 URL 구성
            api_url = f"{self.space_url}/analyze_pose" if not self.space_url.endswith("/analyze_pose") else self.space_url
            
            # FastAPI 파일 업로드 형식으로 요청
            files = {
                "image": ("image.jpg", img_bytes, "image/jpeg")
            }
            
            # API 호출
            response = requests.post(
                api_url,
                files=files,
                timeout=30
            )
            
            if response.status_code != 200:
                return None
            
            # 응답 파싱
            try:
                result = response.json()
            except Exception as e:
                return None
            
            # FastAPI 응답 형식: {"landmarks": [...]} 또는 {"error": "..."} 또는 {"people": [...]}
            if "error" in result:
                print(f"[PoseLandmarkService] ❌ API 오류 응답: {result['error']}")
                return None
            
            # 여러 사람 감지 지원: 여러 형식 처리
            # 형식 1: {"people": [{"landmarks": [...]}, ...]}
            if "people" in result and isinstance(result["people"], list):
                print(f"[PoseLandmarkService] 여러 사람 감지됨 (people 형식): {len(result['people'])}명")
                all_landmarks_list = []
                for person in result["people"]:
                    if "landmarks" in person and person["landmarks"] is not None:
                        all_landmarks_list.append(person["landmarks"])
                if all_landmarks_list:
                    best_landmarks = self._select_best_person(all_landmarks_list)
                    if best_landmarks:
                        return best_landmarks
            
            # 형식 2: {"detections": [{"landmarks": [...]}, ...]} 또는 {"pose_landmarks": [[...], [...]]}
            if "detections" in result and isinstance(result["detections"], list):
                print(f"[PoseLandmarkService] 여러 사람 감지됨 (detections 형식): {len(result['detections'])}명")
                all_landmarks_list = []
                for detection in result["detections"]:
                    if "landmarks" in detection and detection["landmarks"] is not None:
                        all_landmarks_list.append(detection["landmarks"])
                if all_landmarks_list:
                    best_landmarks = self._select_best_person(all_landmarks_list)
                    if best_landmarks:
                        return best_landmarks
            
            if "pose_landmarks" in result and isinstance(result["pose_landmarks"], list):
                pose_landmarks_data = result["pose_landmarks"]
                if len(pose_landmarks_data) > 0 and isinstance(pose_landmarks_data[0], list):
                    print(f"[PoseLandmarkService] 여러 사람 감지됨 (pose_landmarks 형식): {len(pose_landmarks_data)}명")
                    best_landmarks = self._select_best_person(pose_landmarks_data)
                    if best_landmarks:
                        return best_landmarks
            
            # 여러 사람 감지 지원: landmarks가 리스트의 리스트일 수 있음
            if "landmarks" in result:
                landmarks_data = result["landmarks"]
                
                if landmarks_data is None:
                    return None
                
                # 여러 사람이 감지된 경우 (리스트의 리스트)
                if isinstance(landmarks_data, list) and len(landmarks_data) > 0:
                    # 첫 번째 요소가 리스트인지 확인 (여러 사람)
                    if isinstance(landmarks_data[0], list):
                        # 여러 사람 중 가장 적합한 사람 선택
                        best_landmarks = self._select_best_person(landmarks_data)
                        if best_landmarks:
                            return best_landmarks
                        else:
                            # 선택 실패 시 첫 번째 사람 사용
                            landmarks = landmarks_data[0]
                    else:
                        # 단일 사람
                        landmarks = landmarks_data
                else:
                    return None
                
                # 랜드마크 형식 변환
                formatted_landmarks = []
                for idx, landmark in enumerate(landmarks):
                    if isinstance(landmark, dict):
                        formatted_landmarks.append({
                            "id": landmark.get("id", idx),
                            "x": landmark.get("x", 0.0),
                            "y": landmark.get("y", 0.0),
                            "z": landmark.get("z", 0.0),
                            "visibility": landmark.get("visibility", 1.0)
                        })
                    elif isinstance(landmark, (list, tuple)) and len(landmark) >= 2:
                        # [x, y] 또는 [x, y, z] 형식
                        formatted_landmarks.append({
                            "id": idx,
                            "x": float(landmark[0]),
                            "y": float(landmark[1]),
                            "z": float(landmark[2]) if len(landmark) > 2 else 0.0,
                            "visibility": float(landmark[3]) if len(landmark) > 3 else 1.0
                        })
                
                if formatted_landmarks:
                    return formatted_landmarks
                else:
                    return None
            
            return None
            
        except Exception as e:
            return None
    
    def _select_best_person(self, all_landmarks: List[List]) -> Optional[List[Dict]]:
        """
        여러 사람 중에서 가장 적합한 사람 선택
        - 전신 랜드마크가 있는 사람 (33개)
        - 가장 크게 보이는 사람 (랜드마크 bounding box 크기)
        
        Args:
            all_landmarks: 여러 사람의 랜드마크 리스트
            
        Returns:
            가장 적합한 사람의 랜드마크 또는 None
        """
        if not all_landmarks:
            return None
        
        best_person = None
        best_score = -1
        
        for person_idx, landmarks in enumerate(all_landmarks):
            # 랜드마크 형식 변환
            formatted_landmarks = []
            for idx, landmark in enumerate(landmarks):
                if isinstance(landmark, dict):
                    formatted_landmarks.append({
                        "id": landmark.get("id", idx),
                        "x": landmark.get("x", 0.0),
                        "y": landmark.get("y", 0.0),
                        "z": landmark.get("z", 0.0),
                        "visibility": landmark.get("visibility", 1.0)
                    })
                elif isinstance(landmark, (list, tuple)) and len(landmark) >= 2:
                    formatted_landmarks.append({
                        "id": idx,
                        "x": float(landmark[0]),
                        "y": float(landmark[1]),
                        "z": float(landmark[2]) if len(landmark) > 2 else 0.0,
                        "visibility": float(landmark[3]) if len(landmark) > 3 else 1.0
                    })
            
            if not formatted_landmarks:
                continue
            
            # 1. 전신 랜드마크 확인 (33개 이상, 주요 랜드마크 visibility 확인)
            required_landmarks = [11, 12, 23, 24, 27, 28]  # 어깨, 엉덩이, 발목
            has_full_body = len(formatted_landmarks) >= 33
            
            # 주요 랜드마크의 visibility 확인
            visible_key_landmarks = 0
            for req_id in required_landmarks:
                if req_id < len(formatted_landmarks):
                    landmark = formatted_landmarks[req_id]
                    if landmark.get("visibility", 0) >= 0.3:
                        visible_key_landmarks += 1
            
            # 전신 랜드마크가 있고 주요 랜드마크가 최소 2개 이상 보이는 경우 선택 가능
            # (여러 사람이 있을 때도 선택할 수 있도록 완화)
            if not (has_full_body and visible_key_landmarks >= 2):
                print(f"[PoseLandmarkService] 사람 {person_idx}: 전신 랜드마크 부족 (개수: {len(formatted_landmarks)}, 주요 랜드마크: {visible_key_landmarks}/6)")
                # 전신이 아니어도 주요 랜드마크가 2개 이상이면 후보로 고려
                if visible_key_landmarks < 2:
                    continue
                # 전신이 아니지만 주요 랜드마크가 있으면 점수에 페널티 적용
                has_full_body = False
            
            # 2. 가장 크게 보이는 사람 선택 (랜드마크 bounding box 크기 계산)
            x_coords = [lm.get("x", 0) for lm in formatted_landmarks if lm.get("visibility", 0) >= 0.3]
            y_coords = [lm.get("y", 0) for lm in formatted_landmarks if lm.get("visibility", 0) >= 0.3]
            
            if not x_coords or not y_coords:
                continue
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            # bounding box 크기 (면적)
            width = max_x - min_x
            height = max_y - min_y
            area = width * height
            
            # 평균 visibility
            avg_visibility = sum(lm.get("visibility", 0) for lm in formatted_landmarks) / len(formatted_landmarks)
            
            # 점수 계산: 면적 * visibility * 전신 보너스
            full_body_bonus = 1.5 if has_full_body else 1.0
            score = area * avg_visibility * full_body_bonus
            
            if score > best_score:
                best_score = score
                best_person = formatted_landmarks
        
        if best_person:
            return best_person
        else:
            # 적합한 사람이 없으면 첫 번째 사람이라도 반환 (최소한의 랜드마크가 있으면)
            if all_landmarks and len(all_landmarks) > 0:
                first_landmarks = all_landmarks[0]
                # 형식 변환
                formatted = []
                for idx, landmark in enumerate(first_landmarks):
                    if isinstance(landmark, dict):
                        formatted.append({
                            "id": landmark.get("id", idx),
                            "x": landmark.get("x", 0.0),
                            "y": landmark.get("y", 0.0),
                            "z": landmark.get("z", 0.0),
                            "visibility": landmark.get("visibility", 1.0)
                        })
                    elif isinstance(landmark, (list, tuple)) and len(landmark) >= 2:
                        formatted.append({
                            "id": idx,
                            "x": float(landmark[0]),
                            "y": float(landmark[1]),
                            "z": float(landmark[2]) if len(landmark) > 2 else 0.0,
                            "visibility": float(landmark[3]) if len(landmark) > 3 else 1.0
                        })
                if formatted:
                    return formatted
            return None

