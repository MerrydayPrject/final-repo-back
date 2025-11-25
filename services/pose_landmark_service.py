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
            print(f"[PoseLandmarkService] 이미지 처리 시작 - 이미지 모드: {image.mode}, 크기: {image.size}")
            
            # 이미지를 바이트로 변환
            buffered = io.BytesIO()
            if image.mode == 'RGBA':
                print("[PoseLandmarkService] RGBA 모드를 RGB로 변환")
                image = image.convert('RGB')
            image.save(buffered, format="JPEG", quality=95)
            img_bytes = buffered.getvalue()
            print(f"[PoseLandmarkService] 이미지 변환 완료 - 크기: {len(img_bytes)} bytes")
            
            # FastAPI 엔드포인트 URL 구성
            api_url = f"{self.space_url}/analyze_pose" if not self.space_url.endswith("/analyze_pose") else self.space_url
            print(f"[PoseLandmarkService] API 요청 URL: {api_url}")
            print(f"[PoseLandmarkService] Base Space URL: {self.space_url}")
            
            # FastAPI 파일 업로드 형식으로 요청
            files = {
                "image": ("image.jpg", img_bytes, "image/jpeg")
            }
            
            print(f"[PoseLandmarkService] API 요청 전송 중...")
            # API 호출
            response = requests.post(
                api_url,
                files=files,
                timeout=30
            )
            
            print(f"[PoseLandmarkService] API 응답 수신 - 상태 코드: {response.status_code}")
            print(f"[PoseLandmarkService] 응답 헤더: {dict(response.headers)}")
            
            if response.status_code != 200:
                print(f"[PoseLandmarkService] ❌ API 호출 실패: {response.status_code}")
                print(f"[PoseLandmarkService] 응답 본문: {response.text[:500]}")  # 처음 500자만 출력
                try:
                    error_json = response.json()
                    print(f"[PoseLandmarkService] 응답 JSON: {error_json}")
                except:
                    print(f"[PoseLandmarkService] JSON 파싱 실패 (텍스트 응답)")
                return None
            
            # 응답 파싱
            print(f"[PoseLandmarkService] 응답 파싱 시작...")
            try:
                result = response.json()
                print(f"[PoseLandmarkService] 응답 JSON 파싱 성공 - 키: {list(result.keys()) if isinstance(result, dict) else '리스트'}")
            except Exception as e:
                print(f"[PoseLandmarkService] ❌ JSON 파싱 실패: {e}")
                print(f"[PoseLandmarkService] 응답 텍스트: {response.text[:500]}")
                return None
            
            # FastAPI 응답 형식: {"landmarks": [...]} 또는 {"error": "..."}
            if "error" in result:
                print(f"[PoseLandmarkService] ❌ API 오류 응답: {result['error']}")
                return None
            
            if "landmarks" in result:
                landmarks = result["landmarks"]
                print(f"[PoseLandmarkService] 랜드마크 데이터 발견 - 타입: {type(landmarks)}")
                
                if landmarks is None:
                    print(f"[PoseLandmarkService] ⚠️ 랜드마크가 None입니다")
                    return None
                
                if not isinstance(landmarks, list):
                    print(f"[PoseLandmarkService] ⚠️ 랜드마크가 리스트가 아닙니다: {type(landmarks)}")
                    return None
                
                print(f"[PoseLandmarkService] 랜드마크 개수: {len(landmarks)}")
                
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
                    else:
                        print(f"[PoseLandmarkService] ⚠️ 알 수 없는 랜드마크 형식 (인덱스 {idx}): {type(landmark)}")
                
                if formatted_landmarks:
                    print(f"[PoseLandmarkService] ✅ 랜드마크 추출 성공 - {len(formatted_landmarks)}개 포인트")
                    return formatted_landmarks
                else:
                    print(f"[PoseLandmarkService] ⚠️ 포맷팅된 랜드마크가 없습니다")
                    return None
            
            # 다른 형식의 응답 처리
            print(f"[PoseLandmarkService] ⚠️ 예상하지 못한 응답 형식")
            print(f"[PoseLandmarkService] 응답 전체: {result}")
            return None
            
        except requests.exceptions.Timeout as e:
            print(f"[PoseLandmarkService] ❌ API 요청 타임아웃: {e}")
            print(f"[PoseLandmarkService] 요청 URL: {api_url if 'api_url' in locals() else 'N/A'}")
            return None
        except requests.exceptions.ConnectionError as e:
            print(f"[PoseLandmarkService] ❌ API 연결 오류: {e}")
            print(f"[PoseLandmarkService] 요청 URL: {api_url if 'api_url' in locals() else 'N/A'}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"[PoseLandmarkService] ❌ API 요청 오류: {type(e).__name__}: {e}")
            print(f"[PoseLandmarkService] 요청 URL: {api_url if 'api_url' in locals() else 'N/A'}")
            import traceback
            traceback.print_exc()
            return None
        except Exception as e:
            print(f"[PoseLandmarkService] ❌ 포즈 랜드마크 추출 오류: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None

