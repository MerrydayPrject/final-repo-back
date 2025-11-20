"""
MediaPipe Image Classifier 서비스
이미지 분류를 통해 사람 여부를 판단
"""
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
import numpy as np
from typing import Optional, List, Dict
from pathlib import Path
import urllib.request
import os


class ImageClassifierService:
    """이미지 분류 서비스"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        초기화
        
        Args:
            model_path: MediaPipe 모델 파일 경로 (None이면 자동 다운로드)
        """
        self.model_path = model_path
        self.classifier = None
        self.is_initialized = False
        
        # MediaPipe 초기화
        self._init_classifier()
    
    def _init_classifier(self):
        """MediaPipe Image Classifier 초기화"""
        try:
            # 모델 파일 경로 설정
            if self.model_path is None:
                model_path = self._get_default_model_path()
            else:
                model_path = Path(self.model_path)
            
            # 모델 파일이 없으면 다운로드
            if not model_path.exists():
                print("모델 파일이 없습니다. 다운로드를 시도합니다...")
                self._download_model(model_path)
            
            # Image Classifier 옵션 설정
            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.ImageClassifierOptions(
                base_options=base_options,
                max_results=10,  # 상위 10개 결과만 가져오기
                score_threshold=0.1  # 최소 신뢰도 0.1
            )
            
            # Image Classifier 생성
            self.classifier = vision.ImageClassifier.create_from_options(options)
            self.is_initialized = True
            print("✅ MediaPipe Image Classifier 초기화 완료!")
            
        except Exception as e:
            print(f"MediaPipe Image Classifier 초기화 오류: {e}")
            self.is_initialized = False
    
    def _get_default_model_path(self) -> Path:
        """기본 모델 경로 반환"""
        models_dir = Path(__file__).parent.parent / "models" / "image_classifier"
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir / "efficientnet_lite0.tflite"
    
    def _download_model(self, model_path: Path):
        """MediaPipe 모델 다운로드"""
        try:
            # EfficientNet-Lite0 모델 URL (float32)
            model_url = "https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/latest/efficientnet_lite0.tflite"
            
            print(f"모델 다운로드 중...")
            print(f"URL: {model_url}")
            print(f"저장 경로: {model_path}")
            
            urllib.request.urlretrieve(model_url, str(model_path))
            
            # 파일 크기 확인
            file_size = model_path.stat().st_size / (1024 * 1024)  # MB
            print(f"✅ 모델 다운로드 완료!")
            print(f"   파일 크기: {file_size:.2f} MB")
            print(f"   저장 위치: {model_path}")
            
        except Exception as e:
            print(f"❌ 모델 다운로드 실패: {e}")
            print(f"수동 다운로드: {model_url}")
            print(f"저장 위치: {model_path}")
            raise
    
    def classify_image(self, image: Image.Image) -> Optional[List[Dict]]:
        """
        이미지를 분류
        
        Args:
            image: PIL Image 객체
            
        Returns:
            분류 결과 리스트 [{"category_name": str, "score": float}, ...]
        """
        if not self.is_initialized:
            print("서비스가 초기화되지 않았습니다.")
            return None
        
        try:
            # PIL Image를 numpy array로 변환
            image_array = np.array(image)
            
            # RGB 형식으로 변환 (MediaPipe는 RGB 필요)
            if len(image_array.shape) == 3 and image_array.shape[2] == 4:
                # RGBA -> RGB
                image_array = image_array[:, :, :3]
            
            # MediaPipe 형식으로 변환
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
            
            # 이미지 분류
            classification_result = self.classifier.classify(mp_image)
            
            # 결과 추출
            if not classification_result.classifications:
                return None
            
            categories = []
            for category in classification_result.classifications[0].categories:
                categories.append({
                    "category_name": category.category_name,
                    "score": category.score
                })
            
            return categories
            
        except Exception as e:
            print(f"이미지 분류 오류: {e}")
            return None
    
    def is_person(self, image: Image.Image, threshold: float = 0.3) -> bool:
        """
        이미지에 사람이 있는지 판단
        
        Args:
            image: PIL Image 객체
            threshold: 사람 관련 클래스의 최소 신뢰도 (기본값: 0.3)
            
        Returns:
            사람이 있으면 True, 없으면 False
        """
        categories = self.classify_image(image)
        if not categories:
            print("❌ 이미지 분류 결과 없음")
            return False
        
        # 사람 관련 키워드 (정확한 매칭만 허용)
        person_keywords = [
            "person", "man", "woman", "girl", "boy", "child", "baby",
            "people", "human", "bride", "groom", "bridegroom",
            "lady", "gentleman", "adult", "teenager", "infant"
        ]
        
        # 동물 관련 키워드 (제외)
        animal_keywords = [
            "animal", "dog", "cat", "bear", "monkey", "ape", "gorilla",
            "orangutan", "chimpanzee", "elephant", "lion", "tiger",
            "bird", "fish", "horse", "cow", "pig", "sheep", "goat",
            "rabbit", "mouse", "rat", "hamster", "squirrel", "deer",
            "wolf", "fox", "panda", "koala", "kangaroo", "zebra",
            "giraffe", "camel", "donkey", "mule", "llama", "alpaca"
        ]
        
        # 상위 결과 확인
        print(f"이미지 분류 결과 (상위 5개):")
        for i, category in enumerate(categories[:5]):
            print(f"  {i+1}. {category['category_name']}: {category['score']:.2f}")
        
        # 상위 결과 중 사람 관련 클래스가 있는지 확인
        for category in categories:
            category_name_lower = category["category_name"].lower()
            score = category["score"]
            
            # 동물 관련 키워드가 포함되어 있으면 즉시 차단
            if any(keyword in category_name_lower for keyword in animal_keywords):
                print(f"❌ 동물 감지: {category['category_name']} (신뢰도: {score:.2f}) - 차단")
                return False
            
            # 사람 관련 키워드가 포함되어 있고 신뢰도가 임계값 이상이면 사람으로 판단
            if any(keyword in category_name_lower for keyword in person_keywords):
                if score >= threshold:
                    print(f"✅ 사람 감지: {category['category_name']} (신뢰도: {score:.2f})")
                    return True
        
        # 사람 관련 클래스가 없으면 차단
        top_category = categories[0]['category_name'] if categories else 'None'
        top_score = categories[0]['score'] if categories else 0
        print(f"❌ 사람 감지 실패 - 상위 분류: {top_category} (신뢰도: {top_score:.2f})")
        return False

