"""
이미지 분류 서비스
MediaPipe 의존성 제거됨 - 현재 비활성화 상태

참고: 이미지 분류 기능이 필요한 경우 HuggingFace Inference API를 사용하도록 변경 가능
"""
from PIL import Image
from typing import Optional, List, Dict


class ImageClassifierService:
    """이미지 분류 서비스 (현재 비활성화)"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        초기화
        
        Args:
            model_path: 모델 파일 경로 (사용하지 않음)
        """
        self.model_path = model_path
        self.is_initialized = False
        print("⚠️  ImageClassifierService는 MediaPipe 의존성 제거로 인해 비활성화되었습니다.")
        print("   이미지 분류가 필요한 경우 HuggingFace Inference API를 사용하도록 변경하세요.")
    
    def classify_image(self, image: Image.Image) -> Optional[List[Dict]]:
        """
        이미지를 분류 (현재 비활성화)
        
        Args:
            image: PIL Image 객체
            
        Returns:
            None (서비스 비활성화)
        """
        print("⚠️  이미지 분류 서비스가 비활성화되었습니다.")
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

