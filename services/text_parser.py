"""텍스트 프롬프트 파싱 서비스"""
import re
from typing import Dict, List, Tuple, Optional


class TextAdjustmentParser:
    """텍스트 프롬프트를 체형 조정 파라미터로 변환"""
    
    # 신체 부위 매핑 (한국어/영어)
    BODY_PARTS = {
        # 어깨
        '어깨': 'shoulder',
        '어깨가': 'shoulder',
        '어깨를': 'shoulder',
        'shoulder': 'shoulder',
        'shoulders': 'shoulder',
        
        # 허리
        '허리': 'waist',
        '허리가': 'waist',
        '허리를': 'waist',
        'waist': 'waist',
        
        # 엉덩이/골반
        '엉덩이': 'hip',
        '엉덩이가': 'hip',
        '엉덩이를': 'hip',
        '골반': 'hip',
        '골반을': 'hip',
        'hip': 'hip',
        'hips': 'hip',
        
        # 다리
        '다리': 'leg',
        '다리가': 'leg',
        '다리를': 'leg',
        'leg': 'leg',
        'legs': 'leg',
        
        # 팔/팔뚝
        '팔': 'arm',
        '팔을': 'arm',
        '팔뚝': 'arm',
        '팔뚝을': 'arm',
        '팔뚝살': 'arm',
        'arm': 'arm',
        'arms': 'arm',
        
        # 목
        '목': 'neck',
        '목을': 'neck',
        'neck': 'neck',
        
        # 가슴
        '가슴': 'chest',
        '가슴을': 'chest',
        'chest': 'chest',
        'bust': 'chest',
    }
    
    # 조정 방향 매핑
    DIRECTIONS = {
        # 넓게/넓히기
        '넓게': ('wider', 1.15),
        '넓히': ('wider', 1.15),
        '넓어': ('wider', 1.15),
        '크게': ('wider', 1.15),
        '키우': ('wider', 1.15),
        'wider': ('wider', 1.15),
        'broad': ('wider', 1.15),
        'bigger': ('wider', 1.15),
        
        # 좁게/가늘게
        '좁게': ('slimmer', 0.85),
        '좁히': ('slimmer', 0.85),
        '가늘게': ('slimmer', 0.85),
        '가늘': ('slimmer', 0.85),
        '날씬': ('slimmer', 0.85),
        '작게': ('slimmer', 0.85),
        'slimmer': ('slimmer', 0.85),
        'slim': ('slimmer', 0.85),
        'narrow': ('slimmer', 0.85),
        'thinner': ('slimmer', 0.85),
        'smaller': ('slimmer', 0.85),
        
        # 길게
        '길게': ('longer', 1.1),
        '길': ('longer', 1.1),
        '늘리': ('longer', 1.1),
        'longer': ('longer', 1.1),
        'elongate': ('longer', 1.1),
        
        # 짧게
        '짧게': ('shorter', 0.9),
        '짧': ('shorter', 0.9),
        'shorter': ('shorter', 0.9),
    }
    
    # 강도 배수
    INTENSITY = {
        '약간': 0.5,
        '조금': 0.5,
        'slightly': 0.5,
        'a bit': 0.5,
        
        '많이': 1.5,
        '크게': 1.5,
        'very': 1.5,
        'much': 1.5,
        'greatly': 1.5,
        
        '매우': 2.0,
        '아주': 2.0,
        'extremely': 2.0,
    }
    
    @staticmethod
    def parse(text: str) -> Dict[str, Dict[str, float]]:
        """
        텍스트 프롬프트를 파싱하여 조정 파라미터 추출
        
        Args:
            text: 입력 텍스트 (예: "어깨 넓게, 허리 가늘게")
            
        Returns:
            {
                'shoulder': {'factor': 1.15, 'direction': 'wider'},
                'waist': {'factor': 0.85, 'direction': 'slimmer'}
            }
        """
        result = {}
        
        # 쉼표, 그리고, and 등으로 분리
        commands = re.split(r'[,،、]+|\s+그리고\s+|\s+and\s+', text)
        
        for command in commands:
            command = command.strip().lower()
            if not command:
                continue
            
            # 신체 부위 찾기
            body_part = None
            for keyword, part in TextAdjustmentParser.BODY_PARTS.items():
                if keyword in command:
                    body_part = part
                    break
            
            if not body_part:
                continue
            
            # 방향 찾기
            direction = None
            base_factor = 1.0
            for keyword, (dir_name, factor) in TextAdjustmentParser.DIRECTIONS.items():
                if keyword in command:
                    direction = dir_name
                    base_factor = factor
                    break
            
            if not direction:
                continue
            
            # 강도 찾기
            intensity_multiplier = 1.0
            for keyword, multiplier in TextAdjustmentParser.INTENSITY.items():
                if keyword in command:
                    intensity_multiplier = multiplier
                    break
            
            # 최종 factor 계산
            if base_factor > 1.0:
                # 넓게/크게: 1.0 이상
                final_factor = 1.0 + (base_factor - 1.0) * intensity_multiplier
            else:
                # 좁게/작게: 1.0 이하
                final_factor = 1.0 - (1.0 - base_factor) * intensity_multiplier
            
            # 범위 제한
            final_factor = max(0.7, min(1.5, final_factor))
            
            result[body_part] = {
                'factor': final_factor,
                'direction': direction
            }
        
        return result
    
    @staticmethod
    def to_service_params(parsed: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        파싱된 결과를 서비스 함수 파라미터로 변환
        
        Args:
            parsed: parse() 결과
            
        Returns:
            {
                'shoulder_factor': 1.15,
                'waist_factor': 0.85,
                'hip_factor': 1.0,
                'leg_factor': 1.0,
                'arm_factor': 1.0,
                'neck_thickness_factor': 1.0,
                'chest_factor': 1.0
            }
        """
        return {
            'shoulder_factor': parsed.get('shoulder', {}).get('factor', 1.0),
            'waist_factor': parsed.get('waist', {}).get('factor', 1.0),
            'hip_factor': parsed.get('hip', {}).get('factor', 1.0),
            'leg_factor': parsed.get('leg', {}).get('factor', 1.0),
            'arm_factor': parsed.get('arm', {}).get('factor', 1.0),
            'neck_thickness_factor': parsed.get('neck', {}).get('factor', 1.0),
            'chest_factor': parsed.get('chest', {}).get('factor', 1.0),
        }


def parse_adjustment_text(text: str) -> Dict[str, float]:
    """
    편의 함수: 텍스트를 바로 서비스 파라미터로 변환
    
    Args:
        text: 입력 텍스트
        
    Returns:
        서비스 함수용 파라미터 딕셔너리
        
    Example:
        >>> parse_adjustment_text("어깨 넓게, 허리 가늘게")
        {'shoulder_factor': 1.15, 'waist_factor': 0.85, 'hip_factor': 1.0, 'leg_factor': 1.0}
    """
    parser = TextAdjustmentParser()
    parsed = parser.parse(text)
    return parser.to_service_params(parsed)
