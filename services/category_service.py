"""카테고리 규칙 관리"""
import json
from pathlib import Path
from typing import List, Optional


def load_category_rules() -> List[dict]:
    """카테고리 규칙 JSON 파일 로드"""
    rules_file = Path("category_rules.json")
    if not rules_file.exists():
        # 기본 규칙으로 파일 생성
        default_rules = [
            {"prefix": "A", "style": "A라인"},
            {"prefix": "Mini", "style": "미니드레스"},
            {"prefix": "B", "style": "벨라인"},
            {"prefix": "P", "style": "프린세스"}
        ]
        with open(rules_file, "w", encoding="utf-8") as f:
            json.dump(default_rules, f, ensure_ascii=False, indent=2)
        return default_rules
    
    try:
        with open(rules_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"카테고리 규칙 로드 오류: {e}")
        return []


def save_category_rules(rules: List[dict]) -> bool:
    """카테고리 규칙 JSON 파일 저장"""
    try:
        rules_file = Path("category_rules.json")
        with open(rules_file, "w", encoding="utf-8") as f:
            json.dump(rules, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"카테고리 규칙 저장 오류: {e}")
        return False


def detect_style_from_filename(filename: str) -> Optional[str]:
    """
    이미지 파일명에서 스타일을 감지 (JSON 규칙 기반)
    
    Args:
        filename: 이미지 파일명 (예: "Adress1.jpg", "Mini_dress.png")
    
    Returns:
        감지된 스타일 문자열 또는 None (감지 실패 시)
    """
    rules = load_category_rules()
    filename_upper = filename.upper()
    
    # 규칙을 우선순위대로 확인 (긴 prefix 우선)
    sorted_rules = sorted(rules, key=lambda x: len(x["prefix"]), reverse=True)
    
    for rule in sorted_rules:
        prefix_upper = rule["prefix"].upper()
        # prefix로 시작하거나 포함하는지 확인
        if filename_upper.startswith(prefix_upper) or prefix_upper in filename_upper:
            return rule["style"]
    
    return None

