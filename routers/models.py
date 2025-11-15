"""모델 관리 라우터"""
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import json
from pathlib import Path

router = APIRouter()


@router.get("/api/models", tags=["모델 관리"])
async def get_models():
    """
    사용 가능한 AI 모델 목록 조회
    
    models_config.json 파일에서 모델 목록을 읽어 반환합니다.
    """
    try:
        config_file = Path("models_config.json")
        if not config_file.exists():
            return JSONResponse({
                "success": False,
                "error": "Config file not found",
                "message": "models_config.json 파일이 없습니다."
            }, status_code=404)
        
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # models_config.json 구조: {"models": [...]}
        models = config.get("models", [])
        
        return JSONResponse({
            "success": True,
            "models": models
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"모델 목록 조회 중 오류: {str(e)}"
        }, status_code=500)


@router.post("/api/models", tags=["모델 관리"])
async def add_model(model_data: dict):
    """
    새로운 모델 추가
    
    models_config.json 파일에 모델 정보를 추가합니다.
    """
    try:
        config_file = Path("models_config.json")
        
        # 기존 모델 목록 읽기
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
                models = config.get("models", [])
        else:
            models = []
        
        # 새 모델 추가
        models.append(model_data)
        
        # 파일 저장 (models_config.json 구조 유지: {"models": [...]})
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump({"models": models}, f, ensure_ascii=False, indent=2)
        
        return JSONResponse({
            "success": True,
            "message": "모델이 추가되었습니다.",
            "models": models
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"모델 추가 중 오류: {str(e)}"
        }, status_code=500)

