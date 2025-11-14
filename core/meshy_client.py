"""Meshy.ai 3D 변환 클라이언트"""
import os
import base64
import requests
import traceback
from pathlib import Path
from typing import Dict, Optional

from config.settings import MESHY_API_KEY, MESHY_API_URL


def create_3d_model_meshy(image_bytes: bytes) -> Dict:
    """
    Meshy.ai API를 사용하여 이미지를 3D 모델로 변환
    
    Args:
        image_bytes: 이미지 바이트 데이터
    
    Returns:
        dict: 작업 정보 (task_id, status 등)
    """
    if not MESHY_API_KEY:
        error_msg = (
            "MESHY_API_KEY가 설정되지 않았습니다!\n\n"
            "해결 방법:\n"
            "1. final-repo-back/.env 파일 생성\n"
            "2. 다음 줄 추가: MESHY_API_KEY=msy_your_api_key_here\n"
            "3. https://www.meshy.ai 에서 API 키 발급\n"
            "4. 서버 재시작"
        )
        print(f"[Meshy API] 오류: {error_msg}")
        raise ValueError(error_msg)
    
    # 이미지를 base64 data URI로 변환
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    # 이미지 포맷 감지 (파일 시그니처 기반)
    image_format = "png"
    if image_bytes[:2] == b'\xff\xd8':  # JPEG magic number
        image_format = "jpeg"
    elif image_bytes[:4] == b'\x89PNG':  # PNG magic number
        image_format = "png"
    
    data_uri = f"data:image/{image_format};base64,{base64_image}"
    
    headers = {
        "Authorization": f"Bearer {MESHY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # API 요청 데이터
    payload = {
        "image_url": data_uri,
        "enable_pbr": True,  # PBR 텍스처 생성
        "should_remesh": True,  # 리메시 활성화
        "should_texture": True,  # 텍스처 생성
        "ai_model": "meshy-4"  # AI 모델 지정
    }
    
    try:
        print(f"[Meshy API] 요청 시작 - 이미지 크기: {len(image_bytes)} bytes")
        print(f"[Meshy API] 엔드포인트: {MESHY_API_URL}/openapi/v1/image-to-3d")
        print(f"[Meshy API] API 키 설정: {'O' if MESHY_API_KEY else 'X'}")
        
        response = requests.post(
            f"{MESHY_API_URL}/openapi/v1/image-to-3d",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"[Meshy API] 응답 상태 코드: {response.status_code}")
        
        # 200 (OK) 또는 202 (Accepted) 모두 성공
        if response.status_code in [200, 202]:
            result = response.json()
            task_id = result.get("result")
            print(f"[Meshy API] 성공! Task ID: {task_id}")
            return {
                "success": True,
                "task_id": task_id,
                "message": "3D 모델 생성 작업이 시작되었습니다."
            }
        else:
            error_detail = response.text
            try:
                error_json = response.json()
                error_detail = error_json.get("message", response.text)
                print(f"[Meshy API] 오류 응답: {error_json}")
            except:
                print(f"[Meshy API] 원시 오류: {response.text}")
            
            return {
                "success": False,
                "error": f"API 오류: {response.status_code}",
                "message": error_detail
            }
            
    except requests.exceptions.Timeout:
        print(f"[Meshy API] 타임아웃 오류")
        return {
            "success": False,
            "error": "요청 시간 초과"
        }
    except Exception as e:
        print(f"[Meshy API] 예외 발생: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


def check_3d_task_status(task_id: str) -> Dict:
    """
    Meshy.ai 3D 생성 작업 상태 확인
    
    Args:
        task_id: 작업 ID
    
    Returns:
        dict: 작업 상태 정보
    """
    if not MESHY_API_KEY:
        return {"success": False, "error": "API 키가 없습니다."}
    
    headers = {
        "Authorization": f"Bearer {MESHY_API_KEY}",
    }
    
    try:
        response = requests.get(
            f"{MESHY_API_URL}/openapi/v1/image-to-3d/{task_id}",
            headers=headers,
            timeout=10
        )
        
        print(f"[Meshy API] 상태 확인 - Task: {task_id}, 응답: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            status = result.get("status")
            progress = result.get("progress", 0)
            
            print(f"[Meshy API] 상태: {status}, 진행률: {progress}%")
            
            return {
                "success": True,
                "status": status,
                "progress": progress,
                "model_urls": result.get("model_urls", {}),
                "thumbnail_url": result.get("thumbnail_url"),
                "texture_urls": result.get("texture_urls", []),
                "message": f"상태: {status}"
            }
        else:
            error_detail = response.text
            try:
                error_json = response.json()
                error_detail = error_json.get("message", response.text)
                print(f"[Meshy API] 상태 확인 오류: {error_json}")
            except:
                print(f"[Meshy API] 상태 확인 원시 오류: {response.text}")
            
            return {
                "success": False,
                "error": f"API 오류: {response.status_code}",
                "message": error_detail
            }
            
    except Exception as e:
        print(f"[Meshy API] 상태 확인 예외: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def download_3d_model(model_url: str) -> Optional[bytes]:
    """
    생성된 3D 모델 다운로드
    
    Args:
        model_url: 모델 다운로드 URL
    
    Returns:
        bytes: 모델 파일 데이터 또는 None
    """
    try:
        response = requests.get(model_url, timeout=30)
        if response.status_code == 200:
            return response.content
        else:
            return None
    except Exception as e:
        print(f"3D 모델 다운로드 실패: {e}")
        return None


def save_3d_models_to_server(task_id: str, model_urls: Dict, thumbnail_url: Optional[str] = None) -> Dict:
    """
    Meshy.ai에서 생성된 3D 모델을 서버에 저장
    
    Args:
        task_id: 작업 ID
        model_urls: 모델 다운로드 URL 딕셔너리
        thumbnail_url: 썸네일 URL (선택)
    
    Returns:
        dict: 저장된 파일 경로들
    """
    saved_files = {}
    save_dir = Path("3d_models") / task_id
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[3D 저장] 저장 디렉토리: {save_dir}")
    
    # 각 포맷별 모델 다운로드 및 저장
    for format_name, url in model_urls.items():
        if not url:
            continue
            
        try:
            print(f"[3D 저장] {format_name.upper()} 다운로드 중...")
            response = requests.get(url, timeout=60)
            
            if response.status_code == 200:
                file_path = save_dir / f"model.{format_name}"
                file_path.write_bytes(response.content)
                
                saved_files[format_name] = str(file_path)
                print(f"[3D 저장] ✓ {format_name.upper()} 저장 완료: {file_path}")
            else:
                print(f"[3D 저장] ✗ {format_name.upper()} 다운로드 실패: {response.status_code}")
                
        except Exception as e:
            print(f"[3D 저장] ✗ {format_name.upper()} 저장 오류: {e}")
    
    # 썸네일 저장
    if thumbnail_url:
        try:
            print(f"[3D 저장] 썸네일 다운로드 중...")
            response = requests.get(thumbnail_url, timeout=30)
            
            if response.status_code == 200:
                thumbnail_path = save_dir / "thumbnail.png"
                thumbnail_path.write_bytes(response.content)
                
                saved_files["thumbnail"] = str(thumbnail_path)
                print(f"[3D 저장] ✓ 썸네일 저장 완료: {thumbnail_path}")
                
        except Exception as e:
            print(f"[3D 저장] ✗ 썸네일 저장 오류: {e}")
    
    return saved_files

