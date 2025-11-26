"""인증 미들웨어"""
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from core.supabase_client import verify_user_token, is_admin_user

security = HTTPBearer()


async def get_current_user(request: Request) -> Optional[dict]:
    """
    요청에서 현재 사용자 정보를 가져옴
    
    Args:
        request: FastAPI Request 객체
        
    Returns:
        사용자 정보 딕셔너리 또는 None
    """
    # Authorization 헤더에서 토큰 추출
    authorization = request.headers.get("Authorization")
    if not authorization:
        return None
    
    # "Bearer <token>" 형식에서 토큰 추출
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            return None
    except ValueError:
        return None
    
    # 토큰 검증
    user_data = await verify_user_token(token)
    return user_data


async def require_admin(request: Request) -> dict:
    """
    관리자 권한이 필요한 엔드포인트에서 사용
    관리자가 아니면 HTTPException 발생
    
    Args:
        request: FastAPI Request 객체
        
    Returns:
        사용자 정보 딕셔너리
        
    Raises:
        HTTPException: 인증 실패 또는 권한 없음
    """
    user_data = await get_current_user(request)
    
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="인증이 필요합니다.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not is_admin_user(user_data):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="관리자 권한이 필요합니다.",
        )
    
    return user_data

