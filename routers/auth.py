"""인증 라우터"""
from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from core.supabase_client import get_supabase_client, get_supabase_admin_client, verify_user_token, is_admin_user

router = APIRouter()


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenVerifyRequest(BaseModel):
    token: str


@router.post("/api/auth/login", tags=["인증"])
async def login(request: Request, login_data: LoginRequest):
    """
    로그인 엔드포인트
    
    Supabase를 통해 이메일/비밀번호로 로그인하고,
    admin 권한이 있는 사용자만 성공 응답을 반환합니다.
    """
    try:
        client = get_supabase_client()
        if not client:
            return JSONResponse(
                {
                    "success": False,
                    "error": "Supabase client not configured",
                    "message": "Supabase 클라이언트가 설정되지 않았습니다."
                },
                status_code=500
            )
        
        # Supabase 로그인
        response = client.auth.sign_in_with_password({
            "email": login_data.email,
            "password": login_data.password
        })
        
        if not response or not response.user:
            return JSONResponse(
                {
                    "success": False,
                    "error": "Invalid credentials",
                    "message": "이메일 또는 비밀번호가 올바르지 않습니다."
                },
                status_code=401
            )
        
        user = response.user
        access_token = response.session.access_token if response.session else None
        
        if not access_token:
            return JSONResponse(
                {
                    "success": False,
                    "error": "No access token",
                    "message": "액세스 토큰을 가져올 수 없습니다."
                },
                status_code=500
            )
        
        # 사용자 정보 구성
        user_data = {
            "id": user.id,
            "email": user.email,
            "user_metadata": user.user_metadata or {},
            "app_metadata": user.app_metadata or {}
        }
        
        print(f"[DEBUG] 로그인 시도 - 이메일: {user.email}, ID: {user.id}")
        print(f"[DEBUG] 초기 user_metadata: {user_data.get('user_metadata')}")
        print(f"[DEBUG] 초기 app_metadata: {user_data.get('app_metadata')}")
        
        # Admin 권한 확인 (profiles 테이블도 확인)
        is_admin = is_admin_user(user_data)
        print(f"[DEBUG] Admin 권한 확인 결과: {is_admin}")
        if not is_admin:
            return JSONResponse(
                {
                    "success": False,
                    "error": "Not an admin user",
                    "message": "관리자 권한이 없습니다."
                },
                status_code=403
            )
        
        return JSONResponse({
            "success": True,
            "data": {
                "user": {
                    "id": user.id,
                    "email": user.email
                },
                "access_token": access_token
            },
            "message": "로그인 성공"
        })
        
    except Exception as e:
        error_message = str(e)
        # Supabase 에러 메시지 처리
        if "Invalid login credentials" in error_message or "Email not confirmed" in error_message:
            return JSONResponse(
                {
                    "success": False,
                    "error": "Invalid credentials",
                    "message": "이메일 또는 비밀번호가 올바르지 않습니다."
                },
                status_code=401
            )
        
        return JSONResponse(
            {
                "success": False,
                "error": str(e),
                "message": f"로그인 중 오류 발생: {error_message}"
            },
            status_code=500
        )


@router.post("/api/auth/logout", tags=["인증"])
async def logout(request: Request):
    """
    로그아웃 엔드포인트
    
    클라이언트 측에서 토큰을 삭제하는 것으로 충분하지만,
    서버 측에서도 세션을 무효화할 수 있습니다.
    """
    try:
        # Authorization 헤더에서 토큰 추출
        authorization = request.headers.get("Authorization")
        if authorization:
            try:
                scheme, token = authorization.split()
                if scheme.lower() == "bearer":
                    client = get_supabase_client()
                    if client:
                        client.auth.sign_out()
            except ValueError:
                pass
        
        return JSONResponse({
            "success": True,
            "message": "로그아웃 성공"
        })
        
    except Exception as e:
        return JSONResponse(
            {
                "success": False,
                "error": str(e),
                "message": f"로그아웃 중 오류 발생: {str(e)}"
            },
            status_code=500
        )


@router.get("/api/auth/verify", tags=["인증"])
async def verify_token(request: Request):
    """
    토큰 검증 엔드포인트
    
    현재 요청의 토큰이 유효한지 확인하고,
    admin 권한이 있는지 확인합니다.
    """
    try:
        # Authorization 헤더에서 토큰 추출
        authorization = request.headers.get("Authorization")
        if not authorization:
            return JSONResponse(
                {
                    "success": False,
                    "error": "No authorization header",
                    "message": "인증 헤더가 없습니다."
                },
                status_code=401
            )
        
        try:
            scheme, token = authorization.split()
            if scheme.lower() != "bearer":
                return JSONResponse(
                    {
                        "success": False,
                        "error": "Invalid authorization scheme",
                        "message": "잘못된 인증 방식입니다."
                    },
                    status_code=401
                )
        except ValueError:
            return JSONResponse(
                {
                    "success": False,
                    "error": "Invalid authorization header",
                    "message": "잘못된 인증 헤더 형식입니다."
                },
                status_code=401
            )
        
        # 토큰 검증
        user_data = await verify_user_token(token)
        if not user_data:
            return JSONResponse(
                {
                    "success": False,
                    "error": "Invalid token",
                    "message": "유효하지 않은 토큰입니다."
                },
                status_code=401
            )
        
        # Admin 권한 확인
        if not is_admin_user(user_data):
            return JSONResponse(
                {
                    "success": False,
                    "error": "Not an admin user",
                    "message": "관리자 권한이 없습니다."
                },
                status_code=403
            )
        
        return JSONResponse({
            "success": True,
            "data": {
                "user": {
                    "id": user_data.get("id"),
                    "email": user_data.get("email")
                },
                "is_admin": True
            },
            "message": "토큰이 유효합니다."
        })
        
    except Exception as e:
        return JSONResponse(
            {
                "success": False,
                "error": str(e),
                "message": f"토큰 검증 중 오류 발생: {str(e)}"
            },
            status_code=500
        )

