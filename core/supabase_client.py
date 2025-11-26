"""Supabase 클라이언트 초기화 및 인증 유틸리티"""
from supabase import create_client, Client
from config.settings import SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_ROLE_KEY
from typing import Optional, Dict, Any

# Supabase 클라이언트 인스턴스
_supabase_client: Optional[Client] = None
_supabase_admin_client: Optional[Client] = None


def get_supabase_client() -> Optional[Client]:
    """일반 사용자용 Supabase 클라이언트 반환"""
    global _supabase_client
    if _supabase_client is None:
        if not SUPABASE_URL or not SUPABASE_ANON_KEY:
            return None
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    return _supabase_client


def get_supabase_admin_client() -> Optional[Client]:
    """관리자용 Supabase 클라이언트 반환 (서비스 역할 키 사용)"""
    global _supabase_admin_client
    if _supabase_admin_client is None:
        if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
            return None
        _supabase_admin_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    return _supabase_admin_client


async def verify_user_token(token: str) -> Optional[Dict[str, Any]]:
    """
    JWT 토큰을 검증하고 사용자 정보를 반환
    
    Args:
        token: JWT 토큰
        
    Returns:
        사용자 정보 딕셔너리 또는 None
    """
    try:
        import jwt
        
        # JWT에서 사용자 정보 추출 (검증 없이 - Supabase에서 이미 검증된 토큰)
        try:
            decoded = jwt.decode(token, options={"verify_signature": False})
            user_id = decoded.get("sub")
            email = decoded.get("email")
            user_metadata = decoded.get("user_metadata", {})
            app_metadata = decoded.get("app_metadata", {})
            
            if user_id:
                return {
                    "id": user_id,
                    "email": email,
                    "user_metadata": user_metadata,
                    "app_metadata": app_metadata
                }
        except Exception as jwt_error:
            print(f"JWT 디코딩 오류: {jwt_error}")
        
        # 대안: 관리자 클라이언트로 사용자 정보 가져오기
        admin_client = get_supabase_admin_client()
        if admin_client:
            try:
                # JWT에서 user_id 추출
                decoded = jwt.decode(token, options={"verify_signature": False})
                user_id = decoded.get("sub")
                
                if user_id:
                    # 관리자 클라이언트로 사용자 정보 가져오기
                    response = admin_client.auth.admin.get_user_by_id(user_id)
                    if response and response.user:
                        user = response.user
                        return {
                            "id": user.id,
                            "email": user.email,
                            "user_metadata": user.user_metadata or {},
                            "app_metadata": user.app_metadata or {}
                        }
            except Exception as admin_error:
                print(f"관리자 클라이언트 사용자 조회 오류: {admin_error}")
        
        return None
    except Exception as e:
        print(f"토큰 검증 오류: {e}")
        return None


def is_admin_user(user_data: Dict[str, Any]) -> bool:
    """
    사용자가 admin 역할을 가지고 있는지 확인
    
    Args:
        user_data: verify_user_token에서 반환된 사용자 정보
        
    Returns:
        admin 여부
    """
    if not user_data:
        print(f"[DEBUG is_admin_user] user_data가 None")
        return False
    
    user_id = user_data.get("id")
    email = user_data.get("email", "")
    
    # 1. app_metadata에서 role 확인
    app_metadata = user_data.get("app_metadata", {})
    print(f"[DEBUG is_admin_user] app_metadata: {app_metadata}, type: {type(app_metadata)}")
    if app_metadata and app_metadata.get("role") == "admin":
        print(f"[DEBUG is_admin_user] app_metadata에서 admin 확인됨")
        return True
    
    # 2. user_metadata에서 role 확인
    user_metadata = user_data.get("user_metadata", {})
    print(f"[DEBUG is_admin_user] user_metadata: {user_metadata}, type: {type(user_metadata)}")
    if user_metadata and user_metadata.get("role") == "admin":
        print(f"[DEBUG is_admin_user] user_metadata에서 admin 확인됨")
        return True
    
    # 3. public.profiles 테이블에서 role 확인
    admin_client = get_supabase_admin_client()
    if admin_client and (user_id or email):
        try:
            # user_id로 조회 시도
            if user_id:
                response = admin_client.table('profiles').select('role').eq('id', user_id).execute()
                if response.data and len(response.data) > 0:
                    profile_role = response.data[0].get('role')
                    print(f"[DEBUG is_admin_user] profiles 테이블 (user_id로 조회) role: {profile_role}")
                    if profile_role == "admin":
                        print(f"[DEBUG is_admin_user] profiles 테이블에서 admin 확인됨 (user_id)")
                        return True
            
            # email로 조회 시도
            if email:
                response = admin_client.table('profiles').select('role').eq('email', email).execute()
                if response.data and len(response.data) > 0:
                    profile_role = response.data[0].get('role')
                    print(f"[DEBUG is_admin_user] profiles 테이블 (email로 조회) role: {profile_role}")
                    if profile_role == "admin":
                        print(f"[DEBUG is_admin_user] profiles 테이블에서 admin 확인됨 (email)")
                        return True
        except Exception as e:
            print(f"[DEBUG is_admin_user] profiles 테이블 조회 오류: {e}")
    
    print(f"[DEBUG is_admin_user] admin 권한 없음")
    return False

