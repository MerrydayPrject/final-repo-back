"""서버 세션 정보 관리 (서버 재시작 감지용)"""
import time
import uuid

# 서버 시작 시간 및 세션 ID
SERVER_START_TIME = time.time()
SERVER_SESSION_ID = str(uuid.uuid4())

def get_server_session():
    """서버 세션 정보 반환"""
    return {
        "server_session_id": SERVER_SESSION_ID,
        "server_start_time": SERVER_START_TIME
    }

