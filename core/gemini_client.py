"""Gemini API 클라이언트 풀 관리자 (다중 API 키 지원)"""
import asyncio
import threading
from typing import List, Optional, Callable, Any
from google import genai
from config.settings import get_gemini_3_api_keys


class GeminiClientPool:
    """
    여러 Gemini API 키를 라운드로빈 방식으로 관리하고,
    Rate limit/에러 발생 시 자동 재시도하는 클라이언트 풀
    """
    
    def __init__(self, api_keys: Optional[List[str]] = None):
        """
        Args:
            api_keys: API 키 리스트 (None이면 환경변수에서 자동 로드)
        """
        if api_keys is None:
            api_keys = get_gemini_3_api_keys()
        
        if not api_keys:
            raise ValueError("Gemini API 키가 설정되지 않았습니다. GEMINI_3_API_KEY 환경변수를 확인하세요.")
        
        self.api_keys = api_keys
        self.clients = {key: genai.Client(api_key=key) for key in api_keys}
        self.current_index = 0
        self.lock = threading.Lock()  # 동기 버전용
        self.async_lock = asyncio.Lock()  # 비동기 버전용
        print(f"[GeminiClientPool] {len(self.api_keys)}개의 API 키로 초기화 완료")
    
    def _get_next_key(self) -> str:
        """라운드로빈 방식으로 다음 API 키 반환 (Thread-safe)"""
        with self.lock:
            key = self.api_keys[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.api_keys)
            return key
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """
        재시도 가능한 에러인지 확인
        
        Args:
            error: 발생한 예외
            
        Returns:
            bool: 재시도 가능하면 True
        """
        error_str = str(error).lower()
        
        # Rate limit 에러 (429)
        if "429" in error_str or "rate limit" in error_str or "quota" in error_str:
            return True
        
        # 일시적인 네트워크 에러
        if "timeout" in error_str or "connection" in error_str:
            return True
        
        # 기타 재시도 가능한 에러
        if "service unavailable" in error_str or "internal error" in error_str:
            return True
        
        return False
    
    def generate_content_with_retry(
        self,
        model: str,
        contents: List[Any],
        max_retries: Optional[int] = None
    ) -> Any:
        """
        여러 API 키를 사용하여 Gemini API 호출 (자동 재시도 포함)
        동기 버전 - 기존 호환성 유지
        
        Args:
            model: 사용할 모델명 (예: "gemini-3-pro-image-preview")
            contents: 생성할 콘텐츠 리스트
            max_retries: 최대 재시도 횟수 (None이면 모든 키 시도)
            
        Returns:
            Gemini API 응답 객체
            
        Raises:
            Exception: 모든 키 실패 시 마지막 에러
        """
        if max_retries is None:
            max_retries = len(self.api_keys)
        
        # 각 요청마다 라운드로빈으로 시작 키 선택
        with self.lock:
            start_key = self.api_keys[self.current_index]
            start_index = self.current_index
            self.current_index = (self.current_index + 1) % len(self.api_keys)
        
        last_error = None
        tried_keys = set()
        
        for attempt in range(max_retries):
            # 시작 키부터 순차적으로 시도
            key_index = (start_index + attempt) % len(self.api_keys)
            api_key = self.api_keys[key_index]
            
            # 이미 시도한 키는 건너뛰기
            if api_key in tried_keys:
                continue
            
            tried_keys.add(api_key)
            client = self.clients[api_key]
            
            try:
                print(f"[GeminiClientPool] API 키 {attempt + 1}/{max_retries} 사용 중 (키 인덱스: {key_index})")
                response = client.models.generate_content(
                    model=model,
                    contents=contents
                )
                print(f"[GeminiClientPool] API 호출 성공 (키 인덱스: {key_index})")
                return response
                
            except Exception as e:
                last_error = e
                error_msg = str(e)
                print(f"[GeminiClientPool] API 키 {attempt + 1}/{max_retries} 실패: {error_msg[:100]}")
                
                # 재시도 불가능한 에러면 즉시 종료
                if not self._is_retryable_error(e):
                    print(f"[GeminiClientPool] 재시도 불가능한 에러로 인해 중단")
                    raise e
                
                # 마지막 시도가 아니면 다음 키로 재시도
                if attempt < max_retries - 1:
                    print(f"[GeminiClientPool] 다음 API 키로 재시도...")
                    continue
                else:
                    # 모든 시도 실패
                    print(f"[GeminiClientPool] 모든 API 키 시도 실패")
                    raise last_error
        
        # 모든 시도 실패
        if last_error:
            raise last_error
        else:
            raise Exception("Gemini API 호출 실패: 알 수 없는 오류")
    
    async def generate_content_with_retry_async(
        self,
        model: str,
        contents: List[Any],
        max_retries: Optional[int] = None
    ) -> Any:
        """
        여러 API 키를 사용하여 Gemini API 호출 (자동 재시도 포함)
        비동기 버전 - 실제 병렬 처리 가능
        
        Args:
            model: 사용할 모델명 (예: "gemini-3-pro-image-preview")
            contents: 생성할 콘텐츠 리스트
            max_retries: 최대 재시도 횟수 (None이면 모든 키 시도)
            
        Returns:
            Gemini API 응답 객체
            
        Raises:
            Exception: 모든 키 실패 시 마지막 에러
        """
        if max_retries is None:
            max_retries = len(self.api_keys)
        
        # 각 요청마다 라운드로빈으로 시작 키 선택 (비동기 lock 사용)
        async with self.async_lock:
            start_key = self.api_keys[self.current_index]
            start_index = self.current_index
            self.current_index = (self.current_index + 1) % len(self.api_keys)
            print(f"[GeminiClientPool] 요청 시작 - 시작 키 인덱스: {start_index}, 다음 인덱스: {self.current_index}")
        
        last_error = None
        tried_keys = set()
        consecutive_503_count = 0  # 503 에러 연속 발생 횟수 추적
        
        for attempt in range(max_retries):
            # 시작 키부터 순차적으로 시도
            key_index = (start_index + attempt) % len(self.api_keys)
            api_key = self.api_keys[key_index]
            
            # 이미 시도한 키는 건너뛰기
            if api_key in tried_keys:
                continue
            
            tried_keys.add(api_key)
            client = self.clients[api_key]
            
            try:
                print(f"[GeminiClientPool] API 키 {attempt + 1}/{max_retries} 사용 중 (키 인덱스: {key_index})")
                # 동기 호출을 비동기로 실행 (별도 스레드에서 실행되어 블로킹되지 않음)
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=model,
                    contents=contents
                )
                print(f"[GeminiClientPool] API 호출 성공 (키 인덱스: {key_index})")
                consecutive_503_count = 0  # 성공 시 카운터 리셋
                return response
                
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                print(f"[GeminiClientPool] API 키 {attempt + 1}/{max_retries} 실패: {str(e)[:100]}")
                
                # 503 에러 또는 overloaded 메시지 체크
                is_503_error = "503" in str(e) or "overloaded" in error_msg
                
                if is_503_error:
                    consecutive_503_count += 1
                    print(f"[Warning] 503 Overloaded. API Key {key_index} Failed. Retrying with next key...")
                    
                    # 503 에러가 연속으로 5번 발생하면 포기
                    if consecutive_503_count >= 5:
                        print(f"[Warning] 503 Overloaded occurred 5 times consecutively. Giving up.")
                        raise last_error
                    
                    await asyncio.sleep(2)
                    # 마지막 시도가 아니면 다음 키로 재시도
                    if attempt < max_retries - 1:
                        continue
                    else:
                        # 모든 시도 실패
                        print(f"[GeminiClientPool] 모든 API 키 시도 실패")
                        raise last_error
                else:
                    # 503이 아닌 다른 에러면 카운터 리셋
                    consecutive_503_count = 0
                
                # 재시도 불가능한 에러면 즉시 종료
                if not self._is_retryable_error(e):
                    print(f"[GeminiClientPool] 재시도 불가능한 에러로 인해 중단")
                    raise e
                
                # 마지막 시도가 아니면 다음 키로 재시도
                if attempt < max_retries - 1:
                    print(f"[GeminiClientPool] 다음 API 키로 재시도...")
                    continue
                else:
                    # 모든 시도 실패
                    print(f"[GeminiClientPool] 모든 API 키 시도 실패")
                    raise last_error
        
        # 모든 시도 실패
        if last_error:
            raise last_error
        else:
            raise Exception("Gemini API 호출 실패: 알 수 없는 오류")


# 전역 인스턴스 (lazy initialization)
_pool_instance: Optional[GeminiClientPool] = None
_pool_lock = threading.Lock()


def get_gemini_client_pool() -> GeminiClientPool:
    """
    전역 GeminiClientPool 인스턴스 반환 (Singleton 패턴)
    
    Returns:
        GeminiClientPool: 클라이언트 풀 인스턴스
    """
    global _pool_instance
    
    if _pool_instance is None:
        with _pool_lock:
            if _pool_instance is None:
                _pool_instance = GeminiClientPool()
    
    return _pool_instance

