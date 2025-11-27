// 관리자 로그인/로그아웃 로직

let accessToken = null;

// URL에서 민감한 정보 제거
function cleanUrl() {
    const url = new URL(window.location.href);
    const sensitiveParams = ['email', 'password', 'token', 'access_token'];
    let hasChanges = false;

    sensitiveParams.forEach(param => {
        if (url.searchParams.has(param)) {
            url.searchParams.delete(param);
            hasChanges = true;
        }
    });

    if (hasChanges) {
        // URL에서 쿼리 파라미터 제거 (히스토리 변경 없이)
        window.history.replaceState({}, document.title, url.pathname);
    }
}

// 서버 재시작 감지 및 토큰 무효화
async function checkServerRestart() {
    try {
        const response = await fetch('/health');
        const data = await response.json();

        const currentSessionId = data.server_session_id;
        const savedSessionId = localStorage.getItem('server_session_id');

        // 서버 세션 ID가 다르면 서버가 재시작된 것
        if (savedSessionId && savedSessionId !== currentSessionId) {
            // 서버가 재시작되었으므로 토큰 삭제
            localStorage.removeItem('admin_access_token');
            accessToken = null;
            console.log('서버가 재시작되어 로그인 상태가 초기화되었습니다.');
        }

        // 현재 서버 세션 ID 저장
        localStorage.setItem('server_session_id', currentSessionId);

        return true;
    } catch (error) {
        console.error('서버 상태 확인 오류:', error);
        return false;
    }
}

// 페이지 로드 시 저장된 토큰 확인
document.addEventListener('DOMContentLoaded', async function () {
    // URL에서 민감한 정보 제거
    cleanUrl();

    // 서버 재시작 확인 (비동기)
    await checkServerRestart();

    // 서버에서 전달된 인증 상태 확인 (인라인 스크립트에서 설정됨)
    const isAuthenticated = window.isAuthenticated || false;
    const userEmail = window.userEmail || null;

    // localStorage에서 토큰 확인
    const savedToken = localStorage.getItem('admin_access_token');
    const loginContainer = document.getElementById('login-container');
    const adminContainer = document.getElementById('admin-container');

    if (savedToken) {
        accessToken = savedToken;
        // 토큰이 있으면 초기에 로그인 폼을 숨기고 관리자 메뉴를 표시 (토큰 검증 중)
        // 토큰 검증이 실패하면 verifyToken 내부에서 로그인 폼을 다시 표시함
        if (loginContainer) loginContainer.style.display = 'none';
        if (adminContainer) adminContainer.style.display = 'block';
        // 토큰 검증 (비동기로 처리되므로 UI 업데이트는 verifyToken 내부에서)
        verifyToken(savedToken);
    } else if (isAuthenticated) {
        // 서버에서 인증되었다고 했지만 토큰이 없는 경우 (이상한 상황)
        // 로그인 폼 표시
        if (loginContainer) loginContainer.style.display = 'block';
        if (adminContainer) adminContainer.style.display = 'none';
    } else {
        // 인증되지 않음 - 로그인 폼 표시
        if (loginContainer) loginContainer.style.display = 'block';
        if (adminContainer) adminContainer.style.display = 'none';
    }

    // 로그인 폼 이벤트 리스너
    const loginForm = document.getElementById('login-form');
    if (loginForm) {
        // 폼의 action과 method 제거하여 URL에 데이터가 추가되지 않도록 함
        loginForm.setAttribute('action', '#');
        loginForm.setAttribute('method', 'post');
        loginForm.addEventListener('submit', handleLogin);
    }

    // 로그아웃 버튼 이벤트 리스너
    const logoutButton = document.getElementById('logout-button');
    if (logoutButton) {
        logoutButton.addEventListener('click', handleLogout);
    }
});

// 로그인 처리
async function handleLogin(event) {
    event.preventDefault();

    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    const errorDiv = document.getElementById('login-error');

    // 에러 메시지 초기화
    errorDiv.style.display = 'none';
    errorDiv.textContent = '';

    // 로그인 버튼 비활성화
    const loginForm = event.target;
    const loginButton = loginForm.querySelector('button[type="submit"]') ||
        document.querySelector('#login-form button[type="submit"]');
    if (loginButton) {
        loginButton.disabled = true;
        loginButton.textContent = '로그인 중...';
    }

    try {
        const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                email: email,
                password: password
            })
        });

        const data = await response.json();

        if (response.ok && data.success && data.data && data.data.access_token) {
            // 토큰 저장
            accessToken = data.data.access_token;
            localStorage.setItem('admin_access_token', accessToken);

            // URL 정리 후 페이지 새로고침
            cleanUrl();
            window.location.href = '/';
        } else {
            // 로그인 실패
            const errorMessage = data.message || data.error || '로그인에 실패했습니다.';
            errorDiv.textContent = errorMessage;
            errorDiv.style.display = 'block';

            // 로그인 버튼 다시 활성화
            if (loginButton) {
                loginButton.disabled = false;
                loginButton.textContent = '로그인';
            }
        }
    } catch (error) {
        console.error('로그인 오류:', error);
        errorDiv.textContent = '로그인 중 오류가 발생했습니다. 네트워크 연결을 확인해주세요.';
        errorDiv.style.display = 'block';

        // 로그인 버튼 다시 활성화
        if (loginButton) {
            loginButton.disabled = false;
            loginButton.textContent = '로그인';
        }
    } finally {
        // URL에서 민감한 정보 제거
        cleanUrl();
    }
}

// 로그아웃 처리
async function handleLogout() {
    try {
        // 서버에 로그아웃 요청
        if (accessToken) {
            await fetch('/api/auth/logout', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${accessToken}`,
                    'Content-Type': 'application/json',
                }
            });
        }
    } catch (error) {
        console.error('로그아웃 오류:', error);
    } finally {
        // 로컬 스토리지에서 토큰 제거
        localStorage.removeItem('admin_access_token');
        accessToken = null;

        // 페이지 새로고침하여 로그인 폼 표시
        window.location.href = '/';
    }
}

// 토큰 검증
async function verifyToken(token) {
    try {
        const response = await fetch('/api/auth/verify', {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json',
            }
        });

        const data = await response.json();

        if (response.ok && data.success && data.data) {
            // 토큰이 유효함 - 관리자 메뉴 표시
            const loginContainer = document.getElementById('login-container');
            const adminContainer = document.getElementById('admin-container');
            const userEmailSpan = document.getElementById('user-email');

            if (loginContainer) loginContainer.style.display = 'none';
            if (adminContainer) adminContainer.style.display = 'block';
            if (userEmailSpan && data.data.user && data.data.user.email) {
                userEmailSpan.textContent = data.data.user.email;
            }
        } else {
            // 토큰이 유효하지 않음
            localStorage.removeItem('admin_access_token');
            accessToken = null;
            // 로그인 폼 표시
            const loginContainer = document.getElementById('login-container');
            const adminContainer = document.getElementById('admin-container');
            if (loginContainer) loginContainer.style.display = 'block';
            if (adminContainer) adminContainer.style.display = 'none';
        }
    } catch (error) {
        console.error('토큰 검증 오류:', error);
        localStorage.removeItem('admin_access_token');
        accessToken = null;
        const loginContainer = document.getElementById('login-container');
        const adminContainer = document.getElementById('admin-container');
        if (loginContainer) loginContainer.style.display = 'block';
        if (adminContainer) adminContainer.style.display = 'none';
    }
}

// API 요청 시 토큰을 헤더에 추가하는 헬퍼 함수
function getAuthHeaders() {
    const token = localStorage.getItem('admin_access_token');
    if (token) {
        return {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
        };
    }
    return {
        'Content-Type': 'application/json',
    };
}

// 전역으로 사용 가능하도록 window 객체에 추가 (필요한 경우)
if (typeof window !== 'undefined') {
    window.getAuthHeaders = getAuthHeaders;
    window.verifyToken = verifyToken;
}

