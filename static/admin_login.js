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

// 서버 재시작 감지 (배포 환경 호환성을 위해 비활성화)
// 배포 환경에서는 서버 세션 ID가 불안정할 수 있으므로 토큰 검증만으로 판단
async function checkServerRestart() {
    // 서버 세션 ID 체크는 비활성화
    // 토큰 검증만으로 인증 상태를 판단하도록 변경
    // 배포 환경에서 로드밸런서나 여러 인스턴스로 인해 세션 ID가 불안정할 수 있음
    return true;
}

// 방문자 수 로드
async function loadVisitorCount() {
    try {
        // 방문자 수 조회만
        const response = await fetch('/visitor/today', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        let count = 0;
        if (response.ok) {
            const data = await response.json();
            count = data.count || 0;
        } else {
            // 조회 실패 시 재시도
            const getResponse = await fetch('/visitor/today', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            if (getResponse.ok) {
                const data = await getResponse.json();
                count = data.count || 0;
            }
        }

        // 관리자 메뉴의 방문자 수 업데이트
        const visitorCountElement = document.getElementById('visitor-count');
        if (visitorCountElement) {
            visitorCountElement.textContent = count;
        }
    } catch (error) {
        console.error('방문자 수 로드 오류:', error);
        // 오류 시에도 기본값 표시
        const visitorCountElement = document.getElementById('visitor-count');
        if (visitorCountElement) {
            visitorCountElement.textContent = '0';
        }
    }
}

// 페이지 로드 시 저장된 토큰 확인
document.addEventListener('DOMContentLoaded', async function () {
    // URL에서 민감한 정보 제거
    cleanUrl();

    // 방문자 수 로드 (로그인 여부와 관계없이 표시)
    loadVisitorCount();

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
    } else {
        // 토큰이 없으면 로그인 폼 표시
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

        // 응답이 JSON인지 확인
        let data;
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            data = await response.json();
        } else {
            // JSON이 아닌 경우 텍스트로 읽기
            const text = await response.text();
            console.error('토큰 검증 응답이 JSON이 아닙니다:', text);
            // 네트워크 오류나 서버 오류는 토큰을 삭제하지 않음 (일시적 오류일 수 있음)
            return;
        }

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
            // 토큰이 유효하지 않음 (401, 403 등 명확한 인증 오류)
            // 401, 403 오류일 때만 토큰 삭제
            if (response.status === 401 || response.status === 403) {
                localStorage.removeItem('admin_access_token');
                accessToken = null;
                // 로그인 폼 표시
                const loginContainer = document.getElementById('login-container');
                const adminContainer = document.getElementById('admin-container');
                if (loginContainer) loginContainer.style.display = 'block';
                if (adminContainer) adminContainer.style.display = 'none';
            } else {
                // 다른 오류(500 등)는 일시적일 수 있으므로 토큰 유지
                console.warn('토큰 검증 중 오류 발생 (토큰 유지):', data.message || data.error);
            }
        }
    } catch (error) {
        console.error('토큰 검증 오류:', error);
        // 네트워크 오류는 일시적일 수 있으므로 토큰을 삭제하지 않음
        // 사용자가 명시적으로 로그아웃하거나 토큰이 만료될 때까지 유지
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

