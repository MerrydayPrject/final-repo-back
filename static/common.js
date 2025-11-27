// 공통 JavaScript - 모든 페이지에서 사용
// "홈으로" 버튼 클릭 시 로그인 상태 유지

// 메인 페이지(로그인 페이지)에서는 실행하지 않음
if (window.location.pathname !== '/') {
    // 홈으로 버튼 클릭 처리
    function handleHomeButtonClick(event) {
        event.preventDefault();
        // localStorage에 토큰이 있으면 그대로 메인 페이지로 이동
        // 메인 페이지의 admin_login.js가 자동으로 토큰을 확인하여 로그인 폼 또는 관리자 메뉴를 표시함
        window.location.href = '/';
    }

    // 페이지 로드 시 모든 "홈으로" 버튼에 이벤트 리스너 추가
    document.addEventListener('DOMContentLoaded', function () {
        const homeButtons = document.querySelectorAll('.btn-home');
        homeButtons.forEach(button => {
            // 기본 링크 동작을 막고 JavaScript로 처리
            button.addEventListener('click', handleHomeButtonClick);
        });
    });
}

