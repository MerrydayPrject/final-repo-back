// 전역 변수
let currentPage = 1;
const itemsPerPage = 20;
let currentSearchModel = null;
let currentTab = 'synthesis'; // 'synthesis', 'body', or 'reviews'
let currentBodyPage = 1;
let currentReviewsPage = 1;

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', async () => {
    // 토큰 확인
    const token = localStorage.getItem('admin_access_token');
    if (!token) {
        // 토큰이 없으면 조용히 로그인 페이지로 이동
        window.location.href = '/';
        return;
    }

    // 토큰 검증
    try {
        // 직접 토큰을 사용하여 검증
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
            window.location.href = '/';
            return;
        }

        if (!response.ok || !data.success) {
            // 토큰이 유효하지 않으면 조용히 로그인 페이지로 이동
            console.log('토큰 검증 실패:', data.message || data.error);
            window.location.href = '/';
            return;
        }
    } catch (error) {
        console.error('토큰 검증 오류:', error);
        // 네트워크 오류 등으로 검증 실패 시 조용히 로그인 페이지로 이동
        window.location.href = '/';
        return;
    }

    loadLogs(currentPage);

    // 탭 버튼 이벤트 리스너
    const tabSynthesis = document.getElementById('tabSynthesis');
    const tabBodyAnalysis = document.getElementById('tabBodyAnalysis');
    const tabReviews = document.getElementById('tabReviews');

    if (tabSynthesis) {
        tabSynthesis.addEventListener('click', () => switchTab('synthesis'));
    }
    if (tabBodyAnalysis) {
        tabBodyAnalysis.addEventListener('click', () => switchTab('body'));
    }
    if (tabReviews) {
        tabReviews.addEventListener('click', () => switchTab('reviews'));
    }

    // 검색 입력 필드에 Enter 키 이벤트 추가
    const searchInput = document.getElementById('search-input');
    if (searchInput) {
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                handleSearch();
            }
        });
    }
});

// 탭 전환
function switchTab(tab) {
    currentTab = tab;

    const synthesisSection = document.getElementById('synthesis-logs-section');
    const bodySection = document.getElementById('body-logs-section');
    const reviewsSection = document.getElementById('reviews-logs-section');
    const tabSynthesis = document.getElementById('tabSynthesis');
    const tabBodyAnalysis = document.getElementById('tabBodyAnalysis');
    const tabReviews = document.getElementById('tabReviews');
    const sectionTitle = document.getElementById('section-title');
    const logsCountLabel = document.getElementById('logs-count-label');
    const searchContainer = document.querySelector('.search-container');

    // 모든 섹션 숨기기
    if (synthesisSection) synthesisSection.style.display = 'none';
    if (bodySection) bodySection.style.display = 'none';
    if (reviewsSection) reviewsSection.style.display = 'none';

    // 모든 탭 버튼 초기화
    if (tabSynthesis) {
        tabSynthesis.classList.remove('active');
        tabSynthesis.style.background = '#fff';
        tabSynthesis.style.color = '#333';
    }
    if (tabBodyAnalysis) {
        tabBodyAnalysis.classList.remove('active');
        tabBodyAnalysis.style.background = '#fff';
        tabBodyAnalysis.style.color = '#333';
    }
    if (tabReviews) {
        tabReviews.classList.remove('active');
        tabReviews.style.background = '#fff';
        tabReviews.style.color = '#333';
    }

    if (tab === 'synthesis') {
        if (synthesisSection) synthesisSection.style.display = 'block';
        if (tabSynthesis) {
            tabSynthesis.classList.add('active');
            tabSynthesis.style.background = '#007bff';
            tabSynthesis.style.color = '#fff';
        }
        if (sectionTitle) sectionTitle.textContent = '📋 합성 로그 목록';
        if (logsCountLabel) logsCountLabel.textContent = '전체 합성:';
        if (searchContainer) searchContainer.style.display = 'block';
        loadLogs(currentPage, currentSearchModel);
    } else if (tab === 'body') {
        if (bodySection) bodySection.style.display = 'block';
        if (tabBodyAnalysis) {
            tabBodyAnalysis.classList.add('active');
            tabBodyAnalysis.style.background = '#007bff';
            tabBodyAnalysis.style.color = '#fff';
        }
        if (sectionTitle) sectionTitle.textContent = '📊 분석 결과 로그 목록';
        if (logsCountLabel) logsCountLabel.textContent = '전체 분석:';
        if (searchContainer) searchContainer.style.display = 'none';
        loadBodyLogs(currentBodyPage);
    } else if (tab === 'reviews') {
        if (reviewsSection) reviewsSection.style.display = 'block';
        if (tabReviews) {
            tabReviews.classList.add('active');
            tabReviews.style.background = '#007bff';
            tabReviews.style.color = '#fff';
        }
        if (sectionTitle) sectionTitle.textContent = '⭐ 리뷰 로그 목록';
        if (logsCountLabel) logsCountLabel.textContent = '전체 리뷰:';
        if (searchContainer) searchContainer.style.display = 'none';
        loadReviews(currentReviewsPage);
    }
}

// 통계 로드
async function loadStats() {
    try {
        const headers = window.getAuthHeaders ? window.getAuthHeaders() : {};
        const response = await fetch('/api/admin/stats', {
            headers: headers
        });
        const data = await response.json();

        if (data.success) {
            const stats = data.data;
            document.getElementById('stat-total').textContent = stats.total;
            document.getElementById('stat-success').textContent = stats.success;
            document.getElementById('stat-failed').textContent = stats.failed;
            document.getElementById('stat-success-rate').textContent = stats.success_rate + '%';
            document.getElementById('stat-avg-time').textContent = stats.average_processing_time + '초';
            document.getElementById('stat-today').textContent = stats.today;
        }
    } catch (error) {
        console.error('통계 로드 오류:', error);
    }
}

// 로그 목록 로드
async function loadLogs(page, model = null) {
    try {
        let url = `/api/admin/logs?page=${page}&limit=${itemsPerPage}`;
        if (model && model.trim() !== '') {
            url += `&model=${encodeURIComponent(model.trim())}`;
        }

        const headers = window.getAuthHeaders ? window.getAuthHeaders() : {};
        const response = await fetch(url, {
            headers: headers
        });

        // 401 오류 처리
        if (response.status === 401) {
            // 인증 오류 시 조용히 로그인 페이지로 이동
            window.location.href = '/';
            return;
        }

        const data = await response.json();

        if (data.success) {
            renderLogs(data.data);
            renderPagination(data.pagination);
            updateLogsCount(data.pagination.total);
            currentPage = page;
        } else {
            showError(data.message || '로그를 불러오는 중 오류가 발생했습니다.');
        }
    } catch (error) {
        console.error('로그 로드 오류:', error);
        const tbody = document.getElementById('logs-tbody');
        if (tbody) {
            tbody.innerHTML = '<tr><td colspan="4" class="loading" style="color: #ef4444;">로그를 불러오는 중 오류가 발생했습니다.</td></tr>';
        }
    }
}

// 검색 처리
function handleSearch() {
    const searchInput = document.getElementById('search-input');
    const searchValue = searchInput ? searchInput.value.trim() : '';
    const clearButton = document.getElementById('search-clear-button');

    currentSearchModel = searchValue || null;
    currentPage = 1; // 검색 시 첫 페이지로 이동

    // 검색어가 있으면 초기화 버튼 표시
    if (clearButton) {
        clearButton.style.display = searchValue ? 'inline-block' : 'none';
    }

    loadLogs(currentPage, currentSearchModel);
}

// 검색 초기화
function clearSearch() {
    const searchInput = document.getElementById('search-input');
    const clearButton = document.getElementById('search-clear-button');

    if (searchInput) {
        searchInput.value = '';
    }
    if (clearButton) {
        clearButton.style.display = 'none';
    }

    currentSearchModel = null;
    currentPage = 1;
    loadLogs(currentPage);
}

// 로그 갯수 업데이트
function updateLogsCount(count) {
    const logsCountElement = document.getElementById('logs-count');
    if (logsCountElement) {
        logsCountElement.textContent = count;
    }
}

// 로그 테이블 렌더링
function renderLogs(logs) {
    const tbody = document.getElementById('logs-tbody');

    if (logs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" class="loading">로그가 없습니다.</td></tr>';
        return;
    }

    tbody.innerHTML = logs.map(log => {
        // 백엔드에서 반환하는 필드명 그대로 사용
        // 백엔드: idx as id, model, run_time, result_url
        const id = log.id !== undefined ? log.id : '-';
        const model = log.model !== undefined ? log.model : '-';
        const runTime = log.run_time !== undefined ? log.run_time : null;
        const resultUrl = log.result_url !== undefined ? log.result_url : '';

        // 처리 시간 포맷팅 (숫자일 경우 소수점 2자리까지)
        let timeDisplay = '-';
        if (runTime !== null && runTime !== undefined) {
            if (typeof runTime === 'number') {
                timeDisplay = runTime.toFixed(2) + '초';
            } else {
                timeDisplay = String(runTime);
            }
        }

        return `
        <tr>
            <td>${id}</td>
            <td>${model}</td>
            <td>${timeDisplay}</td>
            <td>
                <button class="btn-detail-emoji" onclick="showDetail(${id})" title="상세보기">
                    ${resultUrl ? '🖼️' : '❌'}
                </button>
            </td>
        </tr>
    `;
    }).join('');
}

// 상태 배지 렌더링
function renderStatusBadge(success) {
    if (success) {
        return '<span class="status-badge status-success">성공</span>';
    } else {
        return '<span class="status-badge status-failed">실패</span>';
    }
}

// 페이지네이션 렌더링
function renderPagination(pagination) {
    const paginationDiv = document.getElementById('pagination');

    if (pagination.total_pages === 0) {
        paginationDiv.innerHTML = '';
        return;
    }

    // 페이지네이션 버튼 생성 함수
    const createPageButton = (pageNum, text, disabled = false, active = false) => {
        if (disabled) {
            return `<button disabled>${text}</button>`;
        }
        const activeClass = active ? ' class="active"' : '';
        return `<button onclick="loadLogsWithSearch(${pageNum})"${activeClass}>${text}</button>`;
    };

    let html = createPageButton(1, '처음', pagination.page === 1);

    // 이전 페이지
    if (pagination.page > 1) {
        html += createPageButton(pagination.page - 1, '이전');
    }

    // 페이지 번호들
    const startPage = Math.max(1, pagination.page - 2);
    const endPage = Math.min(pagination.total_pages, pagination.page + 2);

    if (startPage > 1) {
        html += '<button disabled>...</button>';
    }

    for (let i = startPage; i <= endPage; i++) {
        html += createPageButton(i, i.toString(), false, i === pagination.page);
    }

    if (endPage < pagination.total_pages) {
        html += '<button disabled>...</button>';
    }

    // 다음 페이지
    if (pagination.page < pagination.total_pages) {
        html += createPageButton(pagination.page + 1, '다음');
    }

    html += createPageButton(pagination.total_pages, '마지막', pagination.page === pagination.total_pages);

    html += `<span class="pagination-info">총 ${pagination.total}개 항목 (${pagination.page}/${pagination.total_pages} 페이지)</span>`;

    paginationDiv.innerHTML = html;
}

// 검색어를 포함한 로그 로드 (페이지네이션용)
function loadLogsWithSearch(page) {
    loadLogs(page, currentSearchModel);
}

// 로그 상세 보기
async function showDetail(logId) {
    try {
        const headers = window.getAuthHeaders ? window.getAuthHeaders() : {};
        const response = await fetch(`/api/admin/logs/${logId}`, {
            headers: headers
        });

        // 401 오류 처리
        if (response.status === 401) {
            // 인증 오류 시 조용히 로그인 페이지로 이동
            window.location.href = '/';
            return;
        }

        const data = await response.json();

        if (data.success) {
            renderDetailModal(data.data);
            openModal();
        } else {
            alert(data.message || '로그를 불러오는 중 오류가 발생했습니다.');
        }
    } catch (error) {
        console.error('상세 로그 로드 오류:', error);
        alert('로그를 불러오는 중 오류가 발생했습니다.');
    }
}

// 상세 모달 렌더링
function renderDetailModal(log) {
    const modalBody = document.getElementById('modal-body');

    // result_url이 있으면 이미지 표시, 없으면 메시지 표시
    const resultImageHtml = log.result_url ? `
        <div class="detail-item">
            <div class="detail-label">결과 이미지</div>
            <div class="image-preview-single">
                <img 
                    id="result-image" 
                    src="/api/admin/s3-image-proxy?url=${encodeURIComponent(log.result_url)}" 
                    alt="Result" 
                    loading="lazy"
                    onload="handleImageLoad(this);"
                    onerror="handleImageError(this, '${escapeHtml(log.result_url)}');"
                    style="opacity: 0; transition: opacity 0.3s;"
                >
                <div id="image-loading" style="text-align: center; padding: 20px; color: #666;">
                    ⏳ 이미지를 불러오는 중...
                </div>
                <div id="image-error" style="display: none; text-align: center; padding: 20px; color: #ef4444;">
                    ❌ 이미지를 불러올 수 없습니다
                    <br><small style="color: #999; word-break: break-all;">${escapeHtml(log.result_url)}</small>
                </div>
            </div>
        </div>
    ` : `
        <div class="detail-item">
            <div class="detail-label">결과 이미지</div>
            <div class="detail-value" style="color: #ef4444; text-align: center; padding: 20px;">
                ❌ 결과 이미지가 없습니다
            </div>
        </div>
    `;

    modalBody.innerHTML = `
        <div class="detail-grid">
            ${resultImageHtml}
        </div>
    `;

    // 이미지 로드 상태 확인
    if (log.result_url) {
        setTimeout(() => {
            const img = document.getElementById('result-image');
            const loading = document.getElementById('image-loading');

            if (img) {
                // 이미지가 이미 로드되어 있으면 loading 숨기기
                if (img.complete && img.naturalHeight !== 0) {
                    if (loading) loading.style.display = 'none';
                    img.style.opacity = '1';
                } else {
                    // 이미지 로딩 중 표시
                    if (loading) loading.style.display = 'block';
                }
            }
        }, 100);
    }
}

// 이미지 로드 성공 처리
function handleImageLoad(img) {
    img.style.opacity = '1';
    const loading = document.getElementById('image-loading');
    if (loading) loading.style.display = 'none';
}

// 이미지 로드 오류 처리
function handleImageError(img, url) {
    img.style.display = 'none';
    const loading = document.getElementById('image-loading');
    const error = document.getElementById('image-error');

    if (loading) loading.style.display = 'none';
    if (error) {
        error.style.display = 'block';
        // URL이 S3인 경우 CORS 문제일 수 있음을 표시
        if (url && (url.includes('s3') || url.includes('amazonaws.com'))) {
            const errorMsg = error.querySelector('small');
            if (errorMsg) {
                errorMsg.textContent = 'S3 이미지 로드 실패 (CORS 또는 네트워크 오류 가능)';
            }
        }
    }
}

// 모달 열기
function openModal() {
    document.getElementById('detail-modal').classList.add('show');
}

// 모달 닫기
function closeModal() {
    document.getElementById('detail-modal').classList.remove('show');
}

// 모달 외부 클릭 시 닫기
document.addEventListener('click', (e) => {
    const modal = document.getElementById('detail-modal');
    if (e.target === modal) {
        closeModal();
    }
});

// ESC 키로 모달 닫기
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeModal();
    }
});

// 유틸리티 함수들
function formatDateTime(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString('ko-KR', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
}

function formatTime(dateString) {
    const date = new Date(dateString);
    return date.toLocaleTimeString('ko-KR', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showError(message) {
    // 간단한 에러 표시 (필요시 토스트 메시지 등으로 변경 가능)
    alert(message);
}

// 체형 분석 로그 목록 로드
async function loadBodyLogs(page) {
    try {
        const url = `/api/admin/body-logs?page=${page}&limit=${itemsPerPage}`;

        const headers = window.getAuthHeaders ? window.getAuthHeaders() : {};
        const response = await fetch(url, {
            headers: headers
        });

        // 401 오류 처리
        if (response.status === 401) {
            // 인증 오류 시 조용히 로그인 페이지로 이동
            window.location.href = '/';
            return;
        }

        const data = await response.json();

        if (data.success) {
            renderBodyLogs(data.data);
            renderBodyPagination(data.pagination);
            updateBodyLogsCount(data.pagination.total);
            currentBodyPage = page;
        } else {
            showError(data.message || '체형 분석 로그를 불러오는 중 오류가 발생했습니다.');
        }
    } catch (error) {
        console.error('체형 분석 로그 로드 오류:', error);
        const tbody = document.getElementById('body-logs-tbody');
        if (tbody) {
            tbody.innerHTML = '<tr><td colspan="8" class="loading" style="color: #ef4444;">로그를 불러오는 중 오류가 발생했습니다.</td></tr>';
        }
    }
}

// 체형 분석 로그 갯수 업데이트
function updateBodyLogsCount(count) {
    const logsCountElement = document.getElementById('logs-count');
    if (logsCountElement) {
        logsCountElement.textContent = count;
    }
}

// 체형 분석 로그 테이블 렌더링
function renderBodyLogs(logs) {
    const tbody = document.getElementById('body-logs-tbody');

    if (!tbody) return;

    if (logs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" class="loading">로그가 없습니다.</td></tr>';
        return;
    }

    // 체형 특징을 부드러운 표현으로 변환하는 함수
    const softFeatureMap = {
        '키가 작은 체형': '키가 작으신 체형',
        '키가 큰 체형': '키가 크신 체형',
        '허리가 짧은 체형': '허리 비율이 짧으신 체형',
        '어깨가 넓은 체형': '균형잡힌 상체체형',
        '어깨가 좁은 체형': '어깨라인이 슬림한 체형',
        '마른 체형': '슬림한 체형',
        '글래머러스한 체형': '곡선미가 돋보이는 체형',
        '팔 라인이 신경 쓰이는 체형': '팔라인이 신경쓰이는 체형',
        '복부가 신경 쓰이는 체형': '' // 표시하지 않음
    };

    tbody.innerHTML = logs.map(log => {
        const id = log.id !== undefined ? log.id : '-';
        const model = log.model !== undefined ? log.model : '-';
        const height = log.height !== undefined && log.height !== null ? log.height + ' cm' : '-';
        const weight = log.weight !== undefined && log.weight !== null ? log.weight + ' kg' : '-';
        const bmi = log.bmi !== undefined && log.bmi !== null ? log.bmi.toFixed(1) : '-';

        // 체형 특징 파싱 및 변환
        let features = [];
        if (log.characteristic) {
            try {
                // JSON 문자열인 경우 파싱
                if (log.characteristic.startsWith('[') || log.characteristic.startsWith('{')) {
                    features = JSON.parse(log.characteristic);
                } else {
                    // 쉼표로 구분된 문자열인 경우
                    features = log.characteristic.split(',').map(f => f.trim()).filter(f => f);
                }
            } catch (e) {
                // 파싱 실패 시 그대로 사용
                features = [log.characteristic];
            }
        }

        // 부드러운 표현으로 변환
        const softFeatures = features.map(feature => {
            return softFeatureMap[feature] !== undefined ? softFeatureMap[feature] : feature;
        }).filter(f => f !== ''); // 빈 문자열 제거

        const featuresDisplay = softFeatures.length > 0 ? softFeatures.join(', ') : '-';
        const processingTime = log.processing_time || '-';

        return `
        <tr>
            <td>${id}</td>
            <td>${model}</td>
            <td>${height}</td>
            <td>${weight}</td>
            <td>${bmi}</td>
            <td style="max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${featuresDisplay}">${featuresDisplay}</td>
            <td>${processingTime}</td>
            <td>
                <button class="btn-detail-emoji" onclick="showBodyDetail(${id})" title="상세보기">
                    📋
                </button>
            </td>
        </tr>
    `;
    }).join('');
}

// 체형 분석 로그 페이지네이션 렌더링
function renderBodyPagination(pagination) {
    const paginationDiv = document.getElementById('body-pagination');

    if (!paginationDiv) return;

    if (pagination.total_pages === 0) {
        paginationDiv.innerHTML = '';
        return;
    }

    const createPageButton = (pageNum, text, disabled = false, active = false) => {
        if (disabled) {
            return `<button disabled>${text}</button>`;
        }
        const activeClass = active ? ' class="active"' : '';
        return `<button onclick="loadBodyLogs(${pageNum})"${activeClass}>${text}</button>`;
    };

    let html = createPageButton(1, '처음', pagination.page === 1);

    if (pagination.page > 1) {
        html += createPageButton(pagination.page - 1, '이전');
    }

    const startPage = Math.max(1, pagination.page - 2);
    const endPage = Math.min(pagination.total_pages, pagination.page + 2);

    if (startPage > 1) {
        html += '<button disabled>...</button>';
    }

    for (let i = startPage; i <= endPage; i++) {
        html += createPageButton(i, i.toString(), false, i === pagination.page);
    }

    if (endPage < pagination.total_pages) {
        html += '<button disabled>...</button>';
    }

    if (pagination.page < pagination.total_pages) {
        html += createPageButton(pagination.page + 1, '다음');
    }

    html += createPageButton(pagination.total_pages, '마지막', pagination.page === pagination.total_pages);

    html += `<span class="pagination-info">총 ${pagination.total}개 항목 (${pagination.page}/${pagination.total_pages} 페이지)</span>`;

    paginationDiv.innerHTML = html;
}

// 체형 분석 로그 상세 보기
async function showBodyDetail(logId) {
    try {
        const headers = window.getAuthHeaders ? window.getAuthHeaders() : {};
        const response = await fetch(`/api/admin/body-logs/${logId}`, {
            headers: headers
        });

        // 401 오류 처리
        if (response.status === 401) {
            // 인증 오류 시 조용히 로그인 페이지로 이동
            window.location.href = '/';
            return;
        }

        const data = await response.json();

        if (data.success) {
            renderBodyDetailModal(data.data);
            openModal();
        } else {
            alert(data.message || '로그를 불러오는 중 오류가 발생했습니다.');
        }
    } catch (error) {
        console.error('체형 분석 상세 로그 로드 오류:', error);
        alert('로그를 불러오는 중 오류가 발생했습니다.');
    }
}

// 체형 분석 상세 모달 렌더링
function renderBodyDetailModal(log) {
    const modalBody = document.getElementById('modal-body');

    if (!modalBody) return;

    const height = log.height !== undefined && log.height !== null ? log.height + ' cm' : '-';
    const weight = log.weight !== undefined && log.weight !== null ? log.weight + ' kg' : '-';
    const bmi = log.bmi !== undefined && log.bmi !== null ? log.bmi.toFixed(1) : '-';

    // characteristic은 쉼표로 구분된 문자열이거나 JSON 문자열일 수 있음
    let features = [];
    if (log.characteristic) {
        try {
            // JSON 문자열인 경우 파싱
            if (log.characteristic.startsWith('[') || log.characteristic.startsWith('{')) {
                features = JSON.parse(log.characteristic);
            } else {
                // 쉼표로 구분된 문자열인 경우
                features = log.characteristic.split(',').map(f => f.trim()).filter(f => f);
            }
        } catch (e) {
            // 파싱 실패 시 그대로 사용
            features = [log.characteristic];
        }
    }

    // 체형 특징을 부드러운 표현으로 변환
    const softFeatureMap = {
        '키가 작은 체형': '키가 작으신 체형',
        '키가 큰 체형': '키가 크신 체형',
        '허리가 짧은 체형': '허리 비율이 짧으신 체형',
        '어깨가 넓은 체형': '균형잡힌 상체체형',
        '어깨가 좁은 체형': '어깨라인이 슬림한 체형',
        '마른 체형': '슬림한 체형',
        '글래머러스한 체형': '곡선미가 돋보이는 체형',
        '팔 라인이 신경 쓰이는 체형': '팔라인이 신경쓰이는 체형',
        '복부가 신경 쓰이는 체형': '' // 표시하지 않음
    };

    // 부드러운 표현으로 변환
    features = features.map(feature => {
        return softFeatureMap[feature] !== undefined ? softFeatureMap[feature] : feature;
    }).filter(f => f !== ''); // 빈 문자열 제거

    const detailedAnalysis = log.analysis_results || '-';
    const runTime = log.run_time !== undefined && log.run_time !== null
        ? (typeof log.run_time === 'number' ? log.run_time.toFixed(2) + '초' : log.run_time)
        : '-';
    const createdAt = log.created_at ? formatDateTime(log.created_at) : '-';
    const processingTime = createdAt !== '-' && runTime !== '-'
        ? `${createdAt} (${runTime})`
        : createdAt !== '-' ? createdAt : runTime;

    const imageHtml = log.image_url ? `
        <div class="detail-item">
            <div class="detail-label">업로드 이미지</div>
            <div class="image-preview-single">
                <img 
                    src="/api/admin/s3-image-proxy?url=${encodeURIComponent(log.image_url)}" 
                    alt="Body Analysis" 
                    loading="lazy"
                    onload="handleImageLoad(this);"
                    onerror="handleImageError(this, '${escapeHtml(log.image_url)}');"
                    style="opacity: 0; transition: opacity 0.3s; max-width: 100%;"
                >
                <div id="image-loading" style="text-align: center; padding: 20px; color: #666;">
                    ⏳ 이미지를 불러오는 중...
                </div>
            </div>
        </div>
    ` : '';

    modalBody.innerHTML = `
        <div class="detail-grid">
            <div class="detail-item">
                <div class="detail-label">ID</div>
                <div class="detail-value">${log.id || '-'}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">키</div>
                <div class="detail-value">${height}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">몸무게</div>
                <div class="detail-value">${weight}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">BMI</div>
                <div class="detail-value">${bmi}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">체형 특징</div>
                <div class="detail-value">${features.join(', ') || '-'}</div>
            </div>
            <div class="detail-item" style="grid-column: 1 / -1;">
                <div class="detail-label">상세 분석</div>
                <div class="detail-value" style="white-space: pre-wrap; max-height: 300px; overflow-y: auto;">${detailedAnalysis}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">처리 시간</div>
                <div class="detail-value">${processingTime}</div>
            </div>
            ${imageHtml}
        </div>
    `;

    if (log.image_url) {
        setTimeout(() => {
            const img = modalBody.querySelector('img');
            const loading = modalBody.querySelector('#image-loading');

            if (img) {
                if (img.complete && img.naturalHeight !== 0) {
                    if (loading) loading.style.display = 'none';
                    img.style.opacity = '1';
                }
            }
        }, 100);
    }
}

// 리뷰 로그 목록 로드
async function loadReviews(page) {
    try {
        const url = `/api/reviews?limit=${itemsPerPage}&offset=${(page - 1) * itemsPerPage}`;

        const response = await fetch(url);
        const data = await response.json();

        if (data.success) {
            renderReviews(data.reviews);
            renderReviewsPagination(data.total, page);
            updateReviewsCount(data.total);
            currentReviewsPage = page;
        } else {
            showError('리뷰 로그를 불러오는 중 오류가 발생했습니다.');
        }
    } catch (error) {
        console.error('리뷰 로그 로드 오류:', error);
        const tbody = document.getElementById('reviews-logs-tbody');
        if (tbody) {
            tbody.innerHTML = '<tr><td colspan="5" class="loading">로그를 불러오는 중 오류가 발생했습니다.</td></tr>';
        }
    }
}

// 리뷰 로그 갯수 업데이트
function updateReviewsCount(count) {
    const logsCountElement = document.getElementById('logs-count');
    if (logsCountElement) {
        logsCountElement.textContent = count;
    }
}

// 리뷰 로그 테이블 렌더링
function renderReviews(reviews) {
    const tbody = document.getElementById('reviews-logs-tbody');

    if (!tbody) return;

    if (reviews.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="loading">리뷰가 없습니다.</td></tr>';
        return;
    }

    // 카테고리 한글 변환
    const categoryMap = {
        'general': '일반피팅',
        'custom': '커스텀피팅',
        'analysis': '체형분석'
    };

    tbody.innerHTML = reviews.map(review => {
        const id = review.idx !== undefined ? review.idx : '-';
        const category = categoryMap[review.category] || review.category || '-';
        const rating = review.rating !== undefined ? '⭐'.repeat(review.rating) + ` (${review.rating})` : '-';
        const content = review.content ? (review.content.length > 50 ? review.content.substring(0, 50) + '...' : review.content) : '-';
        const createdAt = review.created_at ? formatDateTime(review.created_at) : '-';

        return `
        <tr>
            <td>${id}</td>
            <td>${category}</td>
            <td>${rating}</td>
            <td style="max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${escapeHtml(review.content || '')}">${escapeHtml(content)}</td>
            <td>${createdAt}</td>
        </tr>
    `;
    }).join('');
}

// 리뷰 로그 페이지네이션 렌더링
function renderReviewsPagination(total, currentPage) {
    const paginationDiv = document.getElementById('reviews-pagination');

    if (!paginationDiv) return;

    const totalPages = Math.ceil(total / itemsPerPage);

    if (totalPages === 0) {
        paginationDiv.innerHTML = '';
        return;
    }

    const createPageButton = (pageNum, text, disabled = false, active = false) => {
        if (disabled) {
            return `<button disabled>${text}</button>`;
        }
        const activeClass = active ? ' class="active"' : '';
        return `<button onclick="loadReviews(${pageNum})"${activeClass}>${text}</button>`;
    };

    let html = createPageButton(1, '처음', currentPage === 1);

    if (currentPage > 1) {
        html += createPageButton(currentPage - 1, '이전');
    }

    const startPage = Math.max(1, currentPage - 2);
    const endPage = Math.min(totalPages, currentPage + 2);

    if (startPage > 1) {
        html += '<button disabled>...</button>';
    }

    for (let i = startPage; i <= endPage; i++) {
        html += createPageButton(i, i.toString(), false, i === currentPage);
    }

    if (endPage < totalPages) {
        html += '<button disabled>...</button>';
    }

    if (currentPage < totalPages) {
        html += createPageButton(currentPage + 1, '다음');
    }

    html += createPageButton(totalPages, '마지막', currentPage === totalPages);

    html += `<span class="pagination-info">총 ${total}개 항목 (${currentPage}/${totalPages} 페이지)</span>`;

    paginationDiv.innerHTML = html;
}


