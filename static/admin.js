// 전역 변수
let currentPage = 1;
const itemsPerPage = 20;
let currentSearchModel = null;
let currentTab = 'synthesis'; // 'synthesis', 'body', 'reviews', 'synthesis-stats', 'visitor-stats', 'custom-fitting', 'profile-logs'
let currentBodyPage = 1;
let currentReviewsPage = 1;
let currentSynthesisStatsPage = 1;
let currentVisitorStatsPage = 1;
let currentCustomFittingPage = 1;
let currentProfileLogsPage = 1;
let currentSearchDate = null; // 날짜 검색용
let currentProfileEndpoint = null; // 프로파일링 로그 엔드포인트 필터

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
            // JSON이 아닌 경우 - 서버 오류일 수 있으므로 토큰이 있으면 페이지 계속 로드
            const text = await response.text();
            console.warn('토큰 검증 응답이 JSON이 아닙니다 (페이지 계속 로드):', text);
            // 토큰이 있으므로 페이지는 계속 로드
            loadLogs(currentPage);
            return;
        }

        if (!response.ok || !data.success) {
            // 401, 403 오류일 때만 리다이렉트 (명확한 인증 오류)
            if (response.status === 401 || response.status === 403) {
                console.log('토큰 검증 실패:', data.message || data.error);
                window.location.href = '/';
                return;
            } else {
                // 다른 오류(500 등)는 일시적일 수 있으므로 페이지는 계속 로드
                console.warn('토큰 검증 중 오류 발생 (페이지 계속 로드):', data.message || data.error);
            }
        }
    } catch (error) {
        console.error('토큰 검증 오류:', error);
        // 네트워크 오류는 일시적일 수 있으므로 페이지는 계속 로드
        // 토큰이 있으면 일단 페이지를 표시하고, API 호출 시 다시 검증
    }

    loadLogs(currentPage);

    // 탭 버튼 이벤트 리스너
    const tabSynthesis = document.getElementById('tabSynthesis');
    const tabBodyAnalysis = document.getElementById('tabBodyAnalysis');
    const tabReviews = document.getElementById('tabReviews');
    const tabSynthesisStats = document.getElementById('tabSynthesisStats');
    const tabVisitorStats = document.getElementById('tabVisitorStats');
    const tabCustomFitting = document.getElementById('tabCustomFitting');
    const tabProfileLogs = document.getElementById('tabProfileLogs');

    if (tabSynthesis) {
        tabSynthesis.addEventListener('click', () => switchTab('synthesis'));
    }
    if (tabBodyAnalysis) {
        tabBodyAnalysis.addEventListener('click', () => switchTab('body'));
    }
    if (tabReviews) {
        tabReviews.addEventListener('click', () => switchTab('reviews'));
    }
    if (tabSynthesisStats) {
        tabSynthesisStats.addEventListener('click', () => switchTab('synthesis-stats'));
    }
    if (tabVisitorStats) {
        tabVisitorStats.addEventListener('click', () => switchTab('visitor-stats'));
    }
    if (tabCustomFitting) {
        tabCustomFitting.addEventListener('click', () => switchTab('custom-fitting'));
    }
    if (tabProfileLogs) {
        tabProfileLogs.addEventListener('click', () => switchTab('profile-logs'));
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

    // 날짜 검색 입력 필드에 Enter 키 이벤트 추가
    const dateSearchInput = document.getElementById('date-search-input');
    if (dateSearchInput) {
        dateSearchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                handleDateSearch();
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
    const synthesisStatsSection = document.getElementById('synthesis-stats-section');
    const visitorStatsSection = document.getElementById('visitor-stats-section');
    const customFittingSection = document.getElementById('custom-fitting-logs-section');
    const profileLogsSection = document.getElementById('profile-logs-section');
    const tabSynthesis = document.getElementById('tabSynthesis');
    const tabBodyAnalysis = document.getElementById('tabBodyAnalysis');
    const tabReviews = document.getElementById('tabReviews');
    const tabSynthesisStats = document.getElementById('tabSynthesisStats');
    const tabVisitorStats = document.getElementById('tabVisitorStats');
    const tabCustomFitting = document.getElementById('tabCustomFitting');
    const tabProfileLogs = document.getElementById('tabProfileLogs');
    const sectionTitle = document.getElementById('section-title');
    const logsCountLabel = document.getElementById('logs-count-label');
    const searchContainerText = document.getElementById('search-container-text');
    const searchContainerDate = document.getElementById('search-container-date');

    // 모든 섹션 숨기기
    if (synthesisSection) synthesisSection.style.display = 'none';
    if (bodySection) bodySection.style.display = 'none';
    if (reviewsSection) reviewsSection.style.display = 'none';
    if (synthesisStatsSection) synthesisStatsSection.style.display = 'none';
    if (visitorStatsSection) visitorStatsSection.style.display = 'none';
    if (customFittingSection) customFittingSection.style.display = 'none';
    if (profileLogsSection) profileLogsSection.style.display = 'none';

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
    if (tabSynthesisStats) {
        tabSynthesisStats.classList.remove('active');
        tabSynthesisStats.style.background = '#fff';
        tabSynthesisStats.style.color = '#333';
    }
    if (tabVisitorStats) {
        tabVisitorStats.classList.remove('active');
        tabVisitorStats.style.background = '#fff';
        tabVisitorStats.style.color = '#333';
    }
    if (tabCustomFitting) {
        tabCustomFitting.classList.remove('active');
        tabCustomFitting.style.background = '#fff';
        tabCustomFitting.style.color = '#333';
    }
    if (tabProfileLogs) {
        tabProfileLogs.classList.remove('active');
        tabProfileLogs.style.background = '#fff';
        tabProfileLogs.style.color = '#333';
    }

    if (tab === 'synthesis') {
        if (synthesisSection) synthesisSection.style.display = 'block';
        if (tabSynthesis) {
            tabSynthesis.classList.add('active');
            tabSynthesis.style.background = '#007bff';
            tabSynthesis.style.color = '#fff';
        }
        if (sectionTitle) sectionTitle.textContent = '📋 합성 로그';
        if (logsCountLabel) logsCountLabel.textContent = '전체 합성:';
        if (searchContainerText) searchContainerText.style.display = 'block';
        if (searchContainerDate) searchContainerDate.style.display = 'none';
        // 다른 탭으로 전환 시 날짜 검색 초기화
        currentSearchDate = null;
        loadLogs(currentPage, currentSearchModel);
    } else if (tab === 'body') {
        if (bodySection) bodySection.style.display = 'block';
        if (tabBodyAnalysis) {
            tabBodyAnalysis.classList.add('active');
            tabBodyAnalysis.style.background = '#007bff';
            tabBodyAnalysis.style.color = '#fff';
        }
        if (sectionTitle) sectionTitle.textContent = '📊 분석 로그';
        if (logsCountLabel) logsCountLabel.textContent = '전체 분석:';
        if (searchContainerText) searchContainerText.style.display = 'none';
        if (searchContainerDate) searchContainerDate.style.display = 'none';
        // 다른 탭으로 전환 시 날짜 검색 초기화
        currentSearchDate = null;
        loadBodyLogs(currentBodyPage);
    } else if (tab === 'reviews') {
        if (reviewsSection) reviewsSection.style.display = 'block';
        if (tabReviews) {
            tabReviews.classList.add('active');
            tabReviews.style.background = '#007bff';
            tabReviews.style.color = '#fff';
        }
        if (sectionTitle) sectionTitle.textContent = '⭐ 리뷰 로그';
        if (logsCountLabel) logsCountLabel.textContent = '전체 리뷰:';
        if (searchContainerText) searchContainerText.style.display = 'none';
        if (searchContainerDate) searchContainerDate.style.display = 'none';
        // 다른 탭으로 전환 시 날짜 검색 초기화
        currentSearchDate = null;
        loadReviews(currentReviewsPage);
    } else if (tab === 'synthesis-stats') {
        if (synthesisStatsSection) synthesisStatsSection.style.display = 'block';
        if (tabSynthesisStats) {
            tabSynthesisStats.classList.add('active');
            tabSynthesisStats.style.background = '#007bff';
            tabSynthesisStats.style.color = '#fff';
        }
        if (sectionTitle) sectionTitle.textContent = '📈 합성 통계';
        if (logsCountLabel) logsCountLabel.textContent = '전체 날짜:';
        if (searchContainerText) searchContainerText.style.display = 'none';
        if (searchContainerDate) searchContainerDate.style.display = 'block';
        // 날짜 검색 입력 필드에 현재 검색 날짜 설정
        const dateSearchInput = document.getElementById('date-search-input');
        const dateSearchClearButton = document.getElementById('date-search-clear-button');
        if (dateSearchInput && currentSearchDate) {
            dateSearchInput.value = currentSearchDate;
        }
        if (dateSearchClearButton) {
            dateSearchClearButton.style.display = currentSearchDate ? 'inline-block' : 'none';
        }
        loadDailySynthesisStats(currentSynthesisStatsPage, currentSearchDate);
    } else if (tab === 'visitor-stats') {
        if (visitorStatsSection) visitorStatsSection.style.display = 'block';
        if (tabVisitorStats) {
            tabVisitorStats.classList.add('active');
            tabVisitorStats.style.background = '#007bff';
            tabVisitorStats.style.color = '#fff';
        }
        if (sectionTitle) sectionTitle.textContent = '👥 조회수 통계';
        if (logsCountLabel) logsCountLabel.textContent = '전체 날짜:';
        if (searchContainerText) searchContainerText.style.display = 'none';
        if (searchContainerDate) searchContainerDate.style.display = 'block';
        // 날짜 검색 입력 필드에 현재 검색 날짜 설정
        const dateSearchInput = document.getElementById('date-search-input');
        const dateSearchClearButton = document.getElementById('date-search-clear-button');
        if (dateSearchInput && currentSearchDate) {
            dateSearchInput.value = currentSearchDate;
        }
        if (dateSearchClearButton) {
            dateSearchClearButton.style.display = currentSearchDate ? 'inline-block' : 'none';
        }
        loadDailyVisitorStats(currentVisitorStatsPage, currentSearchDate);
    } else if (tab === 'custom-fitting') {
        if (customFittingSection) customFittingSection.style.display = 'block';
        if (tabCustomFitting) {
            tabCustomFitting.classList.add('active');
            tabCustomFitting.style.background = '#007bff';
            tabCustomFitting.style.color = '#fff';
        }
        if (sectionTitle) sectionTitle.textContent = '👔 커스텀피팅로그';
        if (logsCountLabel) logsCountLabel.textContent = '전체 커스텀피팅:';
        if (searchContainerText) searchContainerText.style.display = 'none';
        if (searchContainerDate) searchContainerDate.style.display = 'none';
        // 다른 탭으로 전환 시 날짜 검색 초기화
        currentSearchDate = null;
        loadCustomFittingLogs(currentCustomFittingPage);
    } else if (tab === 'profile-logs') {
        if (profileLogsSection) profileLogsSection.style.display = 'block';
        if (tabProfileLogs) {
            tabProfileLogs.classList.add('active');
            tabProfileLogs.style.background = '#007bff';
            tabProfileLogs.style.color = '#fff';
        }
        if (sectionTitle) sectionTitle.textContent = '⏱️ 피팅 프로파일링';
        if (logsCountLabel) logsCountLabel.textContent = '전체 프로파일링:';
        if (searchContainerText) searchContainerText.style.display = 'none';
        if (searchContainerDate) searchContainerDate.style.display = 'none';
        currentSearchDate = null;
        // 기본값은 일반 피팅
        if (!currentProfileEndpoint) {
            currentProfileEndpoint = '/tryon/compare';
        }
        loadProfileLogs(currentProfileLogsPage, currentProfileEndpoint);
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

// 날짜 검색 처리
function handleDateSearch() {
    const dateSearchInput = document.getElementById('date-search-input');
    const dateValue = dateSearchInput ? dateSearchInput.value.trim() : '';
    const clearButton = document.getElementById('date-search-clear-button');

    currentSearchDate = dateValue || null;
    
    // 통계 탭에 따라 해당 페이지 초기화 및 데이터 로드
    if (currentTab === 'synthesis-stats') {
        currentSynthesisStatsPage = 1;
        loadDailySynthesisStats(currentSynthesisStatsPage, currentSearchDate);
    } else if (currentTab === 'visitor-stats') {
        currentVisitorStatsPage = 1;
        loadDailyVisitorStats(currentVisitorStatsPage, currentSearchDate);
    }

    // 검색어가 있으면 초기화 버튼 표시
    if (clearButton) {
        clearButton.style.display = dateValue ? 'inline-block' : 'none';
    }
}

// 날짜 검색 초기화
function clearDateSearch() {
    const dateSearchInput = document.getElementById('date-search-input');
    const clearButton = document.getElementById('date-search-clear-button');

    if (dateSearchInput) {
        dateSearchInput.value = '';
    }
    if (clearButton) {
        clearButton.style.display = 'none';
    }

    currentSearchDate = null;
    
    // 통계 탭에 따라 해당 페이지 초기화 및 데이터 로드
    if (currentTab === 'synthesis-stats') {
        currentSynthesisStatsPage = 1;
        loadDailySynthesisStats(currentSynthesisStatsPage);
    } else if (currentTab === 'visitor-stats') {
        currentVisitorStatsPage = 1;
        loadDailyVisitorStats(currentVisitorStatsPage);
    }
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

// 날짜별 합성 통계 로드
async function loadDailySynthesisStats(page, date = null) {
    try {
        let url = `/api/admin/daily-synthesis-stats?page=${page}&limit=${itemsPerPage}`;
        if (date && date.trim() !== '') {
            url += `&date=${encodeURIComponent(date.trim())}`;
        }

        const headers = window.getAuthHeaders ? window.getAuthHeaders() : {};
        const response = await fetch(url, {
            headers: headers
        });

        // 401 오류 처리
        if (response.status === 401) {
            window.location.href = '/';
            return;
        }

        const data = await response.json();

        if (data.success) {
            renderDailySynthesisStats(data.data);
            renderDailySynthesisStatsPagination(data.pagination);
            updateSynthesisStatsCount(data.pagination.total);
            currentSynthesisStatsPage = page;
        } else {
            showError('날짜별 합성 통계를 불러오는 중 오류가 발생했습니다.');
        }
    } catch (error) {
        console.error('날짜별 합성 통계 로드 오류:', error);
        const tbody = document.getElementById('synthesis-stats-tbody');
        if (tbody) {
            tbody.innerHTML = '<tr><td colspan="2" class="loading">데이터를 불러오는 중 오류가 발생했습니다.</td></tr>';
        }
    }
}

// 날짜별 합성 통계 테이블 렌더링
function renderDailySynthesisStats(stats) {
    const tbody = document.getElementById('synthesis-stats-tbody');

    if (!tbody) return;

    if (stats.length === 0) {
        tbody.innerHTML = '<tr><td colspan="2" class="loading">데이터가 없습니다.</td></tr>';
        return;
    }

    tbody.innerHTML = stats.map(stat => {
        const date = stat.date || stat.synthesis_date || '-';
        const count = stat.count !== undefined ? stat.count : 0;

        return `
        <tr>
            <td>${date}</td>
            <td>${count}</td>
        </tr>
    `;
    }).join('');
}

// 날짜별 합성 통계 페이지네이션 렌더링
function renderDailySynthesisStatsPagination(pagination) {
    const paginationDiv = document.getElementById('synthesis-stats-pagination');

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
        const dateParam = currentSearchDate ? `, '${currentSearchDate}'` : '';
        return `<button onclick="loadDailySynthesisStats(${pageNum}${dateParam})"${activeClass}>${text}</button>`;
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

// 날짜별 합성 통계 카운트 업데이트
function updateSynthesisStatsCount(count) {
    const logsCountElement = document.getElementById('logs-count');
    if (logsCountElement) {
        logsCountElement.textContent = count;
    }
}

// 날짜별 조회수 통계 로드
async function loadDailyVisitorStats(page, date = null) {
    try {
        let url = `/api/admin/daily-visitor-stats?page=${page}&limit=${itemsPerPage}`;
        if (date && date.trim() !== '') {
            url += `&date=${encodeURIComponent(date.trim())}`;
        }

        const headers = window.getAuthHeaders ? window.getAuthHeaders() : {};
        const response = await fetch(url, {
            headers: headers
        });

        // 401 오류 처리
        if (response.status === 401) {
            window.location.href = '/';
            return;
        }

        const data = await response.json();

        if (data.success) {
            renderDailyVisitorStats(data.data);
            renderDailyVisitorStatsPagination(data.pagination);
            updateVisitorStatsCount(data.pagination.total);
            currentVisitorStatsPage = page;
        } else {
            showError('날짜별 조회수 통계를 불러오는 중 오류가 발생했습니다.');
        }
    } catch (error) {
        console.error('날짜별 조회수 통계 로드 오류:', error);
        const tbody = document.getElementById('visitor-stats-tbody');
        if (tbody) {
            tbody.innerHTML = '<tr><td colspan="2" class="loading">데이터를 불러오는 중 오류가 발생했습니다.</td></tr>';
        }
    }
}

// 날짜별 조회수 통계 테이블 렌더링
function renderDailyVisitorStats(stats) {
    const tbody = document.getElementById('visitor-stats-tbody');

    if (!tbody) return;

    if (stats.length === 0) {
        tbody.innerHTML = '<tr><td colspan="2" class="loading">데이터가 없습니다.</td></tr>';
        return;
    }

    tbody.innerHTML = stats.map(stat => {
        const date = stat.date || stat.visit_date || '-';
        const count = stat.count !== undefined ? stat.count : 0;

        return `
        <tr>
            <td>${date}</td>
            <td>${count}</td>
        </tr>
    `;
    }).join('');
}

// 날짜별 조회수 통계 페이지네이션 렌더링
function renderDailyVisitorStatsPagination(pagination) {
    const paginationDiv = document.getElementById('visitor-stats-pagination');

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
        const dateParam = currentSearchDate ? `, '${currentSearchDate}'` : '';
        return `<button onclick="loadDailyVisitorStats(${pageNum}${dateParam})"${activeClass}>${text}</button>`;
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

// 날짜별 조회수 통계 카운트 업데이트
function updateVisitorStatsCount(count) {
    const logsCountElement = document.getElementById('logs-count');
    if (logsCountElement) {
        logsCountElement.textContent = count;
    }
}

// 커스텀 피팅 로그 목록 로드
async function loadCustomFittingLogs(page) {
    try {
        const url = `/api/admin/custom-fitting-logs?page=${page}&limit=${itemsPerPage}`;

        const headers = window.getAuthHeaders ? window.getAuthHeaders() : {};
        const response = await fetch(url, {
            headers: headers
        });

        // 401 오류 처리
        if (response.status === 401) {
            window.location.href = '/';
            return;
        }

        const data = await response.json();

        if (data.success) {
            renderCustomFittingLogs(data.data);
            renderCustomFittingPagination(data.pagination);
            updateCustomFittingCount(data.pagination.total);
            currentCustomFittingPage = page;
        } else {
            showError(data.message || '커스텀 피팅 로그를 불러오는 중 오류가 발생했습니다.');
        }
    } catch (error) {
        console.error('커스텀 피팅 로그 로드 오류:', error);
        const tbody = document.getElementById('custom-fitting-logs-tbody');
        if (tbody) {
            tbody.innerHTML = '<tr><td colspan="4" class="loading" style="color: #ef4444;">로그를 불러오는 중 오류가 발생했습니다.</td></tr>';
        }
    }
}

// 커스텀 피팅 로그 테이블 렌더링
function renderCustomFittingLogs(logs) {
    const tbody = document.getElementById('custom-fitting-logs-tbody');

    if (!tbody) return;

    if (logs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" class="loading">로그가 없습니다.</td></tr>';
        return;
    }

    tbody.innerHTML = logs.map(log => {
        const id = log.id !== undefined ? log.id : '-';
        
        // 생성일시 포맷팅 (안전하게 처리)
        let createdAt = '-';
        if (log.created_at) {
            try {
                let dateStr = String(log.created_at).trim();
                
                // 빈 문자열 체크
                if (!dateStr || dateStr === 'None' || dateStr === 'null') {
                    createdAt = '-';
                } else {
                    // MySQL datetime 형식 (YYYY-MM-DD HH:MM:SS) 처리
                    // 공백을 T로 변환
                    if (dateStr.includes(' ') && !dateStr.includes('T')) {
                        dateStr = dateStr.replace(' ', 'T');
                    }
                    
                    // T가 없으면 추가 (날짜만 있는 경우)
                    if (!dateStr.includes('T')) {
                        dateStr = dateStr + 'T00:00:00';
                    }
                    
                    // 타임존 정보가 없으면 UTC로 가정 (Z 추가하지 않음, 로컬 시간으로 표시)
                    const date = new Date(dateStr);
                    
                    if (!isNaN(date.getTime())) {
                        // 유효한 날짜인 경우 포맷팅
                        createdAt = formatDateTime(dateStr);
                    } else {
                        // 파싱 실패 시 원본 문자열 표시
                        console.warn('날짜 파싱 실패:', log.created_at);
                        createdAt = String(log.created_at);
                    }
                }
            } catch (e) {
                // 에러 발생 시 원본 문자열 표시
                console.error('날짜 포맷팅 오류:', e, log.created_at);
                createdAt = String(log.created_at) || '-';
            }
        }
        
        const runTime = log.run_time !== undefined && log.run_time !== null
            ? (typeof log.run_time === 'number' ? log.run_time.toFixed(2) + '초' : log.run_time)
            : '-';
        const dressUrl = log.dress_url || '';

        // 의상 이미지 표시 (있으면 이미지, 없으면 메시지)
        // 클릭 시 모달 팝업으로 표시
        const dressImageHtml = dressUrl
            ? `<img src="/api/admin/s3-image-proxy?url=${encodeURIComponent(dressUrl)}" alt="의상 이미지" style="max-width: 100px; max-height: 100px; cursor: pointer;" onclick="showDressImageModal('${escapeHtml(dressUrl)}')" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';"><span style="display: none; color: #999;">이미지 없음</span>`
            : '<span style="color: #999;">-</span>';

        return `
        <tr>
            <td>${id}</td>
            <td>${createdAt}</td>
            <td>${runTime}</td>
            <td>${dressImageHtml}</td>
        </tr>
    `;
    }).join('');
}

// 커스텀 피팅 로그 페이지네이션 렌더링
function renderCustomFittingPagination(pagination) {
    const paginationDiv = document.getElementById('custom-fitting-pagination');

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
        return `<button onclick="loadCustomFittingLogs(${pageNum})"${activeClass}>${text}</button>`;
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

// 커스텀 피팅 로그 카운트 업데이트
function updateCustomFittingCount(count) {
    const logsCountElement = document.getElementById('logs-count');
    if (logsCountElement) {
        logsCountElement.textContent = count;
    }
}

// 의상 이미지 모달 표시
function showDressImageModal(dressUrl) {
    const modalBody = document.getElementById('modal-body');
    if (!modalBody) return;

    const dressImageHtml = dressUrl ? `
        <div class="detail-item" style="grid-column: 1 / -1;">
            <div class="detail-label">의상 이미지</div>
            <div class="image-preview-single">
                <img 
                    id="dress-image" 
                    src="/api/admin/s3-image-proxy?url=${encodeURIComponent(dressUrl)}" 
                    alt="의상 이미지" 
                    loading="lazy"
                    onload="handleImageLoad(this);"
                    onerror="handleImageError(this, '${escapeHtml(dressUrl)}');"
                    style="opacity: 0; transition: opacity 0.3s; max-width: 100%;"
                >
                <div id="image-loading" style="text-align: center; padding: 20px; color: #666;">
                    ⏳ 이미지를 불러오는 중...
                </div>
                <div id="image-error" style="display: none; text-align: center; padding: 20px; color: #ef4444;">
                    ❌ 이미지를 불러올 수 없습니다
                    <br><small style="color: #999; word-break: break-all;">${escapeHtml(dressUrl)}</small>
                </div>
            </div>
        </div>
    ` : `
        <div class="detail-item">
            <div class="detail-label">의상 이미지</div>
            <div class="detail-value" style="color: #ef4444; text-align: center; padding: 20px;">
                ❌ 의상 이미지가 없습니다
            </div>
        </div>
    `;

    modalBody.innerHTML = `
        <div class="detail-grid">
            ${dressImageHtml}
        </div>
    `;

    // 이미지 로드 상태 확인
    if (dressUrl) {
        setTimeout(() => {
            const img = document.getElementById('dress-image');
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

    // 모달 제목 변경
    const modalTitle = document.querySelector('#detail-modal .modal-header h3');
    if (modalTitle) {
        modalTitle.textContent = '👔 의상 이미지';
    }

    openModal();
}

// 프로파일링 로그 로드
async function loadProfileLogs(page = 1, endpoint = null) {
    try {
        currentProfileLogsPage = page;
        currentProfileEndpoint = endpoint || '/tryon/compare';
        
        // 테이블 헤더 업데이트
        updateProfileLogsTableHeader(currentProfileEndpoint);
        
        const headers = window.getAuthHeaders ? window.getAuthHeaders() : {};
        let url = `/api/admin/tryon-profile-logs?page=${page}&limit=${itemsPerPage}`;
        if (currentProfileEndpoint) {
            url += `&endpoint=${encodeURIComponent(currentProfileEndpoint)}`;
        }
        
        const response = await fetch(url, {
            headers: headers
        });
        
        if (response.status === 401) {
            window.location.href = '/';
            return;
        }
        
        const data = await response.json();
        
        if (data.success) {
            renderProfileLogs(data.data);
            renderProfileLogsPagination(data.pagination);
            updateLogsCount(data.pagination.total);
        } else {
            const tbody = document.getElementById('profile-logs-tbody');
            const isCustom = currentProfileEndpoint === '/tryon/compare/custom';
            const colspan = isCustom ? 9 : 7;
            if (tbody) {
                tbody.innerHTML = `<tr><td colspan="${colspan}" class="loading">${data.message || '로그를 불러오는 중 오류가 발생했습니다.'}</td></tr>`;
            }
        }
    } catch (error) {
        console.error('프로파일링 로그 로드 오류:', error);
        const tbody = document.getElementById('profile-logs-tbody');
        const isCustom = currentProfileEndpoint === '/tryon/compare/custom';
        const colspan = isCustom ? 9 : 7;
        if (tbody) {
            tbody.innerHTML = `<tr><td colspan="${colspan}" class="loading">로그를 불러오는 중 오류가 발생했습니다.</td></tr>`;
        }
    }
}

// 프로파일링 로그 테이블 헤더 업데이트
function updateProfileLogsTableHeader(endpoint) {
    const thead = document.getElementById('profile-logs-thead');
    if (!thead) return;
    
    const isCustom = endpoint === '/tryon/compare/custom';
    
    if (isCustom) {
        // 커스텀 피팅: ID, 카테고리, 생성일시, 서버 총시간, 인물 예외처리, 드레스 예외처리, 누끼처리, Gemini 호출, 상세보기
        thead.innerHTML = `
            <tr>
                <th>ID</th>
                <th>카테고리</th>
                <th>생성일시</th>
                <th>서버 총 시간 (ms)</th>
                <th>인물 예외처리 (ms)</th>
                <th>드레스 예외처리 (ms)</th>
                <th>누끼 처리 (ms)</th>
                <th>Gemini 호출 (ms)</th>
                <th>상세보기</th>
            </tr>
        `;
    } else {
        // 일반 피팅: ID, 카테고리, 생성일시, 서버 총시간, 인물 예외처리, Gemini 호출, 상세보기
        thead.innerHTML = `
            <tr>
                <th>ID</th>
                <th>카테고리</th>
                <th>생성일시</th>
                <th>서버 총 시간 (ms)</th>
                <th>인물 예외처리 (ms)</th>
                <th>Gemini 호출 (ms)</th>
                <th>상세보기</th>
            </tr>
        `;
    }
}

// 프로파일링 로그 렌더링
function renderProfileLogs(logs) {
    const tbody = document.getElementById('profile-logs-tbody');
    if (!tbody) return;
    
    const isCustom = currentProfileEndpoint === '/tryon/compare/custom';
    const colspan = isCustom ? 9 : 7;
    
    if (logs.length === 0) {
        tbody.innerHTML = `<tr><td colspan="${colspan}" class="loading">로그가 없습니다.</td></tr>`;
        return;
    }
    
    tbody.innerHTML = logs.map(log => {
        const id = log.id !== undefined ? log.id : '-';
        const endpoint = log.endpoint || '-';
        const category = endpoint === '/tryon/compare' ? '일반 피팅' : endpoint === '/tryon/compare/custom' ? '커스텀 피팅' : endpoint;
        const createdAt = log.created_at ? new Date(log.created_at).toLocaleString('ko-KR') : '-';
        const serverTotalMs = log.server_total_ms !== null && log.server_total_ms !== undefined ? 
            (typeof log.server_total_ms === 'number' ? log.server_total_ms.toFixed(2) : log.server_total_ms) : '-';
        const geminiCallMs = log.gemini_call_ms !== null && log.gemini_call_ms !== undefined ? 
            (typeof log.gemini_call_ms === 'number' ? log.gemini_call_ms.toFixed(2) : log.gemini_call_ms) : '-';
        const cutoutMs = log.cutout_ms !== null && log.cutout_ms !== undefined ? 
            (typeof log.cutout_ms === 'number' ? log.cutout_ms.toFixed(2) : log.cutout_ms) : '-';
        
        // 프론트엔드 프로파일링 데이터에서 추출
        const frontProfile = log.front_profile || {};
        const personValidateMs = frontProfile.person_validate_ms !== null && frontProfile.person_validate_ms !== undefined ? 
            (typeof frontProfile.person_validate_ms === 'number' ? frontProfile.person_validate_ms.toFixed(2) : frontProfile.person_validate_ms) : '-';
        const dressValidateMs = frontProfile.dress_validate_ms !== null && frontProfile.dress_validate_ms !== undefined ? 
            (typeof frontProfile.dress_validate_ms === 'number' ? frontProfile.dress_validate_ms.toFixed(2) : frontProfile.dress_validate_ms) : '-';
        
        if (isCustom) {
            // 커스텀 피팅: ID, 카테고리, 생성일시, 서버 총시간, 인물 예외처리, 드레스 예외처리, 누끼처리, Gemini 호출, 상세보기
            return `
            <tr>
                <td>${id}</td>
                <td>${category}</td>
                <td>${createdAt}</td>
                <td>${serverTotalMs}</td>
                <td>${personValidateMs}</td>
                <td>${dressValidateMs}</td>
                <td>${cutoutMs}</td>
                <td>${geminiCallMs}</td>
                <td>
                    <button class="btn-detail-emoji" onclick="showProfileDetail(${id})" title="상세보기">
                        📋
                    </button>
                </td>
            </tr>
        `;
        } else {
            // 일반 피팅: ID, 카테고리, 생성일시, 서버 총시간, 인물 예외처리, Gemini 호출, 상세보기
            return `
            <tr>
                <td>${id}</td>
                <td>${category}</td>
                <td>${createdAt}</td>
                <td>${serverTotalMs}</td>
                <td>${personValidateMs}</td>
                <td>${geminiCallMs}</td>
                <td>
                    <button class="btn-detail-emoji" onclick="showProfileDetail(${id})" title="상세보기">
                        📋
                    </button>
                </td>
            </tr>
        `;
        }
    }).join('');
}

// 프로파일링 로그 페이지네이션 렌더링
function renderProfileLogsPagination(pagination) {
    const paginationDiv = document.getElementById('profile-logs-pagination');
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
        return `<button onclick="loadProfileLogsWithFilter(${pageNum})"${activeClass}>${text}</button>`;
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

// 프로파일링 로그 필터링
function filterProfileLogs(endpoint) {
    currentProfileEndpoint = endpoint;
    currentProfileLogsPage = 1;
    
    // 필터 버튼 활성화 상태 업데이트
    const generalBtn = document.getElementById('profile-filter-general');
    const customBtn = document.getElementById('profile-filter-custom');
    
    if (generalBtn) {
        generalBtn.classList.remove('active');
        generalBtn.style.background = '#fff';
        generalBtn.style.color = '#333';
    }
    if (customBtn) {
        customBtn.classList.remove('active');
        customBtn.style.background = '#fff';
        customBtn.style.color = '#333';
    }
    
    if (endpoint === '/tryon/compare' && generalBtn) {
        generalBtn.classList.add('active');
        generalBtn.style.background = '#007bff';
        generalBtn.style.color = '#fff';
    } else if (endpoint === '/tryon/compare/custom' && customBtn) {
        customBtn.classList.add('active');
        customBtn.style.background = '#007bff';
        customBtn.style.color = '#fff';
    }
    
    loadProfileLogs(1, endpoint);
}

// 프로파일링 로그 필터링 포함 로드
function loadProfileLogsWithFilter(page) {
    loadProfileLogs(page, currentProfileEndpoint);
}


// 프로파일링 로그 상세보기
async function showProfileDetail(logId) {
    try {
        const headers = window.getAuthHeaders ? window.getAuthHeaders() : {};
        const response = await fetch(`/api/admin/tryon-profile-logs/${logId}`, {
            headers: headers
        });
        
        if (response.status === 401) {
            window.location.href = '/';
            return;
        }
        
        const data = await response.json();
        
        if (data.success) {
            renderProfileDetailModal(data.data);
            openModal();
        } else {
            alert(data.message || '로그를 불러오는 중 오류가 발생했습니다.');
        }
    } catch (error) {
        console.error('프로파일링 로그 상세 로드 오류:', error);
        alert('로그를 불러오는 중 오류가 발생했습니다.');
    }
}

// 프로파일링 로그 상세 모달 렌더링
function renderProfileDetailModal(log) {
    const modalBody = document.getElementById('modal-body');
    if (!modalBody) return;
    
    const frontProfile = log.front_profile || {};
    const category = log.endpoint === '/tryon/compare' ? '일반 피팅' : log.endpoint === '/tryon/compare/custom' ? '커스텀 피팅' : log.endpoint;
    const isCustom = category === '커스텀 피팅';
    
    // 인물/드레스 예외처리 시간 추출
    const personValidateMs = frontProfile.person_validate_ms !== null && frontProfile.person_validate_ms !== undefined ? 
        (typeof frontProfile.person_validate_ms === 'number' ? frontProfile.person_validate_ms.toFixed(2) + ' ms' : frontProfile.person_validate_ms) : '-';
    const dressValidateMs = frontProfile.dress_validate_ms !== null && frontProfile.dress_validate_ms !== undefined ? 
        (typeof frontProfile.dress_validate_ms === 'number' ? frontProfile.dress_validate_ms.toFixed(2) + ' ms' : frontProfile.dress_validate_ms) : '-';
    
    // 일반 피팅: 카테고리 / 생성일시 / 서버 총시간 / 인물 예외처리 / Gemini 호출
    // 커스텀 피팅: 카테고리 / 생성일시 / 서버 총시간 / 인물 예외처리 / 드레스 예외처리 / 누끼처리 / Gemini 호출
    let detailItems = `
        <div class="detail-item">
            <div class="detail-label">카테고리</div>
            <div class="detail-value">${category}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">생성일시</div>
            <div class="detail-value">${log.created_at ? new Date(log.created_at).toLocaleString('ko-KR') : '-'}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">서버 총 시간</div>
            <div class="detail-value">${log.server_total_ms !== null && log.server_total_ms !== undefined ? log.server_total_ms.toFixed(2) + ' ms' : '-'}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">인물 예외처리 시간</div>
            <div class="detail-value">${personValidateMs}</div>
        </div>
    `;
    
    // 커스텀 피팅만 추가 항목
    if (isCustom) {
        detailItems += `
        <div class="detail-item">
            <div class="detail-label">드레스 예외처리 시간</div>
            <div class="detail-value">${dressValidateMs}</div>
        </div>
        `;
        
        if (log.cutout_ms !== null && log.cutout_ms !== undefined) {
            detailItems += `
        <div class="detail-item">
            <div class="detail-label">누끼 처리 시간</div>
            <div class="detail-value">${log.cutout_ms.toFixed(2)} ms</div>
        </div>
            `;
        }
    }
    
    detailItems += `
        <div class="detail-item">
            <div class="detail-label">Gemini 호출 시간</div>
            <div class="detail-value">${log.gemini_call_ms !== null && log.gemini_call_ms !== undefined ? log.gemini_call_ms.toFixed(2) + ' ms' : '-'}</div>
        </div>
    `;
    
    // 에러 단계가 있으면 추가
    if (log.error_stage) {
        detailItems += `
        <div class="detail-item">
            <div class="detail-label">에러 단계</div>
            <div class="detail-value" style="color: #ef4444;">${log.error_stage}</div>
        </div>
        `;
    }
    
    modalBody.innerHTML = `
        <div class="detail-grid">
            ${detailItems}
        </div>
    `;
    
    // 모달 제목 변경
    const modalTitle = document.querySelector('#detail-modal .modal-header h3');
    if (modalTitle) {
        modalTitle.textContent = '⏱️ 프로파일링 로그 상세';
    }
}


