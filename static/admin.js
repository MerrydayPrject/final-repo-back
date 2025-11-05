// 전역 변수
let currentPage = 1;
const itemsPerPage = 20;
let currentSearchModel = null;

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', () => {
    loadLogs(currentPage);
    
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

// 통계 로드
async function loadStats() {
    try {
        const response = await fetch('/api/admin/stats');
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
        
        const response = await fetch(url);
        const data = await response.json();
        
        if (data.success) {
            renderLogs(data.data);
            renderPagination(data.pagination);
            updateLogsCount(data.pagination.total);
            currentPage = page;
        } else {
            showError('로그를 불러오는 중 오류가 발생했습니다.');
        }
    } catch (error) {
        console.error('로그 로드 오류:', error);
        document.getElementById('logs-tbody').innerHTML = 
            '<tr><td colspan="4" class="loading">로그를 불러오는 중 오류가 발생했습니다.</td></tr>';
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
        const response = await fetch(`/api/admin/logs/${logId}`);
        const data = await response.json();
        
        if (data.success) {
            renderDetailModal(data.data);
            openModal();
        } else {
            alert('로그를 불러오는 중 오류가 발생했습니다.');
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

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showError(message) {
    // 간단한 에러 표시 (필요시 토스트 메시지 등으로 변경 가능)
    alert(message);
}


