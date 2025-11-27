// 전역 변수
let currentPage = 1;
const itemsPerPage = 20;
let currentSearchModel = null;

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

// 로그 목록 로드
async function loadLogs(page, model = null) {
    try {
        let url = `/api/admin/logs?page=${page}&limit=${itemsPerPage}`;
        if (model && model.trim() !== '') {
            url += `&model=${encodeURIComponent(model.trim())}`;
        }

        // 토큰 가져오기
        const token = localStorage.getItem('admin_access_token');
        const headers = {
            'Content-Type': 'application/json',
        };
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }

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
        const model = log.model !== undefined ? log.model : '-';
        const personUrl = log.person_url !== undefined ? log.person_url : '';
        const dressUrl = log.dress_url !== undefined ? log.dress_url : '';
        const resultUrl = log.result_url !== undefined ? log.result_url : '';
        const id = log.id !== undefined ? log.id : '-';

        // 이미지 URL 생성
        const personImageUrl = personUrl
            ? `/api/admin/s3-image-proxy?url=${encodeURIComponent(personUrl)}`
            : '';
        const dressImageUrl = dressUrl
            ? `/api/admin/s3-image-proxy?url=${encodeURIComponent(dressUrl)}`
            : '';

        // 결과 이미지 URL 생성
        const resultImageUrl = resultUrl
            ? `/api/admin/s3-image-proxy?url=${encodeURIComponent(resultUrl)}`
            : '';

        // 인물사진 셀
        const personImageCell = personImageUrl
            ? `<img src="${personImageUrl}" alt="인물사진" class="table-image" onerror="handleImageError(this)" loading="lazy">`
            : '<span class="no-image">-</span>';

        // 의상사진 셀
        const dressImageCell = dressImageUrl
            ? `<img src="${dressImageUrl}" alt="의상사진" class="table-image" onerror="handleImageError(this)" loading="lazy">`
            : '<span class="no-image">-</span>';

        // 결과 이미지 셀
        const resultImageCell = resultImageUrl
            ? `<img src="${resultImageUrl}" alt="결과 이미지" class="table-image" onerror="handleImageError(this)" loading="lazy">`
            : '<span class="no-image">-</span>';

        return `
        <tr>
            <td class="model-cell">${escapeHtml(model)}</td>
            <td class="image-cell">${personImageCell}</td>
            <td class="image-cell">${dressImageCell}</td>
            <td class="image-cell">${resultImageCell}</td>
        </tr>
    `;
    }).join('');
}

// 이미지 로드 오류 처리 (테이블용)
function handleImageError(img) {
    img.style.display = 'none';
    const parent = img.parentElement;
    if (parent && !parent.querySelector('.no-image')) {
        parent.innerHTML = '<span class="no-image">이미지 없음</span>';
    }
}

// 이미지 로드 오류 처리 (모달용)
function handleModalImageError(img, url) {
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
        // 토큰 가져오기
        const token = localStorage.getItem('admin_access_token');
        const headers = {
            'Content-Type': 'application/json',
        };
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }

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
                    onerror="handleModalImageError(this, '${escapeHtml(log.result_url)}');"
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
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showError(message) {
    // 간단한 에러 표시 (필요시 토스트 메시지 등으로 변경 가능)
    alert(message);
}

