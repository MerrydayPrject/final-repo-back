// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', () => {
    loadDresses();
    setupEventListeners();
});

// 이벤트 리스너 설정
function setupEventListeners() {
    const imageNameInput = document.getElementById('image-name-input');
    const addDressBtn = document.getElementById('add-dress-btn');
    const clearFormBtn = document.getElementById('clear-form-btn');
    const refreshBtn = document.getElementById('refresh-btn');

    // 이미지명 입력 시 스타일 자동 감지 (요소가 존재하는 경우에만)
    if (imageNameInput) {
        imageNameInput.addEventListener('input', handleImageNameChange);
    }

    // 드레스 추가 버튼 (요소가 존재하는 경우에만)
    if (addDressBtn) {
        addDressBtn.addEventListener('click', handleAddDress);
    }

    // 폼 초기화 버튼 (요소가 존재하는 경우에만)
    if (clearFormBtn) {
        clearFormBtn.addEventListener('click', clearForm);
    }

    // 새로고침 버튼
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            loadDresses();
        });
    }

    // DB 정보 내보내기 버튼
    const exportBtn = document.getElementById('export-btn');
    if (exportBtn) {
        exportBtn.addEventListener('click', handleExportData);
    }

    // DB 정보 가져오기 버튼
    const importBtn = document.getElementById('import-btn');
    const importFileInput = document.getElementById('import-file-input');
    if (importBtn && importFileInput) {
        importBtn.addEventListener('click', () => {
            importFileInput.click();
        });
        importFileInput.addEventListener('change', handleImportData);
    }
}

// 이미지명 입력 시 스타일 감지
function handleImageNameChange(e) {
    const imageName = e.target.value.trim();
    const styleDisplay = document.getElementById('style-display');
    const addDressBtn = document.getElementById('add-dress-btn');

    if (!styleDisplay || !addDressBtn) {
        return;
    }

    if (!imageName) {
        styleDisplay.value = '';
        styleDisplay.classList.remove('valid', 'invalid');
        addDressBtn.disabled = true;
        return;
    }

    const detectedStyle = detectStyleFromFilename(imageName);

    if (detectedStyle) {
        styleDisplay.value = detectedStyle;
        styleDisplay.classList.add('valid');
        styleDisplay.classList.remove('invalid');
        addDressBtn.disabled = false;
    } else {
        styleDisplay.value = '스타일을 감지할 수 없습니다';
        styleDisplay.classList.add('invalid');
        styleDisplay.classList.remove('valid');
        addDressBtn.disabled = true;
    }
}

// 파일명에서 스타일 감지 (서버의 detect_style_from_filename 함수와 동일한 로직)
function detectStyleFromFilename(filename) {
    const filenameUpper = filename.toUpperCase();

    // 1. "A"로 시작하는지 확인
    if (filenameUpper.startsWith('A')) {
        return 'A라인';
    }

    // 2. "Mini" 포함 여부 확인 (대소문자 구분 없음)
    if (filenameUpper.includes('MINI')) {
        return '미니드레스';
    }

    // 3. "B"로 시작하는지 확인
    if (filenameUpper.startsWith('B')) {
        return '벨라인';
    }

    // 4. "P"로 시작하는지 확인
    if (filenameUpper.startsWith('P')) {
        return '프린세스';
    }

    // 5. 위 조건에 해당하지 않으면 null 반환 (삽입 불가)
    return null;
}

// 드레스 목록 로드
async function loadDresses() {
    const tbody = document.getElementById('dresses-tbody');
    const totalCount = document.getElementById('total-count');

    tbody.innerHTML = '<tr><td colspan="4" class="loading">데이터를 불러오는 중...</td></tr>';

    try {
        const response = await fetch('/api/admin/dresses');
        const data = await response.json();

        if (data.success) {
            renderDresses(data.data);
            totalCount.textContent = `총 ${data.total}개`;
        } else {
            tbody.innerHTML = `<tr><td colspan="5" class="loading" style="color: #ef4444;">${data.message || '드레스 목록을 불러오는 중 오류가 발생했습니다.'}</td></tr>`;
        }
    } catch (error) {
        console.error('드레스 목록 로드 오류:', error);
        tbody.innerHTML = '<tr><td colspan="5" class="loading" style="color: #ef4444;">드레스 목록을 불러오는 중 오류가 발생했습니다.</td></tr>';
    }
}

// 드레스 목록 렌더링
function renderDresses(dresses) {
    const tbody = document.getElementById('dresses-tbody');

    if (dresses.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="loading">등록된 드레스가 없습니다.</td></tr>';
        return;
    }

    tbody.innerHTML = dresses.map(dress => {
        // 백엔드 프록시를 통해 S3 이미지 제공 (CORS 문제 우회)
        // S3 URL만 사용, 로컬 경로는 사용하지 않음
        let imageUrl = null;
        if (dress.url && (dress.url.startsWith('http://') || dress.url.startsWith('https://'))) {
            // S3 URL이면 프록시 사용
            imageUrl = `/api/images/${dress.image_name}`;
        }
        // S3 URL이 없으면 null (이미지 없음 표시)
        const styleClass = getStyleClass(dress.style);

        return `
            <tr>
                <td>${dress.id}</td>
                <td class="image-name-cell">${escapeHtml(dress.image_name)}</td>
                <td><span class="style-badge ${styleClass}">${escapeHtml(dress.style)}</span></td>
                <td class="image-preview-cell">
                    ${imageUrl 
                        ? `<img 
                            src="${imageUrl}" 
                            alt="${escapeHtml(dress.image_name)}"
                            class="image-preview"
                            onerror="console.error('이미지 로드 실패:', '${imageUrl}', event); this.onerror=null; this.parentElement.innerHTML='<div class=\\'image-preview error\\' title=\\'${imageUrl}\\'>이미지 없음</div>';"
                            loading="lazy"
                        >`
                        : '<div class="image-preview error">S3 URL 없음</div>'
                    }
                </td>
                <td class="action-cell">
                    <button 
                        class="btn-delete" 
                        onclick="handleDeleteDress(${dress.id}, '${dress.image_name.replace(/'/g, "\\'")}')"
                        title="삭제"
                    >
                        🗑️ 삭제
                    </button>
                </td>
            </tr>
        `;
    }).join('');
}

// 스타일에 따른 CSS 클래스 반환
function getStyleClass(style) {
    if (style === 'A라인') return 'aline';
    if (style === '미니드레스') return 'mini';
    if (style === '벨라인') return 'bell';
    if (style === '프린세스') return 'princess';
    return '';
}

// 드레스 추가
async function handleAddDress() {
    const imageNameInput = document.getElementById('image-name-input');
    const styleDisplay = document.getElementById('style-display');
    const addDressBtn = document.getElementById('add-dress-btn');
    const messageBar = document.getElementById('add-message');

    if (!imageNameInput || !styleDisplay || !addDressBtn) {
        return;
    }

    const imageName = imageNameInput.value.trim();
    const style = styleDisplay.value;

    if (!imageName || !style) {
        showMessage('이미지명과 스타일을 모두 입력해주세요.', 'error');
        return;
    }

    // 스타일 검증 (다시 한번 확인)
    const detectedStyle = detectStyleFromFilename(imageName);
    if (!detectedStyle || detectedStyle !== style) {
        showMessage('파일명에서 스타일을 정확히 감지할 수 없습니다.', 'error');
        return;
    }

    // 버튼 비활성화
    addDressBtn.disabled = true;
    addDressBtn.textContent = '추가 중...';

    try {
        const response = await fetch('/api/admin/dresses', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image_name: imageName,
                style: style
            })
        });

        const data = await response.json();

        if (data.success) {
            showMessage(data.message || '드레스가 성공적으로 추가되었습니다.', 'success');
            clearForm();
            // 목록 새로고침
            setTimeout(() => {
                loadDresses();
            }, 500);
        } else {
            const errorMessage = data.message || '드레스 추가 중 오류가 발생했습니다.';
            alert(`❌ 드레스 추가 실패\n\n${errorMessage}`);
            showMessage(errorMessage, 'error');
        }
    } catch (error) {
        console.error('드레스 추가 오류:', error);
        const errorMessage = '드레스 추가 중 오류가 발생했습니다.';
        alert(`❌ 드레스 추가 실패\n\n${errorMessage}`);
        showMessage(errorMessage, 'error');
    } finally {
        addDressBtn.disabled = false;
        addDressBtn.textContent = '추가';
    }
}

// 폼 초기화
function clearForm() {
    const imageNameInput = document.getElementById('image-name-input');
    const styleDisplay = document.getElementById('style-display');
    const addDressBtn = document.getElementById('add-dress-btn');
    const messageBar = document.getElementById('add-message');

    if (!imageNameInput || !styleDisplay || !addDressBtn) {
        return;
    }

    imageNameInput.value = '';
    styleDisplay.value = '';
    styleDisplay.classList.remove('valid', 'invalid');
    addDressBtn.disabled = true;
    hideMessage();
}

// 메시지 표시
function showMessage(message, type) {
    const messageBar = document.getElementById('add-message');
    if (!messageBar) {
        console.log(`[${type}] ${message}`);
        return;
    }
    messageBar.textContent = message;
    messageBar.className = `message-bar ${type} show`;
}

// 메시지 숨기기
function hideMessage() {
    const messageBar = document.getElementById('add-message');
    if (!messageBar) {
        return;
    }
    messageBar.classList.remove('show');
}

// 드레스 삭제
async function handleDeleteDress(dressId, imageName) {
    if (!confirm(`정말로 드레스 '${imageName}'을(를) 삭제하시겠습니까?\n\n이 작업은 되돌릴 수 없으며, S3의 이미지와 데이터베이스의 레코드가 모두 삭제됩니다.`)) {
        return;
    }

    try {
        const response = await fetch(`/api/admin/dresses/${dressId}`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        const data = await response.json();

        if (data.success) {
            // 성공 메시지 표시
            alert(data.message || '드레스가 성공적으로 삭제되었습니다.');
            // 목록 새로고침
            loadDresses();
        } else {
            alert(data.message || '드레스 삭제 중 오류가 발생했습니다.');
        }
    } catch (error) {
        console.error('드레스 삭제 오류:', error);
        alert('드레스 삭제 중 오류가 발생했습니다.');
    }
}

// DB 정보 가져오기
async function handleImportData(e) {
    const file = e.target.files[0];
    if (!file) {
        return;
    }
    
    // 파일 형식 확인
    const fileName = file.name.toLowerCase();
    if (!fileName.endsWith('.json') && !fileName.endsWith('.csv')) {
        alert('❌ 파일 형식 오류\n\n지원하는 파일 형식은 JSON 또는 CSV입니다.');
        e.target.value = '';
        return;
    }
    
    if (!confirm(`파일 "${file.name}"을(를) 가져오시겠습니까?\n\n중복된 항목은 자동으로 건너뜁니다.`)) {
        e.target.value = '';
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/api/admin/dresses/import', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            const summary = data.summary;
            const failedResults = data.results.filter(r => !r.success);
            
            let message = `✅ 가져오기 완료\n\n`;
            message += `총: ${summary.total}개\n`;
            message += `성공: ${summary.success}개\n`;
            message += `실패: ${summary.failed}개`;
            
            if (failedResults.length > 0) {
                const errorMessages = failedResults.slice(0, 5).map(r => {
                    const dressName = r.row.dress_name || r.row.dressName || '알 수 없음';
                    return `• ${dressName}: ${r.error || '가져오기 실패'}`;
                }).join('\n');
                
                if (failedResults.length > 5) {
                    message += `\n\n실패한 항목 (최대 5개):\n${errorMessages}\n...`;
                } else {
                    message += `\n\n실패한 항목:\n${errorMessages}`;
                }
            }
            
            alert(message);
            
            // 목록 새로고침
            setTimeout(() => {
                loadDresses();
            }, 500);
        } else {
            alert(`❌ 가져오기 실패\n\n${data.message || '데이터 가져오기 중 오류가 발생했습니다.'}`);
        }
    } catch (error) {
        console.error('가져오기 오류:', error);
        alert('❌ 가져오기 실패\n\n데이터 가져오기 중 오류가 발생했습니다.');
    } finally {
        e.target.value = '';
    }
}

// DB 정보 내보내기
async function handleExportData() {
    // 형식 선택
    const format = confirm('JSON 형식으로 내보내시겠습니까?\n\n확인: JSON\n취소: CSV') ? 'json' : 'csv';
    
    try {
        const response = await fetch(`/api/admin/dresses/export?format=${format}`);
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ message: '내보내기 실패' }));
            alert(`❌ 내보내기 실패\n\n${errorData.message || '데이터 내보내기 중 오류가 발생했습니다.'}`);
            return;
        }
        
        // 파일 다운로드
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        
        // Content-Disposition 헤더에서 파일명 추출
        const contentDisposition = response.headers.get('Content-Disposition');
        let filename = `dresses_export_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.${format}`;
        
        if (contentDisposition) {
            const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
            if (filenameMatch && filenameMatch[1]) {
                filename = filenameMatch[1].replace(/['"]/g, '');
            }
        }
        
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        alert(`✅ 내보내기 완료\n\n파일명: ${filename}`);
    } catch (error) {
        console.error('내보내기 오류:', error);
        alert('❌ 내보내기 실패\n\n데이터 내보내기 중 오류가 발생했습니다.');
    }
}

// HTML 이스케이프
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

