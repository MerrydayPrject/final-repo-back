// 업로드된 파일들을 저장
let uploadedFiles = [];
let fileStyles = {}; // 파일명 -> 스타일 매핑
let processedImages = {}; // 파일명 -> 처리된 이미지 Blob 매핑

// 스타일 옵션
const STYLE_OPTIONS = ['A라인', '미니드레스', '벨라인', '프린세스'];

// 카테고리 규칙
let categoryRules = [];

// 스타일 옵션 가져오기 (카테고리 규칙에서 동적으로)
function getStyleOptions() {
    if (categoryRules.length > 0) {
        // 카테고리 규칙에서 고유한 스타일 추출
        const uniqueStyles = [...new Set(categoryRules.map(r => r.style))];
        return uniqueStyles.sort();
    }
    // 규칙이 없으면 기본 옵션 사용
    return STYLE_OPTIONS;
}

// DOM 요소
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const uploadButton = document.getElementById('upload-button');
const imagesSection = document.getElementById('images-section');
const imagesGrid = document.getElementById('images-grid');
const infoBar = document.getElementById('info-bar');
const selectedCount = document.getElementById('selected-count');
const selectAll = document.getElementById('select-all');
const uploadAllBtn = document.getElementById('upload-all-btn');
const clearAllBtn = document.getElementById('clear-all-btn');

// 카테고리 규칙 관련 DOM 요소
const rulesList = document.getElementById('rules-list');
const refreshRulesBtn = document.getElementById('refresh-rules-btn');
const addRuleBtn = document.getElementById('add-rule-btn');
const rulePrefixInput = document.getElementById('rule-prefix');
const ruleStyleInput = document.getElementById('rule-style');

// 파일명에서 스타일 감지 (서버의 카테고리 규칙 사용)
function detectStyleFromFilename(filename) {
    if (categoryRules.length === 0) {
        // 규칙이 로드되지 않은 경우 기본 규칙 사용
        const filenameUpper = filename.toUpperCase();
        if (filenameUpper.startsWith('A')) return 'A라인';
        if (filenameUpper.includes('MINI')) return '미니드레스';
        if (filenameUpper.startsWith('B')) return '벨라인';
        if (filenameUpper.startsWith('P')) return '프린세스';
        return null;
    }
    
    const filenameUpper = filename.toUpperCase();
    
    // 규칙을 우선순위대로 확인 (긴 prefix 우선)
    const sortedRules = [...categoryRules].sort((a, b) => b.prefix.length - a.prefix.length);
    
    for (const rule of sortedRules) {
        const prefixUpper = rule.prefix.toUpperCase();
        // prefix로 시작하거나 포함하는지 확인
        if (filenameUpper.startsWith(prefixUpper) || filenameUpper.includes(prefixUpper)) {
            return rule.style;
        }
    }
    
    return null;
}

// 파일 입력 처리
function handleFiles(files) {
    if (files.length === 0) return;
    
    Array.from(files).forEach(file => {
        if (!file.type.startsWith('image/')) {
            showInfo('이미지 파일만 업로드할 수 있습니다.', 'error');
            return;
        }
        
        // 중복 체크
        if (uploadedFiles.some(f => f.name === file.name)) {
            return;
        }
        
        uploadedFiles.push(file);
        
        // 파일명에서 스타일 자동 감지
        const detectedStyle = detectStyleFromFilename(file.name);
        fileStyles[file.name] = detectedStyle || '';
        
        // 미리보기 생성
        createImagePreview(file);
    });
    
    updateUI();
    showInfo(`${files.length}개의 이미지가 추가되었습니다.`, 'success');
}

// 이미지 미리보기 생성
function createImagePreview(file) {
    const reader = new FileReader();
    
    reader.onload = (e) => {
        const card = document.createElement('div');
        card.className = 'image-card';
        card.dataset.fileName = file.name;
        
        const detectedStyle = fileStyles[file.name];
        const styleClass = detectedStyle ? 'detected' : 'undetected';
        
        card.innerHTML = `
            <div class="image-card-header">
                <input type="checkbox" class="image-checkbox" data-file-name="${file.name}">
                <span class="image-name">${escapeHtml(file.name)}</span>
            </div>
            <img src="${e.target.result}" alt="${escapeHtml(file.name)}" class="image-preview" data-file-name="${file.name}">
            <div class="image-actions">
                <button class="btn-remove-bg" data-file-name="${file.name}">
                    ✂️ 누끼 따기
                </button>
                <span class="processing-status" data-file-name="${file.name}" style="display: none;"></span>
            </div>
            <div class="style-selection">
                <label class="style-label">스타일:</label>
                <select class="style-dropdown" data-file-name="${file.name}">
                    <option value="">스타일 선택</option>
                    ${getStyleOptions().map(style => 
                        `<option value="${style}" ${fileStyles[file.name] === style ? 'selected' : ''}>${style}</option>`
                    ).join('')}
                </select>
            </div>
            <div class="style-info ${styleClass}">
                ${detectedStyle ? `자동 감지: ${detectedStyle}` : '스타일을 감지할 수 없습니다'}
            </div>
        `;
        
        imagesGrid.appendChild(card);
        
        // 체크박스 이벤트
        const checkbox = card.querySelector('.image-checkbox');
        checkbox.addEventListener('change', updateSelectedCount);
        
        // 드롭다운 이벤트
        const dropdown = card.querySelector('.style-dropdown');
        dropdown.addEventListener('change', (e) => {
            fileStyles[file.name] = e.target.value;
            updateUI();
        });
        
        // 누끼 따기 버튼 이벤트
        const removeBgBtn = card.querySelector('.btn-remove-bg');
        removeBgBtn.addEventListener('click', () => handleRemoveBackground(file.name));
    };
    
    reader.readAsDataURL(file);
}

// 배경 제거 (누끼 따기)
async function handleRemoveBackground(fileName) {
    const file = uploadedFiles.find(f => f.name === fileName);
    if (!file) return;
    
    const card = document.querySelector(`[data-file-name="${fileName}"]`);
    const previewImg = card.querySelector('.image-preview');
    const removeBgBtn = card.querySelector('.btn-remove-bg');
    const statusSpan = card.querySelector('.processing-status');
    
    // 버튼 비활성화 및 상태 표시
    removeBgBtn.disabled = true;
    removeBgBtn.textContent = '처리 중...';
    statusSpan.style.display = 'block';
    statusSpan.textContent = '배경 제거 중...';
    statusSpan.className = 'processing-status processing';
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('/api/segment', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            // 처리된 이미지로 미리보기 업데이트
            previewImg.src = data.result_image;
            
            // base64를 Blob으로 변환하여 저장
            const base64Data = data.result_image.split(',')[1];
            const byteCharacters = atob(base64Data);
            const byteNumbers = new Array(byteCharacters.length);
            for (let i = 0; i < byteCharacters.length; i++) {
                byteNumbers[i] = byteCharacters.charCodeAt(i);
            }
            const byteArray = new Uint8Array(byteNumbers);
            const blob = new Blob([byteArray], { type: 'image/png' });
            
            // 처리된 이미지 저장
            processedImages[fileName] = blob;
            
            // 원본 파일을 처리된 파일로 교체
            const processedFile = new File([blob], fileName, { type: 'image/png' });
            const index = uploadedFiles.findIndex(f => f.name === fileName);
            if (index !== -1) {
                uploadedFiles[index] = processedFile;
            }
            
            statusSpan.textContent = '✓ 배경 제거 완료';
            statusSpan.className = 'processing-status success';
            removeBgBtn.textContent = '✓ 누끼 완료';
            removeBgBtn.disabled = true;
            
            showInfo('배경이 성공적으로 제거되었습니다.', 'success');
        } else {
            statusSpan.textContent = '✗ 처리 실패';
            statusSpan.className = 'processing-status error';
            removeBgBtn.disabled = false;
            removeBgBtn.textContent = '✂️ 누끼 따기';
            showInfo(data.message || '배경 제거 중 오류가 발생했습니다.', 'error');
        }
    } catch (error) {
        console.error('배경 제거 오류:', error);
        statusSpan.textContent = '✗ 처리 실패';
        statusSpan.className = 'processing-status error';
        removeBgBtn.disabled = false;
        removeBgBtn.textContent = '✂️ 누끼 따기';
        showInfo('배경 제거 중 오류가 발생했습니다.', 'error');
    }
}

// UI 업데이트
function updateUI() {
    if (uploadedFiles.length > 0) {
        imagesSection.style.display = 'block';
    } else {
        imagesSection.style.display = 'none';
    }
    
    updateSelectedCount();
}

// 선택된 개수 업데이트
function updateSelectedCount() {
    const checkedBoxes = document.querySelectorAll('.image-checkbox:checked');
    const count = checkedBoxes.length;
    selectedCount.textContent = `선택됨: ${count}개`;
    
    // 업로드 버튼 활성화/비활성화
    uploadAllBtn.disabled = count === 0 || !hasValidStyles();
}

// 유효한 스타일이 모두 선택되었는지 확인
function hasValidStyles() {
    const checkedBoxes = document.querySelectorAll('.image-checkbox:checked');
    for (const checkbox of checkedBoxes) {
        const fileName = checkbox.dataset.fileName;
        if (!fileStyles[fileName] || fileStyles[fileName] === '') {
            return false;
        }
    }
    return true;
}

// 전체 선택/해제
selectAll.addEventListener('change', (e) => {
    const checkboxes = document.querySelectorAll('.image-checkbox');
    checkboxes.forEach(cb => {
        cb.checked = e.target.checked;
    });
    updateSelectedCount();
});

// 모든 이미지 업로드
uploadAllBtn.addEventListener('click', async () => {
    const checkedBoxes = document.querySelectorAll('.image-checkbox:checked');
    if (checkedBoxes.length === 0) {
        showInfo('업로드할 이미지를 선택해주세요.', 'error');
        return;
    }
    
    const filesToUpload = [];
    const stylesData = [];
    
    checkedBoxes.forEach(checkbox => {
        const fileName = checkbox.dataset.fileName;
        const file = uploadedFiles.find(f => f.name === fileName);
        const style = fileStyles[fileName];
        
        if (file && style) {
            filesToUpload.push(file);
            stylesData.push({
                file: fileName,
                style: style
            });
        }
    });
    
    if (filesToUpload.length === 0) {
        showInfo('업로드할 이미지가 없습니다.', 'error');
        return;
    }
    
    // 업로드 버튼 비활성화
    uploadAllBtn.disabled = true;
    uploadAllBtn.textContent = '업로드 중...';
    
    try {
        const formData = new FormData();
        filesToUpload.forEach(file => {
            formData.append('files', file);
        });
        formData.append('styles', JSON.stringify(stylesData));
        
        // FormData를 사용하는 경우 Authorization 헤더만 추가 (Content-Type은 브라우저가 자동 설정)
        const token = localStorage.getItem('admin_access_token');
        const headers = {};
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }
        
        const response = await fetch('/api/admin/dresses/upload', {
            method: 'POST',
            headers: headers,
            body: formData
        });
        
        // 401 오류 처리
        if (response.status === 401) {
            alert('인증이 필요합니다. 로그인 페이지로 이동합니다.');
            window.location.href = '/';
            return;
        }
        
        const data = await response.json();
        
        if (data.success) {
            // 실패한 항목 확인
            const failedResults = data.results.filter(r => !r.success);
            
            if (failedResults.length > 0) {
                // 실패한 항목들에 대한 에러 메시지 수집
                const errorMessages = failedResults.map(r => {
                    return `• ${r.file_name}: ${r.error || '업로드 실패'}`;
                }).join('\n');
                
                const errorSummary = `업로드 중 일부 항목이 실패했습니다:\n\n${errorMessages}`;
                alert(errorSummary);
                showInfo(`${data.summary.failed}개의 이미지 업로드가 실패했습니다.`, 'error');
            }
            
            // 성공 메시지 표시
            if (data.summary.success > 0) {
                showInfo(data.message || '업로드가 완료되었습니다.', 'success');
            }
            
            // 업로드 성공한 이미지 제거
            const uploadedFileNames = data.results
                .filter(r => r.success)
                .map(r => r.file_name);
            
            uploadedFileNames.forEach(fileName => {
                const index = uploadedFiles.findIndex(f => f.name === fileName);
                if (index !== -1) {
                    uploadedFiles.splice(index, 1);
                }
                delete fileStyles[fileName];
                
                const card = document.querySelector(`[data-file-name="${fileName}"]`);
                if (card) {
                    card.remove();
                }
            });
            
            updateUI();
            
            // 3초 후 드레스 관리 페이지로 이동하거나 새로고침
            setTimeout(() => {
                if (uploadedFiles.length === 0) {
                    window.location.href = '/admin/dress-manage';
                }
            }, 3000);
        } else {
            const errorMessage = data.message || '업로드 중 오류가 발생했습니다.';
            alert(`❌ 업로드 실패\n\n${errorMessage}`);
            showInfo(errorMessage, 'error');
        }
    } catch (error) {
        console.error('업로드 오류:', error);
        const errorMessage = '업로드 중 오류가 발생했습니다.';
        alert(`❌ 업로드 실패\n\n${errorMessage}`);
        showInfo(errorMessage, 'error');
    } finally {
        uploadAllBtn.disabled = false;
        uploadAllBtn.textContent = '📤 선택된 이미지 업로드';
    }
});

// 모두 지우기
clearAllBtn.addEventListener('click', () => {
    if (confirm('모든 이미지를 제거하시겠습니까?')) {
        uploadedFiles = [];
        fileStyles = {};
        imagesGrid.innerHTML = '';
        updateUI();
        showInfo('모든 이미지가 제거되었습니다.', 'success');
    }
});

// 드래그 & 드롭 이벤트
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    handleFiles(e.dataTransfer.files);
});

uploadArea.addEventListener('click', () => {
    fileInput.click();
});

uploadButton.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
    e.target.value = ''; // 같은 파일을 다시 선택할 수 있도록
});

// 메시지 표시
function showInfo(message, type) {
    infoBar.textContent = message;
    infoBar.className = `info-bar ${type} show`;
    
    setTimeout(() => {
        infoBar.classList.remove('show');
    }, 5000);
}

// 카테고리 규칙 로드
async function loadCategoryRules() {
    try {
        const headers = window.getAuthHeaders ? window.getAuthHeaders() : {};
        const response = await fetch('/api/admin/category-rules', {
            headers: headers
        });
        
        // 401 오류 처리
        if (response.status === 401) {
            alert('인증이 필요합니다. 로그인 페이지로 이동합니다.');
            window.location.href = '/';
            return;
        }
        
        const data = await response.json();
        
        if (data.success) {
            categoryRules = data.data;
            renderRules(data.data);
            
            // 이미 업로드된 이미지의 드롭다운 옵션 업데이트
            updateStyleDropdowns();
        } else {
            rulesList.innerHTML = '<div class="loading-placeholder" style="color: #ef4444;">규칙을 불러오는 중 오류가 발생했습니다.</div>';
        }
    } catch (error) {
        console.error('규칙 로드 오류:', error);
        rulesList.innerHTML = '<div class="loading-placeholder" style="color: #ef4444;">규칙을 불러오는 중 오류가 발생했습니다.</div>';
    }
}

// 카테고리 규칙 렌더링
function renderRules(rules) {
    if (rules.length === 0) {
        rulesList.innerHTML = '<div class="loading-placeholder">등록된 규칙이 없습니다.</div>';
        return;
    }
    
    rulesList.innerHTML = rules.map(rule => `
        <div class="rule-item">
            <div class="rule-content">
                <span class="rule-prefix">${escapeHtml(rule.prefix)}</span>
                <span class="rule-arrow">→</span>
                <span class="rule-style">${escapeHtml(rule.style)}</span>
            </div>
            <button class="btn-delete-rule" data-prefix="${escapeHtml(rule.prefix)}">
                🗑️ 삭제
            </button>
        </div>
    `).join('');
    
    // 삭제 버튼 이벤트 추가
    rulesList.querySelectorAll('.btn-delete-rule').forEach(btn => {
        btn.addEventListener('click', () => handleDeleteRule(btn.dataset.prefix));
    });
}

// 카테고리 규칙 추가
async function handleAddRule() {
    const prefix = rulePrefixInput.value.trim();
    const style = ruleStyleInput.value.trim();
    
    if (!prefix || !style) {
        alert('접두사와 스타일을 모두 입력해주세요.');
        return;
    }
    
    addRuleBtn.disabled = true;
    addRuleBtn.textContent = '추가 중...';
    
    try {
        const headers = window.getAuthHeaders ? window.getAuthHeaders() : {
            'Content-Type': 'application/json',
        };
        const response = await fetch('/api/admin/category-rules', {
            method: 'POST',
            headers: headers,
            body: JSON.stringify({
                prefix: prefix,
                style: style
            })
        });
        
        // 401 오류 처리
        if (response.status === 401) {
            alert('인증이 필요합니다. 로그인 페이지로 이동합니다.');
            window.location.href = '/';
            return;
        }
        
        const data = await response.json();
        
        if (data.success) {
            alert(`✅ 규칙 추가 완료\n\n접두사: ${prefix}\n스타일: ${style}`);
            rulePrefixInput.value = '';
            ruleStyleInput.value = '';
            await loadCategoryRules();
            // 이미 업로드된 이미지의 스타일 재감지
            reDetectStyles();
            showInfo('카테고리 규칙이 추가되었습니다.', 'success');
        } else {
            alert(`❌ 규칙 추가 실패\n\n${data.message || '규칙 추가 중 오류가 발생했습니다.'}`);
            showInfo(data.message || '규칙 추가 중 오류가 발생했습니다.', 'error');
        }
    } catch (error) {
        console.error('규칙 추가 오류:', error);
        alert('❌ 규칙 추가 실패\n\n규칙 추가 중 오류가 발생했습니다.');
        showInfo('규칙 추가 중 오류가 발생했습니다.', 'error');
    } finally {
        addRuleBtn.disabled = false;
        addRuleBtn.textContent = '➕ 규칙 추가';
    }
}

// 카테고리 규칙 삭제
async function handleDeleteRule(prefix) {
    if (!confirm(`정말로 규칙 '${prefix}'을(를) 삭제하시겠습니까?`)) {
        return;
    }
    
    try {
        const headers = window.getAuthHeaders ? window.getAuthHeaders() : {
            'Content-Type': 'application/json',
        };
        const response = await fetch('/api/admin/category-rules', {
            method: 'DELETE',
            headers: headers,
            body: JSON.stringify({
                prefix: prefix
            })
        });
        
        // 401 오류 처리
        if (response.status === 401) {
            alert('인증이 필요합니다. 로그인 페이지로 이동합니다.');
            window.location.href = '/';
            return;
        }
        
        const data = await response.json();
        
        if (data.success) {
            alert(`✅ 규칙 삭제 완료\n\n접두사: ${prefix}`);
            await loadCategoryRules();
            // 이미 업로드된 이미지의 스타일 재감지
            reDetectStyles();
            showInfo('카테고리 규칙이 삭제되었습니다.', 'success');
        } else {
            alert(`❌ 규칙 삭제 실패\n\n${data.message || '규칙 삭제 중 오류가 발생했습니다.'}`);
            showInfo(data.message || '규칙 삭제 중 오류가 발생했습니다.', 'error');
        }
    } catch (error) {
        console.error('규칙 삭제 오류:', error);
        alert('❌ 규칙 삭제 실패\n\n규칙 삭제 중 오류가 발생했습니다.');
        showInfo('규칙 삭제 중 오류가 발생했습니다.', 'error');
    }
}

// 드롭다운 옵션 업데이트
function updateStyleDropdowns() {
    const styleOptions = getStyleOptions();
    const dropdowns = document.querySelectorAll('.style-dropdown');
    
    dropdowns.forEach(dropdown => {
        const currentValue = dropdown.value;
        const fileName = dropdown.dataset.fileName;
        
        // 기존 옵션 제거 (기본 옵션 제외)
        const options = dropdown.querySelectorAll('option:not([value=""])');
        options.forEach(opt => opt.remove());
        
        // 새 옵션 추가
        styleOptions.forEach(style => {
            const option = document.createElement('option');
            option.value = style;
            option.textContent = style;
            if (style === currentValue) {
                option.selected = true;
            }
            dropdown.appendChild(option);
        });
    });
}

// 이미 업로드된 이미지의 스타일 재감지
function reDetectStyles() {
    uploadedFiles.forEach(file => {
        const detectedStyle = detectStyleFromFilename(file.name);
        if (detectedStyle) {
            fileStyles[file.name] = detectedStyle;
            
            // 카드 업데이트
            const card = document.querySelector(`[data-file-name="${file.name}"]`);
            if (card) {
                const dropdown = card.querySelector('.style-dropdown');
                const styleInfo = card.querySelector('.style-info');
                
                if (dropdown) {
                    dropdown.value = detectedStyle;
                }
                
                if (styleInfo) {
                    styleInfo.textContent = `자동 감지: ${detectedStyle}`;
                    styleInfo.className = 'style-info detected';
                }
            }
        }
    });
    
    // 드롭다운 옵션 업데이트
    updateStyleDropdowns();
}

// 페이지 로드 시 규칙 로드
document.addEventListener('DOMContentLoaded', async () => {
    // admin_login.js가 로드될 때까지 대기
    let retryCount = 0;
    const maxRetries = 10;
    
    while (!window.getAuthHeaders && retryCount < maxRetries) {
        await new Promise(resolve => setTimeout(resolve, 100));
        retryCount++;
    }
    
    // 토큰 확인
    const token = localStorage.getItem('admin_access_token');
    if (!token) {
        alert('로그인이 필요합니다. 로그인 페이지로 이동합니다.');
        window.location.href = '/';
        return;
    }
    
    // 토큰 검증
    try {
        const headers = window.getAuthHeaders ? window.getAuthHeaders() : {};
        const response = await fetch('/api/auth/verify', {
            headers: headers
        });
        const data = await response.json();
        if (!response.ok || !data.success) {
            alert('인증이 필요합니다. 로그인 페이지로 이동합니다.');
            window.location.href = '/';
            return;
        }
    } catch (error) {
        console.error('토큰 검증 오류:', error);
        alert('인증 확인 중 오류가 발생했습니다. 로그인 페이지로 이동합니다.');
        window.location.href = '/';
        return;
    }
    
    loadCategoryRules();
});

// 새로고침 버튼
refreshRulesBtn.addEventListener('click', () => {
    loadCategoryRules();
});

// 규칙 추가 버튼
addRuleBtn.addEventListener('click', handleAddRule);

// Enter 키로 규칙 추가
rulePrefixInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        ruleStyleInput.focus();
    }
});

ruleStyleInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        handleAddRule();
    }
});

// HTML 이스케이프
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
