// 전역 변수
let uploadedFiles = [];
let results = [];
let currentFilter = 'all';
let incorrectAnswers = new Set(); // 오답 체크 상태 관리

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', () => {
    setupUploadArea();
    setupThumbnailGridDragDrop();
});

// 업로드 영역 설정
function setupUploadArea() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');

    // 클릭 이벤트
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    // 파일 선택 이벤트
    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    // 드래그 앤 드롭 이벤트
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });
}

// 파일 처리
function handleFiles(files) {
    const maxFiles = 100;
    const maxSize = 5 * 1024 * 1024; // 5MB
    
    // 유효한 파일만 필터링
    const validFiles = Array.from(files).filter(file => {
        // 파일 크기 체크
        if (file.size > maxSize) {
            return false;
        }
        
        // 이미지 파일 체크
        if (!file.type.startsWith('image/')) {
            return false;
        }
        
        // 중복 체크
        if (uploadedFiles.some(f => f.name === file.name && f.size === file.size)) {
            return false;
        }
        
        return true;
    });
    
    // 현재 업로드 가능한 파일 수 계산
    const remainingSlots = maxFiles - uploadedFiles.length;
    
    if (remainingSlots <= 0) {
        alert(`최대 ${maxFiles}장까지만 업로드할 수 있습니다.`);
        return;
    }
    
    // 100장 제한을 넘으면 자동으로 잘라내기
    let filesToAdd = validFiles.slice(0, remainingSlots);
    const totalFiles = uploadedFiles.length + validFiles.length;
    
    if (totalFiles > maxFiles) {
        // 한 번만 알림 표시
        alert(`최대 ${maxFiles}장까지만 업로드할 수 있습니다. ${filesToAdd.length}장만 추가됩니다.`);
    }
    
    // 파일 추가
    filesToAdd.forEach(file => {
        uploadedFiles.push(file);
        addThumbnail(file);
    });
    
    // 파일이 추가되면 업로드 영역 숨기기
    if (filesToAdd.length > 0 && uploadedFiles.length > 0) {
        const uploadArea = document.getElementById('upload-area');
        if (uploadArea) {
            uploadArea.style.display = 'none';
        }
    }
}

// 썸네일 추가
function addThumbnail(file) {
    const grid = document.getElementById('thumbnail-grid');
    const reader = new FileReader();

    reader.onload = (e) => {
        const item = document.createElement('div');
        item.className = 'thumbnail-item';
        item.dataset.filename = file.name;

        // 파일명을 안전하게 처리 (특수문자 이스케이프)
        const safeFilename = file.name.replace(/'/g, "\\'").replace(/"/g, '&quot;');
        
        item.innerHTML = `
            <img src="${e.target.result}" alt="${file.name}">
            <button class="remove-btn" onclick="removeFile('${safeFilename}')" data-filename="${safeFilename}">&times;</button>
        `;

        grid.appendChild(item);
    };

    reader.readAsDataURL(file);
}

// 파일 제거
function removeFile(filename) {
    // 특수문자 처리
    const decodedFilename = filename.replace(/\\'/g, "'").replace(/&quot;/g, '"');
    
    uploadedFiles = uploadedFiles.filter(f => f.name !== decodedFilename);
    const item = document.querySelector(`.thumbnail-item[data-filename="${filename}"]`);
    if (item) {
        item.remove();
    }

    // 모든 파일이 제거되면 업로드 영역 다시 보이기
    if (uploadedFiles.length === 0) {
        const uploadArea = document.getElementById('upload-area');
        if (uploadArea) {
            uploadArea.style.display = 'block';
        }
    }
}

// 썸네일 그리드에 드래그 앤 드롭 설정
function setupThumbnailGridDragDrop() {
    const thumbnailGrid = document.getElementById('thumbnail-grid');
    
    if (!thumbnailGrid) return;

    // 드래그 오버 이벤트
    thumbnailGrid.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        thumbnailGrid.classList.add('dragover');
    });

    // 드래그 리브 이벤트
    thumbnailGrid.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        thumbnailGrid.classList.remove('dragover');
    });

    // 드롭 이벤트
    thumbnailGrid.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        thumbnailGrid.classList.remove('dragover');
        
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFiles(e.dataTransfer.files);
        }
    });
}

// 배치 처리
async function processBatch() {
    if (uploadedFiles.length === 0) {
        alert('업로드할 이미지가 없습니다.');
        return;
    }

    const model = document.getElementById('model-select').value;
    const mode = document.getElementById('mode-select').value;
    const processBtn = document.getElementById('process-btn');
    const progressSection = document.getElementById('progress-section');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');

    // UI 업데이트
    processBtn.disabled = true;
    progressSection.style.display = 'block';
    document.getElementById('results-section').style.display = 'none';
    document.getElementById('filter-section').style.display = 'none';
    document.getElementById('stats-section').style.display = 'none';

    // FormData 생성
    const formData = new FormData();
    uploadedFiles.forEach(file => {
        formData.append('files', file);
    });
    formData.append('model', model);
    formData.append('mode', mode);

    try {
        const response = await fetch('/api/dress/batch-check', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`서버 오류: ${response.status}`);
        }

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.message || '처리 실패');
        }

        const rawResults = data.results || [];
        // 각 결과에 원본 인덱스 추가
        results = rawResults.map((result, index) => ({
            ...result,
            _originalIndex: index
        }));
        incorrectAnswers.clear(); // 새 결과 처리 시 오답 체크 초기화
        
        // 결과 표시 (통계 섹션도 함께 표시됨)
        displayResults(results);
        
        // 통계 업데이트 (displayResults 이후에 호출하여 섹션이 표시된 후 업데이트)
        updateStats();
        updateProgress(100, '완료');

    } catch (error) {
        console.error('처리 오류:', error);
        alert(`처리 중 오류가 발생했습니다: ${error.message}`);
        updateProgress(0, '오류 발생');
    } finally {
        processBtn.disabled = false;
    }
}

// 진행률 업데이트
function updateProgress(percent, text) {
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    
    if (progressBar) {
        progressBar.style.width = `${percent}%`;
        progressBar.textContent = `${percent}%`;
    }
    if (progressText) {
        progressText.textContent = text;
    }
}

// 결과 표시
function displayResults(resultsToShow) {
    const grid = document.getElementById('results-grid');
    if (!grid) {
        console.error('results-grid 요소를 찾을 수 없습니다.');
        return;
    }
    grid.innerHTML = '';

    resultsToShow.forEach((result, filteredIndex) => {
        // 원본 인덱스 사용 (필터링과 무관하게 항상 원본 results의 인덱스)
        let originalIndex = result._originalIndex;
        
        // _originalIndex가 없으면 results 배열에서 찾기
        if (originalIndex === undefined) {
            // filename과 confidence로 매칭 시도
            const foundIndex = results.findIndex(r => 
                r.filename === result.filename && 
                r.confidence === result.confidence &&
                r.dress === result.dress
            );
            originalIndex = foundIndex !== -1 ? foundIndex : filteredIndex;
        }
        
        const card = document.createElement('div');
        card.className = `result-card ${result.dress ? 'dress' : 'not-dress'}`;
        card.dataset.index = originalIndex;

        const statusEmoji = result.dress ? '🟢' : '🔴';
        const statusText = result.dress ? '드레스' : '일반 옷';
        const confidencePercent = (result.confidence * 100).toFixed(1);
        const isChecked = incorrectAnswers.has(originalIndex);

        card.innerHTML = `
            <img src="${result.thumbnail || ''}" alt="${result.filename}">
            <div class="result-info">
                <div class="status">${statusEmoji} ${statusText}</div>
                <div class="confidence">신뢰도: ${confidencePercent}%</div>
                <div>카테고리: ${result.category || 'N/A'}</div>
                <div style="font-size: 12px; color: #999; margin-top: 5px;">${result.filename}</div>
                <div class="checkbox-container" style="margin-top: 15px; padding-top: 10px; border-top: 1px solid rgba(255, 255, 255, 0.2); display: block; visibility: visible;">
                    <label style="display: flex !important; align-items: center; gap: 8px; cursor: pointer; font-size: 13px; color: #f6f6f6; margin: 0; padding: 5px 0;">
                        <input type="checkbox" ${isChecked ? 'checked' : ''} onchange="toggleIncorrect(${originalIndex})" style="width: 20px !important; height: 20px !important; min-width: 20px !important; min-height: 20px !important; cursor: pointer; margin: 0 !important; flex-shrink: 0; appearance: checkbox; -webkit-appearance: checkbox; background-color: rgba(255, 255, 255, 0.2); border: 2px solid #999; border-radius: 3px;">
                        <span style="color: #f6f6f6; font-weight: 500;">오답(틀림)</span>
                    </label>
                </div>
            </div>
        `;

        grid.appendChild(card);
    });

    // 섹션 표시 (요소가 존재하는 경우에만)
    const resultsSection = document.getElementById('results-section');
    const filterSection = document.getElementById('filter-section');
    const statsSection = document.getElementById('stats-section');
    
    if (resultsSection) resultsSection.style.display = 'block';
    if (filterSection) filterSection.style.display = 'block';
    if (statsSection) statsSection.style.display = 'block';
}

// 필터 적용
function filterResults(filter) {
    currentFilter = filter;

    // 필터 버튼 활성화 상태 업데이트
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');

    let filtered = results;

    switch (filter) {
        case 'dress':
            filtered = results.filter(r => r.dress === true);
            break;
        case 'not-dress':
            filtered = results.filter(r => r.dress === false);
            break;
        default:
            filtered = results;
    }

    displayResults(filtered);
}

// 통계 업데이트
function updateStats() {
    // 항상 전체 results 기준으로 계산 (필터와 무관)
    const total = results.length;
    const incorrect = incorrectAnswers.size;
    const correct = total - incorrect;
    const accuracy = total > 0 ? (correct / total) * 100 : 0;
    const errorRate = total > 0 ? (incorrect / total) * 100 : 0;

    // 요소가 존재하는지 확인 후 업데이트
    const statTotal = document.getElementById('stat-total');
    const statCorrect = document.getElementById('stat-correct');
    const statIncorrect = document.getElementById('stat-incorrect');
    const statAccuracy = document.getElementById('stat-accuracy');
    const statErrorRate = document.getElementById('stat-error-rate');

    if (statTotal) statTotal.textContent = total;
    if (statCorrect) statCorrect.textContent = correct;
    if (statIncorrect) statIncorrect.textContent = incorrect;
    if (statAccuracy) statAccuracy.textContent = accuracy.toFixed(1) + '%';
    if (statErrorRate) statErrorRate.textContent = errorRate.toFixed(1) + '%';
}

// 오답 체크 토글
function toggleIncorrect(index) {
    if (incorrectAnswers.has(index)) {
        incorrectAnswers.delete(index);
    } else {
        incorrectAnswers.add(index);
    }
    updateStats();
}

// 초기화
function resetAll() {
    uploadedFiles = [];
    results = [];
    currentFilter = 'all';
    incorrectAnswers.clear();

    document.getElementById('thumbnail-grid').innerHTML = '';
    document.getElementById('results-section').style.display = 'none';
    document.getElementById('filter-section').style.display = 'none';
    document.getElementById('stats-section').style.display = 'none';
    document.getElementById('progress-section').style.display = 'none';
    document.getElementById('file-input').value = '';
    
    // 업로드 영역 다시 보이기
    const uploadArea = document.getElementById('upload-area');
    if (uploadArea) {
        uploadArea.style.display = 'block';
    }
}

// 재실행
function rerunProcess() {
    if (uploadedFiles.length === 0) {
        alert('업로드된 이미지가 없습니다.');
        return;
    }

    results = [];
    incorrectAnswers.clear();
    document.getElementById('results-section').style.display = 'none';
    document.getElementById('filter-section').style.display = 'none';
    document.getElementById('stats-section').style.display = 'none';
    
    processBatch();
}

