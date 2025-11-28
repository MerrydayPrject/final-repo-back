// 전역 변수
let uploadedFiles = [];
let results = [];
let currentFilter = 'all';

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
    let hasNewFiles = false;

    Array.from(files).forEach(file => {
        // 파일 수 제한
        if (uploadedFiles.length >= maxFiles) {
            alert(`최대 ${maxFiles}장까지만 업로드할 수 있습니다.`);
            return;
        }

        // 파일 크기 체크
        if (file.size > maxSize) {
            alert(`${file.name} 파일이 5MB를 초과합니다.`);
            return;
        }

        // 이미지 파일 체크
        if (!file.type.startsWith('image/')) {
            alert(`${file.name}은(는) 이미지 파일이 아닙니다.`);
            return;
        }

        // 중복 체크
        if (uploadedFiles.some(f => f.name === file.name && f.size === file.size)) {
            return;
        }

        uploadedFiles.push(file);
        addThumbnail(file);
        hasNewFiles = true;
    });

    // 파일이 추가되면 업로드 영역 숨기기
    if (hasNewFiles && uploadedFiles.length > 0) {
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

        results = data.results || [];
        displayResults(results);
        updateStats(results);
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
    
    progressBar.style.width = `${percent}%`;
    progressBar.textContent = `${percent}%`;
    progressText.textContent = text;
}

// 결과 표시
function displayResults(resultsToShow) {
    const grid = document.getElementById('results-grid');
    grid.innerHTML = '';

    resultsToShow.forEach((result, index) => {
        const card = document.createElement('div');
        card.className = `result-card ${result.dress ? 'dress' : 'not-dress'}`;
        card.dataset.index = index;

        const statusEmoji = result.dress ? '🟢' : '🔴';
        const statusText = result.dress ? '드레스' : '일반 옷';
        const confidencePercent = (result.confidence * 100).toFixed(1);

        card.innerHTML = `
            <img src="${result.thumbnail || ''}" alt="${result.filename}">
            <div class="result-info">
                <div class="status">${statusEmoji} ${statusText}</div>
                <div class="confidence">신뢰도: ${confidencePercent}%</div>
                <div>카테고리: ${result.category || 'N/A'}</div>
                <div style="font-size: 12px; color: #999; margin-top: 5px;">${result.filename}</div>
            </div>
        `;

        grid.appendChild(card);
    });

    document.getElementById('results-section').style.display = 'block';
    document.getElementById('filter-section').style.display = 'block';
    document.getElementById('stats-section').style.display = 'block';
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
        case 'low-confidence':
            filtered = results.filter(r => r.confidence < 0.7);
            break;
        default:
            filtered = results;
    }

    displayResults(filtered);
}

// 통계 업데이트
function updateStats(resultsData) {
    const total = resultsData.length;
    const dressCount = resultsData.filter(r => r.dress === true).length;
    const notDressCount = resultsData.filter(r => r.dress === false).length;
    const avgConfidence = resultsData.length > 0
        ? resultsData.reduce((sum, r) => sum + r.confidence, 0) / resultsData.length
        : 0;

    document.getElementById('stat-total').textContent = total;
    document.getElementById('stat-dress').textContent = dressCount;
    document.getElementById('stat-not-dress').textContent = notDressCount;
    document.getElementById('stat-avg-confidence').textContent = (avgConfidence * 100).toFixed(1) + '%';
}

// 초기화
function resetAll() {
    uploadedFiles = [];
    results = [];
    currentFilter = 'all';

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
    document.getElementById('results-section').style.display = 'none';
    document.getElementById('filter-section').style.display = 'none';
    document.getElementById('stats-section').style.display = 'none';
    
    processBatch();
}

