/****************************************************
 * 전역 상태
 ****************************************************/
let uploadedFiles = [];
let results = [];
let currentFilter = 'all';


/****************************************************
 * 초기화
 ****************************************************/
document.addEventListener('DOMContentLoaded', () => {
    setupUploadArea();
    setupThumbnailGridDragDrop();
});


/****************************************************
 * 업로드 영역 설정
 ****************************************************/
function setupUploadArea() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');

    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', e => handleFiles(e.target.files));

    uploadArea.addEventListener('dragover', e => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', e => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });
}


/****************************************************
 * 파일 처리
 ****************************************************/
function handleFiles(files) {
    const maxFiles = 100;
    const maxSize = 5 * 1024 * 1024;

    let added = false;

    Array.from(files).forEach(file => {
        if (uploadedFiles.length >= maxFiles) {
            alert(`최대 ${maxFiles}장까지만 업로드할 수 있습니다.`);
            return;
        }
        if (file.size > maxSize) {
            alert(`${file.name} 파일이 5MB를 초과합니다.`);
            return;
        }
        if (!file.type.startsWith('image/')) {
            alert(`${file.name}은(는) 이미지 파일이 아닙니다.`);
            return;
        }
        if (uploadedFiles.some(f => f.name === file.name && f.size === file.size)) {
            return;
        }

        uploadedFiles.push(file);
        addThumbnail(file);
        added = true;
    });

    if (added && uploadedFiles.length > 0) {
        document.getElementById('upload-area').style.display = 'none';
    }
}


/****************************************************
 * 썸네일 추가
 ****************************************************/
function addThumbnail(file) {
    const grid = document.getElementById('thumbnail-grid');
    const reader = new FileReader();

    reader.onload = e => {
        const safeName = file.name.replace(/'/g, "\\'").replace(/"/g, '&quot;');

        const item = document.createElement('div');
        item.className = 'thumbnail-item';
        item.dataset.filename = file.name;

        item.innerHTML = `
            <img src="${e.target.result}" alt="${file.name}">
            <button class="remove-btn" onclick="removeFile('${safeName}')">&times;</button>
        `;

        grid.appendChild(item);
    };

    reader.readAsDataURL(file);
}


/****************************************************
 * 파일 제거
 ****************************************************/
function removeFile(filename) {
    const decoded = filename.replace(/\\'/g, "'").replace(/&quot;/g, '"');

    uploadedFiles = uploadedFiles.filter(f => f.name !== decoded);

    const item = document.querySelector(`.thumbnail-item[data-filename="${decoded}"]`);
    if (item) item.remove();

    if (uploadedFiles.length === 0) {
        document.getElementById('upload-area').style.display = 'block';
    }
}


/****************************************************
 * 썸네일 그리드 Drag & Drop
 ****************************************************/
function setupThumbnailGridDragDrop() {
    const grid = document.getElementById('thumbnail-grid');
    if (!grid) return;

    grid.addEventListener('dragover', e => {
        e.preventDefault();
        grid.classList.add('dragover');
    });

    grid.addEventListener('dragleave', e => {
        e.preventDefault();
        grid.classList.remove('dragover');
    });

    grid.addEventListener('drop', e => {
        e.preventDefault();
        grid.classList.remove('dragover');
        if (e.dataTransfer.files?.length) {
            handleFiles(e.dataTransfer.files);
        }
    });
}


/****************************************************
 * 서버로 이미지 배치 처리
 ****************************************************/
async function processBatch() {
    if (uploadedFiles.length === 0) {
        alert('업로드할 이미지가 없습니다.');
        return;
    }

    const model = document.getElementById('model-select').value;
    const mode = document.getElementById('mode-select').value;
    const progressSection = document.getElementById('progress-section');

    document.getElementById('process-btn').disabled = true;
    progressSection.style.display = 'block';

    const formData = new FormData();
    uploadedFiles.forEach(f => formData.append('files', f));
    formData.append('model', model);
    formData.append('mode', mode);

    try {
        const response = await fetch('/api/dress/batch-check', { method: 'POST', body: formData });
        if (!response.ok) throw new Error(`서버 오류: ${response.status}`);

        const data = await response.json();
        if (!data.success) throw new Error(data.message || '처리 실패');

        results = data.results || [];
        displayResults(results);
        updateStats(results);
        updateProgress(100, '완료');
    } catch (err) {
        alert(`처리 중 오류 발생: ${err.message}`);
        updateProgress(0, '오류');
    } finally {
        document.getElementById('process-btn').disabled = false;
    }
}


/****************************************************
 * 진행률 업데이트
 ****************************************************/
function updateProgress(percent, text) {
    const bar = document.getElementById('progress-bar');
    document.getElementById('progress-text').textContent = text;

    bar.style.width = `${percent}%`;
    bar.textContent = `${percent}%`;
}


/****************************************************
 * 실제 GroundTruth 얻기
 ****************************************************/
function getGroundTruth(result) {
    if (typeof result.manual === 'boolean') return result.manual;

    const keys = ['groundTruth', 'actualDress', 'actual', 'isDress', 'label'];

    for (const k of keys) {
        if (typeof result[k] === 'boolean') return result[k];
    }
    return null;
}


/****************************************************
 * 통계 + 혼동행렬 계산
 ****************************************************/
function updateStats(resultsData) {
    const total = resultsData.length;
    const dressCount = resultsData.filter(r => r.dress).length;
    const avgConf = total ? resultsData.reduce((a, r) => a + r.confidence, 0) / total : 0;

    const conf = resultsData.reduce(
        (acc, r) => {
            const actual = getGroundTruth(r);
            if (typeof actual !== 'boolean') return acc;

            const pred = Boolean(r.dress);
            if (pred && actual) acc.tp++;
            else if (pred && !actual) acc.fp++;
            else if (!pred && actual) acc.fn++;
            else acc.tn++;

            return acc;
        },
        { tp: 0, fp: 0, fn: 0, tn: 0 }
    );

    const precision = conf.tp + conf.fp ? conf.tp / (conf.tp + conf.fp) : null;
    const recall = conf.tp + conf.fn ? conf.tp / (conf.tp + conf.fn) : null;
    const f1 = precision && recall ? (2 * precision * recall) / (precision + recall) : null;

    const pct = v => (typeof v === 'number' ? (v * 100).toFixed(1) + '%' : 'N/A');

    // UI 업데이트
    document.getElementById('stat-total').textContent = total;
    document.getElementById('stat-dress').textContent = dressCount;
    document.getElementById('stat-not-dress').textContent = total - dressCount;
    document.getElementById('stat-avg-confidence').textContent = pct(avgConf);
    document.getElementById('stat-precision').textContent = pct(precision);
    document.getElementById('stat-recall').textContent = pct(recall);
    document.getElementById('stat-f1').textContent = pct(f1);

    // 혼동행렬
    document.getElementById('matrix-tp').textContent = conf.tp;
    document.getElementById('matrix-fp').textContent = conf.fp;
    document.getElementById('matrix-fn').textContent = conf.fn;
    document.getElementById('matrix-tn').textContent = conf.tn;
}

// 결과 표시 (체크박스 이벤트 포함)
function displayResults(resultsToShow) {
    const grid = document.getElementById('results-grid');
    grid.innerHTML = '';

    data.forEach(result => {
        const card = document.createElement('div');
        card.className = `result-card ${result.dress ? 'dress' : 'not-dress'}`;

        card.innerHTML = `
            <img src="${result.thumbnail || ''}" alt="${result.filename}">
            <div class="result-info">
                <div class="status">${result.dress ? '🟢 드레스' : '🔴 일반 옷'}</div>
                <div class="confidence">신뢰도: ${(result.confidence * 100).toFixed(1)}%</div>
                <div>카테고리: ${result.category || 'N/A'}</div>
                <div style="font-size:12px;color:#999">${result.filename}</div>
                <label style="margin-top:6px;">
                    <input type="checkbox" class="manual-toggle" ${result.dress ? 'checked' : ''}>
                    수동 라벨 적용
                </label>
            </div>
        `;

        grid.appendChild(card);

        // 체크박스 이벤트
        card.querySelector('.manual-toggle').addEventListener('change', async e => {
            const isDress = e.target.checked;

            // 개발자/사용자가 바꾼 판별값 적용
            result.dress = isDress;
            result.manual = isDress; // 서버용 수동 라벨

            // UI 업데이트
            card.className = `result-card ${isDress ? 'dress' : 'not-dress'}`;
            card.querySelector('.status').textContent = isDress ? '🟢 드레스' : '🔴 일반 옷';

            // 전체 results 기준으로 통계/혼동행렬 업데이트
            updateStats(results);

            try {
                await fetch('/api/dress/manual-label', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ filename: result.filename, dress: isDress })
                });
            } catch (err) {
                console.error('수동 라벨 저장 실패:', err);
            }
        });
    });

    document.getElementById('results-section').style.display = 'block';
    document.getElementById('filter-section').style.display = 'block';
    document.getElementById('stats-section').style.display = 'block';
}

// 수동 라벨 우선 getGroundTruth
function getGroundTruth(result) {
    // 수동 라벨이 있으면 그것이 실제값
    if (typeof result.manual === 'boolean') return result.manual;

    // 기존 필드 탐색
    return ['groundTruth','actualDress','actual','isDress','label']
        .map(k => result[k])
        .find(v => typeof v === 'boolean') || null;
}


/****************************************************
 * 필터 기능
 ****************************************************/
function filterResults(filter, event) {
    currentFilter = filter;

    document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
    if (event) event.target.classList.add('active');

    let filtered = results;

    if (filter === 'dress') filtered = results.filter(r => r.dress);
    else if (filter === 'not-dress') filtered = results.filter(r => !r.dress);
    else if (filter === 'low-confidence') filtered = results.filter(r => r.confidence < 0.7);

    displayResults(filtered);
}


/****************************************************
 * 초기화 / 재실행
 ****************************************************/
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
    document.getElementById('upload-area').style.display = 'block';
}

function rerunProcess() {
    if (!uploadedFiles.length) {
        alert('업로드된 이미지가 없습니다.');
        return;
    }
    results = [];
    document.getElementById('results-section').style.display = 'none';
    document.getElementById('filter-section').style.display = 'none';
    document.getElementById('stats-section').style.display = 'none';

    processBatch();
}
