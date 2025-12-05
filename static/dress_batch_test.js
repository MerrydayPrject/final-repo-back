/****************************************************
 * 전역 상태
 ****************************************************/
let uploadedFiles = [];
let results = [];
let currentFilter = 'all';
let showCurrentSessionOnly = true; // 기본값: 현재 세션만 보기


/****************************************************
 * 초기화
 ****************************************************/
document.addEventListener('DOMContentLoaded', () => {
    setupUploadArea();
    setupThumbnailGridDragDrop();
    // 체크박스 초기 상태 설정 (기본값: 현재 세션만 보기)
    const checkbox = document.getElementById('show-all-data');
    if (checkbox) {
        checkbox.checked = false; // 체크 해제 = 현재 세션만 보기
    }
    refreshMetrics(); // 초기 성능지표 로드
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
    const maxSize = 5 * 1024 * 1024; // 5MB

    let added = false;

    // 파일이 이미 최대 개수를 초과했는지 체크
    if (uploadedFiles.length >= maxFiles) {
        alert(`최대 ${maxFiles}장까지만 업로드할 수 있습니다.`);
        return;
    }

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

    // 필터링된 파일 추가 처리 (파일 추가 부분은 추가로 작성)
    // validFiles 배열을 사용해 유효한 파일만 추가
}

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

    // 결과, 필터, 성능지표 섹션 숨기기
    document.getElementById('results-section').style.display = 'none';
    document.getElementById('filter-section').style.display = 'none';
    document.getElementById('metrics-section').style.display = 'none';

    const formData = new FormData();
    uploadedFiles.forEach(f => formData.append('files', f));
    formData.append('model', model);
    formData.append('mode', mode);

    try {
        const response = await fetch('/api/dress/batch-check', { method: 'POST', body: formData });
        if (!response.ok) throw new Error(`서버 오류: ${response.status}`);

        const data = await response.json();
        if (!data.success) throw new Error(data.message || '처리 실패');

        const rawResults = data.results || [];
        // 각 결과에 원본 인덱스 추가
        results = rawResults.map((result, index) => ({
            ...result,
            _originalIndex: index
        }));
        
        // 결과 표시
        displayResults(results);
        
        // 성능지표 업데이트
        refreshMetrics();
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
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    
    // progress-bar가 존재하는 경우에만 진행률 스타일 업데이트
    if (progressBar) {
        progressBar.style.width = `${percent}%`;
        progressBar.textContent = `${percent}%`;
    }

    // progress-text가 존재하는 경우에만 텍스트 업데이트
    if (progressText) {
        progressText.textContent = text;
    }
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


/**************************************************** 
 * 결과 표시 
 ****************************************************/
function displayResults(data) {
    const grid = document.getElementById('results-grid');
    if (!grid) {
        console.error('results-grid 요소를 찾을 수 없습니다.');
        return;
    }
    grid.innerHTML = '';

    // 결과를 표시할 배열 (필터링된 결과로 가정)
    const resultsToShow = data; // data가 이미 필터링된 데이터라면 이대로 사용
    resultsToShow.forEach((result, filteredIndex) => {
        // 원본 인덱스 사용 (필터링과 무관하게 항상 원본 results의 인덱스)
        let originalIndex = result._originalIndex;
        
        // _originalIndex가 없으면 results 배열에서 찾기
        if (originalIndex === undefined) {
            // filename과 confidence로 매칭 시도
            const foundIndex = data.findIndex(r => 
                r.filename === result.filename && 
                r.confidence === result.confidence &&
                r.dress === result.dress
            );
            originalIndex = foundIndex !== -1 ? foundIndex : filteredIndex;
        }
        
        const card = document.createElement('div');
        card.className = `result-card ${result.dress ? 'dress' : 'not-dress'}`;
        card.dataset.index = originalIndex;
        if (result.record_id) {
            card.dataset.recordId = result.record_id;
        }

        const statusEmoji = result.dress ? '🟢' : '🔴';
        const statusText = result.dress ? '드레스' : '일반 옷';
        const recordId = result.record_id || null;
        const verifiedDress = result.verified_dress;
        const isVerified = result.is_verified || false;

        // 검수 상태에 따른 라디오버튼 체크 상태
        const dressChecked = isVerified && verifiedDress === true ? 'checked' : '';
        const notDressChecked = isVerified && verifiedDress === false ? 'checked' : '';

        // 이미지 src와 filename 이스케이프 처리
        const safeThumbnail = (result.thumbnail || '').replace(/'/g, "\\'").replace(/"/g, '&quot;');
        const safeFilename = (result.filename || '').replace(/'/g, "\\'").replace(/"/g, '&quot;');

        card.innerHTML = `
            <img src="${safeThumbnail}" alt="${safeFilename}" 
                 onclick="openImageModal('${safeThumbnail}', '${safeFilename}')" 
                 style="cursor: pointer;">
            <div class="result-info">
                <div class="status">${statusEmoji} ${statusText}</div>
                <div style="font-size: 12px; color: #999; margin-top: 5px;">${result.filename}</div>
                ${recordId ? `
                <div class="verification-options">
                    <div style="font-size: 12px; color: #999; margin-bottom: 8px;">정답 선택 (검수):</div>
                    <label style="display: flex; align-items: center; gap: 8px; margin-bottom: 0;">
                        <input type="radio" name="verify_${recordId}" value="true" ${dressChecked} 
                               ${isVerified ? 'disabled' : ''}>
                        <span>드레스</span>
                    </label>
                    <div class="option-desc">웨딩드레스, 파티드레스 등 한 벌로 된 여성용 의류</div>
                    <label style="display: flex; align-items: center; gap: 8px; margin-bottom: 0;">
                        <input type="radio" name="verify_${recordId}" value="false" ${notDressChecked}
                               ${isVerified ? 'disabled' : ''}>
                        <span>일반 옷</span>
                    </label>
                    <div class="option-desc">상의, 하의, 아우터 등 드레스가 아닌 의류</div>
                    ${!isVerified ? `
                    <button class="btn-verify" onclick="saveVerification(${recordId})" id="verify-btn-${recordId}">
                        검수 완료
                    </button>
                    ` : '<div style="font-size: 11px; color: #28a745; margin-top: 5px;">✓ 검수 완료</div>'}
                </div>
                ` : '<div style="font-size: 11px; color: #999; margin-top: 10px;">검수 불가 (DB 저장 실패)</div>'}
            </div>
        `;

        grid.appendChild(card);



        // 체크박스 이벤트
        card.querySelector('.manual-toggle').addEventListener('change', async e => {
            const isDress = e.target.checked;
            result.dress = isDress;
            result.manual = isDress;

            card.className = `result-card ${isDress ? 'dress' : 'not-dress'}`;
            card.querySelector('.status').textContent = isDress ? '🟢 드레스' : '🔴 일반 옷';

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

    // 섹션 표시 (요소가 존재하는 경우에만)
    const resultsSection = document.getElementById('results-section');
    const filterSection = document.getElementById('filter-section');
    const metricsSection = document.getElementById('metrics-section');
    
    if (resultsSection) resultsSection.style.display = 'block';
    if (filterSection) filterSection.style.display = 'block';
    if (metricsSection) metricsSection.style.display = 'block';
}


/****************************************************
 * 필터 기능
 ****************************************************/
function filterResults(filter, event) {
    currentFilter = filter;

    // 활성화된 필터 버튼을 업데이트
    document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
    if (event) event.target.classList.add('active');

    // 필터링된 결과 배열
    let filtered = results;

    // 필터 적용
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

    // 필터링된 결과를 표시
    displayResults(filtered);
}




/****************************************************
 * 초기화 / 재실행
 ****************************************************/

// 이미지 모달 열기
// 이미지 모달 열기
function openImageModal(imageSrc, filename) {
    const modal = document.getElementById('image-modal');
    const modalImg = document.getElementById('modal-image');
    if (modal && modalImg) {
        modalImg.src = imageSrc;
        modalImg.alt = filename;
        modal.style.display = 'block';
    }
}

// 이미지 모달 닫기
function closeImageModal(event) {
    if (event && event.target.classList.contains('image-modal-content')) {
        return;
    }
    const modal = document.getElementById('image-modal');
    if (modal) {
        modal.style.display = 'none';
    }
}

// ESC 키로 모달 닫기
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeImageModal();
    }
});


// 검수 저장
async function saveVerification(recordId) {
    if (!recordId) {
        alert('레코드 ID가 없습니다.');
        return;
    }

    const radioTrue = document.querySelector(`input[name="verify_${recordId}"][value="true"]`);
    const radioFalse = document.querySelector(`input[name="verify_${recordId}"][value="false"]`);
    
    let verifiedDress = null;
    if (radioTrue && radioTrue.checked) {
        verifiedDress = true;
    } else if (radioFalse && radioFalse.checked) {
        verifiedDress = false;
    } else {
        alert('드레스 또는 일반 옷을 선택해주세요.');
        return;
    }

    const verifyBtn = document.getElementById(`verify-btn-${recordId}`);
    if (verifyBtn) {
        verifyBtn.disabled = true;
        verifyBtn.textContent = '저장 중...';
    }

    try {
        const response = await fetch('/api/dress/verify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                record_id: recordId,
                verified_dress: verifiedDress
            })
        });

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.message || '검수 저장 실패');
        }

        const resultIndex = results.findIndex(r => r.record_id === recordId);
        if (resultIndex !== -1) {
            results[resultIndex].verified_dress = verifiedDress;
            results[resultIndex].is_verified = true;
        }

        const card = document.querySelector(`[data-record-id="${recordId}"]`);
        if (card) {
            const verifyOptions = card.querySelector('.verification-options');
            if (verifyOptions) {
                verifyOptions.innerHTML = `
                    <div style="font-size: 12px; color: #999; margin-bottom: 8px;">정답 선택 (검수):</div>
                    <label style="display: flex; align-items: center; gap: 8px; margin-bottom: 0;">
                        <input type="radio" name="verify_${recordId}" value="true" ${verifiedDress ? 'checked' : ''} disabled>
                        <span>드레스</span>
                    </label>
                    <div class="option-desc">웨딩드레스, 파티드레스 등 한 벌로 된 여성용 의류</div>
                    <label style="display: flex; align-items: center; gap: 8px; margin-bottom: 0;">
                        <input type="radio" name="verify_${recordId}" value="false" ${!verifiedDress ? 'checked' : ''} disabled>
                        <span>일반 옷</span>
                    </label>
                    <div class="option-desc">상의, 하의, 아우터 등 드레스가 아닌 의류</div>
                    <div style="font-size: 11px; color: #28a745; margin-top: 5px;">✓ 검수 완료</div>
                `;
            }
        }

        // 성능지표 새로고침
        refreshMetrics();

        console.log('검수 결과가 저장되었습니다.');
    } catch (error) {
        console.error('검수 저장 오류:', error);
        alert(`검수 저장 중 오류가 발생했습니다: ${error.message}`);
        
        if (verifyBtn) {
            verifyBtn.disabled = false;
            verifyBtn.textContent = '검수 완료';
        }
    }
}


// DB 전체 데이터 보기 토글
function toggleShowAllData() {
    const checkbox = document.getElementById('show-all-data');
    showCurrentSessionOnly = checkbox ? !checkbox.checked : true; // 체크 해제 시 현재 세션만 보기
    refreshMetrics();
}

// 현재 세션의 성능지표 계산
function calculateCurrentSessionMetrics() {
    const verifiedResults = results.filter(r => r.is_verified && r.verified_dress !== undefined);
    
    if (verifiedResults.length === 0) {
        return null;
    }
    
    let TP = 0, FP = 0, FN = 0, TN = 0;
    
    verifiedResults.forEach(result => {
        const predicted = result.dress;
        const verified = result.verified_dress;
        
        if (predicted && verified) {
            TP++;
        } else if (predicted && !verified) {
            FP++;
        } else if (!predicted && verified) {
            FN++;
        } else {
            TN++;
        }
    });
    
    const sampleCount = verifiedResults.length;
    
    const precision = (TP + FP) > 0 ? TP / (TP + FP) : 0.0;
    const recall = (TP + FN) > 0 ? TP / (TP + FN) : 0.0;
    const f1 = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0.0;
    const accuracy = sampleCount > 0 ? (TP + TN) / sampleCount : 0.0;
    
    return {
        confusion_matrix: { TP, FP, FN, TN },
        metrics: {
            precision: round(precision, 4),
            recall: round(recall, 4),
            f1: round(f1, 4),
            accuracy: round(accuracy, 4)
        },
        sample_count: sampleCount
    };
}

// 반올림 헬퍼 함수
function round(value, decimals) {
    return Math.round(value * Math.pow(10, decimals)) / Math.pow(10, decimals);
}

// 성능지표 조회 및 표시
async function refreshMetrics() {
    const metricsContent = document.getElementById('metrics-content');
    if (!metricsContent) return;

    // 현재 세션만 보기 옵션이 켜져 있으면 클라이언트에서 계산
    if (showCurrentSessionOnly) {
        const sessionMetrics = calculateCurrentSessionMetrics();
        
        if (!sessionMetrics) {
            metricsContent.innerHTML = '<p style="color: #999; text-align: center; padding: 20px;">현재 세션에서 검수 완료된 데이터가 없습니다.</p>';
            return;
        }
        
        const cm = sessionMetrics.confusion_matrix;
        const metrics = sessionMetrics.metrics;
        const sampleCount = sessionMetrics.sample_count;
        
        // Confusion Matrix HTML
        const cmHtml = `
            <div class="confusion-matrix">
                <div class="confusion-matrix-header"></div>
                <div class="confusion-matrix-header">예측: 드레스</div>
                <div class="confusion-matrix-header">예측: 일반옷</div>
                
                <div class="confusion-matrix-header">실제: 드레스</div>
                <div class="confusion-matrix-cell tp">${cm.TP}</div>
                <div class="confusion-matrix-cell fn">${cm.FN}</div>
                
                <div class="confusion-matrix-header">실제: 일반옷</div>
                <div class="confusion-matrix-cell fp">${cm.FP}</div>
                <div class="confusion-matrix-cell tn">${cm.TN}</div>
            </div>
        `;

        // Metrics HTML
        const metricsHtml = `
            <div class="metrics-grid">
                <div class="metric-item">
                    <div class="value">${(metrics.precision * 100).toFixed(2)}%</div>
                    <div class="label">Precision</div>
                </div>
                <div class="metric-item">
                    <div class="value">${(metrics.recall * 100).toFixed(2)}%</div>
                    <div class="label">Recall</div>
                </div>
                <div class="metric-item">
                    <div class="value">${(metrics.f1 * 100).toFixed(2)}%</div>
                    <div class="label">F1 Score</div>
                </div>
                <div class="metric-item">
                    <div class="value">${(metrics.accuracy * 100).toFixed(2)}%</div>
                    <div class="label">Accuracy</div>
                </div>
                <div class="metric-item">
                    <div class="value">${sampleCount}</div>
                    <div class="label">샘플 수 (현재 세션)</div>
                </div>
            </div>
        `;

        metricsContent.innerHTML = cmHtml + metricsHtml;
        return;
    }

    // 전체 데이터 조회 (기존 로직)
    const days = document.getElementById('metrics-days')?.value;
    const limit = document.getElementById('metrics-limit')?.value;

    try {
        let url = '/api/dress/metrics?';
        const params = new URLSearchParams();
        if (days) params.append('days', days);
        if (limit) params.append('limit', limit);
        url += params.toString();

        const response = await fetch(url);
        const data = await response.json();

        if (!data.success) {
            throw new Error(data.message || '성능지표 조회 실패');
        }

        const cm = data.confusion_matrix;
        const metrics = data.metrics;
        const sampleCount = data.sample_count;

        if (sampleCount === 0) {
            metricsContent.innerHTML = '<p style="color: #999; text-align: center; padding: 20px;">검수 완료된 데이터가 없습니다.</p>';
            return;
        }

        // Confusion Matrix HTML
        const cmHtml = `
            <div class="confusion-matrix">
                <div class="confusion-matrix-header"></div>
                <div class="confusion-matrix-header">예측: 드레스</div>
                <div class="confusion-matrix-header">예측: 일반옷</div>
                
                <div class="confusion-matrix-header">실제: 드레스</div>
                <div class="confusion-matrix-cell tp">${cm.TP}</div>
                <div class="confusion-matrix-cell fn">${cm.FN}</div>
                
                <div class="confusion-matrix-header">실제: 일반옷</div>
                <div class="confusion-matrix-cell fp">${cm.FP}</div>
                <div class="confusion-matrix-cell tn">${cm.TN}</div>
            </div>
        `;

        // Metrics HTML
        const metricsHtml = `
            <div class="metrics-grid">
                <div class="metric-item">
                    <div class="value">${(metrics.precision * 100).toFixed(2)}%</div>
                    <div class="label">Precision</div>
                </div>
                <div class="metric-item">
                    <div class="value">${(metrics.recall * 100).toFixed(2)}%</div>
                    <div class="label">Recall</div>
                </div>
                <div class="metric-item">
                    <div class="value">${(metrics.f1 * 100).toFixed(2)}%</div>
                    <div class="label">F1 Score</div>
                </div>
                <div class="metric-item">
                    <div class="value">${(metrics.accuracy * 100).toFixed(2)}%</div>
                    <div class="label">Accuracy</div>
                </div>
                <div class="metric-item">
                    <div class="value">${sampleCount}</div>
                    <div class="label">샘플 수</div>
                </div>
            </div>
        `;

        metricsContent.innerHTML = cmHtml + metricsHtml;
    } catch (error) {
        console.error('성능지표 조회 오류:', error);
        metricsContent.innerHTML = `<p style="color: #dc3545; text-align: center; padding: 20px;">오류: ${error.message}</p>`;
    }
}

// 초기화
function resetAll() {
    uploadedFiles = [];
    results = [];
    currentFilter = 'all';

    document.getElementById('thumbnail-grid').innerHTML = '';
    document.getElementById('results-section').style.display = 'none';
    document.getElementById('filter-section').style.display = 'none';
    document.getElementById('metrics-section').style.display = 'none';
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
    document.getElementById('metrics-section').style.display = 'none';

    processBatch();
}

// 전체 검수 완료
async function batchVerifyAll() {
    const pendingVerifications = [];
    
    results.forEach(result => {
        if (!result.record_id || result.is_verified) {
            return; // Skip already verified or missing record IDs
        }
        
        const radioTrue = document.querySelector(`input[name="verify_${result.record_id}"][value="true"]`);
        const radioFalse = document.querySelector(`input[name="verify_${result.record_id}"][value="false"]`);
        
        let verifiedDress = null;
        if (radioTrue && radioTrue.checked) {
            verifiedDress = true;
        } else if (radioFalse && radioFalse.checked) {
            verifiedDress = false;
        }
        
        if (verifiedDress !== null) {
            pendingVerifications.push({
                recordId: result.record_id,
                verifiedDress: verifiedDress
            });
        }
    });

    if (pendingVerifications.length === 0) {
        alert('검수할 항목이 없습니다. 라디오버튼을 선택해주세요.');
        return;
    }

    const batchBtn = document.querySelector('.btn-batch-verify');
    if (batchBtn) {
        batchBtn.disabled = true;
        batchBtn.textContent = `검수 중... (${pendingVerifications.length}개)`;
    }

    let successCount = 0;
    let failCount = 0;

    for (const item of pendingVerifications) {
        try {
            const response = await fetch('/api/dress/verify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    record_id: item.recordId,
                    verified_dress: item.verifiedDress
                })
            });

            const data = await response.json();

            if (data.success) {
                const resultIndex = results.findIndex(r => r.record_id === item.recordId);
                if (resultIndex !== -1) {
                    results[resultIndex].verified_dress = item.verifiedDress;
                    results[resultIndex].is_verified = true;
                }

                const card = document.querySelector(`[data-record-id="${item.recordId}"]`);
                if (card) {
                    const verifyOptions = card.querySelector('.verification-options');
                    if (verifyOptions) {
                        verifyOptions.innerHTML = `
                            <div style="font-size: 12px; color: #999; margin-bottom: 8px;">정답 선택 (검수):</div>
                            <label style="display: flex; align-items: center; gap: 8px; margin-bottom: 0;">
                                <input type="radio" name="verify_${item.recordId}" value="true" ${item.verifiedDress ? 'checked' : ''} disabled>
                                <span>드레스</span>
                            </label>
                            <div class="option-desc">웨딩드레스, 파티드레스 등 한 벌로 된 여성용 의류</div>
                            <label style="display: flex; align-items: center; gap: 8px; margin-bottom: 0;">
                                <input type="radio" name="verify_${item.recordId}" value="false" ${!item.verifiedDress ? 'checked' : ''} disabled>
                                <span>일반 옷</span>
                            </label>
                            <div class="option-desc">상의, 하의, 아우터 등 드레스가 아닌 의류</div>
                            <div style="font-size: 11px; color: #28a745; margin-top: 5px;">✓ 검수 완료</div>
                        `;
                    }
                }

                successCount++;
            } else {
                failCount++;
            }
        } catch (error) {
            console.error(`검수 저장 오류 (record_id: ${item.recordId}):`, error);
            failCount++;
        }
    }

    // 성능지표 새로고침
    refreshMetrics();

    // 버튼 상태 복원
    if (batchBtn) {
        batchBtn.disabled = false;
        batchBtn.textContent = '전체 검수 완료';
    }

    // 결과 알림
    if (failCount === 0) {
        alert(`${successCount}개 항목이 검수 완료되었습니다.`);
    } else {
        alert(`${successCount}개 항목 검수 완료, ${failCount}개 항목 실패했습니다.`);
    }
}
