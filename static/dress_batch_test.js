// 전역 변수
let uploadedFiles = [];
let results = [];
let currentFilter = 'all';
let showCurrentSessionOnly = true; // 기본값: 현재 세션만 보기

// 페이지 로드 시 초기화
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
    document.getElementById('metrics-section').style.display = 'none';

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
        
        // 결과 표시
        displayResults(results);
        
        // 성능지표 업데이트
        refreshMetrics();
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
    });

    // 섹션 표시 (요소가 존재하는 경우에만)
    const resultsSection = document.getElementById('results-section');
    const filterSection = document.getElementById('filter-section');
    const metricsSection = document.getElementById('metrics-section');
    
    if (resultsSection) resultsSection.style.display = 'block';
    if (filterSection) filterSection.style.display = 'block';
    if (metricsSection) metricsSection.style.display = 'block';
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
    // 이미지 자체를 클릭한 경우는 닫지 않음
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

    // 선택된 라디오버튼 확인
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

        // 결과 배열 업데이트
        const resultIndex = results.findIndex(r => r.record_id === recordId);
        if (resultIndex !== -1) {
            results[resultIndex].verified_dress = verifiedDress;
            results[resultIndex].is_verified = true;
        }

        // UI 업데이트 - 검수 완료 표시
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

        // 성능지표 새로고침 (현재 세션 모드면 자동 업데이트)
        refreshMetrics();

        // 성공 메시지
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
    // 현재 페이지에서 검수 완료된 결과만 필터링
    const verifiedResults = results.filter(r => r.is_verified && r.verified_dress !== undefined);
    
    if (verifiedResults.length === 0) {
        return null;
    }
    
    // Confusion Matrix 계산
    let TP = 0, FP = 0, FN = 0, TN = 0;
    
    verifiedResults.forEach(result => {
        const predicted = result.dress; // 예측값
        const verified = result.verified_dress; // 실제값
        
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
    
    // 성능 지표 계산
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
    document.getElementById('metrics-section').style.display = 'none';
    
    processBatch();
}

// 전체 검수 완료
async function batchVerifyAll() {
    // 검수 가능한 모든 항목 찾기 (아직 검수 안 된 것 중 라디오버튼이 선택된 것)
    const pendingVerifications = [];
    
    results.forEach(result => {
        if (!result.record_id || result.is_verified) {
            return; // record_id가 없거나 이미 검수 완료된 것은 제외
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
    
    // 각 항목에 대해 검수 저장
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
                // 결과 배열 업데이트
                const resultIndex = results.findIndex(r => r.record_id === item.recordId);
                if (resultIndex !== -1) {
                    results[resultIndex].verified_dress = item.verifiedDress;
                    results[resultIndex].is_verified = true;
                }
                
                // UI 업데이트
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

