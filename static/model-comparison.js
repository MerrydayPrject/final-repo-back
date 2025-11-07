// 전역 변수
let models = [];
let modelModals = {}; // 각 모델별 모달 데이터 저장

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', () => {
    loadModels();
});

// 모델 목록 로드
async function loadModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        
        if (data.success) {
            models = data.models;
            renderModelButtons();
            createModelModals();
        } else {
            showError('모델 목록을 불러오는 중 오류가 발생했습니다.');
        }
    } catch (error) {
        console.error('모델 로드 오류:', error);
        showError('모델 목록을 불러오는 중 오류가 발생했습니다.');
    }
}

// 모델 버튼 렌더링
function renderModelButtons() {
    const grid = document.getElementById('models-button-grid');
    
    if (models.length === 0) {
        grid.innerHTML = '<div class="no-models">사용 가능한 모델이 없습니다.</div>';
        return;
    }
    
    const buttonsHtml = models.map(model => {
        const isGemini = model.id === 'gemini-compose';
        const geminiClass = isGemini ? 'gemini-model-card' : '';
        const geminiBadge = isGemini ? '<div class="model-badge">NEW</div>' : '';
        
        return `
            <button class="model-button-card ${geminiClass}" onclick="openModelModal('${model.id}')">
                <div class="model-button-icon">${isGemini ? '✨' : '🤖'}</div>
                <div class="model-button-content">
                    <h3>${model.name}</h3>
                    <p>${model.description}</p>
                    <span class="model-category">${model.category === 'composition' ? '합성' : '세그멘테이션'}</span>
                </div>
                ${geminiBadge}
            </button>
        `;
    }).join('');
    
    // 모델 추가 버튼 추가
    const addButtonHtml = `
        <button class="add-model-button" onclick="openAddModelModal()">
            <div class="add-model-icon">➕</div>
            <div class="add-model-text">모델 추가</div>
        </button>
    `;
    
    grid.innerHTML = buttonsHtml + addButtonHtml;
}

// 모델별 모달 생성
function createModelModals() {
    const container = document.getElementById('model-modals-container');
    
    container.innerHTML = models.map(model => {
        const inputFields = generateInputFields(model);
        const parameterFields = generateParameterFields(model);
        
        return `
            <div class="model-modal" id="modal-${model.id}">
                <div class="model-modal-content">
                    <div class="model-modal-header">
                        <div class="model-modal-title">
                            <div class="model-modal-icon">${model.id === 'gemini-compose' ? '✨' : '🤖'}</div>
                            <div>
                                <h2>${model.name}</h2>
                                <p>${model.description}</p>
                            </div>
                        </div>
                        <button class="model-modal-close" onclick="closeModelModal('${model.id}')">&times;</button>
                    </div>
                    <div class="model-modal-body">
                        <div class="model-upload-section">
                            ${inputFields}
                        </div>
                        ${parameterFields}
                        <div class="model-action-section">
                            <button class="model-run-btn" id="run-btn-${model.id}" onclick="runModelTest('${model.id}')">
                                <span class="btn-icon">🚀</span>
                                테스트 실행
                            </button>
                        </div>
                        <div class="model-loading" id="loading-${model.id}" style="display: none;">
                            <div class="model-spinner"></div>
                            <p>처리 중...</p>
                        </div>
                        <div class="model-result-section" id="result-${model.id}" style="display: none;">
                            <div class="model-result-header">
                                <div class="model-processing-time">
                                    <span>처리 시간: </span>
                                    <span id="time-${model.id}">-</span>
                                </div>
                            </div>
                            <div class="model-result-images" id="result-images-${model.id}">
                                <!-- 결과 이미지가 여기에 표시됨 -->
                            </div>
                            <div class="model-result-actions">
                                <button class="model-download-btn" id="download-btn-${model.id}" onclick="downloadModelResult('${model.id}')" style="display: none;">
                                    <span class="btn-icon">💾</span>
                                    결과 다운로드
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }).join('');
    
    // 각 모달에 드래그 앤 드롭 설정
    models.forEach(model => {
        setupModalDragAndDrop(model);
    });
}

// 입력 필드 생성
function generateInputFields(model) {
    if (model.input_type === 'dual_image') {
        return `
            <div class="model-upload-row">
                <div class="model-upload-item">
                    <label class="model-upload-label">
                        <span class="upload-icon">👤</span>
                        사람 이미지
                    </label>
                    <div class="model-upload-area" id="upload-${model.id}-person">
                        <input type="file" id="input-${model.id}-person" accept="image/*" style="display: none;" onchange="handleModelImageUpload(event, '${model.id}', 'person')">
                        <div class="model-upload-content">
                            <div class="model-upload-icon">📁</div>
                            <p>이미지를 드래그하거나 클릭</p>
                            <button class="model-upload-btn" onclick="document.getElementById('input-${model.id}-person').click()">파일 선택</button>
                        </div>
                        <div class="model-preview-container" id="preview-${model.id}-person" style="display: none;">
                            <img id="img-${model.id}-person" alt="Person Preview">
                            <button class="model-remove-btn" onclick="removeModelImage('${model.id}', 'person')">&times;</button>
                        </div>
                    </div>
                </div>
                <div class="model-upload-item">
                    <label class="model-upload-label">
                        <span class="upload-icon">👗</span>
                        드레스 이미지
                    </label>
                    <div class="model-upload-area" id="upload-${model.id}-dress">
                        <input type="file" id="input-${model.id}-dress" accept="image/*" style="display: none;" onchange="handleModelImageUpload(event, '${model.id}', 'dress')">
                        <div class="model-upload-content">
                            <div class="model-upload-icon">📁</div>
                            <p>이미지를 드래그하거나 클릭</p>
                            <button class="model-upload-btn" onclick="document.getElementById('input-${model.id}-dress').click()">파일 선택</button>
                        </div>
                        <div class="model-preview-container" id="preview-${model.id}-dress" style="display: none;">
                            <img id="img-${model.id}-dress" alt="Dress Preview">
                            <button class="model-remove-btn" onclick="removeModelImage('${model.id}', 'dress')">&times;</button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    } else {
        return `
            <div class="model-upload-item">
                <label class="model-upload-label">
                    <span class="upload-icon">📁</span>
                    이미지 파일
                </label>
                <div class="model-upload-area" id="upload-${model.id}-single">
                    <input type="file" id="input-${model.id}-single" accept="image/*" style="display: none;" onchange="handleModelImageUpload(event, '${model.id}', 'single')">
                    <div class="model-upload-content">
                        <div class="model-upload-icon">📁</div>
                        <p>이미지를 드래그하거나 클릭</p>
                        <button class="model-upload-btn" onclick="document.getElementById('input-${model.id}-single').click()">파일 선택</button>
                    </div>
                    <div class="model-preview-container" id="preview-${model.id}-single" style="display: none;">
                        <img id="img-${model.id}-single" alt="Preview">
                        <button class="model-remove-btn" onclick="removeModelImage('${model.id}', 'single')">&times;</button>
                    </div>
                </div>
            </div>
        `;
    }
}

// 파라미터 필드 생성
function generateParameterFields(model) {
    if (!model.parameters || Object.keys(model.parameters).length === 0) {
        return '';
    }
    
    const paramsHtml = Object.entries(model.parameters).map(([key, param]) => {
        if (param.type === 'checkbox') {
            return `
                <div class="model-parameter-item">
                    <label>
                        <input type="checkbox" 
                               id="param-${model.id}-${key}" 
                               ${param.default ? 'checked' : ''}>
                        ${param.label}
                    </label>
                </div>
            `;
        } else if (param.type === 'select') {
            const options = (param.options || []).map(opt => 
                `<option value="${opt}" ${opt === param.default ? 'selected' : ''}>${opt}</option>`
            ).join('');
            return `
                <div class="model-parameter-item">
                    <label>${param.label}</label>
                    <select id="param-${model.id}-${key}" ${param.required ? 'required' : ''}>
                        ${options}
                    </select>
                </div>
            `;
        } else {
            return `
                <div class="model-parameter-item">
                    <label>${param.label}</label>
                    <input type="${param.type}" 
                           id="param-${model.id}-${key}" 
                           placeholder="${param.placeholder || ''}" 
                           value="${param.default || ''}"
                           ${param.required ? 'required' : ''}>
                </div>
            `;
        }
    }).join('');
    
    return `
        <div class="model-parameters-section">
            <h3>파라미터 설정</h3>
            ${paramsHtml}
        </div>
    `;
}

// 모달 열기
function openModelModal(modelId) {
    const modal = document.getElementById(`modal-${modelId}`);
    if (modal) {
        modal.classList.add('show');
    }
}

// 모달 닫기
function closeModelModal(modelId) {
    const modal = document.getElementById(`modal-${modelId}`);
    if (modal) {
        modal.classList.remove('show');
        // 결과 초기화
        document.getElementById(`result-${modelId}`).style.display = 'none';
        delete modelModals[modelId];
    }
}

// 드래그 앤 드롭 설정
function setupModalDragAndDrop(model) {
    if (model.input_type === 'dual_image') {
        ['person', 'dress'].forEach(type => {
            const area = document.getElementById(`upload-${model.id}-${type}`);
            if (!area) return;
            
            area.addEventListener('dragover', (e) => {
                e.preventDefault();
                area.classList.add('drag-over');
            });
            
            area.addEventListener('dragleave', () => {
                area.classList.remove('drag-over');
            });
            
            area.addEventListener('drop', (e) => {
                e.preventDefault();
                area.classList.remove('drag-over');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    const input = document.getElementById(`input-${model.id}-${type}`);
                    if (input) {
                        input.files = files;
                        input.dispatchEvent(new Event('change'));
                    }
                }
            });
        });
    } else {
        const area = document.getElementById(`upload-${model.id}-single`);
        if (!area) return;
        
        area.addEventListener('dragover', (e) => {
            e.preventDefault();
            area.classList.add('drag-over');
        });
        
        area.addEventListener('dragleave', () => {
            area.classList.remove('drag-over');
        });
        
        area.addEventListener('drop', (e) => {
            e.preventDefault();
            area.classList.remove('drag-over');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const input = document.getElementById(`input-${model.id}-single`);
                if (input) {
                    input.files = files;
                    input.dispatchEvent(new Event('change'));
                }
            }
        });
    }
}

// 이미지 업로드 처리
function handleModelImageUpload(event, modelId, type) {
    const file = event.target.files[0];
    if (!file) {
        console.warn('파일이 선택되지 않았습니다.');
        return;
    }
    
    if (!file.type.startsWith('image/')) {
        alert('이미지 파일만 업로드 가능합니다.');
        return;
    }
    
    // 모달 데이터 저장 (파일 읽기 전에 먼저 저장)
    if (!modelModals[modelId]) {
        modelModals[modelId] = {};
    }
    modelModals[modelId][type] = file;
    console.log(`이미지 업로드 완료: ${modelId} - ${type}`, file.name, file.size);
    
    const reader = new FileReader();
    reader.onload = (e) => {
        const previewId = `preview-${modelId}-${type}`;
        const imgId = `img-${modelId}-${type}`;
        const uploadAreaId = `upload-${modelId}-${type}`;
        
        const previewElement = document.getElementById(previewId);
        const imgElement = document.getElementById(imgId);
        const uploadAreaElement = document.getElementById(uploadAreaId);
        
        if (imgElement && previewElement && uploadAreaElement) {
            imgElement.src = e.target.result;
            previewElement.style.display = 'block';
            const uploadContent = uploadAreaElement.querySelector('.model-upload-content');
            if (uploadContent) {
                uploadContent.style.display = 'none';
            }
        }
    };
    reader.onerror = (error) => {
        console.error('파일 읽기 오류:', error);
        alert('이미지 파일을 읽는 중 오류가 발생했습니다.');
        // 파일 읽기 실패 시 저장된 파일 제거
        if (modelModals[modelId]) {
            delete modelModals[modelId][type];
        }
    };
    reader.readAsDataURL(file);
}

// 이미지 제거
function removeModelImage(modelId, type) {
    const previewId = `preview-${modelId}-${type}`;
    const uploadAreaId = `upload-${modelId}-${type}`;
    const inputId = `input-${modelId}-${type}`;
    
    document.getElementById(previewId).style.display = 'none';
    document.querySelector(`#${uploadAreaId} .model-upload-content`).style.display = 'block';
    document.getElementById(inputId).value = '';
    document.getElementById(`result-${modelId}`).style.display = 'none';
    
    if (modelModals[modelId]) {
        delete modelModals[modelId][type];
    }
}

// 모델 테스트 실행
async function runModelTest(modelId) {
    const model = models.find(m => m.id === modelId);
    if (!model) return;
    
    // 이미지 검증 (더 엄격한 검증)
    if (model.input_type === 'dual_image') {
        const personFile = modelModals[modelId]?.person;
        const dressFile = modelModals[modelId]?.dress;
        
        if (!personFile || !dressFile) {
            alert('사람 이미지와 드레스 이미지를 모두 업로드해주세요.');
            return;
        }
        
        // 파일이 실제로 존재하는지 확인
        if (!(personFile instanceof File) || !(dressFile instanceof File)) {
            alert('이미지 파일이 올바르지 않습니다. 다시 업로드해주세요.');
            return;
        }
    } else {
        const singleFile = modelModals[modelId]?.single;
        if (!singleFile || !(singleFile instanceof File)) {
            alert('이미지를 업로드해주세요.');
            return;
        }
    }
    
    // 파라미터 검증
    if (model.parameters) {
        for (const [key, param] of Object.entries(model.parameters)) {
            const input = document.getElementById(`param-${modelId}-${key}`);
            if (param.required && (!input || !input.value.trim())) {
                alert(`${param.label}을(를) 입력해주세요.`);
                return;
            }
        }
    }
    
    // gemini-compose 모델인 경우: 프롬프트 생성 및 확인 프로세스
    if (modelId === 'gemini-compose' && model.input_type === 'dual_image') {
        await runGeminiComposeWithPromptCheck(modelId, model);
        return;
    }
    const loadingDiv = document.getElementById(`loading-${modelId}`);
    const resultDiv = document.getElementById(`result-${modelId}`);
    const runBtn = document.getElementById(`run-btn-${modelId}`);
    
    // UI 상태 변경
    loadingDiv.style.display = 'flex';
    resultDiv.style.display = 'none';
    runBtn.disabled = true;
    
    const startTime = performance.now();
    
    try {
        const formData = new FormData();
        
        // 입력 이미지 추가
        if (model.input_type === 'dual_image') {
            const personFile = modelModals[modelId]['person'];
            const dressFile = modelModals[modelId]['dress'];
            
            console.log('이미지 파일 확인:', { personFile, dressFile, modelModals: modelModals[modelId] });
            
            if (!personFile || !dressFile) {
                console.error('이미지 파일이 없습니다:', { personFile, dressFile });
                alert('이미지 파일이 없습니다. 다시 업로드해주세요.');
                loadingDiv.style.display = 'none';
                runBtn.disabled = false;
                return;
            }
            
            formData.append(model.inputs[0].name, personFile);
            formData.append(model.inputs[1].name, dressFile);
            console.log(`FormData에 이미지 추가: ${model.inputs[0].name}, ${model.inputs[1].name}`);
        } else {
            const singleFile = modelModals[modelId]['single'];
            if (!singleFile) {
                console.error('이미지 파일이 없습니다:', singleFile);
                alert('이미지 파일이 없습니다. 다시 업로드해주세요.');
                loadingDiv.style.display = 'none';
                runBtn.disabled = false;
                return;
            }
            formData.append(model.inputs[0].name, singleFile);
            console.log(`FormData에 이미지 추가: ${model.inputs[0].name}`);
        }
        
        // 모델명과 prompt 추가 (로그 저장용)
        formData.append('model_name', model.id);
        
        // prompt는 models_config.json에서 가져오거나 기본값 사용
        const prompt = model.prompt || '';
        if (prompt) {
            formData.append('prompt', prompt);
        }
        
        // 파라미터 추가
        let url = model.endpoint;
        if (model.parameters) {
            for (const [key, param] of Object.entries(model.parameters)) {
                const input = document.getElementById(`param-${modelId}-${key}`);
                if (input) {
                    if (param.type === 'checkbox') {
                        // checkbox는 문자열로 변환 (백엔드에서 str로 받음)
                        formData.append(key, input.checked ? 'true' : 'false');
                    } else {
                        if (input.value) {
                            formData.append(key, input.value);
                        }
                    }
                }
            }
        }
        
        const response = await fetch(url, {
            method: model.method,
            body: formData
        });
        
        const data = await response.json();
        const endTime = performance.now();
        const processingTime = ((endTime - startTime) / 1000).toFixed(2);
        
        loadingDiv.style.display = 'none';
        runBtn.disabled = false;
        
        if (data.success) {
            displayModelResult(modelId, model, data, processingTime);
        } else {
            alert(`오류 발생: ${data.message || data.error}`);
        }
    } catch (error) {
        loadingDiv.style.display = 'none';
        runBtn.disabled = false;
        alert(`테스트 실행 중 오류 발생: ${error.message}`);
    }
}

// 결과 표시
function displayModelResult(modelId, model, data, processingTime) {
    const resultDiv = document.getElementById(`result-${modelId}`);
    const resultImagesDiv = document.getElementById(`result-images-${modelId}`);
    const timeSpan = document.getElementById(`time-${modelId}`);
    const downloadBtn = document.getElementById(`download-btn-${modelId}`);
    
    timeSpan.textContent = `${processingTime}초`;
    
    let imagesHtml = '';
    
    if (model.input_type === 'dual_image') {
        imagesHtml = `
            <div class="model-result-image-item">
                <div class="model-result-image-label">사람 이미지</div>
                <img src="${data.person_image || ''}" alt="Person">
            </div>
            <div class="model-result-image-item">
                <div class="model-result-image-label">드레스 이미지</div>
                <img src="${data.dress_image || ''}" alt="Dress">
            </div>
            <div class="model-result-image-item highlight">
                <div class="model-result-image-label">합성 결과 ✨</div>
                <img src="${data.result_image || ''}" alt="Result" id="result-img-${modelId}">
            </div>
        `;
    } else {
        imagesHtml = `
            <div class="model-result-image-item">
                <div class="model-result-image-label">원본</div>
                <img src="${data.original_image || ''}" alt="Original">
            </div>
            <div class="model-result-image-item highlight">
                <div class="model-result-image-label">결과</div>
                <img src="${data.result_image || ''}" alt="Result" id="result-img-${modelId}">
            </div>
        `;
    }
    
    // 결과 이미지 저장
    if (!modelModals[modelId]) {
        modelModals[modelId] = {};
    }
    modelModals[modelId].resultImage = data.result_image;
    
    resultImagesDiv.innerHTML = imagesHtml;
    resultDiv.style.display = 'block';
    
    if (data.result_image) {
        downloadBtn.style.display = 'flex';
    }
}

// 결과 다운로드
function downloadModelResult(modelId) {
    const resultImage = modelModals[modelId]?.resultImage;
    if (!resultImage) {
        alert('다운로드할 결과 이미지가 없습니다.');
        return;
    }
    
    const link = document.createElement('a');
    link.href = resultImage;
    link.download = `result-${modelId}-${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// 에러 표시
function showError(message) {
    alert(message);
}

// 모달 외부 클릭 시 닫기
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('model-modal')) {
        const modelId = e.target.id.replace('modal-', '');
        closeModelModal(modelId);
    }
});

// ESC 키로 모달 닫기
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        // 모델 테스트 모달 닫기
        models.forEach(model => {
            const modal = document.getElementById(`modal-${model.id}`);
            if (modal && modal.classList.contains('show')) {
                closeModelModal(model.id);
            }
        });
        // 모델 추가 모달 닫기
        const addModal = document.getElementById('modal-add-model');
        if (addModal && addModal.classList.contains('show')) {
            closeAddModelModal();
        }
    }
});

// 모델 추가 모달 열기
function openAddModelModal() {
    const modal = document.getElementById('modal-add-model');
    if (modal) {
        modal.classList.add('show');
        // 폼 초기화
        document.getElementById('add-model-form')?.reset();
    }
}

// 모델 추가 모달 닫기
function closeAddModelModal() {
    const modal = document.getElementById('modal-add-model');
    if (modal) {
        modal.classList.remove('show');
        // 폼 초기화
        const form = document.querySelector('.add-model-form');
        if (form) {
            form.querySelectorAll('input, textarea, select').forEach(input => {
                input.value = '';
            });
        }
    }
}

// 모델 추가 제출
async function submitAddModel() {
    const modelId = document.getElementById('add-model-id').value.trim();
    const modelName = document.getElementById('add-model-name').value.trim();
    const description = document.getElementById('add-model-description').value.trim();
    const endpoint = document.getElementById('add-model-endpoint').value.trim();
    const method = document.getElementById('add-model-method').value;
    const inputType = document.getElementById('add-model-input-type').value;
    const category = document.getElementById('add-model-category').value;
    
    // 유효성 검사
    if (!modelId || !modelName || !description || !endpoint) {
        alert('필수 항목을 모두 입력해주세요.');
        return;
    }
    
    // 모델 ID 형식 검사
    if (!/^[a-z0-9-]+$/.test(modelId)) {
        alert('모델 ID는 영문자, 숫자, 하이픈만 사용 가능합니다.');
        return;
    }
    
    // 중복 체크
    if (models.some(m => m.id === modelId)) {
        alert('이미 존재하는 모델 ID입니다.');
        return;
    }
    
    // 입력 타입에 따른 inputs 생성
    let inputs = [];
    if (inputType === 'dual_image') {
        inputs = [
            {"name": "person_image", "label": "사람 이미지", "required": true},
            {"name": "dress_image", "label": "드레스 이미지", "required": true}
        ];
    } else {
        inputs = [
            {"name": "file", "label": "이미지 파일", "required": true}
        ];
    }
    
    const newModel = {
        id: modelId,
        name: modelName,
        description: description,
        endpoint: endpoint,
        method: method,
        input_type: inputType,
        inputs: inputs,
        category: category
    };
    
    try {
        const response = await fetch('/api/models', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(newModel)
        });
        
        const data = await response.json();
        
        if (data.success) {
            alert('모델이 성공적으로 추가되었습니다!');
            closeAddModelModal();
            // 모델 목록 다시 로드
            loadModels();
        } else {
            alert(`오류 발생: ${data.message || data.error}`);
        }
    } catch (error) {
        console.error('모델 추가 오류:', error);
        alert(`모델 추가 중 오류 발생: ${error.message}`);
    }
}

// 모달 외부 클릭 시 닫기 (모델 추가 모달 포함)
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('model-modal')) {
        const modalId = e.target.id;
        if (modalId === 'modal-add-model') {
            closeAddModelModal();
        } else {
            const modelId = modalId.replace('modal-', '');
            closeModelModal(modelId);
        }
    }
});

// ===================== Gemini Compose 프롬프트 생성 플로우 =====================

async function runGeminiComposeWithPromptCheck(modelId, model) {
    const personFile = modelModals[modelId]?.person;
    const dressFile = modelModals[modelId]?.dress;
    
    if (!personFile || !dressFile) {
        alert('사람 이미지와 드레스 이미지를 모두 업로드해주세요.');
        return;
    }
    
    const loadingDiv = document.getElementById(`loading-${modelId}`);
    const runBtn = document.getElementById(`run-btn-${modelId}`);
    
    try {
        loadingDiv.style.display = 'flex';
        runBtn.disabled = true;
        runBtn.textContent = '프롬프트 생성 중...';
        
        // 1. 프롬프트 생성 API 호출
        const formData = new FormData();
        formData.append('person_image', personFile);
        formData.append('dress_image', dressFile);
        
        const response = await fetch('/api/generate-prompt', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.message || `프롬프트 생성 실패: ${response.status}`);
        }
        
        const data = await response.json();
        
        loadingDiv.style.display = 'none';
        runBtn.disabled = false;
        runBtn.textContent = '테스트 실행';
        
        if (data.success) {
            // 2. 프롬프트 확인 모달 표시
            showPromptConfirmModal(modelId, model, data.prompt);
        } else {
            throw new Error(data.message || '프롬프트 생성에 실패했습니다');
        }
    } catch (error) {
        console.error('프롬프트 생성 오류:', error);
        alert(`프롬프트 생성 실패: ${error.message}`);
        
        loadingDiv.style.display = 'none';
        runBtn.disabled = false;
        runBtn.textContent = '테스트 실행';
    }
}

function showPromptConfirmModal(modelId, model, generatedPrompt) {
    // HTML escape 함수
    const escapeHtml = (text) => {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    };
    
    const modal = document.createElement('div');
    modal.className = 'prompt-confirm-modal';
    modal.id = `prompt-modal-${modelId}`;
    modal.innerHTML = `
        <div class="prompt-confirm-overlay"></div>
        <div class="prompt-confirm-content">
            <div class="prompt-confirm-header">
                <h3><i class="fas fa-magic"></i> AI가 생성한 프롬프트</h3>
                <button class="prompt-close-button" onclick="closePromptConfirmModal()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="prompt-confirm-body">
                <div class="prompt-preview">
                    <label>생성된 프롬프트:</label>
                    <div class="prompt-text">${escapeHtml(generatedPrompt).replace(/\n/g, '<br>')}</div>
                </div>
                <div class="prompt-actions">
                    <p class="prompt-info">
                        <i class="fas fa-info-circle"></i>
                        이 프롬프트를 사용하여 이미지 합성을 진행하시겠습니까?
                    </p>
                    <div class="button-group">
                        <button class="btn-secondary" onclick="closePromptConfirmModal()">
                            <i class="fas fa-times"></i> 취소
                        </button>
                        <button class="btn-primary" onclick="confirmAndRunCompose('${modelId}')">
                            <i class="fas fa-check"></i> 확인 및 합성 시작
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // 기존 모달이 있으면 제거
    const existingModal = document.getElementById(`prompt-modal-${modelId}`);
    if (existingModal) {
        existingModal.remove();
    }
    
    document.body.appendChild(modal);
    
    // 생성된 프롬프트를 저장
    if (!modelModals[modelId]) {
        modelModals[modelId] = {};
    }
    modelModals[modelId].generatedPrompt = generatedPrompt;
    
    // 모달 스타일 추가
    ensurePromptModalStyles();
    
    // 오버레이 클릭 시 닫기
    modal.querySelector('.prompt-confirm-overlay').addEventListener('click', closePromptConfirmModal);
}

function ensurePromptModalStyles() {
    if (document.getElementById('prompt-modal-styles')) return;
    
    const style = document.createElement('style');
    style.id = 'prompt-modal-styles';
    style.textContent = `
        .prompt-confirm-modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 10000;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .prompt-confirm-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            backdrop-filter: blur(5px);
        }
        
        .prompt-confirm-content {
            position: relative;
            background: white;
            border-radius: 12px;
            max-width: 700px;
            width: 90%;
            max-height: 80vh;
            display: flex;
            flex-direction: column;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-50px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .prompt-confirm-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .prompt-confirm-header h3 {
            margin: 0;
            font-size: 1.3rem;
            color: #333;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .prompt-confirm-header h3 i {
            color: #8B5CF6;
        }
        
        .prompt-close-button {
            background: none;
            border: none;
            font-size: 1.5rem;
            cursor: pointer;
            color: #999;
            transition: color 0.2s;
            padding: 5px 10px;
        }
        
        .prompt-close-button:hover {
            color: #333;
        }
        
        .prompt-confirm-body {
            padding: 20px;
            overflow-y: auto;
            flex: 1;
        }
        
        .prompt-preview {
            margin-bottom: 20px;
        }
        
        .prompt-preview label {
            display: block;
            font-weight: 600;
            margin-bottom: 10px;
            color: #555;
        }
        
        .prompt-text {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #8B5CF6;
            max-height: 300px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            line-height: 1.6;
            color: #333;
            white-space: pre-wrap;
        }
        
        .prompt-actions {
            border-top: 1px solid #e0e0e0;
            padding-top: 20px;
        }
        
        .prompt-info {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 20px;
            padding: 15px;
            background: #e8f4f8;
            border-radius: 8px;
            color: #0277bd;
        }
        
        .prompt-info i {
            font-size: 1.2rem;
        }
        
        .button-group {
            display: flex;
            gap: 10px;
            justify-content: flex-end;
        }
        
        .button-group button {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .btn-secondary {
            background: #e0e0e0;
            color: #333;
        }
        
        .btn-secondary:hover {
            background: #d0d0d0;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
    `;
    document.head.appendChild(style);
}

function closePromptConfirmModal() {
    const modals = document.querySelectorAll('.prompt-confirm-modal');
    modals.forEach(modal => modal.remove());
}

async function confirmAndRunCompose(modelId) {
    closePromptConfirmModal();
    
    const model = models.find(m => m.id === modelId);
    if (!model) return;
    
    const prompt = modelModals[modelId]?.generatedPrompt;
    if (!prompt) {
        alert('프롬프트를 찾을 수 없습니다. 다시 시도해주세요.');
        return;
    }
    
    const loadingDiv = document.getElementById(`loading-${modelId}`);
    const resultDiv = document.getElementById(`result-${modelId}`);
    const runBtn = document.getElementById(`run-btn-${modelId}`);
    
    try {
        loadingDiv.style.display = 'flex';
        resultDiv.style.display = 'none';
        runBtn.disabled = true;
        runBtn.textContent = '이미지 합성 중...';
        
        const formData = new FormData();
        formData.append('person_image', modelModals[modelId].person);
        formData.append('dress_image', modelModals[modelId].dress);
        formData.append('model_name', modelId);
        formData.append('prompt', prompt);
        
        const startTime = performance.now();
        const response = await fetch(model.endpoint, {
            method: 'POST',
            body: formData
        });
        
        const processingTime = (performance.now() - startTime) / 1000;
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.message || `HTTP ${response.status}`);
        }
        
        const data = await response.json();
        
        loadingDiv.style.display = 'none';
        runBtn.disabled = false;
        runBtn.textContent = '테스트 실행';
        
        if (data.success) {
            displayModelResult(modelId, model, data, processingTime);
        } else {
            throw new Error(data.message || '이미지 합성 실패');
        }
    } catch (error) {
        console.error('이미지 합성 오류:', error);
        alert(`이미지 합성 실패: ${error.message}`);
        
        loadingDiv.style.display = 'none';
        runBtn.disabled = false;
        runBtn.textContent = '테스트 실행';
    }
}
