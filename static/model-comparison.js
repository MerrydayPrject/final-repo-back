// 전역 변수
let models = [];
let modelModals = {}; // 각 모델별 모달 데이터 저장
let promptImages = {
    person: null,
    dress: null
}; // 프롬프트 비교용 이미지 저장

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', () => {
    loadModels();
    setupPromptDragAndDrop();
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
        
        return `
            <button class="model-button-card" onclick="openModelModal('${model.id}')">
                <div class="model-button-icon">🤖</div>
                <div class="model-button-content">
                    <h3>${model.name}</h3>
                    <p>${model.description}</p>
                    <span class="model-category">${model.category === 'composition' ? '합성' : '세그멘테이션'}</span>
                </div>
            </button>
        `;
    }).join('');
    
    // 버전 선택 카드 추가 (XAI + Gemini 2.5 Flash V1/V2/V2.5 선택)
    const versionSelectCardHtml = `
        <div class="model-button-card" style="display: flex; flex-direction: column; align-items: center; gap: 15px;">
            <div class="model-button-icon">🎯</div>
            <div class="model-button-content">
                <h3>XAI + Gemini 2.5 Flash (버전 선택)</h3>
                <p>V1/V2/V2.5/V3/V3 커스텀/V4/V4 커스텀 중 선택하여 실행할 수 있습니다</p>
                <span class="model-category">합성</span>
            </div>
            <div style="width: 100%; padding: 0 10px;">
                <select id="flash-version-select" style="width: 100%; padding: 8px; border: 2px solid #e5e7eb; border-radius: 6px; font-size: 0.9em; cursor: pointer;">
                    <option value="v1">V1 (배경 포함)</option>
                    <option value="v2">V2 (SegFormer B2 Parsing)</option>
                    <option value="v2.5">V2.5 (인물 전처리 + SegFormer B2 Parsing)</option>
                    <option value="v3">V3 (2단계 Gemini 플로우)</option>
                    <option value="v3-custom">V3 커스텀 (의상 누끼 자동 처리)</option>
                    <option value="v4">V4 (Gemini 3 Flash)</option>
                    <option value="v4-custom">V4 커스텀 (의상 누끼 자동 처리 + Gemini 3)</option>
                </select>
            </div>
            <button class="model-run-btn" onclick="runVersionSelectedFlash()" style="width: calc(100% - 20px); padding: 12px; font-size: 1em; margin: 0 10px;">
                <span class="btn-icon">🚀</span>
                합성
            </button>
        </div>
    `;
    
    // V4V5 비교 버튼 추가
    const v4v5CompareCardHtml = `
        <div class="model-button-card" style="display: flex; flex-direction: column; align-items: center; gap: 15px;">
            <div class="model-button-icon">⚖️</div>
            <div class="model-button-content">
                <h3>V4V5 비교</h3>
                <p>V4와 V5 파이프라인을 병렬 실행하여 결과를 비교합니다</p>
                <span class="model-category">합성</span>
            </div>
            <div style="width: 100%; padding: 0 10px;">
                <select id="v4v5-pipeline-select-button" style="width: 100%; padding: 8px; border: 2px solid #e5e7eb; border-radius: 6px; font-size: 0.9em; cursor: pointer;">
                    <option value="normal">V4V5일반 (누끼 처리 없음)</option>
                    <option value="custom">V4V5커스텀 (누끼 처리 포함)</option>
                </select>
            </div>
            <button class="model-run-btn" onclick="openV4V5CompareModal()" style="width: calc(100% - 20px); padding: 12px; font-size: 1em; margin: 0 10px;">
                <span class="btn-icon">🚀</span>
                비교 실행
            </button>
        </div>
    `;
    
    // 모델 추가 버튼 추가
    const addButtonHtml = `
        <button class="add-model-button" onclick="openAddModelModal()">
            <div class="add-model-icon">➕</div>
            <div class="add-model-text">모델 추가</div>
        </button>
    `;
    
    grid.innerHTML = buttonsHtml + versionSelectCardHtml + v4v5CompareCardHtml + addButtonHtml;
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
                            <div class="model-modal-icon">🤖</div>
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
    
    // V4V5 비교 모달 추가
    const v4v5ModalHtml = `
        <div class="model-modal" id="modal-v4v5-compare">
            <div class="model-modal-content">
                <div class="model-modal-header">
                    <div class="model-modal-title">
                        <div class="model-modal-icon">⚖️</div>
                        <div>
                            <h2>V4V5 비교</h2>
                            <p>V4와 V5 파이프라인을 병렬 실행하여 결과를 비교합니다</p>
                        </div>
                    </div>
                    <button class="model-modal-close" onclick="closeV4V5CompareModal()">&times;</button>
                </div>
                <div class="model-modal-body">
                    <div class="model-upload-section">
                        <div class="model-upload-row">
                            <div class="model-upload-item">
                                <label class="model-upload-label">
                                    <span class="upload-icon">👤</span>
                                    인물 이미지
                                </label>
                                <div class="model-upload-area" id="upload-v4v5-person">
                                    <input type="file" id="input-v4v5-person" accept="image/*" style="display: none;" onchange="handleV4V5ImageUpload(event, 'person')">
                                    <div class="model-upload-content">
                                        <div class="model-upload-icon">📁</div>
                                        <p>이미지를 드래그하거나 클릭</p>
                                        <button class="model-upload-btn" onclick="event.stopPropagation(); document.getElementById('input-v4v5-person').click();">파일 선택</button>
                                    </div>
                                    <div class="model-preview-container" id="preview-v4v5-person" style="display: none;">
                                        <img id="img-v4v5-person" alt="Person Preview">
                                        <button class="model-remove-btn" onclick="removeV4V5Image('person')">&times;</button>
                                    </div>
                                </div>
                            </div>
                            <div class="model-upload-item">
                                <label class="model-upload-label">
                                    <span class="upload-icon">👗</span>
                                    의상 이미지
                                </label>
                                <div class="model-upload-area" id="upload-v4v5-garment">
                                    <input type="file" id="input-v4v5-garment" accept="image/*" style="display: none;" onchange="handleV4V5ImageUpload(event, 'garment')">
                                    <div class="model-upload-content">
                                        <div class="model-upload-icon">📁</div>
                                        <p>이미지를 드래그하거나 클릭</p>
                                        <button class="model-upload-btn" onclick="event.stopPropagation(); document.getElementById('input-v4v5-garment').click();">파일 선택</button>
                                    </div>
                                    <div class="model-preview-container" id="preview-v4v5-garment" style="display: none;">
                                        <img id="img-v4v5-garment" alt="Garment Preview">
                                        <button class="model-remove-btn" onclick="removeV4V5Image('garment')">&times;</button>
                                    </div>
                                </div>
                            </div>
                            <div class="model-upload-item">
                                <label class="model-upload-label">
                                    <span class="upload-icon">🖼️</span>
                                    배경 이미지
                                </label>
                                <div class="model-upload-area" id="upload-v4v5-background">
                                    <input type="file" id="input-v4v5-background" accept="image/*" style="display: none;" onchange="handleV4V5ImageUpload(event, 'background')">
                                    <div class="model-upload-content">
                                        <div class="model-upload-icon">📁</div>
                                        <p>이미지를 드래그하거나 클릭</p>
                                        <button class="model-upload-btn" onclick="event.stopPropagation(); document.getElementById('input-v4v5-background').click();">파일 선택</button>
                                    </div>
                                    <div class="model-preview-container" id="preview-v4v5-background" style="display: none;">
                                        <img id="img-v4v5-background" alt="Background Preview">
                                        <button class="model-remove-btn" onclick="removeV4V5Image('background')">&times;</button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="model-action-section">
                        <button class="model-run-btn" id="run-btn-v4v5" onclick="runV4V5Compare()">
                            <span class="btn-icon">🚀</span>
                            비교 실행
                        </button>
                    </div>
                    <div class="model-loading" id="loading-v4v5" style="display: none;">
                        <div class="model-spinner"></div>
                        <p id="loading-v4v5-text">V4와 V5 파이프라인을 병렬 실행 중...</p>
                    </div>
                    <div class="model-result-section" id="result-v4v5" style="display: none;">
                        <div class="model-result-header">
                            <div class="model-processing-time">
                                <span>전체 처리 시간: </span>
                                <span id="time-v4v5">-</span>
                            </div>
                        </div>
                        <div class="model-result-images" id="result-images-v4v5">
                            <!-- 결과 이미지가 여기에 표시됨 -->
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // 기존 V4V5 비교 모달이 있으면 제거 후 추가 (중복 방지)
    const existingV4V5Modal = document.getElementById('modal-v4v5-compare');
    if (existingV4V5Modal) {
        existingV4V5Modal.remove();
    }
    container.insertAdjacentHTML('beforeend', v4v5ModalHtml);
    
    // V4V5 모달 드래그 앤 드롭 설정
    setupV4V5DragAndDrop();
}

// 입력 필드 생성
function generateInputFields(model) {
    if (model.input_type === 'dual_image') {
        // xai-gemini-unified 모델인 경우 배경 이미지도 추가
        const hasBackground = model.id === 'xai-gemini-unified' && model.inputs.some(input => input.name === 'background_image');
        
        let backgroundField = '';
        if (hasBackground) {
            backgroundField = `
                <div class="model-upload-item">
                    <label class="model-upload-label">
                        <span class="upload-icon">🖼️</span>
                        배경 이미지
                    </label>
                    <div class="model-upload-area" id="upload-${model.id}-background">
                        <input type="file" id="input-${model.id}-background" accept="image/*" style="display: none;" onchange="handleModelImageUpload(event, '${model.id}', 'background')">
                        <div class="model-upload-content">
                            <div class="model-upload-icon">📁</div>
                            <p>이미지를 드래그하거나 클릭</p>
                            <button class="model-upload-btn" onclick="document.getElementById('input-${model.id}-background').click()">파일 선택</button>
                        </div>
                        <div class="model-preview-container" id="preview-${model.id}-background" style="display: none;">
                            <img id="img-${model.id}-background" alt="Background Preview">
                            <button class="model-remove-btn" onclick="removeModelImage('${model.id}', 'background')">&times;</button>
                        </div>
                    </div>
                </div>
            `;
        }
        
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
                ${backgroundField}
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
        const types = ['person', 'dress'];
        // xai-gemini-unified 모델인 경우 배경 이미지도 추가
        if (model.id === 'xai-gemini-unified' && model.inputs.some(input => input.name === 'background_image')) {
            types.push('background');
        }
        
        types.forEach(type => {
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
        
        // xai-gemini-unified 모델인 경우 배경 이미지도 검증
        const hasBackground = modelId === 'xai-gemini-unified' && model.inputs.some(input => input.name === 'background_image');
        const backgroundFile = hasBackground ? modelModals[modelId]?.background : null;
        
        if (!personFile || !dressFile) {
            alert('사람 이미지와 드레스 이미지를 모두 업로드해주세요.');
            return;
        }
        
        if (hasBackground && !backgroundFile) {
            alert('배경 이미지를 업로드해주세요.');
            return;
        }
        
        // 파일이 실제로 존재하는지 확인
        if (!(personFile instanceof File) || !(dressFile instanceof File)) {
            alert('이미지 파일이 올바르지 않습니다. 다시 업로드해주세요.');
            return;
        }
        
        if (hasBackground && !(backgroundFile instanceof File)) {
            alert('배경 이미지 파일이 올바르지 않습니다. 다시 업로드해주세요.');
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
    if ((modelId === 'gemini-compose' || modelId === 'gpt4o-gemini') && model.input_type === 'dual_image') {
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
            
            // xai-gemini-unified 모델인 경우 배경 이미지도 추가
            const hasBackground = modelId === 'xai-gemini-unified' && model.inputs.some(input => input.name === 'background_image');
            const backgroundFile = hasBackground ? modelModals[modelId]['background'] : null;
            
            console.log('이미지 파일 확인:', { personFile, dressFile, backgroundFile, modelModals: modelModals[modelId] });
            
            if (!personFile || !dressFile) {
                console.error('이미지 파일이 없습니다:', { personFile, dressFile });
                alert('이미지 파일이 없습니다. 다시 업로드해주세요.');
                loadingDiv.style.display = 'none';
                runBtn.disabled = false;
                return;
            }
            
            if (hasBackground && !backgroundFile) {
                console.error('배경 이미지 파일이 없습니다:', backgroundFile);
                alert('배경 이미지를 업로드해주세요.');
                loadingDiv.style.display = 'none';
                runBtn.disabled = false;
                return;
            }
            
            formData.append(model.inputs[0].name, personFile);
            formData.append(model.inputs[1].name, dressFile);
            if (hasBackground && backgroundFile) {
                formData.append(model.inputs[2].name, backgroundFile);
            }
            console.log(`FormData에 이미지 추가: ${model.inputs[0].name}, ${model.inputs[1].name}${hasBackground ? `, ${model.inputs[2].name}` : ''}`);
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
        // 통합 트라이온 엔드포인트는 person_image, dress_image를 반환하지 않을 수 있음
        if (data.person_image && data.dress_image) {
            imagesHtml = `
                <div class="model-result-image-item">
                    <div class="model-result-image-label">사람 이미지</div>
                    <img src="${data.person_image}" alt="Person">
                </div>
                <div class="model-result-image-item">
                    <div class="model-result-image-label">드레스 이미지</div>
                    <img src="${data.dress_image}" alt="Dress">
                </div>
                <div class="model-result-image-item highlight">
                    <div class="model-result-image-label">합성 결과 ✨</div>
                    <img src="${data.result_image || ''}" alt="Result" id="result-img-${modelId}">
                </div>
            `;
        } else {
            // 통합 트라이온 엔드포인트의 경우: 프롬프트와 결과만 표시
            imagesHtml = `
                <div class="model-result-image-item highlight" style="grid-column: 1 / -1;">
                    <div class="model-result-image-label">합성 결과 ✨</div>
                    <img src="${data.result_image || ''}" alt="Result" id="result-img-${modelId}">
                </div>
            `;
            // 프롬프트가 있으면 표시
            if (data.prompt) {
                imagesHtml = `
                    <div style="grid-column: 1 / -1; margin-bottom: 15px; padding: 15px; background: #f8f9fa; border-radius: 4px;">
                        <h4 style="margin: 0 0 10px 0; color: #555;">생성된 프롬프트:</h4>
                        <div style="font-size: 13px; line-height: 1.6; white-space: pre-wrap; word-wrap: break-word; max-height: 200px; overflow-y: auto;">
                            ${data.prompt}
                        </div>
                    </div>
                    ${imagesHtml}
                `;
            }
        }
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
        const modalId = e.target.id;
        if (modalId === 'modal-v4v5-compare') {
            closeV4V5CompareModal();
        } else if (modalId === 'modal-add-model') {
            closeAddModelModal();
        } else {
            const modelId = modalId.replace('modal-', '');
            closeModelModal(modelId);
        }
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
        // 버전 선택 모달 닫기
        const versionSelectV2Modal = document.getElementById('modal-version-select-v2');
        if (versionSelectV2Modal && versionSelectV2Modal.classList.contains('show')) {
            closeVersionSelectModal('v2');
        }
        // V2.5 모달 닫기
        const v25Modal = document.getElementById('modal-v25');
        if (v25Modal && v25Modal.classList.contains('show')) {
            closeV25Modal();
        }
        // V3 모달 닫기
        const v3Modal = document.getElementById('modal-v3');
        if (v3Modal && v3Modal.classList.contains('show')) {
            closeV3Modal();
        }
        // V3 커스텀 모달 닫기
        const v3CustomModal = document.getElementById('modal-v3-custom');
        if (v3CustomModal && v3CustomModal.classList.contains('show')) {
            closeV3CustomModal();
        }
        // V4 모달 닫기
        const v4Modal = document.getElementById('modal-v4');
        if (v4Modal && v4Modal.classList.contains('show')) {
            closeV4Modal();
        }
        // V4 커스텀 모달 닫기
        const v4CustomModal = document.getElementById('modal-v4-custom');
        if (v4CustomModal && v4CustomModal.classList.contains('show')) {
            closeV4CustomModal();
        }
        // 모델 추가 모달 닫기
        const addModal = document.getElementById('modal-add-model');
        if (addModal && addModal.classList.contains('show')) {
            closeAddModelModal();
        }
        // V4V5 비교 모달 닫기
        const v4v5Modal = document.getElementById('modal-v4v5-compare');
        if (v4v5Modal && v4v5Modal.classList.contains('show')) {
            closeV4V5CompareModal();
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
        } else if (modalId === 'modal-version-select-v2') {
            closeVersionSelectModal('v2');
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
        
        // GPT-4o → Gemini 2.5 Flash V1 합성의 경우 GPT-4o로 프롬프트 생성
        const promptLLM = model.prompt_llm || (modelId === 'gpt4o-gemini' ? 'gpt-4o' : '');
        if (promptLLM) {
            formData.append('prompt_llm', promptLLM);
        }
        
        const promptEndpoint = model.prompt_generation_endpoint || '/api/gemini/generate-prompt';
        const response = await fetch(promptEndpoint, {
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
            const llmName = data.llm || data.model || data.provider || promptLLM || '알 수 없음';
            showPromptConfirmModal(modelId, model, data.prompt, llmName);
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

function showPromptConfirmModal(modelId, model, generatedPrompt, llmName = '알 수 없음') {
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
                    <div class="prompt-llm-info">
                        <span class="prompt-llm-label">프롬프트 생성 모델:</span>
                        <span class="prompt-llm-name">${escapeHtml(llmName)}</span>
                    </div>
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
    modelModals[modelId].promptLLM = llmName;
    
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

        .prompt-llm-info {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 12px;
            font-size: 0.95rem;
            color: #444;
        }

        .prompt-llm-label {
            font-weight: 600;
        }

        .prompt-llm-name {
            background: #eef2ff;
            color: #4338ca;
            padding: 4px 10px;
            border-radius: 999px;
            font-weight: 600;
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

// ==================== 프롬프트 비교 기능 ====================

// 프롬프트 이미지 업로드 처리
function handlePromptImageUpload(event, type) {
    const file = event.target.files[0];
    if (!file) return;
    
    if (!file.type.startsWith('image/')) {
        alert('이미지 파일만 업로드 가능합니다.');
        return;
    }
    
    promptImages[type] = file;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        const previewId = `prompt-preview-${type}`;
        const imgId = `prompt-img-${type}`;
        const contentId = `prompt-content-${type}`;
        
        const previewElement = document.getElementById(previewId);
        const imgElement = document.getElementById(imgId);
        const contentElement = document.getElementById(contentId);
        
        if (imgElement && previewElement && contentElement) {
            imgElement.src = e.target.result;
            previewElement.style.display = 'block';
            contentElement.style.display = 'none';
        }
    };
    reader.readAsDataURL(file);
}

// 프롬프트 이미지 제거
function removePromptImage(type) {
    const previewId = `prompt-preview-${type}`;
    const contentId = `prompt-content-${type}`;
    const inputId = `prompt-input-${type}`;
    
    document.getElementById(previewId).style.display = 'none';
    document.getElementById(contentId).style.display = 'block';
    document.getElementById(inputId).value = '';
    promptImages[type] = null;
}

// 프롬프트 드래그 앤 드롭 설정
function setupPromptDragAndDrop() {
    ['person', 'dress'].forEach(type => {
        const area = document.getElementById(`prompt-upload-${type}`);
        if (!area) return;
        
        area.addEventListener('dragover', (e) => {
            e.preventDefault();
            area.style.borderColor = '#007bff';
        });
        
        area.addEventListener('dragleave', () => {
            area.style.borderColor = '#ddd';
        });
        
        area.addEventListener('drop', (e) => {
            e.preventDefault();
            area.style.borderColor = '#ddd';
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const input = document.getElementById(`prompt-input-${type}`);
                if (input) {
                    input.files = files;
                    input.dispatchEvent(new Event('change'));
                }
            }
        });
    });
}

// 프롬프트 비교 실행
async function runPromptComparison() {
    // 이미지 검증
    if (!promptImages.person || !promptImages.dress) {
        alert('사람 이미지와 드레스 이미지를 모두 업로드해주세요.');
        return;
    }
    
    const loadingDiv = document.getElementById('prompt-loading');
    const resultsDiv = document.getElementById('prompt-results');
    const resultsGrid = document.getElementById('prompt-results-grid');
    const compareBtn = document.getElementById('prompt-compare-btn');
    
    // UI 상태 변경
    loadingDiv.style.display = 'block';
    resultsDiv.style.display = 'none';
    compareBtn.disabled = true;
    compareBtn.textContent = '생성 중...';
    
    const startTime = performance.now();
    
    try {
        // 세 모델의 프롬프트 생성 API를 병렬 호출
        const formDataGemini = new FormData();
        formDataGemini.append('person_image', promptImages.person);
        formDataGemini.append('dress_image', promptImages.dress);
        
        const formDataGPT4o = new FormData();
        formDataGPT4o.append('person_image', promptImages.person);
        formDataGPT4o.append('dress_image', promptImages.dress);
        
        const formDataXAI = new FormData();
        formDataXAI.append('person_image', promptImages.person);
        formDataXAI.append('dress_image', promptImages.dress);
        
        const [geminiResponse, gpt4oResponse, xaiResponse] = await Promise.allSettled([
            fetch('/api/gemini/generate-prompt', {
                method: 'POST',
                body: formDataGemini
            }),
            fetch('/api/gpt4o-gemini/generate-prompt', {
                method: 'POST',
                body: formDataGPT4o
            }),
            fetch('/api/xai/generate-prompt', {
                method: 'POST',
                body: formDataXAI
            })
        ]);
        
        const endTime = performance.now();
        const processingTime = ((endTime - startTime) / 1000).toFixed(2);
        
        // 결과 처리
        const results = [];
        
        // Gemini 결과
        if (geminiResponse.status === 'fulfilled' && geminiResponse.value.ok) {
            const data = await geminiResponse.value.json();
            results.push({
                model: 'Gemini',
                prompt: data.prompt || '프롬프트 생성 실패',
                success: data.success || false,
                error: data.error || null,
                llm: data.llm || 'gemini'
            });
        } else {
            results.push({
                model: 'Gemini',
                prompt: '프롬프트 생성 실패',
                success: false,
                error: geminiResponse.reason?.message || '알 수 없는 오류',
                llm: 'gemini'
            });
        }
        
        // GPT-4o 결과
        if (gpt4oResponse.status === 'fulfilled' && gpt4oResponse.value.ok) {
            const data = await gpt4oResponse.value.json();
            results.push({
                model: 'GPT-4o',
                prompt: data.prompt || '프롬프트 생성 실패',
                success: data.success || false,
                error: data.error || null,
                llm: data.llm || 'gpt-4o'
            });
        } else {
            results.push({
                model: 'GPT-4o',
                prompt: '프롬프트 생성 실패',
                success: false,
                error: gpt4oResponse.reason?.message || '알 수 없는 오류',
                llm: 'gpt-4o'
            });
        }
        
        // x.ai 결과
        if (xaiResponse.status === 'fulfilled' && xaiResponse.value.ok) {
            const data = await xaiResponse.value.json();
            results.push({
                model: 'x.ai',
                prompt: data.prompt || '프롬프트 생성 실패',
                success: data.success || false,
                error: data.error || null,
                llm: data.llm || 'xai'
            });
        } else {
            results.push({
                model: 'x.ai',
                prompt: '프롬프트 생성 실패',
                success: false,
                error: xaiResponse.reason?.message || '알 수 없는 오류',
                llm: 'xai'
            });
        }
        
        // 결과 표시 (프롬프트만 먼저 표시)
        displayPromptResults(results, processingTime);
        
        // 각 프롬프트로 Gemini 이미지 합성 실행
        await generateImagesWithPrompts(results);
        
    } catch (error) {
        console.error('프롬프트 비교 오류:', error);
        alert(`프롬프트 비교 중 오류 발생: ${error.message}`);
    } finally {
        loadingDiv.style.display = 'none';
        compareBtn.disabled = false;
        compareBtn.textContent = '🚀 프롬프트 비교 실행';
    }
}

// 프롬프트 결과 표시
function displayPromptResults(results, processingTime) {
    const resultsDiv = document.getElementById('prompt-results');
    const resultsGrid = document.getElementById('prompt-results-grid');
    
    const resultsHtml = results.map((result, index) => {
        const statusClass = result.success ? 'success' : 'error';
        const statusText = result.success ? '✅ 성공' : '❌ 실패';
        
        return `
            <div class="prompt-result-card" id="prompt-card-${index}" style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <h3 style="margin: 0; color: #333;">${result.model}</h3>
                    <span style="padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold; background: ${result.success ? '#d4edda' : '#f8d7da'}; color: ${result.success ? '#155724' : '#721c24'};">
                        ${statusText}
                    </span>
                </div>
                <div style="margin-bottom: 10px; font-size: 12px; color: #666;">
                    모델: ${result.llm}
                </div>
                ${result.error ? `
                    <div style="padding: 10px; background: #f8d7da; border-radius: 4px; margin-bottom: 15px; color: #721c24; font-size: 14px;">
                        오류: ${result.error}
                    </div>
                ` : ''}
                <div style="background: #f8f9fa; padding: 15px; border-radius: 4px; max-height: 400px; overflow-y: auto; margin-bottom: 15px;">
                    <div style="font-size: 12px; color: #666; margin-bottom: 8px;">생성된 프롬프트:</div>
                    <div style="white-space: pre-wrap; word-wrap: break-word; font-size: 14px; line-height: 1.6; color: #333;">
                        ${result.prompt}
                    </div>
                </div>
                <div style="margin-bottom: 15px; font-size: 12px; color: #666;">
                    길이: ${result.prompt.length}자
                </div>
                <div id="image-result-${index}" style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #dee2e6;">
                    <div style="font-size: 12px; color: #666; margin-bottom: 10px; font-weight: bold;">Gemini 이미지 합성 결과:</div>
                    <div id="image-loading-${index}" style="text-align: center; padding: 20px;">
                        <div style="display: inline-block; width: 30px; height: 30px; border: 3px solid #f3f3f3; border-top: 3px solid #007bff; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                        <p style="margin-top: 10px; font-size: 12px; color: #666;">이미지 생성 중...</p>
                    </div>
                    <div id="image-content-${index}" style="display: none;">
                        <!-- 이미지 결과가 여기에 표시됨 -->
                    </div>
                </div>
            </div>
        `;
    }).join('');
    
    resultsGrid.innerHTML = resultsHtml + `
        <div style="grid-column: 1 / -1; text-align: center; padding: 15px; background: #e9ecef; border-radius: 4px; margin-top: 10px;">
            <span style="color: #666;">프롬프트 생성 시간: ${processingTime}초</span>
        </div>
    `;
    
    resultsDiv.style.display = 'block';
}

// 각 프롬프트로 Gemini 이미지 합성 실행
async function generateImagesWithPrompts(results) {
    const imagePromises = results.map(async (result, index) => {
        // 프롬프트 생성 실패한 경우 이미지 합성 건너뛰기
        if (!result.success || !result.prompt) {
            const loadingDiv = document.getElementById(`image-loading-${index}`);
            const contentDiv = document.getElementById(`image-content-${index}`);
            if (loadingDiv) loadingDiv.style.display = 'none';
            if (contentDiv) {
                contentDiv.innerHTML = '<div style="padding: 10px; background: #f8d7da; border-radius: 4px; color: #721c24; font-size: 14px;">프롬프트 생성 실패로 이미지 합성을 건너뜁니다.</div>';
                contentDiv.style.display = 'block';
            }
            return;
        }
        
        try {
            const formData = new FormData();
            formData.append('person_image', promptImages.person);
            formData.append('dress_image', promptImages.dress);
            formData.append('prompt', result.prompt);
            formData.append('model_name', `prompt-comparison-${result.llm}`);
            
            const startTime = performance.now();
            const response = await fetch('/api/compose-dress', {
                method: 'POST',
                body: formData
            });
            const endTime = performance.now();
            const imageTime = ((endTime - startTime) / 1000).toFixed(2);
            
            const loadingDiv = document.getElementById(`image-loading-${index}`);
            const contentDiv = document.getElementById(`image-content-${index}`);
            
            if (response.ok) {
                const data = await response.json();
                
                if (data.success && data.result_image) {
                    if (loadingDiv) loadingDiv.style.display = 'none';
                    if (contentDiv) {
                        contentDiv.innerHTML = `
                            <div style="text-align: center;">
                                <img src="${data.result_image}" alt="Generated Image" style="max-width: 100%; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                                <div style="margin-top: 10px; font-size: 12px; color: #666;">
                                    생성 시간: ${imageTime}초
                                </div>
                            </div>
                        `;
                        contentDiv.style.display = 'block';
                    }
                } else {
                    if (loadingDiv) loadingDiv.style.display = 'none';
                    if (contentDiv) {
                        contentDiv.innerHTML = `<div style="padding: 10px; background: #f8d7da; border-radius: 4px; color: #721c24; font-size: 14px;">이미지 생성 실패: ${data.message || data.error || '알 수 없는 오류'}</div>`;
                        contentDiv.style.display = 'block';
                    }
                }
            } else {
                const errorData = await response.json().catch(() => ({}));
                if (loadingDiv) loadingDiv.style.display = 'none';
                if (contentDiv) {
                    contentDiv.innerHTML = `<div style="padding: 10px; background: #f8d7da; border-radius: 4px; color: #721c24; font-size: 14px;">이미지 생성 실패: ${errorData.message || errorData.error || `HTTP ${response.status}`}</div>`;
                    contentDiv.style.display = 'block';
                }
            }
        } catch (error) {
            console.error(`${result.model} 이미지 생성 오류:`, error);
            const loadingDiv = document.getElementById(`image-loading-${index}`);
            const contentDiv = document.getElementById(`image-content-${index}`);
            if (loadingDiv) loadingDiv.style.display = 'none';
            if (contentDiv) {
                contentDiv.innerHTML = `<div style="padding: 10px; background: #f8d7da; border-radius: 4px; color: #721c24; font-size: 14px;">이미지 생성 중 오류 발생: ${error.message}</div>`;
                contentDiv.style.display = 'block';
            }
        }
    });
    
    // 모든 이미지 생성이 완료될 때까지 대기
    await Promise.allSettled(imagePromises);
}

// ==================== 버전 선택 카드 관련 함수 ====================

// 버전 선택 카드의 "합성" 버튼 클릭 핸들러
function runVersionSelectedFlash() {
    const selectedVersion = document.getElementById('flash-version-select').value;
    
    if (selectedVersion === 'v1') {
        // V1 선택: 기존 V1 모달 호출
        const v1Model = models.find(m => m.id === 'xai-gemini-unified');
        if (v1Model) {
            openModelModal('xai-gemini-unified');
        } else {
            alert('V1 모델을 찾을 수 없습니다. models_config.json을 확인해주세요.');
        }
    } else if (selectedVersion === 'v2') {
        // V2 선택: V2 전용 모달 호출
        openVersionSelectModal('v2');
    } else if (selectedVersion === 'v2.5') {
        // V2.5 선택: V2.5 전용 모달 호출
        openV25Modal();
    } else if (selectedVersion === 'v3') {
        // V3 선택: V3 전용 모달 호출
        openV3Modal();
    } else if (selectedVersion === 'v3-custom') {
        // V3 커스텀 선택: V3 커스텀 전용 모달 호출
        openV3CustomModal();
    } else if (selectedVersion === 'v4') {
        // V4 선택: V4 전용 모달 호출
        openV4Modal();
    } else if (selectedVersion === 'v4-custom') {
        // V4 커스텀 선택: V4 커스텀 전용 모달 호출
        openV4CustomModal();
    }
}

// 버전 선택 모달 열기
function openVersionSelectModal(version) {
    // V2 모달이 이미 있는지 확인
    let modal = document.getElementById('modal-version-select-v2');
    
    if (!modal) {
        // V2 모달 생성
        createVersionSelectModal('v2');
        modal = document.getElementById('modal-version-select-v2');
    }
    
    if (modal) {
        modal.classList.add('show');
        // 모달 데이터 초기화
        if (!modelModals['version-select-v2']) {
            modelModals['version-select-v2'] = {};
        }
    }
}

// 버전 선택 모달 생성
function createVersionSelectModal(version) {
    const container = document.getElementById('model-modals-container');
    
    if (version === 'v2') {
        // V2 모달 생성 (person_image, garment_image, background_image)
        const modalHtml = `
            <div class="model-modal" id="modal-version-select-v2">
                <div class="model-modal-content">
                    <div class="model-modal-header">
                        <div class="model-modal-title">
                            <div class="model-modal-icon">🎯</div>
                            <div>
                                <h2>XAI + Gemini 2.5 Flash V2</h2>
                                <p>SegFormer B2 Parsing + X.AI 프롬프트 생성 + Gemini 이미지 합성</p>
                            </div>
                        </div>
                        <button class="model-modal-close" onclick="closeVersionSelectModal('v2')">&times;</button>
                    </div>
                    <div class="model-modal-body">
                        <div class="model-upload-section">
                            <div class="model-upload-row">
                                <div class="model-upload-item">
                                    <label class="model-upload-label">
                                        <span class="upload-icon">👤</span>
                                        사람 이미지
                                    </label>
                                    <div class="model-upload-area" id="upload-version-select-v2-person">
                                        <input type="file" id="input-version-select-v2-person" accept="image/*" style="display: none;" onchange="handleModelImageUpload(event, 'version-select-v2', 'person')">
                                        <div class="model-upload-content">
                                            <div class="model-upload-icon">📁</div>
                                            <p>이미지를 드래그하거나 클릭</p>
                                            <button class="model-upload-btn" onclick="document.getElementById('input-version-select-v2-person').click()">파일 선택</button>
                                        </div>
                                        <div class="model-preview-container" id="preview-version-select-v2-person" style="display: none;">
                                            <img id="img-version-select-v2-person" alt="Person Preview">
                                            <button class="model-remove-btn" onclick="removeModelImage('version-select-v2', 'person')">&times;</button>
                                        </div>
                                    </div>
                                </div>
                                <div class="model-upload-item">
                                    <label class="model-upload-label">
                                        <span class="upload-icon">👗</span>
                                        의상 이미지
                                    </label>
                                    <div class="model-upload-area" id="upload-version-select-v2-dress">
                                        <input type="file" id="input-version-select-v2-dress" accept="image/*" style="display: none;" onchange="handleModelImageUpload(event, 'version-select-v2', 'dress')">
                                        <div class="model-upload-content">
                                            <div class="model-upload-icon">📁</div>
                                            <p>이미지를 드래그하거나 클릭</p>
                                            <button class="model-upload-btn" onclick="document.getElementById('input-version-select-v2-dress').click()">파일 선택</button>
                                        </div>
                                        <div class="model-preview-container" id="preview-version-select-v2-dress" style="display: none;">
                                            <img id="img-version-select-v2-dress" alt="Dress Preview">
                                            <button class="model-remove-btn" onclick="removeModelImage('version-select-v2', 'dress')">&times;</button>
                                        </div>
                                    </div>
                                </div>
                                <div class="model-upload-item">
                                    <label class="model-upload-label">
                                        <span class="upload-icon">🖼️</span>
                                        배경 이미지
                                    </label>
                                    <div class="model-upload-area" id="upload-version-select-v2-background">
                                        <input type="file" id="input-version-select-v2-background" accept="image/*" style="display: none;" onchange="handleModelImageUpload(event, 'version-select-v2', 'background')">
                                        <div class="model-upload-content">
                                            <div class="model-upload-icon">📁</div>
                                            <p>이미지를 드래그하거나 클릭</p>
                                            <button class="model-upload-btn" onclick="document.getElementById('input-version-select-v2-background').click()">파일 선택</button>
                                        </div>
                                        <div class="model-preview-container" id="preview-version-select-v2-background" style="display: none;">
                                            <img id="img-version-select-v2-background" alt="Background Preview">
                                            <button class="model-remove-btn" onclick="removeModelImage('version-select-v2', 'background')">&times;</button>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="model-action-section">
                            <button class="model-run-btn" id="run-btn-version-select-v2" onclick="runVersionSelectV2()">
                                <span class="btn-icon">🚀</span>
                                합성 실행
                            </button>
                        </div>
                        <div class="model-loading" id="loading-version-select-v2" style="display: none;">
                            <div class="model-spinner"></div>
                            <p>처리 중...</p>
                        </div>
                        <div class="model-result-section" id="result-version-select-v2" style="display: none;">
                            <div class="model-result-header">
                                <div class="model-processing-time">
                                    <span>처리 시간: </span>
                                    <span id="time-version-select-v2">-</span>
                                </div>
                            </div>
                            <div class="model-result-images" id="result-images-version-select-v2">
                                <!-- 결과 이미지가 여기에 표시됨 -->
                            </div>
                            <div class="model-result-actions">
                                <button class="model-download-btn" id="download-btn-version-select-v2" onclick="downloadModelResult('version-select-v2')" style="display: none;">
                                    <span class="btn-icon">💾</span>
                                    결과 다운로드
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        container.insertAdjacentHTML('beforeend', modalHtml);
        
        // 드래그 앤 드롭 설정
        setupVersionSelectModalDragAndDrop('v2');
    }
}

// 버전 선택 모달 닫기
function closeVersionSelectModal(version) {
    const modalId = `modal-version-select-${version}`;
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.remove('show');
        // 결과 초기화
        const resultDiv = document.getElementById(`result-version-select-${version}`);
        if (resultDiv) {
            resultDiv.style.display = 'none';
        }
        delete modelModals[`version-select-${version}`];
    }
}

// V2 모달 드래그 앤 드롭 설정
function setupVersionSelectModalDragAndDrop(version) {
    if (version === 'v2') {
        const types = ['person', 'dress', 'background'];
        
        types.forEach(type => {
            const area = document.getElementById(`upload-version-select-v2-${type}`);
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
                    const input = document.getElementById(`input-version-select-v2-${type}`);
                    if (input) {
                        input.files = files;
                        input.dispatchEvent(new Event('change'));
                    }
                }
            });
        });
    }
}

// V2 합성 실행
async function runVersionSelectV2() {
    const modelId = 'version-select-v2';
    const personFile = modelModals[modelId]?.person;
    const dressFile = modelModals[modelId]?.dress;
    const backgroundFile = modelModals[modelId]?.background;
    
    if (!personFile || !dressFile || !backgroundFile) {
        alert('사람 이미지, 의상 이미지, 배경 이미지를 모두 업로드해주세요.');
        return;
    }
    
    // 파일이 실제로 존재하는지 확인
    if (!(personFile instanceof File) || !(dressFile instanceof File) || !(backgroundFile instanceof File)) {
        alert('이미지 파일이 올바르지 않습니다. 다시 업로드해주세요.');
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
        formData.append('person_image', personFile);
        formData.append('garment_image', dressFile);
        formData.append('background_image', backgroundFile);
        
        const response = await fetch('/api/compose_xai_gemini_v2', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        const endTime = performance.now();
        const processingTime = ((endTime - startTime) / 1000).toFixed(2);
        
        loadingDiv.style.display = 'none';
        runBtn.disabled = false;
        
        if (data.success) {
            // 결과 표시를 위한 임시 모델 객체 생성
            const tempModel = {
                id: modelId,
                name: 'XAI + Gemini 2.5 Flash V2',
                input_type: 'dual_image'
            };
            displayModelResult(modelId, tempModel, data, processingTime);
        } else {
            alert(`오류 발생: ${data.message || data.error}`);
        }
    } catch (error) {
        loadingDiv.style.display = 'none';
        runBtn.disabled = false;
        alert(`V2 합성 실행 중 오류 발생: ${error.message}`);
    }
}

// ==================== V2.5 모달 관련 함수 ====================

// V2.5 모달 열기
function openV25Modal() {
    // V2.5 모달이 이미 있는지 확인
    let modal = document.getElementById('modal-v25');
    
    if (!modal) {
        // V2.5 모달 생성
        createV25Modal();
        modal = document.getElementById('modal-v25');
    }
    
    if (modal) {
        modal.classList.add('show');
        // 모달 데이터 초기화
        if (!modelModals['v25']) {
            modelModals['v25'] = {};
        }
    }
}

// V2.5 모달 생성
function createV25Modal() {
    const container = document.getElementById('model-modals-container');
    
    const modalHtml = `
        <div class="model-modal" id="modal-v25">
            <div class="model-modal-content">
                <div class="model-modal-header">
                    <div class="model-modal-title">
                        <div class="model-modal-icon">✨</div>
                        <div>
                            <h2>XAI + Gemini 2.5 V2.5</h2>
                            <p>인물 전처리 + SegFormer B2 Parsing + XAI 프롬프트 생성 + Gemini 이미지 합성</p>
                        </div>
                    </div>
                    <button class="model-modal-close" onclick="closeV25Modal()">&times;</button>
                </div>
                <div class="model-modal-body">
                    <div class="model-upload-section">
                        <div class="model-upload-row">
                            <div class="model-upload-item">
                                <label class="model-upload-label">
                                    <span class="upload-icon">👤</span>
                                    인물 이미지
                                </label>
                                <div class="model-upload-area" id="upload-v25-person">
                                    <input type="file" id="input-v25-person" accept="image/*" style="display: none;" onchange="handleModelImageUpload(event, 'v25', 'person')">
                                    <div class="model-upload-content">
                                        <div class="model-upload-icon">📁</div>
                                        <p>이미지를 드래그하거나 클릭</p>
                                        <button class="model-upload-btn" onclick="document.getElementById('input-v25-person').click()">파일 선택</button>
                                    </div>
                                    <div class="model-preview-container" id="preview-v25-person" style="display: none;">
                                        <img id="img-v25-person" alt="Person Preview">
                                        <button class="model-remove-btn" onclick="removeModelImage('v25', 'person')">&times;</button>
                                    </div>
                                </div>
                            </div>
                            <div class="model-upload-item">
                                <label class="model-upload-label">
                                    <span class="upload-icon">👗</span>
                                    의상 이미지
                                </label>
                                <div class="model-upload-area" id="upload-v25-dress">
                                    <input type="file" id="input-v25-dress" accept="image/*" style="display: none;" onchange="handleModelImageUpload(event, 'v25', 'dress')">
                                    <div class="model-upload-content">
                                        <div class="model-upload-icon">📁</div>
                                        <p>이미지를 드래그하거나 클릭</p>
                                        <button class="model-upload-btn" onclick="document.getElementById('input-v25-dress').click()">파일 선택</button>
                                    </div>
                                    <div class="model-preview-container" id="preview-v25-dress" style="display: none;">
                                        <img id="img-v25-dress" alt="Dress Preview">
                                        <button class="model-remove-btn" onclick="removeModelImage('v25', 'dress')">&times;</button>
                                    </div>
                                </div>
                            </div>
                            <div class="model-upload-item">
                                <label class="model-upload-label">
                                    <span class="upload-icon">🖼️</span>
                                    배경 이미지
                                </label>
                                <div class="model-upload-area" id="upload-v25-background">
                                    <input type="file" id="input-v25-background" accept="image/*" style="display: none;" onchange="handleModelImageUpload(event, 'v25', 'background')">
                                    <div class="model-upload-content">
                                        <div class="model-upload-icon">📁</div>
                                        <p>이미지를 드래그하거나 클릭</p>
                                        <button class="model-upload-btn" onclick="document.getElementById('input-v25-background').click()">파일 선택</button>
                                    </div>
                                    <div class="model-preview-container" id="preview-v25-background" style="display: none;">
                                        <img id="img-v25-background" alt="Background Preview">
                                        <button class="model-remove-btn" onclick="removeModelImage('v25', 'background')">&times;</button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="model-action-section">
                        <button class="model-run-btn" id="run-btn-v25" onclick="runV25Compose()">
                            <span class="btn-icon">🚀</span>
                            합성 실행
                        </button>
                    </div>
                    <div class="model-loading" id="loading-v25" style="display: none;">
                        <div class="model-spinner"></div>
                        <p>처리 중...</p>
                    </div>
                    <div class="model-result-section" id="result-v25" style="display: none;">
                        <div class="model-result-header">
                            <div class="model-processing-time">
                                <span>처리 시간: </span>
                                <span id="time-v25">-</span>
                            </div>
                        </div>
                        <div class="model-result-images" id="result-images-v25">
                            <!-- 결과 이미지가 여기에 표시됨 -->
                        </div>
                        <div class="model-result-actions">
                            <button class="model-download-btn" id="download-btn-v25" onclick="downloadModelResult('v25')" style="display: none;">
                                <span class="btn-icon">💾</span>
                                결과 다운로드
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    container.insertAdjacentHTML('beforeend', modalHtml);
    
    // 드래그 앤 드롭 설정
    setupV25ModalDragAndDrop();
}

// V2.5 모달 닫기
function closeV25Modal() {
    const modal = document.getElementById('modal-v25');
    if (modal) {
        modal.classList.remove('show');
        // 결과 초기화
        const resultDiv = document.getElementById('result-v25');
        if (resultDiv) {
            resultDiv.style.display = 'none';
        }
        delete modelModals['v25'];
    }
}

// V2.5 모달 드래그 앤 드롭 설정
function setupV25ModalDragAndDrop() {
    const types = ['person', 'dress', 'background'];
    
    types.forEach(type => {
        const area = document.getElementById(`upload-v25-${type}`);
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
                const input = document.getElementById(`input-v25-${type}`);
                if (input) {
                    input.files = files;
                    input.dispatchEvent(new Event('change'));
                }
            }
        });
    });
}

// V2.5 합성 실행
async function runV25Compose() {
    const modelId = 'v25';
    const personFile = modelModals[modelId]?.person;
    const dressFile = modelModals[modelId]?.dress;
    const backgroundFile = modelModals[modelId]?.background;
    
    if (!personFile || !dressFile || !backgroundFile) {
        alert('인물 이미지, 의상 이미지, 배경 이미지를 모두 업로드해주세요.');
        return;
    }
    
    // 파일이 실제로 존재하는지 확인
    if (!(personFile instanceof File) || !(dressFile instanceof File) || !(backgroundFile instanceof File)) {
        alert('이미지 파일이 올바르지 않습니다. 다시 업로드해주세요.');
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
        formData.append('person_image', personFile);
        formData.append('garment_image', dressFile);
        formData.append('background_image', backgroundFile);
        formData.append('use_person_preprocess', 'true');
        
        const response = await fetch('/fit/v2.5/compose', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        const endTime = performance.now();
        const processingTime = ((endTime - startTime) / 1000).toFixed(2);
        
        loadingDiv.style.display = 'none';
        runBtn.disabled = false;
        
        if (data.success) {
            // 결과 표시를 위한 임시 모델 객체 생성
            const tempModel = {
                id: modelId,
                name: 'XAI + Gemini 2.5 V2.5',
                input_type: 'dual_image'
            };
            displayModelResult(modelId, tempModel, data, processingTime);
        } else {
            alert(`오류 발생: ${data.message || data.error}`);
        }
    } catch (error) {
        loadingDiv.style.display = 'none';
        runBtn.disabled = false;
        alert(`V2.5 합성 실행 중 오류 발생: ${error.message}`);
    }
}

// V3 모달 열기
function openV3Modal() {
    let modal = document.getElementById('modal-v3');
    
    if (!modal) {
        createV3Modal();
        modal = document.getElementById('modal-v3');
    }
    
    if (modal) {
        modal.classList.add('show');
        if (!modelModals['v3']) {
            modelModals['v3'] = {};
        }
    }
}

// V3 모달 생성
function createV3Modal() {
    const container = document.getElementById('model-modals-container');
    
    const modalHtml = `
        <div class="model-modal" id="modal-v3">
            <div class="model-modal-content">
                <div class="model-modal-header">
                    <div class="model-modal-title">
                        <div class="model-modal-icon">🎯</div>
                        <div>
                            <h2>XAI + Gemini 2.5 Flash V3</h2>
                            <p>2단계 Gemini 플로우: 의상 교체 + 배경 합성 + 조명 보정</p>
                        </div>
                    </div>
                    <button class="model-modal-close" onclick="closeV3Modal()">&times;</button>
                </div>
                <div class="model-modal-body">
                    <div class="model-upload-section">
                        <div class="model-upload-row">
                            <div class="model-upload-item">
                                <label class="model-upload-label">
                                    <span class="upload-icon">👤</span>
                                    사람 이미지
                                </label>
                                <div class="model-upload-area" id="upload-v3-person">
                                    <input type="file" id="input-v3-person" accept="image/*" style="display: none;" onchange="handleModelImageUpload(event, 'v3', 'person')">
                                    <div class="model-upload-content">
                                        <div class="model-upload-icon">📁</div>
                                        <p>이미지를 드래그하거나 클릭</p>
                                        <button class="model-upload-btn" onclick="document.getElementById('input-v3-person').click()">파일 선택</button>
                                    </div>
                                    <div class="model-preview-container" id="preview-v3-person" style="display: none;">
                                        <img id="img-v3-person" alt="Person Preview">
                                        <button class="model-remove-btn" onclick="removeModelImage('v3', 'person')">&times;</button>
                                    </div>
                                </div>
                            </div>
                            <div class="model-upload-item">
                                <label class="model-upload-label">
                                    <span class="upload-icon">👗</span>
                                    의상 이미지
                                </label>
                                <div class="model-upload-area" id="upload-v3-dress">
                                    <input type="file" id="input-v3-dress" accept="image/*" style="display: none;" onchange="handleModelImageUpload(event, 'v3', 'dress')">
                                    <div class="model-upload-content">
                                        <div class="model-upload-icon">📁</div>
                                        <p>이미지를 드래그하거나 클릭</p>
                                        <button class="model-upload-btn" onclick="document.getElementById('input-v3-dress').click()">파일 선택</button>
                                    </div>
                                    <div class="model-preview-container" id="preview-v3-dress" style="display: none;">
                                        <img id="img-v3-dress" alt="Dress Preview">
                                        <button class="model-remove-btn" onclick="removeModelImage('v3', 'dress')">&times;</button>
                                    </div>
                                </div>
                            </div>
                            <div class="model-upload-item">
                                <label class="model-upload-label">
                                    <span class="upload-icon">🖼️</span>
                                    배경 이미지
                                </label>
                                <div class="model-upload-area" id="upload-v3-background">
                                    <input type="file" id="input-v3-background" accept="image/*" style="display: none;" onchange="handleModelImageUpload(event, 'v3', 'background')">
                                    <div class="model-upload-content">
                                        <div class="model-upload-icon">📁</div>
                                        <p>이미지를 드래그하거나 클릭</p>
                                        <button class="model-upload-btn" onclick="document.getElementById('input-v3-background').click()">파일 선택</button>
                                    </div>
                                    <div class="model-preview-container" id="preview-v3-background" style="display: none;">
                                        <img id="img-v3-background" alt="Background Preview">
                                        <button class="model-remove-btn" onclick="removeModelImage('v3', 'background')">&times;</button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="model-action-section">
                        <button class="model-run-btn" id="run-btn-v3" onclick="runV3Compose()">
                            <span class="btn-icon">🚀</span>
                            V3 합성 실행
                        </button>
                    </div>
                    <div class="model-loading" id="loading-v3" style="display: none;">
                        <div class="loading-spinner"></div>
                        <p>V3 파이프라인 실행 중...</p>
                    </div>
                    <div class="model-result" id="result-v3" style="display: none;">
                        <div class="model-result-header">
                            <h3>결과</h3>
                            <div class="model-result-meta">
                                <div class="model-result-time">
                                    <span>처리 시간: </span>
                                    <span id="time-v3">-</span>
                                </div>
                            </div>
                        </div>
                        <div class="model-result-images" id="result-images-v3">
                            <!-- 결과 이미지가 여기에 표시됨 -->
                        </div>
                        <div class="model-result-actions">
                            <button class="model-download-btn" id="download-btn-v3" onclick="downloadModelResult('v3')" style="display: none;">
                                <span class="btn-icon">💾</span>
                                결과 다운로드
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    container.insertAdjacentHTML('beforeend', modalHtml);
    
    // 드래그 앤 드롭 설정
    setupV3ModalDragAndDrop();
}

// V3 모달 닫기
function closeV3Modal() {
    const modal = document.getElementById('modal-v3');
    if (modal) {
        modal.classList.remove('show');
        const resultDiv = document.getElementById('result-v3');
        if (resultDiv) {
            resultDiv.style.display = 'none';
        }
        delete modelModals['v3'];
    }
}

// V3 모달 드래그 앤 드롭 설정
function setupV3ModalDragAndDrop() {
    const types = ['person', 'dress', 'background'];
    
    types.forEach(type => {
        const area = document.getElementById(`upload-v3-${type}`);
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
                const input = document.getElementById(`input-v3-${type}`);
                if (input) {
                    input.files = files;
                    input.dispatchEvent(new Event('change'));
                }
            }
        });
    });
}

// V3 합성 실행
async function runV3Compose() {
    const modelId = 'v3';
    const personFile = modelModals[modelId]?.person;
    const dressFile = modelModals[modelId]?.dress;
    const backgroundFile = modelModals[modelId]?.background;
    
    if (!personFile || !dressFile || !backgroundFile) {
        alert('인물 이미지, 의상 이미지, 배경 이미지를 모두 업로드해주세요.');
        return;
    }
    
    if (!(personFile instanceof File) || !(dressFile instanceof File) || !(backgroundFile instanceof File)) {
        alert('이미지 파일이 올바르지 않습니다. 다시 업로드해주세요.');
        return;
    }
    
    const loadingDiv = document.getElementById(`loading-${modelId}`);
    const resultDiv = document.getElementById(`result-${modelId}`);
    const runBtn = document.getElementById(`run-btn-${modelId}`);
    
    loadingDiv.style.display = 'flex';
    resultDiv.style.display = 'none';
    runBtn.disabled = true;
    
    const startTime = performance.now();
    
    try {
        const formData = new FormData();
        formData.append('person_image', personFile);
        formData.append('garment_image', dressFile);
        formData.append('background_image', backgroundFile);
        
        const response = await fetch('/fit/v3/compose', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        const endTime = performance.now();
        const processingTime = ((endTime - startTime) / 1000).toFixed(2);
        
        loadingDiv.style.display = 'none';
        runBtn.disabled = false;
        
        if (data.success) {
            const tempModel = {
                id: modelId,
                name: 'XAI + Gemini 2.5 V3',
                input_type: 'dual_image'
            };
            displayModelResult(modelId, tempModel, data, processingTime);
        } else {
            alert(`오류 발생: ${data.message || data.error}`);
        }
    } catch (error) {
        loadingDiv.style.display = 'none';
        runBtn.disabled = false;
        alert(`V3 합성 실행 중 오류 발생: ${error.message}`);
    }
}

// ==================== V4 모달 관련 함수 ====================

// V4 모달 열기
function openV4Modal() {
    let modal = document.getElementById('modal-v4');
    
    if (!modal) {
        createV4Modal();
        modal = document.getElementById('modal-v4');
    }
    
    if (modal) {
        modal.classList.add('show');
        if (!modelModals['v4']) {
            modelModals['v4'] = {};
        }
    }
}

// V4 모달 생성
function createV4Modal() {
    const container = document.getElementById('model-modals-container');
    
    const modalHtml = `
        <div class="model-modal" id="modal-v4">
            <div class="model-modal-content">
                <div class="model-modal-header">
                    <div class="model-modal-title">
                        <div class="model-modal-icon">🎯</div>
                        <div>
                            <h2>XAI + Gemini 3 Flash V4</h2>
                            <p>2단계 Gemini 3 Flash 플로우: 의상 교체 + 배경 합성 + 조명 보정</p>
                        </div>
                    </div>
                    <button class="model-modal-close" onclick="closeV4Modal()">&times;</button>
                </div>
                <div class="model-modal-body">
                    <div class="model-upload-section">
                        <div class="model-upload-row">
                            <div class="model-upload-item">
                                <label class="model-upload-label">
                                    <span class="upload-icon">👤</span>
                                    사람 이미지
                                </label>
                                <div class="model-upload-area" id="upload-v4-person">
                                    <input type="file" id="input-v4-person" accept="image/*" style="display: none;" onchange="handleModelImageUpload(event, 'v4', 'person')">
                                    <div class="model-upload-content">
                                        <div class="model-upload-icon">📁</div>
                                        <p>이미지를 드래그하거나 클릭</p>
                                        <button class="model-upload-btn" onclick="document.getElementById('input-v4-person').click()">파일 선택</button>
                                    </div>
                                    <div class="model-preview-container" id="preview-v4-person" style="display: none;">
                                        <img id="img-v4-person" alt="Person Preview">
                                        <button class="model-remove-btn" onclick="removeModelImage('v4', 'person')">&times;</button>
                                    </div>
                                </div>
                            </div>
                            <div class="model-upload-item">
                                <label class="model-upload-label">
                                    <span class="upload-icon">👗</span>
                                    의상 이미지
                                </label>
                                <div class="model-upload-area" id="upload-v4-dress">
                                    <input type="file" id="input-v4-dress" accept="image/*" style="display: none;" onchange="handleModelImageUpload(event, 'v4', 'dress')">
                                    <div class="model-upload-content">
                                        <div class="model-upload-icon">📁</div>
                                        <p>이미지를 드래그하거나 클릭</p>
                                        <button class="model-upload-btn" onclick="document.getElementById('input-v4-dress').click()">파일 선택</button>
                                    </div>
                                    <div class="model-preview-container" id="preview-v4-dress" style="display: none;">
                                        <img id="img-v4-dress" alt="Dress Preview">
                                        <button class="model-remove-btn" onclick="removeModelImage('v4', 'dress')">&times;</button>
                                    </div>
                                </div>
                            </div>
                            <div class="model-upload-item">
                                <label class="model-upload-label">
                                    <span class="upload-icon">🖼️</span>
                                    배경 이미지
                                </label>
                                <div class="model-upload-area" id="upload-v4-background">
                                    <input type="file" id="input-v4-background" accept="image/*" style="display: none;" onchange="handleModelImageUpload(event, 'v4', 'background')">
                                    <div class="model-upload-content">
                                        <div class="model-upload-icon">📁</div>
                                        <p>이미지를 드래그하거나 클릭</p>
                                        <button class="model-upload-btn" onclick="document.getElementById('input-v4-background').click()">파일 선택</button>
                                    </div>
                                    <div class="model-preview-container" id="preview-v4-background" style="display: none;">
                                        <img id="img-v4-background" alt="Background Preview">
                                        <button class="model-remove-btn" onclick="removeModelImage('v4', 'background')">&times;</button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="model-action-section">
                        <button class="model-run-btn" id="run-btn-v4" onclick="runV4Compose()">
                            <span class="btn-icon">🚀</span>
                            V4 합성 실행
                        </button>
                    </div>
                    <div class="model-loading" id="loading-v4" style="display: none;">
                        <div class="loading-spinner"></div>
                        <p>V4 파이프라인 실행 중...</p>
                    </div>
                    <div class="model-result" id="result-v4" style="display: none;">
                        <div class="model-result-header">
                            <h3>결과</h3>
                            <div class="model-result-meta">
                                <div class="model-result-time">
                                    <span>처리 시간: </span>
                                    <span id="time-v4">-</span>
                                </div>
                            </div>
                        </div>
                        <div class="model-result-images" id="result-images-v4">
                            <!-- 결과 이미지가 여기에 표시됨 -->
                        </div>
                        <div class="model-result-actions">
                            <button class="model-download-btn" id="download-btn-v4" onclick="downloadModelResult('v4')" style="display: none;">
                                <span class="btn-icon">💾</span>
                                결과 다운로드
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    container.insertAdjacentHTML('beforeend', modalHtml);
    
    // 드래그 앤 드롭 설정
    setupV4ModalDragAndDrop();
}

// V4 모달 닫기
function closeV4Modal() {
    const modal = document.getElementById('modal-v4');
    if (modal) {
        modal.classList.remove('show');
        const resultDiv = document.getElementById('result-v4');
        if (resultDiv) {
            resultDiv.style.display = 'none';
        }
        delete modelModals['v4'];
    }
}

// V4 모달 드래그 앤 드롭 설정
function setupV4ModalDragAndDrop() {
    const types = ['person', 'dress', 'background'];
    
    types.forEach(type => {
        const area = document.getElementById(`upload-v4-${type}`);
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
                const input = document.getElementById(`input-v4-${type}`);
                if (input) {
                    input.files = files;
                    input.dispatchEvent(new Event('change'));
                }
            }
        });
    });
}

// V4 합성 실행
async function runV4Compose() {
    const modelId = 'v4';
    const personFile = modelModals[modelId]?.person;
    const dressFile = modelModals[modelId]?.dress;
    const backgroundFile = modelModals[modelId]?.background;
    
    if (!personFile || !dressFile || !backgroundFile) {
        alert('인물 이미지, 의상 이미지, 배경 이미지를 모두 업로드해주세요.');
        return;
    }
    
    if (!(personFile instanceof File) || !(dressFile instanceof File) || !(backgroundFile instanceof File)) {
        alert('이미지 파일이 올바르지 않습니다. 다시 업로드해주세요.');
        return;
    }
    
    const loadingDiv = document.getElementById(`loading-${modelId}`);
    const resultDiv = document.getElementById(`result-${modelId}`);
    const runBtn = document.getElementById(`run-btn-${modelId}`);
    
    loadingDiv.style.display = 'flex';
    resultDiv.style.display = 'none';
    runBtn.disabled = true;
    
    const startTime = performance.now();
    
    try {
        const formData = new FormData();
        formData.append('person_image', personFile);
        formData.append('garment_image', dressFile);
        formData.append('background_image', backgroundFile);
        
        const response = await fetch('/fit/v4/compose', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        const endTime = performance.now();
        const processingTime = ((endTime - startTime) / 1000).toFixed(2);
        
        loadingDiv.style.display = 'none';
        runBtn.disabled = false;
        
        if (data.success) {
            const tempModel = {
                id: modelId,
                name: 'XAI + Gemini 3 Flash V4',
                input_type: 'dual_image'
            };
            displayModelResult(modelId, tempModel, data, processingTime);
        } else {
            alert(`오류 발생: ${data.message || data.error}`);
        }
    } catch (error) {
        loadingDiv.style.display = 'none';
        runBtn.disabled = false;
        alert(`V4 합성 실행 중 오류 발생: ${error.message}`);
    }
}

// ==================== V3 커스텀 모달 관련 함수 ====================

// V3 커스텀 모달 열기
function openV3CustomModal() {
    let modal = document.getElementById('modal-v3-custom');
    
    if (!modal) {
        createV3CustomModal();
        modal = document.getElementById('modal-v3-custom');
    }
    
    if (modal) {
        modal.classList.add('show');
        if (!modelModals['v3-custom']) {
            modelModals['v3-custom'] = {};
        }
    }
}

// V3 커스텀 모달 생성
function createV3CustomModal() {
    const container = document.getElementById('model-modals-container');
    
    const modalHtml = `
        <div class="model-modal" id="modal-v3-custom">
            <div class="model-modal-content">
                <div class="model-modal-header">
                    <div class="model-modal-title">
                        <div class="model-modal-icon">🎯</div>
                        <div>
                            <h2>XAI + Gemini 2.5 Flash V3 커스텀</h2>
                            <p>의상 누끼 자동 처리 + 2단계 Gemini 플로우: 의상 교체 + 배경 합성 + 조명 보정</p>
                        </div>
                    </div>
                    <button class="model-modal-close" onclick="closeV3CustomModal()">&times;</button>
                </div>
                <div class="model-modal-body">
                    <div class="model-upload-section">
                        <div class="model-upload-row">
                            <div class="model-upload-item">
                                <label class="model-upload-label">
                                    <span class="upload-icon">👤</span>
                                    사람 이미지
                                </label>
                                <div class="model-upload-area" id="upload-v3-custom-person">
                                    <input type="file" id="input-v3-custom-person" accept="image/*" style="display: none;" onchange="handleModelImageUpload(event, 'v3-custom', 'person')">
                                    <div class="model-upload-content">
                                        <div class="model-upload-icon">📁</div>
                                        <p>이미지를 드래그하거나 클릭</p>
                                        <button class="model-upload-btn" onclick="document.getElementById('input-v3-custom-person').click()">파일 선택</button>
                                    </div>
                                    <div class="model-preview-container" id="preview-v3-custom-person" style="display: none;">
                                        <img id="img-v3-custom-person" alt="Person Preview">
                                        <button class="model-remove-btn" onclick="removeModelImage('v3-custom', 'person')">&times;</button>
                                    </div>
                                </div>
                            </div>
                            <div class="model-upload-item">
                                <label class="model-upload-label">
                                    <span class="upload-icon">👗</span>
                                    의상 이미지
                                </label>
                                <div class="model-upload-area" id="upload-v3-custom-dress">
                                    <input type="file" id="input-v3-custom-dress" accept="image/*" style="display: none;" onchange="handleModelImageUpload(event, 'v3-custom', 'dress')">
                                    <div class="model-upload-content">
                                        <div class="model-upload-icon">📁</div>
                                        <p>이미지를 드래그하거나 클릭</p>
                                        <button class="model-upload-btn" onclick="document.getElementById('input-v3-custom-dress').click()">파일 선택</button>
                                    </div>
                                    <div class="model-preview-container" id="preview-v3-custom-dress" style="display: none;">
                                        <img id="img-v3-custom-dress" alt="Dress Preview">
                                        <button class="model-remove-btn" onclick="removeModelImage('v3-custom', 'dress')">&times;</button>
                                    </div>
                                </div>
                            </div>
                            <div class="model-upload-item">
                                <label class="model-upload-label">
                                    <span class="upload-icon">🖼️</span>
                                    배경 이미지
                                </label>
                                <div class="model-upload-area" id="upload-v3-custom-background">
                                    <input type="file" id="input-v3-custom-background" accept="image/*" style="display: none;" onchange="handleModelImageUpload(event, 'v3-custom', 'background')">
                                    <div class="model-upload-content">
                                        <div class="model-upload-icon">📁</div>
                                        <p>이미지를 드래그하거나 클릭</p>
                                        <button class="model-upload-btn" onclick="document.getElementById('input-v3-custom-background').click()">파일 선택</button>
                                    </div>
                                    <div class="model-preview-container" id="preview-v3-custom-background" style="display: none;">
                                        <img id="img-v3-custom-background" alt="Background Preview">
                                        <button class="model-remove-btn" onclick="removeModelImage('v3-custom', 'background')">&times;</button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="model-action-section">
                        <button class="model-run-btn" id="run-btn-v3-custom" onclick="runV3CustomCompose()">
                            <span class="btn-icon">🚀</span>
                            V3 커스텀 합성 실행
                        </button>
                    </div>
                    <div class="model-loading" id="loading-v3-custom" style="display: none;">
                        <div class="loading-spinner"></div>
                        <p>V3 커스텀 파이프라인 실행 중...</p>
                    </div>
                    <div class="model-result" id="result-v3-custom" style="display: none;">
                        <div class="model-result-header">
                            <h3>결과</h3>
                            <div class="model-result-meta">
                                <div class="model-result-time">
                                    <span>처리 시간: </span>
                                    <span id="time-v3-custom">-</span>
                                </div>
                            </div>
                        </div>
                        <div class="model-result-images" id="result-images-v3-custom">
                            <!-- 결과 이미지가 여기에 표시됨 -->
                        </div>
                        <div class="model-result-actions">
                            <button class="model-download-btn" id="download-btn-v3-custom" onclick="downloadModelResult('v3-custom')" style="display: none;">
                                <span class="btn-icon">💾</span>
                                결과 다운로드
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    container.insertAdjacentHTML('beforeend', modalHtml);
    
    // 드래그 앤 드롭 설정
    setupV3CustomModalDragAndDrop();
}

// V3 커스텀 모달 닫기
function closeV3CustomModal() {
    const modal = document.getElementById('modal-v3-custom');
    if (modal) {
        modal.classList.remove('show');
        const resultDiv = document.getElementById('result-v3-custom');
        if (resultDiv) {
            resultDiv.style.display = 'none';
        }
        delete modelModals['v3-custom'];
    }
}

// V3 커스텀 모달 드래그 앤 드롭 설정
function setupV3CustomModalDragAndDrop() {
    const types = ['person', 'dress', 'background'];
    
    types.forEach(type => {
        const area = document.getElementById(`upload-v3-custom-${type}`);
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
                const input = document.getElementById(`input-v3-custom-${type}`);
                if (input) {
                    input.files = files;
                    input.dispatchEvent(new Event('change'));
                }
            }
        });
    });
}

// V3 커스텀 합성 실행
async function runV3CustomCompose() {
    const modelId = 'v3-custom';
    const personFile = modelModals[modelId]?.person;
    const dressFile = modelModals[modelId]?.dress;
    const backgroundFile = modelModals[modelId]?.background;
    
    if (!personFile || !dressFile || !backgroundFile) {
        alert('인물 이미지, 의상 이미지, 배경 이미지를 모두 업로드해주세요.');
        return;
    }
    
    if (!(personFile instanceof File) || !(dressFile instanceof File) || !(backgroundFile instanceof File)) {
        alert('이미지 파일이 올바르지 않습니다. 다시 업로드해주세요.');
        return;
    }
    
    const loadingDiv = document.getElementById(`loading-${modelId}`);
    const resultDiv = document.getElementById(`result-${modelId}`);
    const runBtn = document.getElementById(`run-btn-${modelId}`);
    
    loadingDiv.style.display = 'flex';
    resultDiv.style.display = 'none';
    runBtn.disabled = true;
    
    const startTime = performance.now();
    
    try {
        const formData = new FormData();
        formData.append('person_image', personFile);
        formData.append('garment_image', dressFile);
        formData.append('background_image', backgroundFile);
        
        const response = await fetch('/fit/custom-v3/compose', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        const endTime = performance.now();
        const processingTime = ((endTime - startTime) / 1000).toFixed(2);
        
        loadingDiv.style.display = 'none';
        runBtn.disabled = false;
        
        if (data.success) {
            const tempModel = {
                id: modelId,
                name: 'XAI + Gemini 2.5 V3 커스텀',
                input_type: 'dual_image'
            };
            displayModelResult(modelId, tempModel, data, processingTime);
        } else {
            alert(`오류 발생: ${data.message || data.error}`);
        }
    } catch (error) {
        loadingDiv.style.display = 'none';
        runBtn.disabled = false;
        alert(`V3 커스텀 합성 실행 중 오류 발생: ${error.message}`);
    }
}

// ==================== V4 커스텀 모달 관련 함수 ====================

// V4 커스텀 모달 열기
function openV4CustomModal() {
    let modal = document.getElementById('modal-v4-custom');
    
    if (!modal) {
        createV4CustomModal();
        modal = document.getElementById('modal-v4-custom');
    }
    
    if (modal) {
        modal.classList.add('show');
        if (!modelModals['v4-custom']) {
            modelModals['v4-custom'] = {};
        }
    }
}

// V4 커스텀 모달 생성
function createV4CustomModal() {
    const container = document.getElementById('model-modals-container');
    
    const modalHtml = `
        <div class="model-modal" id="modal-v4-custom">
            <div class="model-modal-content">
                <div class="model-modal-header">
                    <div class="model-modal-title">
                        <div class="model-modal-icon">🎯</div>
                        <div>
                            <h2>XAI + Gemini 3 Flash V4 커스텀</h2>
                            <p>의상 누끼 자동 처리 + 2단계 Gemini 3 플로우: 의상 교체 + 배경 합성 + 조명 보정</p>
                        </div>
                    </div>
                    <button class="model-modal-close" onclick="closeV4CustomModal()">&times;</button>
                </div>
                <div class="model-modal-body">
                    <div class="model-upload-section">
                        <div class="model-upload-row">
                            <div class="model-upload-item">
                                <label class="model-upload-label">
                                    <span class="upload-icon">👤</span>
                                    사람 이미지
                                </label>
                                <div class="model-upload-area" id="upload-v4-custom-person">
                                    <input type="file" id="input-v4-custom-person" accept="image/*" style="display: none;" onchange="handleModelImageUpload(event, 'v4-custom', 'person')">
                                    <div class="model-upload-content">
                                        <div class="model-upload-icon">📁</div>
                                        <p>이미지를 드래그하거나 클릭</p>
                                        <button class="model-upload-btn" onclick="document.getElementById('input-v4-custom-person').click()">파일 선택</button>
                                    </div>
                                    <div class="model-preview-container" id="preview-v4-custom-person" style="display: none;">
                                        <img id="img-v4-custom-person" alt="Person Preview">
                                        <button class="model-remove-btn" onclick="removeModelImage('v4-custom', 'person')">&times;</button>
                                    </div>
                                </div>
                            </div>
                            <div class="model-upload-item">
                                <label class="model-upload-label">
                                    <span class="upload-icon">👗</span>
                                    의상 이미지
                                </label>
                                <div class="model-upload-area" id="upload-v4-custom-dress">
                                    <input type="file" id="input-v4-custom-dress" accept="image/*" style="display: none;" onchange="handleModelImageUpload(event, 'v4-custom', 'dress')">
                                    <div class="model-upload-content">
                                        <div class="model-upload-icon">📁</div>
                                        <p>이미지를 드래그하거나 클릭</p>
                                        <button class="model-upload-btn" onclick="document.getElementById('input-v4-custom-dress').click()">파일 선택</button>
                                    </div>
                                    <div class="model-preview-container" id="preview-v4-custom-dress" style="display: none;">
                                        <img id="img-v4-custom-dress" alt="Dress Preview">
                                        <button class="model-remove-btn" onclick="removeModelImage('v4-custom', 'dress')">&times;</button>
                                    </div>
                                </div>
                            </div>
                            <div class="model-upload-item">
                                <label class="model-upload-label">
                                    <span class="upload-icon">🖼️</span>
                                    배경 이미지
                                </label>
                                <div class="model-upload-area" id="upload-v4-custom-background">
                                    <input type="file" id="input-v4-custom-background" accept="image/*" style="display: none;" onchange="handleModelImageUpload(event, 'v4-custom', 'background')">
                                    <div class="model-upload-content">
                                        <div class="model-upload-icon">📁</div>
                                        <p>이미지를 드래그하거나 클릭</p>
                                        <button class="model-upload-btn" onclick="document.getElementById('input-v4-custom-background').click()">파일 선택</button>
                                    </div>
                                    <div class="model-preview-container" id="preview-v4-custom-background" style="display: none;">
                                        <img id="img-v4-custom-background" alt="Background Preview">
                                        <button class="model-remove-btn" onclick="removeModelImage('v4-custom', 'background')">&times;</button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="model-action-section">
                        <button class="model-run-btn" id="run-btn-v4-custom" onclick="runV4CustomCompose()">
                            <span class="btn-icon">🚀</span>
                            V4 커스텀 합성 실행
                        </button>
                    </div>
                    <div class="model-loading" id="loading-v4-custom" style="display: none;">
                        <div class="loading-spinner"></div>
                        <p>V4 커스텀 파이프라인 실행 중...</p>
                    </div>
                    <div class="model-result" id="result-v4-custom" style="display: none;">
                        <div class="model-result-header">
                            <h3>결과</h3>
                            <div class="model-result-meta">
                                <div class="model-result-time">
                                    <span>처리 시간: </span>
                                    <span id="time-v4-custom">-</span>
                                </div>
                            </div>
                        </div>
                        <div class="model-result-images" id="result-images-v4-custom">
                            <!-- 결과 이미지가 여기에 표시됨 -->
                        </div>
                        <div class="model-result-actions">
                            <button class="model-download-btn" id="download-btn-v4-custom" onclick="downloadModelResult('v4-custom')" style="display: none;">
                                <span class="btn-icon">💾</span>
                                결과 다운로드
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    container.insertAdjacentHTML('beforeend', modalHtml);
    
    // 드래그 앤 드롭 설정
    setupV4CustomModalDragAndDrop();
}

// V4 커스텀 모달 닫기
function closeV4CustomModal() {
    const modal = document.getElementById('modal-v4-custom');
    if (modal) {
        modal.classList.remove('show');
        const resultDiv = document.getElementById('result-v4-custom');
        if (resultDiv) {
            resultDiv.style.display = 'none';
        }
        delete modelModals['v4-custom'];
    }
}

// V4 커스텀 모달 드래그 앤 드롭 설정
function setupV4CustomModalDragAndDrop() {
    const types = ['person', 'dress', 'background'];
    
    types.forEach(type => {
        const area = document.getElementById(`upload-v4-custom-${type}`);
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
                const input = document.getElementById(`input-v4-custom-${type}`);
                if (input) {
                    input.files = files;
                    input.dispatchEvent(new Event('change'));
                }
            }
        });
    });
}

// V4 커스텀 합성 실행
async function runV4CustomCompose() {
    const modelId = 'v4-custom';
    const personFile = modelModals[modelId]?.person;
    const dressFile = modelModals[modelId]?.dress;
    const backgroundFile = modelModals[modelId]?.background;
    
    if (!personFile || !dressFile || !backgroundFile) {
        alert('인물 이미지, 의상 이미지, 배경 이미지를 모두 업로드해주세요.');
        return;
    }
    
    if (!(personFile instanceof File) || !(dressFile instanceof File) || !(backgroundFile instanceof File)) {
        alert('이미지 파일이 올바르지 않습니다. 다시 업로드해주세요.');
        return;
    }
    
    const loadingDiv = document.getElementById(`loading-${modelId}`);
    const resultDiv = document.getElementById(`result-${modelId}`);
    const runBtn = document.getElementById(`run-btn-${modelId}`);
    
    loadingDiv.style.display = 'flex';
    resultDiv.style.display = 'none';
    runBtn.disabled = true;
    
    const startTime = performance.now();
    
    try {
        const formData = new FormData();
        formData.append('person_image', personFile);
        formData.append('garment_image', dressFile);
        formData.append('background_image', backgroundFile);
        
        const response = await fetch('/fit/custom-v4/compose', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        const endTime = performance.now();
        const processingTime = ((endTime - startTime) / 1000).toFixed(2);
        
        loadingDiv.style.display = 'none';
        runBtn.disabled = false;
        
        if (data.success) {
            const tempModel = {
                id: modelId,
                name: 'XAI + Gemini 3 V4 커스텀',
                input_type: 'dual_image'
            };
            displayModelResult(modelId, tempModel, data, processingTime);
        } else {
            alert(`오류 발생: ${data.message || data.error}`);
        }
    } catch (error) {
        loadingDiv.style.display = 'none';
        runBtn.disabled = false;
        alert(`V4 커스텀 합성 실행 중 오류 발생: ${error.message}`);
    }
}

// ==================== V4V5 비교 기능 ====================

// V4V5 비교 이미지 저장
let v4v5Images = {
    person: null,
    garment: null,
    background: null
};

// V4V5 비교 모달 열기
function openV4V5CompareModal() {
    const modal = document.getElementById('modal-v4v5-compare');
    if (modal) {
        // 버튼 카드의 드롭다운 선택값에 따라 모달 제목 업데이트
        const pipelineSelect = document.getElementById('v4v5-pipeline-select-button');
        const selectedPipeline = pipelineSelect ? pipelineSelect.value : 'normal';
        const isCustom = selectedPipeline === 'custom';
        
        const modalTitle = modal.querySelector('.model-modal-title h2');
        const modalDescription = modal.querySelector('.model-modal-title p');
        
        if (modalTitle) {
            modalTitle.textContent = isCustom ? 'V4V5커스텀 비교' : 'V4V5일반 비교';
        }
        if (modalDescription) {
            modalDescription.textContent = isCustom 
                ? 'CustomV4와 CustomV5 파이프라인을 병렬 실행하여 결과를 비교합니다 (누끼 처리 포함)'
                : 'V4와 V5 파이프라인을 병렬 실행하여 결과를 비교합니다 (누끼 처리 없음)';
        }
        
        modal.classList.add('show');
    }
}

// V4V5 비교 모달 닫기
function closeV4V5CompareModal() {
    const modal = document.getElementById('modal-v4v5-compare');
    if (modal) {
        modal.classList.remove('show');
    }
}

// V4V5 이미지 업로드 처리
function handleV4V5ImageUpload(event, type) {
    const file = event.target.files[0];
    if (!file) return;
    
    if (!file.type.startsWith('image/')) {
        alert('이미지 파일만 업로드 가능합니다.');
        return;
    }
    
    v4v5Images[type] = file;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        const previewContainer = document.getElementById(`preview-v4v5-${type}`);
        const uploadContent = previewContainer.previousElementSibling;
        const img = document.getElementById(`img-v4v5-${type}`);
        
        img.src = e.target.result;
        uploadContent.style.display = 'none';
        previewContainer.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

// V4V5 이미지 제거
function removeV4V5Image(type) {
    v4v5Images[type] = null;
    
    const previewContainer = document.getElementById(`preview-v4v5-${type}`);
    const uploadContent = previewContainer.previousElementSibling;
    const input = document.getElementById(`input-v4v5-${type}`);
    
    previewContainer.style.display = 'none';
    uploadContent.style.display = 'flex';
    input.value = '';
}

// V4V5 드래그 앤 드롭 설정
function setupV4V5DragAndDrop() {
    const types = ['person', 'garment', 'background'];
    
    types.forEach(type => {
        const uploadArea = document.getElementById(`upload-v4v5-${type}`);
        const input = document.getElementById(`input-v4v5-${type}`);
        
        if (!uploadArea || !input) return;
        
        // 기존 이벤트 리스너 제거를 위한 클론 (중복 방지)
        const newUploadArea = uploadArea.cloneNode(true);
        uploadArea.parentNode.replaceChild(newUploadArea, uploadArea);
        const newInput = document.getElementById(`input-v4v5-${type}`);
        
        // 드래그 앤 드롭 이벤트
        newUploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            newUploadArea.classList.add('drag-over');
        });
        
        newUploadArea.addEventListener('dragleave', () => {
            newUploadArea.classList.remove('drag-over');
        });
        
        newUploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            newUploadArea.classList.remove('drag-over');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                if (file.type.startsWith('image/')) {
                    newInput.files = files;
                    handleV4V5ImageUpload({ target: { files: [file] } }, type);
                } else {
                    alert('이미지 파일만 업로드 가능합니다.');
                }
            }
        });
        
        // uploadArea 클릭 시 파일 선택 (단, 버튼 클릭은 제외)
        newUploadArea.addEventListener('click', (e) => {
            // 버튼이나 input 클릭이 아닌 경우에만 파일 선택 창 열기
            if (!e.target.closest('button') && e.target !== newInput && !e.target.closest('.model-upload-btn')) {
                newInput.click();
            }
        });
    });
}

// V4V5 비교 실행
async function runV4V5Compare() {
    const personFile = v4v5Images.person;
    const garmentFile = v4v5Images.garment;
    const backgroundFile = v4v5Images.background;
    
    if (!personFile || !garmentFile || !backgroundFile) {
        alert('인물 이미지, 의상 이미지, 배경 이미지를 모두 업로드해주세요.');
        return;
    }
    
    const loadingDiv = document.getElementById('loading-v4v5');
    const loadingText = document.getElementById('loading-v4v5-text');
    const resultDiv = document.getElementById('result-v4v5');
    const runBtn = document.getElementById('run-btn-v4v5');
    const resultImagesDiv = document.getElementById('result-images-v4v5');
    const timeSpan = document.getElementById('time-v4v5');
    
    // 선택된 파이프라인 확인 (버튼 카드의 드롭다운에서 가져옴)
    const pipelineSelect = document.getElementById('v4v5-pipeline-select-button');
    const selectedPipeline = pipelineSelect ? pipelineSelect.value : 'normal';
    const isCustom = selectedPipeline === 'custom';
    
    // 로딩 텍스트 업데이트
    const pipelineName = isCustom ? 'V4V5커스텀' : 'V4V5일반';
    loadingText.textContent = `${pipelineName} 파이프라인을 병렬 실행 중...`;
    
    loadingDiv.style.display = 'flex';
    resultDiv.style.display = 'none';
    runBtn.disabled = true;
    
    const startTime = performance.now();
    
    try {
        const formData = new FormData();
        formData.append('person_image', personFile);
        formData.append('garment_image', garmentFile);
        formData.append('background_image', backgroundFile);
        
        // 선택된 파이프라인에 따라 다른 API 호출
        const endpoint = isCustom ? '/tryon/compare/custom' : '/tryon/compare';
        
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        const endTime = performance.now();
        const processingTime = ((endTime - startTime) / 1000).toFixed(2);
        
        loadingDiv.style.display = 'none';
        runBtn.disabled = false;
        
        if (data.success) {
            timeSpan.textContent = `${data.total_time || processingTime}초`;
            
            // V4와 V5 결과 표시 (커스텀인 경우 라벨 변경)
            const v4Label = isCustom ? 'CustomV4 결과' : 'V4 결과';
            const v5Label = isCustom ? 'CustomV5 결과' : 'V5 결과';
            
            resultImagesDiv.innerHTML = `
                <div class="model-result-image-item">
                    <div class="model-result-image-label">${v4Label}</div>
                    <img src="${data.v4_result.result_image || ''}" alt="V4 Result" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'400\\' height=\\'300\\'%3E%3Ctext x=\\'50%25\\' y=\\'50%25\\' text-anchor=\\'middle\\'%3E이미지 로드 실패%3C/text%3E%3C/svg%3E'">
                    <div style="margin-top: 10px; padding: 10px; background: #f9fafb; border-radius: 8px;">
                        <div style="font-size: 0.9em; color: #666; margin-bottom: 5px;">
                            <strong>상태:</strong> ${data.v4_result.success ? '✅ 성공' : '❌ 실패'}
                        </div>
                        ${data.v4_result.llm ? `<div style="font-size: 0.85em; color: #666; margin-top: 5px;"><strong>LLM:</strong> ${data.v4_result.llm}</div>` : ''}
                        ${data.v4_result.prompt ? `<div style="font-size: 0.85em; color: #666; margin-top: 5px;"><strong>프롬프트:</strong> ${data.v4_result.prompt.substring(0, 100)}...</div>` : ''}
                        ${data.v4_result.message ? `<div style="font-size: 0.85em; color: #666; margin-top: 5px;">${data.v4_result.message}</div>` : ''}
                    </div>
                </div>
                <div class="model-result-image-item">
                    <div class="model-result-image-label">${v5Label}</div>
                    <img src="${data.v5_result.result_image || ''}" alt="V5 Result" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'400\\' height=\\'300\\'%3E%3Ctext x=\\'50%25\\' y=\\'50%25\\' text-anchor=\\'middle\\'%3E이미지 로드 실패%3C/text%3E%3C/svg%3E'">
                    <div style="margin-top: 10px; padding: 10px; background: #f9fafb; border-radius: 8px;">
                        <div style="font-size: 0.9em; color: #666; margin-bottom: 5px;">
                            <strong>상태:</strong> ${data.v5_result.success ? '✅ 성공' : '❌ 실패'}
                        </div>
                        ${data.v5_result.llm ? `<div style="font-size: 0.85em; color: #666; margin-top: 5px;"><strong>LLM:</strong> ${data.v5_result.llm}</div>` : ''}
                        ${data.v5_result.prompt ? `<div style="font-size: 0.85em; color: #666; margin-top: 5px;"><strong>프롬프트:</strong> ${data.v5_result.prompt.substring(0, 100)}...</div>` : ''}
                        ${data.v5_result.message ? `<div style="font-size: 0.85em; color: #666; margin-top: 5px;">${data.v5_result.message}</div>` : ''}
                    </div>
                </div>
            `;
            
            resultDiv.style.display = 'block';
        } else {
            alert(`오류 발생: ${data.message || '알 수 없는 오류'}`);
        }
    } catch (error) {
        loadingDiv.style.display = 'none';
        runBtn.disabled = false;
        alert(`${pipelineName} 비교 실행 중 오류 발생: ${error.message}`);
    }
}
