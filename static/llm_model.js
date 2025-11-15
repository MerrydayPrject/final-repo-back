// 전역 변수
let uploadedImages = {
    person: null,
    dress: null
};
let models = [];
let selectedModel = null;
let generatedPrompt = null;
let resultImage = null;

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', () => {
    setupDragAndDrop();
    loadModels();
    setupModelSelectListeners();
});

// 모델 선택 변경 이벤트 리스너 설정
function setupModelSelectListeners() {
    const promptLLMSelect = document.getElementById('prompt-llm-select');
    const imageLLMSelect = document.getElementById('image-llm-select');
    
    if (promptLLMSelect) {
        promptLLMSelect.addEventListener('change', () => {
            resetAllImages();
        });
    }
    
    if (imageLLMSelect) {
        imageLLMSelect.addEventListener('change', () => {
            resetAllImages();
        });
    }
}

// 모든 이미지와 결과 초기화
function resetAllImages() {
    // 업로드된 이미지 제거
    ['person', 'dress'].forEach(type => {
        const previewId = `preview-${type}`;
        const uploadAreaId = `upload-${type}`;
        const inputId = `input-${type}`;
        
        const previewElement = document.getElementById(previewId);
        const uploadAreaElement = document.getElementById(uploadAreaId);
        const inputElement = document.getElementById(inputId);
        
        if (previewElement) {
            previewElement.style.display = 'none';
        }
        
        if (uploadAreaElement) {
            const uploadContent = uploadAreaElement.querySelector('.llm-upload-content');
            if (uploadContent) {
                uploadContent.style.display = 'flex';
            }
        }
        
        if (inputElement) {
            inputElement.value = '';
        }
        
        uploadedImages[type] = null;
    });
    
    // 결과 섹션 숨기기
    const resultDiv = document.getElementById('result');
    if (resultDiv) {
        resultDiv.style.display = 'none';
    }
    
    // 전역 변수 초기화
    generatedPrompt = null;
    resultImage = null;
    selectedModel = null;
}

// 모델 목록 로드
async function loadModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        
        if (data.success) {
            // composition 카테고리 및 dual_image 타입 모델만 필터링
            models = data.models.filter(model => 
                model.category === 'composition' && 
                model.input_type === 'dual_image'
            );
        } else {
            console.error('모델 목록 로드 실패:', data.message);
        }
    } catch (error) {
        console.error('모델 로드 오류:', error);
    }
}

// 선택한 LLM 조합에 맞는 모델 찾기
function findModelByLLMSelection(promptLLM, imageLLM) {
    // GPT-4o + Gemini 조합
    if (promptLLM === 'gpt-4o' && imageLLM === 'gemini') {
        return models.find(m => m.id === 'gpt4o-gemini');
    }
    
    // Gemini + Gemini 조합
    if (promptLLM === 'gemini' && imageLLM === 'gemini') {
        return models.find(m => m.id === 'gemini-compose');
    }
    
    // prompt_llm과 일치하는 모델 찾기
    return models.find(model => {
        if (model.prompt_llm) {
            if (promptLLM === 'gpt-4o' && model.prompt_llm === 'gpt-4o') {
                return true;
            }
            if (promptLLM === 'gemini' && model.prompt_llm === 'gemini') {
                return true;
            }
        }
        return false;
    }) || null;
}

// 드래그 앤 드롭 설정
function setupDragAndDrop() {
    ['person', 'dress'].forEach(type => {
        const area = document.getElementById(`upload-${type}`);
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
                const input = document.getElementById(`input-${type}`);
                if (input) {
                    input.files = files;
                    input.dispatchEvent(new Event('change'));
                }
            }
        });
    });
}

// 이미지 업로드 처리
function handleImageUpload(event, type) {
    const file = event.target.files[0];
    if (!file) {
        console.warn('파일이 선택되지 않았습니다.');
        return;
    }
    
    if (!file.type.startsWith('image/')) {
        alert('이미지 파일만 업로드 가능합니다.');
        return;
    }
    
    uploadedImages[type] = file;
    console.log(`이미지 업로드 완료: ${type}`, file.name, file.size);
    
    const reader = new FileReader();
    reader.onload = (e) => {
        const previewId = `preview-${type}`;
        const imgId = `img-${type}`;
        const uploadAreaId = `upload-${type}`;
        
        const previewElement = document.getElementById(previewId);
        const imgElement = document.getElementById(imgId);
        const uploadAreaElement = document.getElementById(uploadAreaId);
        
        if (imgElement && previewElement && uploadAreaElement) {
            imgElement.src = e.target.result;
            previewElement.style.display = 'block';
            const uploadContent = uploadAreaElement.querySelector('.llm-upload-content');
            if (uploadContent) {
                uploadContent.style.display = 'none';
            }
        }
    };
    reader.onerror = (error) => {
        console.error('파일 읽기 오류:', error);
        alert('이미지 파일을 읽는 중 오류가 발생했습니다.');
        delete uploadedImages[type];
    };
    reader.readAsDataURL(file);
}

// 이미지 제거
function removeImage(type) {
    const previewId = `preview-${type}`;
    const uploadAreaId = `upload-${type}`;
    const inputId = `input-${type}`;
    
    document.getElementById(previewId).style.display = 'none';
    document.querySelector(`#${uploadAreaId} .llm-upload-content`).style.display = 'flex';
    document.getElementById(inputId).value = '';
    document.getElementById('result').style.display = 'none';
    
    uploadedImages[type] = null;
}

// LLM 테스트 실행
async function runLLMTest() {
    // 이미지 검증
    if (!uploadedImages.person || !uploadedImages.dress) {
        alert('사람 이미지와 드레스 이미지를 모두 업로드해주세요.');
        return;
    }
    
    if (!(uploadedImages.person instanceof File) || !(uploadedImages.dress instanceof File)) {
        alert('이미지 파일이 올바르지 않습니다. 다시 업로드해주세요.');
        return;
    }
    
    // 선택한 LLM 조합에 맞는 모델 찾기
    const promptLLM = document.getElementById('prompt-llm-select').value;
    const imageLLM = document.getElementById('image-llm-select').value;
    
    // 이미지 생성용 x.ai 선택 시 별도 처리
    if (imageLLM === 'xai') {
        await runXAIImageGeneration();
        return;
    }
    
    // 프롬프트 생성용 x.ai 선택 시 프롬프트 생성 플로우 실행
    if (promptLLM === 'xai') {
        // 사용자가 선택한 이미지 생성용 모델 찾기
        // imageLLM에 맞는 모델 찾기 (promptLLM은 무시하고 imageLLM만 사용)
        if (imageLLM === 'gemini') {
            selectedModel = models.find(m => m.id === 'gemini-compose');
        } else {
            // 다른 이미지 생성 모델의 경우 해당 모델 찾기
            selectedModel = models.find(model => {
                // imageLLM과 일치하는 모델 찾기
                return model.input_type === 'dual_image' && 
                       model.category === 'composition' &&
                       (!model.image_llm || model.image_llm === imageLLM);
            });
        }
        // 모델을 찾지 못하면 경고
        if (!selectedModel) {
            alert(`선택하신 이미지 생성 모델(${imageLLM})에 맞는 모델을 찾을 수 없습니다.`);
            return;
        }
        await runPromptGenerationFlow();
        return;
    }
    
    selectedModel = findModelByLLMSelection(promptLLM, imageLLM);
    
    if (!selectedModel) {
        alert(`선택하신 조합(${promptLLM} + ${imageLLM})에 맞는 모델을 찾을 수 없습니다.`);
        return;
    }
    
    // gemini-compose와 gpt4o-gemini 모델은 강제로 프롬프트 생성 플로우 실행 (model-comparison.js와 동일)
    if ((selectedModel.id === 'gemini-compose' || selectedModel.id === 'gpt4o-gemini') && selectedModel.input_type === 'dual_image') {
        await runPromptGenerationFlow();
        return;
    }
    
    // 모델에 따라 처리 분기
    const needsPromptGeneration = selectedModel.prompt_generation_endpoint || 
                                  selectedModel.requires_prompt_generation ||
                                  (selectedModel.prompt_llm && !selectedModel.prompt);
    
    if (needsPromptGeneration) {
        // 프롬프트 생성이 필요한 경우 (model-comparison.js의 runGeminiComposeWithPromptCheck와 동일)
        await runPromptGenerationFlow();
    } else if (selectedModel.prompt) {
        // 고정 프롬프트가 있는 경우 바로 합성
        generatedPrompt = selectedModel.prompt;
        await runComposeDirectly();
    } else {
        // 프롬프트 없이 합성
        await runComposeDirectly();
    }
}

// 프롬프트 생성 플로우 실행 (model-comparison.js의 runGeminiComposeWithPromptCheck와 동일)
async function runPromptGenerationFlow() {
    const loadingDiv = document.getElementById('loading');
    const resultDiv = document.getElementById('result');
    const runBtn = document.getElementById('run-btn');
    const loadingText = document.getElementById('loading-text');
    
    try {
        loadingDiv.style.display = 'flex';
        resultDiv.style.display = 'none';
        runBtn.disabled = true;
        loadingText.textContent = '프롬프트 생성 중...';
        
        // 1. 프롬프트 생성 API 호출 (model-comparison.js와 동일한 방식)
        const formData = new FormData();
        formData.append('person_image', uploadedImages.person);
        formData.append('dress_image', uploadedImages.dress);
        
        // 프롬프트 LLM 선택 확인
        const promptLLMSelect = document.getElementById('prompt-llm-select');
        const promptLLM = promptLLMSelect ? promptLLMSelect.value : '';
        
        // GPT-4o → Gemini 2.5 Flash V1 합성의 경우 GPT-4o로 프롬프트 생성
        const modelPromptLLM = selectedModel ? (selectedModel.prompt_llm || (selectedModel.id === 'gpt4o-gemini' ? 'gpt-4o' : '')) : '';
        const finalPromptLLM = promptLLM || modelPromptLLM;
        
        // 프롬프트 엔드포인트 결정
        let promptEndpoint = '/api/gemini/generate-prompt';
        if (finalPromptLLM === 'gpt-4o-v2-short') {
            promptEndpoint = '/api/prompt/generate-short';
        } else if (finalPromptLLM === 'xai') {
            promptEndpoint = '/api/xai/generate-prompt';
        } else if (finalPromptLLM === 'gpt-4o') {
            promptEndpoint = '/api/gpt4o-gemini/generate-prompt';
        } else if (selectedModel && selectedModel.prompt_generation_endpoint) {
            promptEndpoint = selectedModel.prompt_generation_endpoint;
        }
        
        // prompt_llm 파라미터는 x.ai와 gpt-4o-v2-short가 아닌 경우에만 추가
        if (finalPromptLLM && finalPromptLLM !== 'xai' && finalPromptLLM !== 'gpt-4o-v2-short') {
            formData.append('prompt_llm', finalPromptLLM);
        }
        
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
        
        if (data.success) {
            generatedPrompt = data.prompt;
            const llmName = data.llm || data.model || data.provider || promptLLM || '알 수 없음';
            // 2. 프롬프트 확인 모달 표시
            showPromptConfirmModal(generatedPrompt, llmName);
        } else {
            throw new Error(data.message || '프롬프트 생성에 실패했습니다');
        }
    } catch (error) {
        console.error('프롬프트 생성 오류:', error);
        alert(`프롬프트 생성 실패: ${error.message}`);
        
        loadingDiv.style.display = 'none';
        runBtn.disabled = false;
    }
}

// 프롬프트 없이 직접 합성 실행 (model-comparison.js의 confirmAndRunCompose와 유사)
async function runComposeDirectly() {
    const loadingDiv = document.getElementById('loading');
    const resultDiv = document.getElementById('result');
    const runBtn = document.getElementById('run-btn');
    const loadingText = document.getElementById('loading-text');
    
    try {
        loadingDiv.style.display = 'flex';
        resultDiv.style.display = 'none';
        runBtn.disabled = true;
        loadingText.textContent = '이미지 합성 중...';
        
        const formData = new FormData();
        formData.append('person_image', uploadedImages.person);
        formData.append('dress_image', uploadedImages.dress);
        formData.append('model_name', selectedModel.id);
        
        if (generatedPrompt || selectedModel.prompt) {
            formData.append('prompt', generatedPrompt || selectedModel.prompt);
        }
        
        const startTime = performance.now();
        const response = await fetch(selectedModel.endpoint, {
            method: selectedModel.method || 'POST',
            body: formData
        });
        
        const processingTime = ((performance.now() - startTime) / 1000).toFixed(2);
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.message || `HTTP ${response.status}`);
        }
        
        const data = await response.json();
        
        loadingDiv.style.display = 'none';
        runBtn.disabled = false;
        
        if (data.success) {
            displayResult(data, processingTime);
        } else {
            throw new Error(data.message || '이미지 합성 실패');
        }
    } catch (error) {
        console.error('이미지 합성 오류:', error);
        alert(`이미지 합성 실패: ${error.message}`);
        
        loadingDiv.style.display = 'none';
        runBtn.disabled = false;
    }
}

// 프롬프트 확인 모달 표시
function showPromptConfirmModal(generatedPrompt, llmName = '알 수 없음') {
    // HTML escape 함수
    const escapeHtml = (text) => {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    };
    
    const modal = document.createElement('div');
    modal.className = 'prompt-confirm-modal';
    modal.id = 'prompt-modal-llm';
    modal.innerHTML = `
        <div class="prompt-confirm-overlay"></div>
        <div class="prompt-confirm-content">
            <div class="prompt-confirm-header">
                <h3>✨ AI가 생성한 프롬프트</h3>
                <button class="prompt-close-button" onclick="closePromptConfirmModal()">×</button>
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
                        이 프롬프트를 사용하여 이미지 합성을 진행하시겠습니까?
                    </p>
                    <div class="button-group">
                        <button class="btn-secondary" onclick="closePromptConfirmModal()">취소</button>
                        <button class="btn-primary" onclick="confirmAndRunCompose()">확인 및 합성 시작</button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // 기존 모달이 있으면 제거
    const existingModal = document.getElementById('prompt-modal-llm');
    if (existingModal) {
        existingModal.remove();
    }
    
    document.body.appendChild(modal);
    
    // 모달 스타일 추가
    ensurePromptModalStyles();
    
    // 오버레이 클릭 시 닫기
    modal.querySelector('.prompt-confirm-overlay').addEventListener('click', closePromptConfirmModal);
}

// 프롬프트 모달 스타일 추가
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
            margin-bottom: 20px;
            padding: 15px;
            background: #e8f4f8;
            border-radius: 8px;
            color: #0277bd;
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

// 프롬프트 확인 모달 닫기
function closePromptConfirmModal() {
    const modals = document.querySelectorAll('.prompt-confirm-modal');
    modals.forEach(modal => modal.remove());
}

// x.ai 이미지 생성 실행
async function runXAIImageGeneration() {
    const loadingDiv = document.getElementById('loading');
    const resultDiv = document.getElementById('result');
    const runBtn = document.getElementById('run-btn');
    const loadingText = document.getElementById('loading-text');
    
    try {
        // 이미지 검증 (x.ai는 텍스트만 사용하지만, 프롬프트 생성을 위해 이미지 필요)
        if (!uploadedImages.person || !uploadedImages.dress) {
            alert('프롬프트 생성을 위해 사람 이미지와 드레스 이미지를 모두 업로드해주세요.');
            return;
        }
        
        loadingDiv.style.display = 'flex';
        resultDiv.style.display = 'none';
        runBtn.disabled = true;
        loadingText.textContent = '프롬프트 생성 중...';
        
        // 1. 프롬프트 생성
        const promptLLM = document.getElementById('prompt-llm-select').value;
        const formData = new FormData();
        formData.append('person_image', uploadedImages.person);
        formData.append('dress_image', uploadedImages.dress);
        
        // GPT-4o-V2 short prompt 선택 시 새로운 엔드포인트 사용
        let promptEndpoint = '/api/gemini/generate-prompt';
        if (promptLLM === 'gpt-4o-v2-short') {
            promptEndpoint = '/api/prompt/generate-short';
        } else if (promptLLM) {
            formData.append('prompt_llm', promptLLM);
        }
        
        const promptResponse = await fetch(promptEndpoint, {
            method: 'POST',
            body: formData
        });
        
        if (!promptResponse.ok) {
            const errorData = await promptResponse.json().catch(() => ({}));
            throw new Error(errorData.message || `프롬프트 생성 실패: ${promptResponse.status}`);
        }
        
        const promptData = await promptResponse.json();
        
        if (!promptData.success || !promptData.prompt) {
            throw new Error(promptData.message || '프롬프트 생성에 실패했습니다');
        }
        
        generatedPrompt = promptData.prompt;
        loadingText.textContent = 'x.ai로 이미지 생성 중...';
        
        // 2. x.ai API 호출 (로깅을 위해 이미지와 프롬프트 LLM 정보도 함께 전달)
        const xaiFormData = new FormData();
        xaiFormData.append('prompt', generatedPrompt);
        xaiFormData.append('person_image', uploadedImages.person);
        xaiFormData.append('dress_image', uploadedImages.dress);
        if (promptLLM) {
            xaiFormData.append('prompt_llm', promptLLM);
        }
        
        const startTime = performance.now();
        const xaiResponse = await fetch('/api/generate-image-xai', {
            method: 'POST',
            body: xaiFormData
        });
        
        const processingTime = ((performance.now() - startTime) / 1000).toFixed(2);
        
        if (!xaiResponse.ok) {
            const errorData = await xaiResponse.json().catch(() => ({}));
            throw new Error(errorData.message || `x.ai API 호출 실패: ${xaiResponse.status}`);
        }
        
        const xaiData = await xaiResponse.json();
        
        loadingDiv.style.display = 'none';
        runBtn.disabled = false;
        
        if (xaiData.success) {
            // 결과 표시 (x.ai는 단일 이미지만 반환)
            displayXAIResult(xaiData, processingTime);
        } else {
            throw new Error(xaiData.message || 'x.ai 이미지 생성 실패');
        }
    } catch (error) {
        console.error('x.ai 이미지 생성 오류:', error);
        alert(`x.ai 이미지 생성 실패: ${error.message}`);
        
        loadingDiv.style.display = 'none';
        runBtn.disabled = false;
    }
}

// x.ai 결과 표시
function displayXAIResult(data, processingTime) {
    const resultDiv = document.getElementById('result');
    const resultImagesDiv = document.getElementById('result-images');
    const timeSpan = document.getElementById('processing-time');
    const downloadBtn = document.getElementById('download-btn');
    
    timeSpan.textContent = `${processingTime}초`;
    
    // 업로드된 이미지의 미리보기 URL 가져오기
    const personImgSrc = document.getElementById('img-person')?.src || '';
    const dressImgSrc = document.getElementById('img-dress')?.src || '';
    
    const imagesHtml = `
        <div class="llm-result-image-item">
            <div class="llm-result-image-label">사람 이미지</div>
            <img src="${personImgSrc}" alt="Person">
        </div>
        <div class="llm-result-image-item">
            <div class="llm-result-image-label">드레스 이미지</div>
            <img src="${dressImgSrc}" alt="Dress">
        </div>
        <div class="llm-result-image-item highlight">
            <div class="llm-result-image-label">x.ai 생성 결과 ✨</div>
            <img src="${data.result_image || ''}" alt="Result" id="result-img">
        </div>
    `;
    
    resultImage = data.result_image;
    
    resultImagesDiv.innerHTML = imagesHtml;
    resultDiv.style.display = 'block';
    
    if (data.result_image) {
        downloadBtn.style.display = 'flex';
    }
}

// 프롬프트 확인 후 이미지 합성 실행 (model-comparison.js와 동일한 방식)
async function confirmAndRunCompose() {
    closePromptConfirmModal();
    
    if (!generatedPrompt) {
        alert('프롬프트를 찾을 수 없습니다. 다시 시도해주세요.');
        return;
    }
    
    // selectedModel이 없으면 사용자가 선택한 이미지 생성용 모델 찾기
    if (!selectedModel) {
        const imageLLM = document.getElementById('image-llm-select')?.value || 'gemini';
        // imageLLM에 맞는 모델 찾기
        if (imageLLM === 'gemini') {
            selectedModel = models.find(m => m.id === 'gemini-compose');
        } else {
            // 다른 이미지 생성 모델의 경우 해당 모델 찾기
            selectedModel = models.find(model => {
                // imageLLM과 일치하는 모델 찾기
                return model.input_type === 'dual_image' && 
                       model.category === 'composition' &&
                       (!model.image_llm || model.image_llm === imageLLM);
            });
        }
        if (!selectedModel) {
            alert(`선택하신 이미지 생성 모델(${imageLLM})에 맞는 모델을 찾을 수 없습니다. 이미지 생성 모델을 선택해주세요.`);
            return;
        }
    }
    
    await runComposeDirectly();
}

// 결과 표시
function displayResult(data, processingTime) {
    const resultDiv = document.getElementById('result');
    const resultImagesDiv = document.getElementById('result-images');
    const timeSpan = document.getElementById('processing-time');
    const downloadBtn = document.getElementById('download-btn');
    
    timeSpan.textContent = `${processingTime}초`;
    
    const imagesHtml = `
        <div class="llm-result-image-item">
            <div class="llm-result-image-label">사람 이미지</div>
            <img src="${data.person_image || ''}" alt="Person">
        </div>
        <div class="llm-result-image-item">
            <div class="llm-result-image-label">드레스 이미지</div>
            <img src="${data.dress_image || ''}" alt="Dress">
        </div>
        <div class="llm-result-image-item highlight">
            <div class="llm-result-image-label">합성 결과 ✨</div>
            <img src="${data.result_image || ''}" alt="Result" id="result-img">
        </div>
    `;
    
    resultImage = data.result_image;
    
    resultImagesDiv.innerHTML = imagesHtml;
    resultDiv.style.display = 'block';
    
    if (data.result_image) {
        downloadBtn.style.display = 'flex';
    }
}

// 결과 다운로드
function downloadResult() {
    if (!resultImage) {
        alert('다운로드할 결과 이미지가 없습니다.');
        return;
    }
    
    const link = document.createElement('a');
    link.href = resultImage;
    link.download = `llm-result-${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

