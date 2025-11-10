// 이미지 보정 테스트 페이지 JavaScript

const API_BASE_URL = window.location.origin;

// DOM 요소
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const uploadContent = document.getElementById('uploadContent');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const removeButton = document.getElementById('removeButton');
const instructionInput = document.getElementById('instructionInput');
const enhanceButton = document.getElementById('enhanceButton');
const resultContent = document.getElementById('resultContent');
const loadingContainer = document.getElementById('loadingContainer');

let selectedFile = null;

// 파일 입력 변경
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFile(file);
    }
});

// 드래그 앤 드롭
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.add('dragging');
});

uploadArea.addEventListener('dragleave', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('dragging');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('dragging');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFile(file);
    }
});

// 업로드 영역 클릭
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// 파일 처리
function handleFile(file) {
    selectedFile = file;
    
    const reader = new FileReader();
    reader.onloadend = () => {
        previewImage.src = reader.result;
        uploadContent.style.display = 'none';
        previewContainer.style.display = 'block';
        updateEnhanceButton();
    };
    reader.readAsDataURL(file);
}

// 이미지 제거
removeButton.addEventListener('click', (e) => {
    e.stopPropagation();
    selectedFile = null;
    previewImage.src = '';
    uploadContent.style.display = 'flex';
    previewContainer.style.display = 'none';
    fileInput.value = '';
    instructionInput.value = '';
    enhanceButton.disabled = true;
    clearResults();
});

// 요청사항 입력 변경
instructionInput.addEventListener('input', () => {
    updateEnhanceButton();
});

// 보정 버튼 활성화 업데이트
function updateEnhanceButton() {
    enhanceButton.disabled = !(selectedFile && instructionInput.value.trim());
}

// 보정 버튼 클릭
enhanceButton.addEventListener('click', async () => {
    if (!selectedFile || !instructionInput.value.trim()) return;
    
    await enhanceImage(selectedFile, instructionInput.value.trim());
});

// 이미지 보정 API 호출
async function enhanceImage(file, instruction) {
    try {
        // 로딩 표시
        showLoading();
        enhanceButton.disabled = true;
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('instruction', instruction);
        formData.append('num_inference_steps', 20);
        formData.append('image_guidance_scale', 1.5);
        formData.append('use_gfpgan', true);
        formData.append('gfpgan_weight', 0.5);
        
        const response = await fetch(`${API_BASE_URL}/api/enhance-image`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResult(data);
        } else {
            showError(data.message || '이미지 보정에 실패했습니다.');
        }
        
    } catch (error) {
        console.error('보정 오류:', error);
        showError('서버 연결 오류가 발생했습니다.');
    } finally {
        hideLoading();
        enhanceButton.disabled = false;
        updateEnhanceButton();
    }
}

// 결과 표시
function displayResult(data) {
    const { result_image, prompt_used } = data;
    
    let html = `
        <div class="result-image-container">
            <img src="${result_image}" alt="보정된 이미지" id="resultImage">
            <button class="download-button" onclick="downloadImage('${result_image}')">이미지 다운로드</button>
        </div>
    `;
    
    if (prompt_used) {
        html += `
            <div class="result-card">
                <div class="result-item">
                    <div class="result-label">사용된 프롬프트</div>
                    <div class="result-value" style="font-size: 13px; color: #666;">${prompt_used}</div>
                </div>
            </div>
        `;
    }
    
    resultContent.innerHTML = html;
}

// 이미지 다운로드
function downloadImage(base64Data) {
    const link = document.createElement('a');
    link.href = base64Data;
    link.download = `enhanced_image_${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// 로딩 표시
function showLoading() {
    resultContent.style.display = 'none';
    loadingContainer.style.display = 'flex';
}

function hideLoading() {
    loadingContainer.style.display = 'none';
    resultContent.style.display = 'block';
}

// 에러 표시
function showError(message) {
    resultContent.innerHTML = `
        <div class="result-placeholder">
            <div class="placeholder-icon">❌</div>
            <p class="placeholder-text">${message}</p>
        </div>
    `;
}

// 결과 초기화
function clearResults() {
    resultContent.innerHTML = `
        <div class="result-placeholder">
            <div class="placeholder-icon">✨</div>
            <p class="placeholder-text">이미지를 업로드하고 보정 요청사항을 입력한 후 보정 버튼을 클릭하세요</p>
        </div>
    `;
}




