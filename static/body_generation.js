// 페이스스왑 페이지 JavaScript

const API_BASE_URL = window.location.origin;

// DOM 요소
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const uploadContent = document.getElementById('uploadContent');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const removeButton = document.getElementById('removeButton');
const generateButton = document.getElementById('generateButton');
const resultContent = document.getElementById('resultContent');
const loadingContainer = document.getElementById('loadingContainer');
const templateSelector = document.getElementById('templateSelector');
const templateSelect = document.getElementById('templateSelect');

let selectedFile = null;
let templates = [];

// 페이지 로드 시 템플릿 목록 가져오기
window.addEventListener('DOMContentLoaded', async () => {
    await loadTemplates();
});

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

// 템플릿 목록 로드
async function loadTemplates() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/body-generation/templates`);
        const data = await response.json();
        
        if (data.success && data.templates.length > 0) {
            templates = data.templates;
            templateSelect.innerHTML = '';
            
            templates.forEach((template, index) => {
                const option = document.createElement('option');
                option.value = template.name;
                option.textContent = `템플릿 ${index + 1}: ${template.name}`;
                if (index === 0) option.selected = true; // 첫 번째 템플릿 기본 선택
                templateSelect.appendChild(option);
            });
            
            templateSelector.style.display = 'block';
        } else {
            templateSelector.style.display = 'none';
        }
    } catch (error) {
        console.error('템플릿 목록 로드 실패:', error);
        templateSelector.style.display = 'none';
    }
}

// 파일 처리
function handleFile(file) {
    selectedFile = file;
    
    const reader = new FileReader();
    reader.onloadend = () => {
        previewImage.src = reader.result;
        uploadContent.style.display = 'none';
        previewContainer.style.display = 'block';
        generateButton.disabled = false;
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
    generateButton.disabled = true;
    clearResults();
});

// 생성 버튼 클릭
generateButton.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    await performFaceSwap(selectedFile);
});

// 페이스스왑 API 호출
async function performFaceSwap(file) {
    try {
        showLoading();
        clearResults();
        
        const formData = new FormData();
        formData.append('file', file);
        
        // 선택된 템플릿 이름 추가
        const selectedTemplate = templateSelect.value;
        if (selectedTemplate) {
            formData.append('template_name', selectedTemplate);
        }
        
        const response = await fetch(`${API_BASE_URL}/api/body-generation`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        hideLoading();
        
        if (data.success) {
            displayResults(data);
        } else {
            showError(data.message || '페이스스왑에 실패했습니다.');
        }
    } catch (error) {
        hideLoading();
        console.error('페이스스왑 오류:', error);
        showError('페이스스왑 중 오류가 발생했습니다: ' + error.message);
    }
}

// 결과 표시
function displayResults(data) {
    resultContent.innerHTML = `
        <div class="result-images">
            <div class="image-item-single">
                <h4>페이스스왑 결과</h4>
                <img src="${data.result_image}" alt="페이스스왑 결과" class="result-image">
            </div>
            <div class="result-info">
                <p class="success-message">✅ ${data.message}</p>
                ${data.template_name ? `<p class="template-info">템플릿: ${data.template_name}</p>` : ''}
                ${data.run_time ? `<p class="time-info">처리 시간: ${data.run_time}초</p>` : ''}
            </div>
        </div>
    `;
}

// 로딩 표시
function showLoading() {
    loadingContainer.style.display = 'block';
    resultContent.style.display = 'none';
    generateButton.disabled = true;
}

// 로딩 숨김
function hideLoading() {
    loadingContainer.style.display = 'none';
    resultContent.style.display = 'block';
    generateButton.disabled = false;
}

// 에러 표시
function showError(message) {
    resultContent.innerHTML = `
        <div class="error-container">
            <div class="error-icon">❌</div>
            <p class="error-message">${message}</p>
        </div>
    `;
}

// 결과 초기화
function clearResults() {
    resultContent.innerHTML = `
        <div class="result-placeholder">
            <div class="placeholder-icon">🖼️</div>
            <p class="placeholder-text">이미지를 업로드하고 생성 버튼을 클릭하세요</p>
        </div>
    `;
}

