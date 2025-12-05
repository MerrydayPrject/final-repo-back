/**
 * 누끼V2 - Ghost Mannequin
 * Gemini3 + GPT-image-1 동시 처리
 */

// 전역 변수
let uploadedFile = null;
let geminiResultImage = null;
let openaiResultImage = null;

// DOM 요소
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const processButton = document.getElementById('processButton');
const loadingSection = document.getElementById('loadingSection');
const resultSection = document.getElementById('resultSection');

// 파일 입력 이벤트
fileInput.addEventListener('change', handleFileSelect);

// 드래그 앤 드랍 이벤트
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
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

/**
 * 파일 선택 핸들러
 */
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

/**
 * 파일 처리
 */
function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('이미지 파일만 업로드할 수 있습니다.');
        return;
    }
    
    uploadedFile = file;
    
    // 미리보기 표시
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewSection.style.display = 'block';
        processButton.style.display = 'flex';
        uploadArea.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

/**
 * 이미지 제거
 */
function removeImage() {
    uploadedFile = null;
    fileInput.value = '';
    previewSection.style.display = 'none';
    processButton.style.display = 'none';
    uploadArea.style.display = 'flex';
}

/**
 * 이미지 처리 시작
 */
async function processImage() {
    if (!uploadedFile) {
        alert('이미지를 먼저 업로드하세요.');
        return;
    }
    
    // UI 상태 변경
    processButton.disabled = true;
    processButton.querySelector('.button-text').textContent = '처리 중...';
    loadingSection.style.display = 'block';
    resultSection.style.display = 'none';
    
    // 상태 초기화
    updateModelStatus('gemini', 'processing', '처리 중');
    updateModelStatus('openai', 'processing', '처리 중');
    
    try {
        const formData = new FormData();
        formData.append('file', uploadedFile);
        
        const response = await fetch('/api/nukki-v2/process', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        // 로딩 숨기기
        loadingSection.style.display = 'none';
        
        // 결과 표시
        displayResults(result);
        
    } catch (error) {
        console.error('처리 오류:', error);
        loadingSection.style.display = 'none';
        
        // 에러 결과 표시
        displayResults({
            success: false,
            gemini3: {
                success: false,
                error: error.message,
                message: '네트워크 오류'
            },
            openai: {
                success: false,
                error: error.message,
                message: '네트워크 오류'
            },
            message: '처리 중 오류가 발생했습니다.'
        });
    } finally {
        processButton.disabled = false;
        processButton.querySelector('.button-text').textContent = 'Ghost Mannequin 생성';
    }
}

/**
 * 결과 표시
 */
function displayResults(result) {
    resultSection.style.display = 'block';
    
    // 전체 결과 정보
    document.getElementById('resultInfo').textContent = result.message || '';
    
    // Gemini3 결과
    const gemini = result.gemini3 || {};
    if (gemini.success && gemini.result_image) {
        geminiResultImage = gemini.result_image;
        updateModelStatus('gemini', 'success', '완료');
        
        document.getElementById('geminiImageWrapper').innerHTML = 
            `<img src="${gemini.result_image}" alt="Gemini3 결과">`;
        
        document.getElementById('geminiTime').textContent = 
            `처리 시간: ${(gemini.run_time || 0).toFixed(2)}초`;
        
        document.getElementById('geminiFooter').style.display = 'flex';
        document.getElementById('geminiError').style.display = 'none';
    } else {
        updateModelStatus('gemini', 'error', '실패');
        
        document.getElementById('geminiImageWrapper').innerHTML = 
            `<div class="placeholder"><span>처리 실패</span></div>`;
        
        document.getElementById('geminiFooter').style.display = 'none';
        document.getElementById('geminiError').style.display = 'block';
        document.getElementById('geminiErrorMsg').textContent = 
            gemini.message || gemini.error || '알 수 없는 오류';
    }
    
    // OpenAI 결과
    const openai = result.openai || {};
    if (openai.success && openai.result_image) {
        openaiResultImage = openai.result_image;
        updateModelStatus('openai', 'success', '완료');
        
        document.getElementById('openaiImageWrapper').innerHTML = 
            `<img src="${openai.result_image}" alt="GPT-image-1 결과">`;
        
        document.getElementById('openaiTime').textContent = 
            `처리 시간: ${(openai.run_time || 0).toFixed(2)}초`;
        
        document.getElementById('openaiFooter').style.display = 'flex';
        document.getElementById('openaiError').style.display = 'none';
    } else {
        updateModelStatus('openai', 'error', '실패');
        
        document.getElementById('openaiImageWrapper').innerHTML = 
            `<div class="placeholder"><span>처리 실패</span></div>`;
        
        document.getElementById('openaiFooter').style.display = 'none';
        document.getElementById('openaiError').style.display = 'block';
        document.getElementById('openaiErrorMsg').textContent = 
            openai.message || openai.error || '알 수 없는 오류';
    }
}

/**
 * 모델 상태 업데이트
 */
function updateModelStatus(model, status, text) {
    const statusElement = document.getElementById(`${model}Status`);
    statusElement.className = `status-badge ${status}`;
    statusElement.textContent = text;
    
    // 로딩 상태도 업데이트
    const loadingElement = document.getElementById(`${model}Loading`);
    if (loadingElement) {
        if (status === 'processing') {
            loadingElement.style.opacity = '1';
        } else {
            loadingElement.style.opacity = '0.5';
            loadingElement.querySelector('span').textContent = 
                status === 'success' ? '완료!' : '실패';
        }
    }
}

/**
 * 이미지 다운로드
 */
function downloadImage(model) {
    let imageData, filename;
    
    if (model === 'gemini') {
        imageData = geminiResultImage;
        filename = 'ghost_mannequin_gemini3.png';
    } else {
        imageData = openaiResultImage;
        filename = 'ghost_mannequin_dalle3.png';
    }
    
    if (!imageData) {
        alert('다운로드할 이미지가 없습니다.');
        return;
    }
    
    // base64 데이터에서 다운로드
    const link = document.createElement('a');
    link.href = imageData;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

/**
 * 모델 재시도
 */
async function retryModel(model) {
    if (!uploadedFile) {
        alert('이미지를 먼저 업로드하세요.');
        return;
    }
    
    // 해당 모델 상태 업데이트
    const modelKey = model === 'gemini3' ? 'gemini' : 'openai';
    updateModelStatus(modelKey, 'processing', '재시도 중');
    
    // 에러 섹션 숨기기
    document.getElementById(`${modelKey}Error`).style.display = 'none';
    document.getElementById(`${modelKey}ImageWrapper`).innerHTML = 
        `<div class="model-loading"><div class="spinner"></div><span>재시도 중...</span></div>`;
    
    try {
        const formData = new FormData();
        formData.append('file', uploadedFile);
        
        const response = await fetch(`/api/nukki-v2/process-single/${model}`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success && result.result_image) {
            if (modelKey === 'gemini') {
                geminiResultImage = result.result_image;
            } else {
                openaiResultImage = result.result_image;
            }
            
            updateModelStatus(modelKey, 'success', '완료');
            
            document.getElementById(`${modelKey}ImageWrapper`).innerHTML = 
                `<img src="${result.result_image}" alt="${model} 결과">`;
            
            document.getElementById(`${modelKey}Time`).textContent = 
                `처리 시간: ${(result.run_time || 0).toFixed(2)}초`;
            
            document.getElementById(`${modelKey}Footer`).style.display = 'flex';
        } else {
            updateModelStatus(modelKey, 'error', '실패');
            
            document.getElementById(`${modelKey}ImageWrapper`).innerHTML = 
                `<div class="placeholder"><span>처리 실패</span></div>`;
            
            document.getElementById(`${modelKey}Error`).style.display = 'block';
            document.getElementById(`${modelKey}ErrorMsg`).textContent = 
                result.message || result.error || '알 수 없는 오류';
        }
        
    } catch (error) {
        console.error('재시도 오류:', error);
        
        updateModelStatus(modelKey, 'error', '실패');
        
        document.getElementById(`${modelKey}ImageWrapper`).innerHTML = 
            `<div class="placeholder"><span>처리 실패</span></div>`;
        
        document.getElementById(`${modelKey}Error`).style.display = 'block';
        document.getElementById(`${modelKey}ErrorMsg`).textContent = 
            '네트워크 오류: ' + error.message;
    }
}

/**
 * 전체 초기화
 */
function resetAll() {
    // 파일 초기화
    uploadedFile = null;
    fileInput.value = '';
    geminiResultImage = null;
    openaiResultImage = null;
    
    // UI 초기화
    previewSection.style.display = 'none';
    processButton.style.display = 'none';
    uploadArea.style.display = 'flex';
    loadingSection.style.display = 'none';
    resultSection.style.display = 'none';
    
    // 결과 카드 초기화
    ['gemini', 'openai'].forEach(model => {
        document.getElementById(`${model}Status`).className = 'status-badge waiting';
        document.getElementById(`${model}Status`).textContent = '대기';
        document.getElementById(`${model}ImageWrapper`).innerHTML = 
            `<div class="placeholder"><span>결과 대기 중</span></div>`;
        document.getElementById(`${model}Footer`).style.display = 'none';
        document.getElementById(`${model}Error`).style.display = 'none';
    });
}

