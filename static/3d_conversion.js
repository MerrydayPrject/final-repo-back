// DOM 요소
const uploadArea = document.getElementById('uploadArea');
const uploadBtn = document.getElementById('uploadBtn');
const imageInput = document.getElementById('imageInput');
const uploadPlaceholder = document.getElementById('uploadPlaceholder');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const changeImageBtn = document.getElementById('changeImageBtn');

const uploadSection = document.getElementById('uploadSection');
const confirmSection = document.getElementById('confirmSection');
const loadingSection = document.getElementById('loadingSection');
const resultSection = document.getElementById('resultSection');
const errorSection = document.getElementById('errorSection');

const convertBtn = document.getElementById('convertBtn');

const originalImage = document.getElementById('originalImage');
const thumbnailImage = document.getElementById('thumbnailImage');
const downloadGrid = document.getElementById('downloadGrid');

const taskId = document.getElementById('taskId');
const processingTime = document.getElementById('processingTime');
const statusText = document.getElementById('statusText');

const newConversionBtn = document.getElementById('newConversionBtn');
const saveToServerBtn = document.getElementById('saveToServerBtn');
const retryBtn = document.getElementById('retryBtn');
const errorMessage = document.getElementById('errorMessage');
const loadingMessage = document.getElementById('loadingMessage');
const saveMessage = document.getElementById('saveMessage');
const saveDetails = document.getElementById('saveDetails');

// 전역 변수
let selectedFile = null;
let currentTaskId = null;
let statusCheckInterval = null;
let startTime = null;

// 초기화
function init() {
    setupEventListeners();
}

// 이벤트 리스너 설정
function setupEventListeners() {
    uploadBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        imageInput.click();
    });
    
    uploadArea.addEventListener('click', (e) => {
        if (e.target !== uploadBtn && !uploadBtn.contains(e.target)) {
            imageInput.click();
        }
    });
    
    imageInput.addEventListener('change', handleImageSelect);
    changeImageBtn.addEventListener('click', () => {
        imageInput.click();
    });
    
    // 드래그 앤 드롭
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    convertBtn.addEventListener('click', startConversion);
    newConversionBtn.addEventListener('click', resetAll);
    saveToServerBtn.addEventListener('click', saveToServer);
    retryBtn.addEventListener('click', hideError);
}

// 드래그 오버
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

// 드래그 리브
function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

// 드롭
function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
        handleFile(files[0]);
    }
}

// 이미지 선택
function handleImageSelect(e) {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFile(file);
    }
}

// 파일 처리
function handleFile(file) {
    selectedFile = file;
    const reader = new FileReader();
    
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadPlaceholder.style.display = 'none';
        previewContainer.style.display = 'flex';
        
        // 확인 섹션 표시
        confirmSection.style.display = 'block';
    };
    
    reader.readAsDataURL(file);
}

// 변환 시작
async function startConversion() {
    if (!selectedFile) {
        alert('이미지를 먼저 업로드해주세요.');
        return;
    }
    
    showLoading();
    startTime = Date.now();
    
    try {
        const formData = new FormData();
        formData.append('image', selectedFile);
        
        const response = await fetch('/api/convert-to-3d', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentTaskId = data.task_id;
            loadingMessage.textContent = '3D 모델 생성 작업이 시작되었습니다. 상태를 확인하고 있습니다...';
            
            // 상태 폴링 시작 (5초마다)
            statusCheckInterval = setInterval(checkTaskStatus, 5000);
            
            // 즉시 한 번 확인
            checkTaskStatus();
        } else {
            throw new Error(data.error || '작업 시작에 실패했습니다.');
        }
        
    } catch (error) {
        console.error('Error:', error);
        showError(error.message);
    }
}

// 작업 상태 확인
async function checkTaskStatus() {
    if (!currentTaskId) return;
    
    try {
        const response = await fetch(`/api/check-3d-status/${currentTaskId}`);
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.message || data.error || '상태 확인 실패');
        }
        
        const status = data.status;
        const progress = data.progress || 0;
        
        // 상태 메시지 한글화
        let statusText = status;
        if (status === 'PENDING') statusText = '대기 중';
        else if (status === 'IN_PROGRESS') statusText = '생성 중';
        else if (status === 'SUCCEEDED') statusText = '완료';
        else if (status === 'FAILED') statusText = '실패';
        else if (status === 'CANCELED') statusText = '취소됨';
        
        // 로딩 메시지 업데이트
        loadingMessage.textContent = `${statusText}... (진행률: ${progress}%)`;
        
        // 완료된 경우
        if (status === 'SUCCEEDED') {
            clearInterval(statusCheckInterval);
            showResult(data);
        } 
        // 실패한 경우
        else if (status === 'FAILED') {
            clearInterval(statusCheckInterval);
            throw new Error(data.message || '3D 모델 생성에 실패했습니다.');
        }
        // 취소된 경우
        else if (status === 'CANCELED') {
            clearInterval(statusCheckInterval);
            throw new Error('작업이 취소되었습니다.');
        }
        
    } catch (error) {
        clearInterval(statusCheckInterval);
        console.error('Status check error:', error);
        showError(error.message);
    }
}

// 로딩 표시
function showLoading() {
    uploadSection.style.display = 'none';
    confirmSection.style.display = 'none';
    resultSection.style.display = 'none';
    errorSection.style.display = 'none';
    loadingSection.style.display = 'block';
}

// 결과 표시
function showResult(data) {
    loadingSection.style.display = 'none';
    resultSection.style.display = 'block';
    
    // 원본 이미지
    originalImage.src = previewImage.src;
    
    // 썸네일 이미지
    if (data.thumbnail_url) {
        thumbnailImage.src = data.thumbnail_url;
    } else {
        thumbnailImage.src = previewImage.src;
    }
    
    // 작업 정보
    taskId.textContent = currentTaskId || '-';
    const totalTime = ((Date.now() - startTime) / 1000).toFixed(0);
    processingTime.textContent = totalTime + '초';
    statusText.textContent = '완료';
    
    // 다운로드 링크 생성
    const modelUrls = data.model_urls || {};
    downloadGrid.innerHTML = '';
    
    const formats = [
        { key: 'glb', label: 'GLB', icon: '📦', desc: '범용 3D 포맷' },
        { key: 'fbx', label: 'FBX', icon: '🎭', desc: 'Maya, Blender 등' },
        { key: 'usdz', label: 'USDZ', icon: '🍎', desc: 'Apple AR' },
        { key: 'obj', label: 'OBJ', icon: '🔷', desc: '범용 메쉬' }
    ];
    
    formats.forEach(format => {
        const url = modelUrls[format.key];
        if (url) {
            const downloadCard = document.createElement('a');
            downloadCard.href = url;
            downloadCard.target = '_blank';
            downloadCard.className = 'download-card';
            downloadCard.innerHTML = `
                <div class="format-icon">${format.icon}</div>
                <div class="format-name">${format.label}</div>
                <div class="format-desc">${format.desc}</div>
            `;
            downloadGrid.appendChild(downloadCard);
        }
    });
    
    // 다운로드 링크가 없으면 메시지 표시
    if (downloadGrid.children.length === 0) {
        downloadGrid.innerHTML = '<p style="text-align: center; color: #666;">다운로드 가능한 모델이 없습니다.</p>';
    }
}

// 에러 표시
function showError(message) {
    loadingSection.style.display = 'none';
    errorSection.style.display = 'block';
    errorMessage.textContent = message;
}

// 에러 숨기기
function hideError() {
    errorSection.style.display = 'none';
    uploadSection.style.display = 'block';
    confirmSection.style.display = 'block';
}

// 서버에 저장
async function saveToServer() {
    if (!currentTaskId) {
        alert('저장할 작업 ID가 없습니다.');
        return;
    }
    
    try {
        saveToServerBtn.disabled = true;
        saveToServerBtn.textContent = '💾 저장 중...';
        saveMessage.style.display = 'none';
        
        const response = await fetch(`/api/save-3d-model/${currentTaskId}`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            // 저장 성공 메시지 표시
            const savedFiles = data.saved_files || {};
            const fileList = Object.keys(savedFiles).map(format => {
                return `<div>📁 ${format.toUpperCase()}: ${savedFiles[format]}</div>`;
            }).join('');
            
            saveDetails.innerHTML = `
                <div style="margin-bottom: 10px;">${data.message}</div>
                ${fileList}
            `;
            saveMessage.style.display = 'block';
            
            saveToServerBtn.textContent = '✅ 저장 완료';
            
            // 3초 후 버튼 텍스트 복원
            setTimeout(() => {
                saveToServerBtn.textContent = '💾 서버에 저장';
                saveToServerBtn.disabled = false;
            }, 3000);
            
        } else {
            throw new Error(data.error || '저장 실패');
        }
        
    } catch (error) {
        console.error('Save error:', error);
        alert('서버 저장 실패: ' + error.message);
        saveToServerBtn.textContent = '💾 서버에 저장';
        saveToServerBtn.disabled = false;
    }
}

// 전체 리셋
function resetAll() {
    selectedFile = null;
    currentTaskId = null;
    
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
    }
    
    uploadPlaceholder.style.display = 'block';
    previewContainer.style.display = 'none';
    
    uploadSection.style.display = 'block';
    confirmSection.style.display = 'none';
    loadingSection.style.display = 'none';
    resultSection.style.display = 'none';
    errorSection.style.display = 'none';
    saveMessage.style.display = 'none';
    
    imageInput.value = '';
}

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', init);
