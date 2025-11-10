// DOM 요소
const uploadArea = document.getElementById('uploadArea');
const uploadBtn = document.getElementById('uploadBtn');
const imageInput = document.getElementById('imageInput');
const uploadPlaceholder = document.getElementById('uploadPlaceholder');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const changeImageBtn = document.getElementById('changeImageBtn');

const uploadSection = document.getElementById('uploadSection');
const modeSection = document.getElementById('modeSection');
const loadingSection = document.getElementById('loadingSection');
const resultSection = document.getElementById('resultSection');
const errorSection = document.getElementById('errorSection');

const modeCards = document.querySelectorAll('.mode-card');
const convertBtn = document.getElementById('convertBtn');

const originalImage = document.getElementById('originalImage');
const depthMapImage = document.getElementById('depthMapImage');
const normalMapImage = document.getElementById('normalMapImage');
const leftImage = document.getElementById('leftImage');
const rightImage = document.getElementById('rightImage');

const depthMapGroup = document.getElementById('depthMapGroup');
const normalMapGroup = document.getElementById('normalMapGroup');
const stereoGroup = document.getElementById('stereoGroup');

const processingTime = document.getElementById('processingTime');
const imageSize = document.getElementById('imageSize');

const newConversionBtn = document.getElementById('newConversionBtn');
const downloadBtn = document.getElementById('downloadBtn');
const retryBtn = document.getElementById('retryBtn');
const errorMessage = document.getElementById('errorMessage');

// 전역 변수
let selectedFile = null;
let selectedMode = null;
let resultData = null;

// 초기화
function init() {
    setupEventListeners();
}

// 이벤트 리스너 설정
function setupEventListeners() {
    uploadBtn.addEventListener('click', () => imageInput.click());
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
    
    // 모드 선택
    modeCards.forEach(card => {
        card.addEventListener('click', () => selectMode(card));
    });
    
    convertBtn.addEventListener('click', startConversion);
    newConversionBtn.addEventListener('click', resetAll);
    downloadBtn.addEventListener('click', downloadResults);
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
        
        // 모드 선택 섹션 표시
        modeSection.style.display = 'block';
    };
    
    reader.readAsDataURL(file);
}

// 모드 선택
function selectMode(card) {
    // 모든 카드에서 selected 제거
    modeCards.forEach(c => c.classList.remove('selected'));
    
    // 선택된 카드에 selected 추가
    card.classList.add('selected');
    selectedMode = card.dataset.mode;
    
    // 변환 버튼 활성화
    convertBtn.disabled = false;
}

// 변환 시작
async function startConversion() {
    if (!selectedFile || !selectedMode) {
        alert('이미지와 변환 모드를 선택해주세요.');
        return;
    }
    
    showLoading();
    
    try {
        const formData = new FormData();
        formData.append('image', selectedFile);
        formData.append('mode', selectedMode);
        
        const response = await fetch('/api/convert-to-3d', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('변환에 실패했습니다.');
        }
        
        const data = await response.json();
        
        if (data.success) {
            resultData = data;
            showResult(data);
        } else {
            throw new Error(data.error || '알 수 없는 오류가 발생했습니다.');
        }
        
    } catch (error) {
        console.error('Error:', error);
        showError(error.message);
    }
}

// 로딩 표시
function showLoading() {
    uploadSection.style.display = 'none';
    modeSection.style.display = 'none';
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
    
    // 처리 정보
    processingTime.textContent = data.processing_time + '초';
    imageSize.textContent = data.image_size;
    
    // Depth Map
    if (data.depth_map) {
        depthMapImage.src = 'data:image/png;base64,' + data.depth_map;
        depthMapGroup.style.display = 'block';
    } else {
        depthMapGroup.style.display = 'none';
    }
    
    // Normal Map
    if (data.normal_map) {
        normalMapImage.src = 'data:image/png;base64,' + data.normal_map;
        normalMapGroup.style.display = 'block';
    } else {
        normalMapGroup.style.display = 'none';
    }
    
    // 입체 효과
    if (data.left_image && data.right_image) {
        leftImage.src = 'data:image/png;base64,' + data.left_image;
        rightImage.src = 'data:image/png;base64,' + data.right_image;
        stereoGroup.style.display = 'block';
    } else {
        stereoGroup.style.display = 'none';
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
    modeSection.style.display = 'block';
}

// 결과 다운로드
function downloadResults() {
    if (!resultData) return;
    
    // Depth Map 다운로드
    if (resultData.depth_map) {
        downloadImage(resultData.depth_map, 'depth_map.png');
    }
    
    // Normal Map 다운로드
    if (resultData.normal_map) {
        downloadImage(resultData.normal_map, 'normal_map.png');
    }
    
    // 입체 이미지 다운로드
    if (resultData.left_image) {
        downloadImage(resultData.left_image, 'left_eye.png');
    }
    if (resultData.right_image) {
        downloadImage(resultData.right_image, 'right_eye.png');
    }
}

// 이미지 다운로드
function downloadImage(base64Data, filename) {
    const link = document.createElement('a');
    link.href = 'data:image/png;base64,' + base64Data;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// 전체 리셋
function resetAll() {
    selectedFile = null;
    selectedMode = null;
    resultData = null;
    
    uploadPlaceholder.style.display = 'block';
    previewContainer.style.display = 'none';
    
    modeCards.forEach(c => c.classList.remove('selected'));
    convertBtn.disabled = true;
    
    uploadSection.style.display = 'block';
    modeSection.style.display = 'none';
    loadingSection.style.display = 'none';
    resultSection.style.display = 'none';
    errorSection.style.display = 'none';
    
    imageInput.value = '';
}

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', init);

