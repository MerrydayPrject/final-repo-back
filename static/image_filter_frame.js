// DOM 요소
const uploadArea = document.getElementById('uploadArea');
const uploadButton = document.getElementById('uploadButton');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const originalImage = document.getElementById('originalImage');
const controlsSection = document.getElementById('controlsSection');
const downloadBtn = document.getElementById('downloadBtn');
const resetBtn = document.getElementById('resetBtn');
const loadingOverlay = document.getElementById('loadingOverlay');

let currentImageFile = null;
let currentResultImageData = null;
let originalImageData = null;
let selectedFilterPreset = 'none';
let selectedFramePreset = 'none';

// 초기 상태 설정
document.addEventListener('DOMContentLoaded', () => {
    const noneFilterBtn = document.querySelector('.filter-btn[data-preset="none"]');
    if (noneFilterBtn) {
        noneFilterBtn.classList.add('active');
    }
    
    const noneFrameBtn = document.querySelector('.frame-btn[data-preset="none"]');
    if (noneFrameBtn) {
        noneFrameBtn.classList.add('active');
    }
});

// 업로드 버튼 클릭 이벤트
uploadButton.addEventListener('click', () => {
    fileInput.click();
});

// 파일 입력 변경 이벤트
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFileUpload(file);
    }
});

// 파일 업로드 처리
function handleFileUpload(file) {
    if (!file.type.startsWith('image/')) {
        alert('이미지 파일만 업로드할 수 있습니다.');
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        alert('파일 크기는 10MB 이하여야 합니다.');
        return;
    }

    currentImageFile = file;
    
    // 미리보기 표시
    const reader = new FileReader();
    reader.onload = (e) => {
        originalImageData = e.target.result; // 원본 이미지 데이터 저장
        originalImage.src = e.target.result; // 원본 이미지 표시
        previewImage.src = e.target.result; // 초기에는 원본과 동일하게
        previewContainer.style.display = 'block';
        controlsSection.style.display = 'block';
        downloadBtn.style.display = 'none';
        
        // 초기 이미지에 필터/프레임 적용
        applyFilterAndFrame();
    };
    reader.readAsDataURL(file);
}

// 필터 버튼 클릭 이벤트
document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        // 활성화 상태 변경
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        selectedFilterPreset = btn.dataset.preset;
        
        // 이미지가 업로드되어 있으면 바로 적용
        if (currentImageFile) {
            applyFilterAndFrame();
        }
    });
});

// 프레임 프리셋 버튼 클릭 이벤트
document.querySelectorAll('.frame-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        // 활성화 상태 변경
        document.querySelectorAll('.frame-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        selectedFramePreset = btn.dataset.preset;
        
        // 이미지가 업로드되어 있으면 바로 적용
        if (currentImageFile) {
            applyFilterAndFrame();
        }
    });
});

// 필터와 프레임 적용 (로딩 표시 포함)
async function applyFilterAndFrame() {
    if (!currentImageFile) {
        return;
    }

    // 로딩 표시
    if (loadingOverlay) {
        loadingOverlay.style.display = 'flex';
    }

    try {
        const formData = new FormData();
        formData.append('file', currentImageFile);
        formData.append('filter_preset', selectedFilterPreset);
        
        // 프레임 프리셋 적용
        if (selectedFramePreset && selectedFramePreset !== 'none') {
            const framePresets = {
                'black': { type: 'solid', color: '#000000', width: 15 },
                'white': { type: 'solid', color: '#FFFFFF', width: 15 },
                'gold': { type: 'solid', color: '#FFD700', width: 20 },
                'red': { type: 'solid', color: '#FF0000', width: 15 },
                'blue': { type: 'solid', color: '#0066FF', width: 15 }
            };
            
            if (framePresets[selectedFramePreset]) {
                const preset = framePresets[selectedFramePreset];
                formData.append('frame_type', preset.type);
                formData.append('frame_color', preset.color);
                formData.append('frame_width', preset.width.toString());
            } else {
                formData.append('frame_type', 'none');
            }
        } else {
            formData.append('frame_type', 'none');
        }

        const response = await fetch('/api/apply-filter-and-frame', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('API 에러 응답:', response.status, errorText);
            return;
        }

        const data = await response.json();

        if (data.success) {
            currentResultImageData = data.result_image;
            previewImage.src = data.result_image;
            downloadBtn.style.display = 'block';
        }
    } catch (error) {
        console.error('이미지 처리 중 오류:', error);
    } finally {
        // 로딩 숨김
        if (loadingOverlay) {
            loadingOverlay.style.display = 'none';
        }
    }
}

// 다운로드 버튼 클릭 이벤트
downloadBtn.addEventListener('click', () => {
    if (currentResultImageData) {
        const link = document.createElement('a');
        link.href = currentResultImageData;
        link.download = 'filtered_image.png';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
});

// 초기화 버튼 클릭 이벤트
resetBtn.addEventListener('click', () => {
    currentImageFile = null;
    currentResultImageData = null;
    selectedFilterPreset = 'none';
    selectedFramePreset = 'none';
    
    previewContainer.style.display = 'none';
    controlsSection.style.display = 'none';
    downloadBtn.style.display = 'none';
    fileInput.value = '';
    originalImageData = null;
    
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    const noneFilterBtn = document.querySelector('.filter-btn[data-preset="none"]');
    if (noneFilterBtn) {
        noneFilterBtn.classList.add('active');
    }
    
    document.querySelectorAll('.frame-btn').forEach(b => b.classList.remove('active'));
    const noneFrameBtn = document.querySelector('.frame-btn[data-preset="none"]');
    if (noneFrameBtn) {
        noneFrameBtn.classList.add('active');
    }
});

