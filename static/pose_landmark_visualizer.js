// MediaPipe Pose 랜드마크 연결 구조
const POSE_CONNECTIONS = [
    // 얼굴 윤곽
    [0, 1], [1, 2], [2, 3], [3, 7],
    [0, 4], [4, 5], [5, 6], [6, 8],
    // 어깨
    [11, 12],
    // 왼쪽 팔
    [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
    // 오른쪽 팔
    [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
    // 몸통
    [11, 23], [12, 24], [23, 24],
    // 왼쪽 다리
    [23, 25], [25, 27], [27, 29], [27, 31],
    // 오른쪽 다리
    [24, 26], [26, 28], [28, 30], [28, 32],
];

// 랜드마크 이름 (참고용)
const LANDMARK_NAMES = {
    0: '코', 1: '왼쪽 눈 안쪽', 2: '왼쪽 눈', 3: '왼쪽 눈 바깥쪽',
    4: '오른쪽 눈 안쪽', 5: '오른쪽 눈', 6: '오른쪽 눈 바깥쪽',
    7: '왼쪽 귀', 8: '오른쪽 귀', 9: '입 왼쪽', 10: '입 오른쪽',
    11: '왼쪽 어깨', 12: '오른쪽 어깨', 13: '왼쪽 팔꿈치', 14: '오른쪽 팔꿈치',
    15: '왼쪽 손목', 16: '오른쪽 손목', 17: '왼쪽 새끼손가락', 18: '오른쪽 새끼손가락',
    19: '왼쪽 검지', 20: '오른쪽 검지', 21: '왼쪽 엄지', 22: '오른쪽 엄지',
    23: '왼쪽 엉덩이', 24: '오른쪽 엉덩이', 25: '왼쪽 무릎', 26: '오른쪽 무릎',
    27: '왼쪽 발목', 28: '오른쪽 발목', 29: '왼쪽 발뒤꿈치', 30: '오른쪽 발뒤꿈치',
    31: '왼쪽 발가락', 32: '오른쪽 발가락'
};

let currentImage = null;
let currentLandmarks = null;
let imageWidth = 0;
let imageHeight = 0;

const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const uploadContent = document.getElementById('uploadContent');
const uploadActions = document.getElementById('uploadActions');
const analyzeButton = document.getElementById('analyzeButton');
const clearButton = document.getElementById('clearButton');
const errorMessage = document.getElementById('errorMessage');
const loadingMessage = document.getElementById('loadingMessage');
const canvas = document.getElementById('landmarkCanvas');
const ctx = canvas.getContext('2d');
const noImageMessage = document.getElementById('noImageMessage');
const infoPanel = document.getElementById('infoPanel');
const landmarkCount = document.getElementById('landmarkCount');
const imageSize = document.getElementById('imageSize');
const bodyRatio = document.getElementById('bodyRatio');

// 파일 선택 이벤트
fileInput.addEventListener('change', handleFileSelect);
uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', handleDragOver);
uploadArea.addEventListener('drop', handleDrop);
analyzeButton.addEventListener('click', analyzeLandmarks);
clearButton.addEventListener('click', clearAll);

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.style.borderColor = '#667eea';
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.style.borderColor = '#ddd';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showError('이미지 파일만 업로드 가능합니다.');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            currentImage = img;
            imageWidth = img.width;
            imageHeight = img.height;
            
            // 캔버스 크기 설정
            const maxWidth = 800;
            const maxHeight = 600;
            let canvasWidth = img.width;
            let canvasHeight = img.height;
            
            if (canvasWidth > maxWidth) {
                canvasHeight = (canvasHeight * maxWidth) / canvasWidth;
                canvasWidth = maxWidth;
            }
            if (canvasHeight > maxHeight) {
                canvasWidth = (canvasWidth * maxHeight) / canvasHeight;
                canvasHeight = maxHeight;
            }
            
            canvas.width = canvasWidth;
            canvas.height = canvasHeight;
            
            // 이미지 그리기
            drawImage();
            
            // UI 업데이트
            uploadContent.style.display = 'none';
            uploadActions.style.display = 'flex';
            noImageMessage.style.display = 'none';
            errorMessage.style.display = 'none';
            infoPanel.style.display = 'none';
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

function drawImage() {
    if (!currentImage) return;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
    
    // 랜드마크가 있으면 그리기
    if (currentLandmarks) {
        drawLandmarks();
    }
}

function drawLandmarks() {
    if (!currentLandmarks || !currentImage) return;
    
    const scaleX = canvas.width / imageWidth;
    const scaleY = canvas.height / imageHeight;
    
    // 상체-하체 차이 계산에 사용되는 랜드마크: 왼쪽 어깨(12)와 왼쪽 발목(27)
    const left_shoulder_landmark = currentLandmarks.find(l => l.id === 12); // 왼쪽 어깨 (파란색)
    const left_ankle_landmark = currentLandmarks.find(l => l.id === 27); // 왼쪽 발목 (노란색)
    
    // 시각화를 위한 변수 (기존 코드 호환성 유지)
    const upper_body_landmark = left_shoulder_landmark;
    const lower_body_landmark = left_ankle_landmark;
    
    // 연결선 그리기
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 2;
    
    POSE_CONNECTIONS.forEach(([startIdx, endIdx]) => {
        const start = currentLandmarks.find(l => l.id === startIdx);
        const end = currentLandmarks.find(l => l.id === endIdx);
        
        if (start && end && start.visibility >= 0.3 && end.visibility >= 0.3) {
            const x1 = start.x * canvas.width;
            const y1 = start.y * canvas.height;
            const x2 = end.x * canvas.width;
            const y2 = end.y * canvas.height;
            
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
        }
    });
    
    // 랜드마크 점 그리기
    currentLandmarks.forEach(landmark => {
        if (landmark.visibility >= 0.3) {
            const x = landmark.x * canvas.width;
            const y = landmark.y * canvas.height;
            
            // 색상 결정
            let color = '#ff0000'; // 기본: 빨간색
            let radius = 4;
            
            // 실제 계산에 사용되는 2개 좌표만 특별히 표시
            if (landmark === upper_body_landmark) {
                // 상체 계산 좌표: 파란색, 큰 원
                color = '#0066ff';
                radius = 8;
            } else if (landmark === lower_body_landmark) {
                // 하체 계산 좌표: 노란색, 큰 원
                color = '#ffcc00';
                radius = 8;
            }
            
            // 점 그리기
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, 2 * Math.PI);
            ctx.fill();
            
            // 계산에 사용되는 좌표는 검은색 테두리로 강조
            if (landmark === upper_body_landmark || landmark === lower_body_landmark) {
                ctx.strokeStyle = '#000000';
                ctx.lineWidth = 3;
                ctx.stroke();
            }
            
            // ID 표시
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 12px Arial';
            ctx.fillText(landmark.id, x + radius + 3, y - radius - 3);
        }
    });
    
    // 계산에 사용되는 두 좌표 사이에 선 그리기 (거리 공식 사용)
    if (left_shoulder_landmark && left_ankle_landmark &&
        left_shoulder_landmark.visibility >= 0.3 && left_ankle_landmark.visibility >= 0.3) {
        const x1 = left_shoulder_landmark.x * canvas.width;
        const y1 = left_shoulder_landmark.y * canvas.height;
        const x2 = left_ankle_landmark.x * canvas.width;
        const y2 = left_ankle_landmark.y * canvas.height;
        
        // 거리 공식: sqrt((x1-x2)² + (y1-y2)²)
        const dx = x1 - x2;
        const dy = y1 - y2;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        // 두 좌표 사이에 선 그리기
        ctx.strokeStyle = '#ff00ff'; // 자홍색
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]); // 점선
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        ctx.setLineDash([]); // 점선 해제
        
        // 거리 값 표시 (정규화된 좌표 기준)
        const normalizedDistance = Math.sqrt(
            (left_shoulder_landmark.x - left_ankle_landmark.x) ** 2 +
            (left_shoulder_landmark.y - left_ankle_landmark.y) ** 2
        );
        ctx.fillStyle = '#ff00ff';
        ctx.font = 'bold 14px Arial';
        ctx.fillText(`차이: ${normalizedDistance.toFixed(3)}`, (x1 + x2) / 2 + 10, (y1 + y2) / 2);
    }
}

async function analyzeLandmarks() {
    if (!currentImage) {
        showError('이미지를 먼저 업로드해주세요.');
        return;
    }
    
    analyzeButton.disabled = true;
    loadingMessage.style.display = 'flex';
    errorMessage.style.display = 'none';
    
    try {
        const formData = new FormData();
        const file = fileInput.files[0];
        formData.append('file', file);
        
        const response = await fetch('/api/pose-landmarks', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.message || '랜드마크 추출 실패');
        }
        
        if (data.success && data.landmarks) {
            currentLandmarks = data.landmarks;
            imageWidth = data.image_width;
            imageHeight = data.image_height;
            
            // 랜드마크 그리기
            drawImage();
            
            // 정보 패널 업데이트
            landmarkCount.textContent = data.landmarks_count;
            imageSize.textContent = `${imageWidth} × ${imageHeight}`;
            
            // 상체-하체 차이 계산 (거리 공식 사용: 왼쪽 어깨(12)와 왼쪽 발목(27) 사이의 거리)
            // sqrt((x1-x2)² + (y1-y2)²)
            const leftShoulder = currentLandmarks.find(l => l.id === 12); // 왼쪽 어깨 (파란색)
            const leftAnkle = currentLandmarks.find(l => l.id === 27); // 왼쪽 발목 (노란색)
            
            if (leftShoulder && leftAnkle && 
                leftShoulder.visibility >= 0.3 && leftAnkle.visibility >= 0.3) {
                const dx = leftShoulder.x - leftAnkle.x;
                const dy = leftShoulder.y - leftAnkle.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                bodyRatio.textContent = distance.toFixed(3);
            } else {
                bodyRatio.textContent = 'N/A';
            }
            
            infoPanel.style.display = 'block';
        } else {
            throw new Error('랜드마크 데이터를 찾을 수 없습니다.');
        }
    } catch (error) {
        showError(error.message || '랜드마크 분석 중 오류가 발생했습니다.');
        console.error('Error:', error);
    } finally {
        analyzeButton.disabled = false;
        loadingMessage.style.display = 'none';
    }
}

function clearAll() {
    currentImage = null;
    currentLandmarks = null;
    imageWidth = 0;
    imageHeight = 0;
    
    fileInput.value = '';
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    uploadContent.style.display = 'flex';
    uploadActions.style.display = 'none';
    noImageMessage.style.display = 'block';
    errorMessage.style.display = 'none';
    infoPanel.style.display = 'none';
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
}

