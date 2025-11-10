// 체형 분석 테스트 페이지 JavaScript

const API_BASE_URL = window.location.origin;

// DOM 요소
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const uploadContent = document.getElementById('uploadContent');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const removeButton = document.getElementById('removeButton');
const analyzeButton = document.getElementById('analyzeButton');
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
        analyzeButton.disabled = false;
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
    analyzeButton.disabled = true;
    clearResults();
});

// 분석 버튼 클릭
analyzeButton.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    await analyzeBody(selectedFile);
});

// 체형 분석 API 호출
async function analyzeBody(file) {
    try {
        // 로딩 표시
        showLoading();
        analyzeButton.disabled = true;
        
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API_BASE_URL}/api/analyze-body`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
        } else {
            showError(data.message || '체형 분석에 실패했습니다.');
        }
        
    } catch (error) {
        console.error('분석 오류:', error);
        showError('서버 연결 오류가 발생했습니다.');
    } finally {
        hideLoading();
        analyzeButton.disabled = false;
    }
}

// 결과 표시
function displayResults(data) {
    const { body_analysis, gemini_analysis, measurements } = data;
    
    let html = '';
    
    // 드레스 스타일 추출 함수 (공통)
    // 실제 드레스 카테고리: 벨라인, 머메이드, 프린세스, A라인, 슬림, 트럼펫
    function extractDressStyles(text, isAvoid = false) {
        const foundStyles = new Set();
        
        // 실제 드레스 카테고리 목록
        const availableCategories = [
            '벨라인', '머메이드', '프린세스', 'A라인', '슬림', '트럼펫'
        ];
        
        // 카테고리 매핑 (분석 텍스트에서 찾을 수 있는 다양한 표현)
        const categoryMapping = {
            '벨라인': ['벨라인', '벨트', '하이웨이스트', '벨티드', '벨트라인'],
            '머메이드': ['머메이드', '물고기', '피쉬', '피쉬테일'],
            '프린세스': ['프린세스', '프린세스라인', '프린세스 라인'],
            'A라인': ['A라인', '에이라인', '에이 라인', '에이-라인'],
            '슬림': ['슬림', '스트레이트', 'H라인', '직선', '피팅', '슬림핏'],
            '트럼펫': ['트럼펫', '플레어', '트럼펫라인', '플레어 실루엣']
        };
        
        // 텍스트에서 카테고리 찾기
        availableCategories.forEach(category => {
            // 직접 매칭
            if (text.includes(category)) {
                foundStyles.add(category);
            }
            
            // 매핑된 키워드로 찾기
            const keywords = categoryMapping[category] || [];
            keywords.forEach(keyword => {
                if (text.includes(keyword) && !foundStyles.has(category)) {
                    foundStyles.add(category);
                }
            });
        });
        
        // 필터링: 실제 카테고리만 포함
        const filtered = Array.from(foundStyles).filter(style => {
            return availableCategories.includes(style);
        });
        
        return filtered.slice(0, 6); // 최대 6개 (모든 카테고리)
    }
    
    // 체형 타입 (간단하게)
    html += `
        <div class="result-card">
            <div class="result-item">
                <div class="result-label">체형 타입</div>
                <div class="body-type-text">${body_analysis.body_type}의 체형에 가깝습니다</div>
            </div>
            ${measurements ? `
                <div class="result-item">
                    <div class="result-label">어깨/엉덩이 비율</div>
                    <div class="result-value">${measurements.shoulder_hip_ratio.toFixed(2)}</div>
                </div>
            ` : ''}
        </div>
    `;
    
    // Gemini 분석 결과 (상세 분석 텍스트만 표시, 별도 리스트는 제외)
    if (gemini_analysis && gemini_analysis.detailed_analysis) {
        // 마크다운 형식 처리
        let analysisText = gemini_analysis.detailed_analysis;
        
        // 추천 드레스 스타일 추출 (추천 섹션만 추출)
        let recommendationSection = analysisText;
        const avoidIndex = analysisText.indexOf('피해야');
        if (avoidIndex !== -1) {
            recommendationSection = analysisText.substring(0, avoidIndex);
        }
        
        const recommendedStyles = extractDressStyles(recommendationSection, false);
        
        // 피해야 할 드레스 스타일 추출
        let avoidSection = '';
        if (avoidIndex !== -1) {
            avoidSection = analysisText.substring(avoidIndex);
        }
        const avoidStyles = extractDressStyles(avoidSection, true);
        
        // 마크다운 볼드를 HTML strong 태그로 변환
        analysisText = analysisText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        // 리스트 항목 정리
        analysisText = analysisText.replace(/\*\s+/g, '• ');
        
        // 빈 줄 정리
        const lines = analysisText.split('\n').filter(line => line.trim());
        
        html += `
            <div class="result-card">
                ${recommendedStyles.length > 0 ? `
                    <div class="result-item">
                        <div class="result-label">추천 드레스 스타일</div>
                        <div class="style-badges">
                            ${recommendedStyles.map(style => `<span class="dress-style-badge recommended">${style}</span>`).join('')}
                        </div>
                    </div>
                ` : ''}
                ${avoidStyles.length > 0 ? `
                    <div class="result-item">
                        <div class="result-label">피해야 할 드레스 스타일</div>
                        <div class="style-badges">
                            ${avoidStyles.map(style => `<span class="dress-style-badge avoid">${style}</span>`).join('')}
                        </div>
                    </div>
                ` : ''}
                <div class="result-title">AI 상세 분석</div>
                <div class="analysis-text">
                    ${lines.map(line => line.trim() ? `<p>${line.trim()}</p>` : '').join('')}
                </div>
            </div>
        `;
    }
    
    resultContent.innerHTML = html;
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
            <div class="placeholder-icon">📊</div>
            <p class="placeholder-text">이미지를 업로드하고 분석 버튼을 클릭하세요</p>
        </div>
    `;
}

