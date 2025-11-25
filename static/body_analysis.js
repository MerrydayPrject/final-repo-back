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
        // 키/몸무게 필수 검증
        const heightInput = document.getElementById('heightInput');
        const weightInput = document.getElementById('weightInput');
        
        if (!heightInput || !heightInput.value || heightInput.value.trim() === '') {
            showError('키를 입력해주세요.');
            return;
        }
        
        if (!weightInput || !weightInput.value || weightInput.value.trim() === '') {
            showError('몸무게를 입력해주세요.');
            return;
        }
        
        const height = parseFloat(heightInput.value);
        const weight = parseFloat(weightInput.value);
        
        if (isNaN(height) || height < 100 || height > 250) {
            showError('키는 100cm 이상 250cm 이하여야 합니다.');
            return;
        }
        
        if (isNaN(weight) || weight < 30 || weight > 200) {
            showError('몸무게는 30kg 이상 200kg 이하여야 합니다.');
            return;
        }
        
        // 로딩 표시
        showLoading();
        analyzeButton.disabled = true;
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('height', height);
        formData.append('weight', weight);
        
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
    
    // 1. 체형 타입 (맨 위)
    html += `
        <div class="result-card">
            <div class="result-item">
                <div class="result-label">체형 타입</div>
                <div class="body-type-text">${body_analysis.body_type}의 체형에 가깝습니다</div>
            </div>
        </div>
    `;
    
    // 2. 체형 특징
    if (body_analysis.body_features && body_analysis.body_features.length > 0) {
        html += `
            <div class="result-card">
                <div class="result-item">
                    <div class="result-label">체형 특징</div>
                    <div class="style-badges">
                        ${body_analysis.body_features.map(feature => {
                            // 부드러운 표현으로 변환
                            let displayFeature = feature;
                            
                            // 체형 특징별 부드러운 표현 매핑
                            const softFeatureMap = {
                                '키가 작은 체형': '키가 작으신 체형',
                                '키가 큰 체형': '키가 크신 체형',
                                '허리가 짧은 체형': '허리 비율이 짧으신 체형',
                                '어깨가 넓은 체형': '균형잡힌 상체체형',
                                '어깨가 좁은 체형': '어깨라인이 슬림한 체형',
                                '마른 체형': '슬림한 체형',
                                '글래머러스한 체형': '곡선미가 돋보이는 체형',
                                '팔 라인이 신경 쓰이는 체형': '팔라인이 신경쓰이는 체형',
                                '복부가 신경 쓰이는 체형': '' // 표시하지 않음
                            };
                            
                            // 매핑된 표현이 있으면 사용, 없으면 원본 사용
                            displayFeature = softFeatureMap[feature] !== undefined 
                                ? softFeatureMap[feature] 
                                : feature;
                            
                            // 빈 문자열이면 표시하지 않음
                            if (!displayFeature) return '';
                            
                            return `<span class="dress-style-badge" style="background: #e3f2fd; color: #1976d2;">${displayFeature}</span>`;
                        }).filter(f => f !== '').join('')}
                    </div>
                </div>
            </div>
        `;
    }
    
    // 3. 추천 드레스 카테고리 2개 (분석글 위)
    if (gemini_analysis && gemini_analysis.detailed_analysis) {
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
        
        // 추천 스타일에서 피해야 할 스타일 제외하고 최대 2개만 선택
        const filteredRecommendedStyles = recommendedStyles
            .filter(style => !avoidStyles.includes(style))
            .slice(0, 2);
        
        if (filteredRecommendedStyles.length > 0) {
            html += `
                <div class="result-card">
                    <div class="result-item">
                        <div class="result-label">추천 드레스 스타일</div>
                        <div class="style-badges">
                            ${filteredRecommendedStyles.map(style => `<span class="dress-style-badge recommended">${style}</span>`).join('')}
                        </div>
                    </div>
                </div>
            `;
        }
    }
    
    // 키워드 하이라이트 함수
    function highlightKeywords(text) {
        let result = text;
        
        // HTML 태그 제거한 순수 텍스트
        const plainText = text.replace(/<[^>]*>/g, '');
        
        // 1. 체형 장점 설명 부분 찾기 (첫 문장에서)
        // 첫 번째 문장 가져오기
        const firstSentence = plainText.split(/[\.。]/)[0].trim();
        
        // 체형 장점을 설명하는 패턴들 (더 구체적으로)
        let advantageText = null;
        
        // 패턴 1: "~체형을 가지고 있습니다" 또는 "~체형입니다"
        const pattern1 = /([^\.]*(?:슬림|늘씬|균형|우아|세련|좋은|탄탄)[^\.]*체형[을를]?\s*(?:가지고\s*있습니다|입니다))/;
        const match1 = firstSentence.match(pattern1);
        if (match1 && match1[1]) {
            advantageText = match1[1].trim();
        }
        
        // 패턴 2: "~돋보이며" 또는 "~돋보입니다"
        if (!advantageText) {
            const pattern2 = /([^\.]*(?:돋보이|좋|균형|우아|세련)[^\.]*(?:며|고|습니다|입니다))/;
            const match2 = firstSentence.match(pattern2);
            if (match2 && match2[1]) {
                advantageText = match2[1].trim();
            }
        }
        
        // 패턴 3: "비율이 좋은", "라인이 ~" 등
        if (!advantageText) {
            const pattern3 = /([^\.]*(?:비율|라인|실루엣|다리|팔|어깨)[이가]\s*(?:좋|돋보이|균형|우아|세련|길|가늘)[^\.]*)/;
            const match3 = firstSentence.match(pattern3);
            if (match3 && match3[1]) {
                advantageText = match3[1].trim();
            }
        }
        
        // 패턴 4: 긍정적 형용사 + 체형 관련 단어
        if (!advantageText) {
            const pattern4 = /([^\.]*(?:슬림|늘씬|균형잡힌|우아한|세련된|좋은|탄탄한)[^\.]*(?:체형|비율|라인|실루엣|인상|느낌))/;
            const match4 = firstSentence.match(pattern4);
            if (match4 && match4[1]) {
                advantageText = match4[1].trim();
            }
        }
        
        // 체형 장점 설명 하이라이트
        if (advantageText && advantageText.length > 8) {
            // 문장 끝에 마침표가 없으면 추가
            if (!advantageText.endsWith('.') && !advantageText.endsWith('다') && !advantageText.endsWith('며')) {
                advantageText += '.';
            }
            
            const escapedText = advantageText.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const regex = new RegExp(`(${escapedText})`, 'g');
            
            result = result.replace(regex, (match) => {
                if (!match.includes('<span class="highlight')) {
                    return `<span class="highlight">${match}</span>`;
                }
                return match;
            });
        }
        
        // 2. 드레스 키워드 2개만 하이라이트
        const styleKeywords = [
            '슬림', '프린세스', 'A라인', '벨라인', '머메이드', '트럼펫', '미니드레스',
            '스트레이트', 'H라인', '에이라인', '플레어', '하이웨이스트', '벨트라인'
        ];
        
        let highlightCount = 0;
        const maxHighlights = 2;
        
        // 키워드 길이순으로 정렬 (긴 키워드부터 매칭)
        const sortedKeywords = [...styleKeywords].sort((a, b) => b.length - a.length);
        
        sortedKeywords.forEach(keyword => {
            if (highlightCount >= maxHighlights) return;
            
            const regex = new RegExp(`(${keyword})`, 'gi');
            result = result.replace(regex, (match, p1, offset, string) => {
                if (highlightCount >= maxHighlights) return match;
                
                // 이미 하이라이트된 부분이 아니고, HTML 태그 내부가 아니면 하이라이트
                const before = string.substring(Math.max(0, offset - 20), offset);
                const after = string.substring(offset + match.length, offset + match.length + 20);
                
                if (!before.includes('<span') && !after.includes('</span>') && 
                    !before.includes('>') && !after.includes('<')) {
                    highlightCount++;
                    return `<span class="highlight">${p1}</span>`;
                }
                return match;
            });
        });
        
        return result;
    }
    
    // 4. AI 상세 분석 (분석 결과)
    if (gemini_analysis && gemini_analysis.detailed_analysis) {
        // 마크다운 형식 처리
        let analysisText = gemini_analysis.detailed_analysis;
        
        // 마크다운 볼드를 HTML strong 태그로 변환
        analysisText = analysisText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        // 리스트 항목 정리
        analysisText = analysisText.replace(/\*\s+/g, '• ');
        
        // 키워드 하이라이트 적용
        analysisText = highlightKeywords(analysisText);
        
        // 빈 줄 정리
        const lines = analysisText.split('\n').filter(line => line.trim());
        
        html += `
            <div class="result-card">
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

