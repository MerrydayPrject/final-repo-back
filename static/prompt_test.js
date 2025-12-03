// 파이프라인 테스트 JavaScript (V4/V5 지원)

// 전역 변수
let uploadedImages = {
    person: null,
    garment: null,
    background: null
};

let promptFiles = {
    stage1: [],
    stage2: [],
    stage3: []
};

let v5PromptFiles = [];
let currentPipeline = 'v4';
let resultImage = null;

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', () => {
    loadPromptFiles();
    loadV5PromptFiles();
    setupDragAndDrop();
});

// 파이프라인 변경 핸들러
function onPipelineChange() {
    const selected = document.querySelector('input[name="pipeline"]:checked').value;
    currentPipeline = selected;
    
    const v4Section = document.getElementById('v4-prompt-section');
    const v5Section = document.getElementById('v5-prompt-section');
    const headerDesc = document.getElementById('header-desc');
    const actionHint = document.getElementById('action-hint');
    const loadingHint = document.getElementById('loading-hint');
    const xaiSection = document.getElementById('xai-section');
    
    if (selected === 'v4') {
        v4Section.style.display = 'block';
        v5Section.style.display = 'none';
        headerDesc.textContent = 'V4 파이프라인: X.AI 분석 → Gemini 3 생성';
        actionHint.textContent = '선택한 Stage 1/2/3 프롬프트로 V4 파이프라인을 실행합니다';
        loadingHint.textContent = 'V4 파이프라인 실행 중입니다. 약 30~60초 소요됩니다.';
    } else {
        v4Section.style.display = 'none';
        v5Section.style.display = 'block';
        headerDesc.textContent = 'V5 파이프라인: Gemini 3 직접 처리 (빠른 응답)';
        actionHint.textContent = '선택한 통합 프롬프트로 V5 파이프라인을 실행합니다';
        loadingHint.textContent = 'V5 파이프라인 실행 중입니다. 약 15~30초 소요됩니다.';
        xaiSection.style.display = 'none';
    }
}

// V4 프롬프트 파일 목록 로드
async function loadPromptFiles() {
    try {
        const response = await fetch('/api/prompts/v4/list');
        const data = await response.json();
        
        if (data.success) {
            promptFiles.stage1 = data.stage1;
            promptFiles.stage2 = data.stage2;
            promptFiles.stage3 = data.stage3;
            
            // 드롭다운 옵션 추가
            const stage1Select = document.getElementById('stage1-select');
            const stage2Select = document.getElementById('stage2-select');
            const stage3Select = document.getElementById('stage3-select');
            
            data.stage1.forEach(file => {
                const option = document.createElement('option');
                option.value = file;
                option.textContent = file.replace('.txt', '');
                stage1Select.appendChild(option);
            });
            
            data.stage2.forEach(file => {
                const option = document.createElement('option');
                option.value = file;
                option.textContent = file.replace('prompt_', '').replace('.txt', '');
                stage2Select.appendChild(option);
            });
            
            data.stage3.forEach(file => {
                const option = document.createElement('option');
                option.value = file;
                option.textContent = file.replace('prompt_', '').replace('.txt', '');
                stage3Select.appendChild(option);
            });
            
            // 테스트 파일 우선 선택
            const testStage1 = data.stage1.find(f => f.includes('test')) || data.stage1[0];
            const testStage2 = data.stage2.find(f => f.includes('test')) || data.stage2[0];
            const testStage3 = data.stage3.find(f => f.includes('test')) || data.stage3[0];
            
            if (testStage1) {
                stage1Select.value = testStage1;
                loadPrompt('stage1');
            }
            if (testStage2) {
                stage2Select.value = testStage2;
                loadPrompt('stage2');
            }
            if (testStage3) {
                stage3Select.value = testStage3;
                loadPrompt('stage3');
            }
        }
    } catch (error) {
        console.error('V4 프롬프트 파일 목록 로드 실패:', error);
    }
}

// V5 프롬프트 파일 목록 로드
async function loadV5PromptFiles() {
    try {
        const response = await fetch('/api/prompts/v5/list');
        const data = await response.json();
        
        if (data.success) {
            v5PromptFiles = data.files;
            
            const select = document.getElementById('v5-unified-select');
            
            data.files.forEach(file => {
                const option = document.createElement('option');
                option.value = file;
                option.textContent = file.replace('prompt_', '').replace('.txt', '');
                select.appendChild(option);
            });
            
            // 첫 번째 파일 선택
            if (data.files.length > 0) {
                select.value = data.files[0];
                loadV5Prompt();
            }
        }
    } catch (error) {
        console.error('V5 프롬프트 파일 목록 로드 실패:', error);
    }
}

// V4 프롬프트 내용 로드
async function loadPrompt(stage) {
    const select = document.getElementById(`${stage}-select`);
    const textarea = document.getElementById(`${stage}-prompt`);
    const filename = select.value;
    
    if (!filename) {
        textarea.value = '';
        updateCharCount(stage);
        return;
    }
    
    try {
        const response = await fetch(`/api/prompts/v4/${filename}`);
        const data = await response.json();
        
        if (data.success) {
            textarea.value = data.content;
            updateCharCount(stage);
        } else {
            console.error('프롬프트 로드 실패:', data.message);
        }
    } catch (error) {
        console.error('프롬프트 로드 오류:', error);
    }
}

// V5 프롬프트 내용 로드
async function loadV5Prompt() {
    const select = document.getElementById('v5-unified-select');
    const textarea = document.getElementById('v5-unified-prompt');
    const filename = select.value;
    
    if (!filename) {
        textarea.value = '';
        updateV5CharCount();
        return;
    }
    
    try {
        const response = await fetch(`/api/prompts/v5/${filename}`);
        const data = await response.json();
        
        if (data.success) {
            textarea.value = data.content;
            updateV5CharCount();
        } else {
            console.error('V5 프롬프트 로드 실패:', data.message);
        }
    } catch (error) {
        console.error('V5 프롬프트 로드 오류:', error);
    }
}

// 문자 수 업데이트
function updateCharCount(stage) {
    const textarea = document.getElementById(`${stage}-prompt`);
    const countSpan = document.getElementById(`${stage}-count`);
    countSpan.textContent = `${textarea.value.length}자`;
}

function updateV5CharCount() {
    const textarea = document.getElementById('v5-unified-prompt');
    const countSpan = document.getElementById('v5-unified-count');
    countSpan.textContent = `${textarea.value.length}자`;
}

// 드래그 앤 드롭 설정
function setupDragAndDrop() {
    ['person', 'garment', 'background'].forEach(type => {
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
        
        // 클릭으로 업로드
        area.addEventListener('click', (e) => {
            if (e.target.tagName !== 'BUTTON' && e.target.tagName !== 'INPUT') {
                document.getElementById(`input-${type}`).click();
            }
        });
    });
}

// 이미지 업로드 처리
function handleImageUpload(event, type) {
    const file = event.target.files[0];
    if (!file) return;
    
    if (!file.type.startsWith('image/')) {
        alert('이미지 파일만 업로드 가능합니다.');
        return;
    }
    
    uploadedImages[type] = file;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        const previewContainer = document.getElementById(`preview-${type}`);
        const img = document.getElementById(`img-${type}`);
        const uploadContent = document.querySelector(`#upload-${type} .upload-content`);
        
        img.src = e.target.result;
        previewContainer.style.display = 'block';
        uploadContent.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// 이미지 제거
function removeImage(type) {
    uploadedImages[type] = null;
    
    const previewContainer = document.getElementById(`preview-${type}`);
    const uploadContent = document.querySelector(`#upload-${type} .upload-content`);
    const input = document.getElementById(`input-${type}`);
    
    previewContainer.style.display = 'none';
    uploadContent.style.display = 'block';
    input.value = '';
}

// 테스트 실행
async function runTest() {
    // 유효성 검사
    if (!uploadedImages.person || !uploadedImages.garment || !uploadedImages.background) {
        alert('인물, 의상, 배경 이미지를 모두 업로드해주세요.');
        return;
    }
    
    if (currentPipeline === 'v4') {
        await runV4Test();
    } else {
        await runV5Test();
    }
}

// V4 테스트 실행
async function runV4Test() {
    const stage1Prompt = document.getElementById('stage1-prompt').value.trim();
    const stage2Prompt = document.getElementById('stage2-prompt').value.trim();
    const stage3Prompt = document.getElementById('stage3-prompt').value.trim();
    
    if (!stage1Prompt || !stage2Prompt || !stage3Prompt) {
        alert('Stage 1, 2, 3 프롬프트를 모두 입력해주세요.');
        return;
    }
    
    // UI 상태 변경
    const runBtn = document.getElementById('run-btn');
    const loading = document.getElementById('loading');
    const resultSection = document.getElementById('result-section');
    const xaiSection = document.getElementById('xai-section');
    
    runBtn.disabled = true;
    loading.style.display = 'block';
    resultSection.style.display = 'none';
    xaiSection.style.display = 'none';
    
    try {
        // FormData 생성 (프롬프트 내용 직접 전송)
        const formData = new FormData();
        formData.append('person_image', uploadedImages.person);
        formData.append('garment_image', uploadedImages.garment);
        formData.append('background_image', uploadedImages.background);
        formData.append('stage1_prompt', stage1Prompt);
        formData.append('stage2_prompt', stage2Prompt);
        formData.append('stage3_prompt', stage3Prompt);
        
        // API 호출 (수정된 프롬프트 내용으로)
        const response = await fetch('/api/prompt-test/run-custom', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        loading.style.display = 'none';
        runBtn.disabled = false;
        
        if (data.success) {
            displayV4Result(data);
        } else {
            alert(`테스트 실패: ${data.message}`);
        }
    } catch (error) {
        console.error('V4 테스트 오류:', error);
        alert(`오류 발생: ${error.message}`);
        
        document.getElementById('loading').style.display = 'none';
        document.getElementById('run-btn').disabled = false;
    }
}

// V5 테스트 실행
async function runV5Test() {
    const unifiedPrompt = document.getElementById('v5-unified-prompt').value.trim();
    
    if (!unifiedPrompt) {
        alert('V5 통합 프롬프트를 입력해주세요.');
        return;
    }
    
    // UI 상태 변경
    const runBtn = document.getElementById('run-btn');
    const loading = document.getElementById('loading');
    const resultSection = document.getElementById('result-section');
    const xaiSection = document.getElementById('xai-section');
    
    runBtn.disabled = true;
    loading.style.display = 'block';
    resultSection.style.display = 'none';
    xaiSection.style.display = 'none';
    
    try {
        // FormData 생성 (프롬프트 내용 직접 전송)
        const formData = new FormData();
        formData.append('person_image', uploadedImages.person);
        formData.append('garment_image', uploadedImages.garment);
        formData.append('background_image', uploadedImages.background);
        formData.append('unified_prompt', unifiedPrompt);
        
        // API 호출 (수정된 프롬프트 내용으로)
        const response = await fetch('/api/prompt-test/run-v5-custom', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        loading.style.display = 'none';
        runBtn.disabled = false;
        
        if (data.success) {
            displayV5Result(data);
        } else {
            alert(`테스트 실패: ${data.message}`);
        }
    } catch (error) {
        console.error('V5 테스트 오류:', error);
        alert(`오류 발생: ${error.message}`);
        
        document.getElementById('loading').style.display = 'none';
        document.getElementById('run-btn').disabled = false;
    }
}

// V4 결과 표시
function displayV4Result(data) {
    const xaiSection = document.getElementById('xai-section');
    const xaiPrompt = document.getElementById('xai-prompt');
    const resultSection = document.getElementById('result-section');
    const runTime = document.getElementById('run-time');
    const resultPrompts = document.getElementById('result-prompts');
    
    // X.AI 프롬프트 표시
    if (data.xai_prompt) {
        xaiPrompt.textContent = data.xai_prompt;
        xaiSection.style.display = 'block';
    }
    
    // 실행 시간
    runTime.textContent = `${data.run_time?.toFixed(2) || '-'}초`;
    
    // 사용된 프롬프트 표시
    if (data.prompts_used) {
        const stage1 = data.prompts_used.stage1 === 'custom' ? '사용자 편집' : data.prompts_used.stage1;
        const stage2 = data.prompts_used.stage2 === 'custom' ? '사용자 편집' : data.prompts_used.stage2;
        const stage3 = data.prompts_used.stage3 === 'custom' ? '사용자 편집' : data.prompts_used.stage3;
        resultPrompts.textContent = `사용 프롬프트: ${stage1} / ${stage2} / ${stage3}`;
    }
    
    // 결과 표시
    const result = data.result;
    const resultImg = document.getElementById('result-image');
    const resultStatus = document.getElementById('result-status');
    
    if (result?.success) {
        resultImg.src = result.result_image;
        resultStatus.textContent = '성공';
        resultStatus.className = 'result-status success';
        resultImage = result.result_image;
    } else {
        resultImg.src = '';
        resultStatus.textContent = result?.message || '실패';
        resultStatus.className = 'result-status error';
        resultImage = null;
    }
    
    resultSection.style.display = 'block';
    
    // 결과 섹션으로 스크롤
    resultSection.scrollIntoView({ behavior: 'smooth' });
}

// V5 결과 표시
function displayV5Result(data) {
    const xaiSection = document.getElementById('xai-section');
    const resultSection = document.getElementById('result-section');
    const runTime = document.getElementById('run-time');
    const resultPrompts = document.getElementById('result-prompts');
    
    // V5는 X.AI 없음
    xaiSection.style.display = 'none';
    
    // 실행 시간
    runTime.textContent = `${data.run_time?.toFixed(2) || '-'}초`;
    
    // 사용된 프롬프트 표시
    if (data.prompt_used) {
        const promptName = data.prompt_used === 'custom' ? '사용자 편집' : data.prompt_used;
        resultPrompts.textContent = `사용 프롬프트: ${promptName} (V5 통합)`;
    }
    
    // 결과 표시
    const result = data.result;
    const resultImg = document.getElementById('result-image');
    const resultStatus = document.getElementById('result-status');
    
    if (result?.success) {
        resultImg.src = result.result_image;
        resultStatus.textContent = '성공';
        resultStatus.className = 'result-status success';
        resultImage = result.result_image;
    } else {
        resultImg.src = '';
        resultStatus.textContent = result?.message || '실패';
        resultStatus.className = 'result-status error';
        resultImage = null;
    }
    
    resultSection.style.display = 'block';
    
    // 결과 섹션으로 스크롤
    resultSection.scrollIntoView({ behavior: 'smooth' });
}

// 이미지 다운로드
function downloadImage() {
    if (!resultImage) {
        alert('다운로드할 이미지가 없습니다.');
        return;
    }
    
    const link = document.createElement('a');
    link.href = resultImage;
    link.download = `prompt-test-${currentPipeline}-${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}
