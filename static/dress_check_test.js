let selectedFile = null;
let previewDataUrl = null;

document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const checkBtn = document.getElementById('check-btn');
    const resetBtn = document.getElementById('reset-btn');

    // 클릭 업로드
    uploadArea.addEventListener('click', () => fileInput.click());

    // 파일 선택
    fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

    // 드래그 앤 드롭
    ['dragenter', 'dragover'].forEach(evt => {
        uploadArea.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            uploadArea.classList.add('dragover');
        });
    });
    ['dragleave', 'drop'].forEach(evt => {
        uploadArea.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            uploadArea.classList.remove('dragover');
        });
    });
    uploadArea.addEventListener('drop', (e) => {
        handleFiles(e.dataTransfer.files);
    });

    checkBtn.addEventListener('click', runCheck);
    resetBtn.addEventListener('click', resetAll);
});

function handleFiles(files) {
    if (!files || files.length === 0) return;

    const file = files[0];
    const maxSize = 5 * 1024 * 1024; // 5MB

    if (!file.type.startsWith('image/')) {
        setStatus('이미지 파일만 업로드해주세요.', true);
        return;
    }

    if (file.size > maxSize) {
        setStatus('파일 크기는 5MB 이하로 업로드해주세요.', true);
        return;
    }

    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewDataUrl = e.target.result;
        const preview = document.getElementById('preview');
        preview.style.display = 'block';
        preview.innerHTML = `<img src="${previewDataUrl}" alt="preview">`;
        setStatus('이미지가 준비되었습니다.');
        document.getElementById('reset-btn').style.display = 'inline-flex';
    };
    reader.readAsDataURL(file);
}

function setStatus(text, isError = false) {
    const el = document.getElementById('status-text');
    if (el) {
        el.textContent = text || '';
        el.style.color = isError ? '#ff9aa9' : '#ccc';
    }
}

async function runCheck() {
    if (!selectedFile) {
        setStatus('이미지를 먼저 업로드해주세요.', true);
        return;
    }

    const model = document.getElementById('model-select').value;
    const mode = document.getElementById('mode-select').value;
    const btn = document.getElementById('check-btn');

    btn.disabled = true;
    setStatus('판별 중입니다...');

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('model', model);
    formData.append('mode', mode);

    try {
        const res = await fetch('/api/dress/check', {
            method: 'POST',
            body: formData
        });

        const data = await res.json();

        if (!res.ok || !data.success) {
            const message = data.message || `요청 실패 (status: ${res.status})`;
            alert(message);
            setStatus(message, true);
            return;
        }

        // --- 웨딩드레스가 아님 → 차단 ---
        if (!data.result?.is_wedding_dress) {
            const message = data.message || '웨딩드레스 이미지가 아니어서 업로드를 차단했습니다.';
            alert(message);
            setStatus(message, true);
            return;
        }

        // --- 웨딩드레스일 때 결과 표시 ---
        renderResult(data.result);
        setStatus('판별을 완료했습니다.');

    } catch (err) {
        console.error(err);
        setStatus(err.message || '처리 중 오류가 발생했습니다.', true);
    } finally {
        btn.disabled = false;
    }
}

function renderResult(result) {
    const section = document.getElementById('result-section');
    const card = document.getElementById('result-card');
    if (!section || !card) return;

    const isDress = !!result.dress;
    const confidence = typeof result.confidence === 'number'
        ? (result.confidence * 100).toFixed(1)
        : '0.0';
    const category = result.category || '-';
    const filename = result.filename || 'uploaded_image';
    const thumb = result.thumbnail || previewDataUrl || '';

    card.innerHTML = `
        <div>
            ${thumb ? `<img src="${thumb}" alt="${filename}">` : ''}
        </div>
        <div>
            <div class="badge ${isDress ? 'green' : 'red'}">
                ${isDress ? '드레스' : '비드레스'}
            </div>
            <div class="metric">파일명: ${filename}</div>
            <div class="metric">신뢰도: ${confidence}%</div>
            <div class="metric">분류: ${category}</div>
            ${result.record_id ? `<div class="metric">기록 ID: ${result.record_id}</div>` : ''}
        </div>
    `;

    section.style.display = 'block';
}

function resetAll() {
    selectedFile = null;
    previewDataUrl = null;
    document.getElementById('file-input').value = '';
    document.getElementById('preview').style.display = 'none';
    document.getElementById('preview').innerHTML = '';
    document.getElementById('result-section').style.display = 'none';
    document.getElementById('status-text').textContent = '';
    document.getElementById('reset-btn').style.display = 'none';
}
