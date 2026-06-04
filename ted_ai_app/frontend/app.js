// ===== DOM Elements =====
const homeView = document.getElementById('home-view');
const learningView = document.getElementById('learning-view');
const topicsContainer = document.getElementById('topics-container');
const btnBackHome = document.getElementById('btn-back-home');
const navLogo = document.getElementById('nav-logo');
const learningTitle = document.getElementById('learning-title');
const topicText = document.getElementById('topic-text');
const btnDeleteCurrent = document.getElementById('btn-delete-current');

const btnProcess = document.getElementById('btn-process');
const fileInput = document.getElementById('video-upload');
const loading = document.getElementById('loading');
const uploadQueue = document.getElementById('upload-queue');
const videoPlayer = document.getElementById('video-player');
const cefrText = document.getElementById('cefr-level-text');
const transcriptBox = document.getElementById('transcript-box');
const mcqContent = document.getElementById('mcq-content');
const vocabContent = document.getElementById('vocab-content');

let transcriptData = [];
let currentVideoId = null;

// ===== Initialization & Navigation =====
document.addEventListener('DOMContentLoaded', loadHomeView);

navLogo.addEventListener('click', showHome);
btnBackHome.addEventListener('click', showHome);
btnDeleteCurrent.addEventListener('click', () => {
    if (currentVideoId) deleteVideo(currentVideoId, learningTitle.textContent, true);
});

function showHome() {
    learningView.classList.add('hidden');
    homeView.classList.remove('hidden');
    videoPlayer.pause();
    loadHomeView();
}

function showLearningView(title) {
    homeView.classList.add('hidden');
    learningView.classList.remove('hidden');
    learningTitle.textContent = title || 'Đang học...';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ===== Load Videos from API =====
async function loadHomeView() {
    try {
        const res = await fetch('/api/videos');
        if (!res.ok) throw new Error('Failed to fetch videos');
        const videos = await res.json();
        renderTopicsGrid(videos);
    } catch (e) {
        console.error(e);
        topicsContainer.innerHTML = '<p style="color:var(--danger)">Không thể tải dữ liệu bài học.</p>';
    }
}

function renderTopicsGrid(videos) {
    topicsContainer.innerHTML = '';
    if (!videos || videos.length === 0) {
        topicsContainer.innerHTML = '<p class="empty-state">Chưa có video nào. Hãy tải lên một hoặc nhiều video để bắt đầu.</p>';
        return;
    }

    const grouped = videos.reduce((acc, video) => {
        const topic = video.topic || 'Khác';
        acc[topic] = acc[topic] || [];
        acc[topic].push(video);
        return acc;
    }, {});

    Object.entries(grouped)
        .sort(([a], [b]) => a.localeCompare(b))
        .forEach(([topic, vids]) => {
            const section = document.createElement('section');
            section.className = 'topic-section';

            const titleRow = document.createElement('div');
            titleRow.className = 'topic-title-row';

            const title = document.createElement('h3');
            title.className = 'topic-title';
            title.textContent = topic;

            const count = document.createElement('span');
            count.className = 'topic-count';
            count.textContent = `${vids.length} video`;

            titleRow.append(title, count);

            const grid = document.createElement('div');
            grid.className = 'video-cards-grid';

            vids.forEach(video => grid.appendChild(createVideoCard(video)));

            section.append(titleRow, grid);
            topicsContainer.appendChild(section);
        });
}

function createVideoCard(video) {
    const card = document.createElement('article');
    card.className = 'video-card';
    card.addEventListener('click', () => loadVideoDetails(video.id));

    const title = document.createElement('div');
    title.className = 'vc-title';
    title.textContent = video.filename;

    const meta = document.createElement('div');
    meta.className = 'vc-meta';

    const cefr = document.createElement('span');
    cefr.className = 'vc-cefr';
    cefr.textContent = video.cefr_level || '—';

    const actions = document.createElement('div');
    actions.className = 'vc-actions';

    const learnBtn = document.createElement('button');
    learnBtn.className = 'btn secondary';
    learnBtn.textContent = 'Học';
    learnBtn.addEventListener('click', event => {
        event.stopPropagation();
        loadVideoDetails(video.id);
    });

    const deleteBtn = document.createElement('button');
    deleteBtn.className = 'btn danger';
    deleteBtn.textContent = 'Xóa';
    deleteBtn.addEventListener('click', event => {
        event.stopPropagation();
        deleteVideo(video.id, video.filename);
    });

    actions.append(learnBtn, deleteBtn);
    meta.append(cefr, actions);
    card.append(title, meta);
    return card;
}

// ===== Load Specific Video =====
async function loadVideoDetails(id) {
    try {
        const res = await fetch(`/api/videos/${id}`);
        if (!res.ok) throw new Error('Failed to fetch video details');
        const data = await res.json();

        setupLearningEnvironment(data);
        showLearningView(data.filename);
    } catch (e) {
        alert('Lỗi khi tải chi tiết video: ' + e.message);
    }
}

function setupLearningEnvironment(data) {
    currentVideoId = data.id || null;
    btnDeleteCurrent.classList.toggle('hidden', !currentVideoId);
    videoPlayer.src = data.video_url;
    cefrText.textContent = data.cefr_level || '—';
    if (topicText) topicText.textContent = data.topic || '—';

    transcriptData = data.transcript || [];
    renderTranscript(transcriptData);
    renderQuestions(data.questions || {});
    renderVocabulary(data.vocabulary || []);
}

async function deleteVideo(id, filename, returnHome = false) {
    const confirmed = confirm(`Xóa video "${filename}" khỏi hệ thống? File lưu trữ và dữ liệu học tập sẽ bị xóa.`);
    if (!confirmed) return;

    try {
        const res = await fetch(`/api/videos/${id}`, { method: 'DELETE' });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || 'Không thể xóa video');
        }

        if (returnHome) {
            currentVideoId = null;
            showHome();
        } else {
            await loadHomeView();
        }
    } catch (e) {
        alert('Lỗi khi xóa video: ' + e.message);
    }
}

// ===== Upload & Process =====
fileInput.addEventListener('change', () => renderUploadQueue([...fileInput.files]));

btnProcess.addEventListener('click', async () => {
    const files = [...fileInput.files];
    if (!files.length) {
        alert('Vui lòng chọn ít nhất một file video.');
        return;
    }

    loading.classList.remove('hidden');
    btnProcess.disabled = true;
    fileInput.disabled = true;

    const loadingText = document.getElementById('loading-text');
    const results = [];
    const failures = [];

    try {
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            updateQueueItem(i, 'processing', 'Đang xử lý');
            loadingText.textContent = `Đang xử lý ${i + 1}/${files.length}: ${file.name}`;

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData,
                });

                if (!response.ok) {
                    const errData = await response.json().catch(() => ({}));
                    throw new Error(errData.detail || `HTTP ${response.status}`);
                }

                const data = await response.json();
                results.push(data);
                updateQueueItem(i, 'done', `${data.topic || 'Khác'} · ${data.cefr_level || '—'}`);
            } catch (err) {
                failures.push({ file: file.name, error: err.message });
                updateQueueItem(i, 'error', err.message);
            }
        }

        await loadHomeView();

        if (files.length === 1 && results.length === 1 && failures.length === 0) {
            setupLearningEnvironment(results[0]);
            showLearningView(results[0].filename);
            return;
        }

        if (failures.length) {
            alert(`Đã xử lý xong ${results.length}/${files.length} video. Có ${failures.length} video lỗi, xem trạng thái trong hàng đợi.`);
        } else {
            alert('Đã phân tích xong toàn bộ video. Danh sách bên dưới đã được nhóm theo chủ đề.');
        }
    } catch (err) {
        alert('Lỗi xử lý video: ' + err.message);
        console.error(err);
    } finally {
        loading.classList.add('hidden');
        loadingText.textContent = 'AI đang phân tích... Xin chờ một lát.';
        btnProcess.disabled = false;
        fileInput.disabled = false;
        fileInput.value = '';
    }
});

function renderUploadQueue(files) {
    uploadQueue.innerHTML = '';
    uploadQueue.classList.toggle('hidden', files.length === 0);
    files.forEach((file, index) => {
        const item = document.createElement('div');
        item.className = 'queue-item';
        item.dataset.index = index;

        const name = document.createElement('span');
        name.className = 'queue-name';
        name.textContent = file.name;

        const status = document.createElement('span');
        status.className = 'queue-status waiting';
        status.textContent = 'Chờ xử lý';

        item.append(name, status);
        uploadQueue.appendChild(item);
    });
}

function updateQueueItem(index, state, text) {
    const item = uploadQueue.querySelector(`[data-index="${index}"]`);
    if (!item) return;
    const status = item.querySelector('.queue-status');
    status.className = `queue-status ${state}`;
    status.textContent = text;
}

// ===== Transcript =====
function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

function renderTranscript(segments) {
    transcriptBox.innerHTML = '';
    segments.forEach((seg, index) => {
        const div = document.createElement('div');
        div.className = 'transcript-line';
        div.dataset.start = seg.start;
        div.dataset.end = seg.end;
        div.dataset.index = index;

        const time = document.createElement('span');
        time.className = 'timestamp';
        time.textContent = formatTime(seg.start);
        div.append(time, document.createTextNode(seg.text));

        div.addEventListener('click', () => {
            videoPlayer.currentTime = parseFloat(seg.start);
            videoPlayer.play();
        });

        transcriptBox.appendChild(div);
    });

    const fullTextEl = document.getElementById('full-transcript-text');
    if (fullTextEl) {
        fullTextEl.textContent = segments.map(s => s.text).join(' ');
    }
}

document.getElementById('btn-full-transcript').addEventListener('click', () => {
    document.getElementById('full-transcript-section').classList.toggle('hidden');
});

videoPlayer.addEventListener('timeupdate', () => {
    const t = videoPlayer.currentTime;
    const lines = transcriptBox.querySelectorAll('.transcript-line');
    let activeEl = null;

    lines.forEach(line => {
        const start = parseFloat(line.dataset.start);
        const end = parseFloat(line.dataset.end);
        if (t >= start && t <= end) {
            line.classList.add('active');
            activeEl = line;
        } else {
            line.classList.remove('active');
        }
    });

    if (activeEl) {
        const boxRect = transcriptBox.getBoundingClientRect();
        const elRect = activeEl.getBoundingClientRect();
        if (elRect.top < boxRect.top || elRect.bottom > boxRect.bottom) {
            activeEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }
});

// ===== Questions =====
function renderVocabulary(vocabList) {
    vocabContent.innerHTML = '';
    if (!vocabList || vocabList.length === 0) {
        vocabContent.innerHTML = '<p style="color:var(--text-muted)">Không có từ vựng nổi bật.</p>';
        return;
    }

    const grid = document.createElement('div');
    grid.className = 'vocab-grid';

    vocabList.forEach(v => {
        const card = document.createElement('div');
        card.className = 'vocab-card';

        const ipaHtml = (v.ipa || '').replace(/Ëˆ|ˈ/g, '<span style="color:var(--primary); font-weight:bold;">ˈ</span>');
        card.innerHTML = `
            <div class="vocab-header">
                <span class="vocab-word">${v.word || ''}</span>
                <span class="vocab-cefr">${v.cefr || '—'}</span>
            </div>
            <div class="vocab-ipa">${ipaHtml}</div>
            <div class="vocab-meaning">${v.meaning || ''}</div>
        `;
        grid.appendChild(card);
    });

    vocabContent.appendChild(grid);
}

function renderQuestions(qData) {
    mcqContent.innerHTML = '';
    const clozeContent = document.getElementById('cloze-content');
    if (clozeContent) clozeContent.innerHTML = '';

    if (qData.error) {
        mcqContent.innerHTML = `<p style="color:var(--danger)">${qData.error}</p>`;
        return;
    }

    if (qData.multiple_choice && qData.multiple_choice.length) {
        qData.multiple_choice.forEach((q, i) => {
            const qDiv = document.createElement('div');
            qDiv.className = 'question-item';

            const optionsHtml = q.options.map((opt, j) => {
                const uid = `mcq-${i}-${j}`;
                return `
                    <label class="option-label" for="${uid}">
                        <input type="radio" name="mcq-${i}" id="${uid}" value="${opt.charAt(0)}">
                        <span>${opt}</span>
                    </label>`;
            }).join('');

            qDiv.innerHTML = `
                <h4>Q${i + 1}. ${q.question}</h4>
                <div class="options">${optionsHtml}</div>
                <button class="btn primary" onclick="checkMCQ(this, '${q.answer}')">Kiểm tra đáp án</button>
                <div class="answer-reveal"></div>
            `;
            mcqContent.appendChild(qDiv);
        });
    } else {
        mcqContent.innerHTML = '<p style="color:var(--text-muted)">Không có câu hỏi trắc nghiệm.</p>';
    }

    if (clozeContent && qData.cloze_test && qData.cloze_test.passage) {
        const cloze = qData.cloze_test;
        const htmlPassage = cloze.passage.replace(/___\((\d+)\)___/g, '<span class="cloze-blank">($1)</span>');
        const questionsHtml = (cloze.questions || []).map((q, i) => {
            const optionsHtml = q.options.map((opt, j) => {
                const uid = `cloze-${i}-${j}`;
                return `
                    <label class="option-label" for="${uid}">
                        <input type="radio" name="cloze-${i}" id="${uid}" value="${opt.charAt(0)}">
                        <span>${opt}</span>
                    </label>`;
            }).join('');

            return `
                <div class="cloze-q-item">
                    <h4>Câu ${q.number}</h4>
                    <div class="options">${optionsHtml}</div>
                    <button class="btn primary" style="font-size:0.8rem; padding: 4px 10px;" onclick="checkMCQ(this, '${q.answer}')">Kiểm tra</button>
                    <div class="answer-reveal" style="font-size:0.8rem; padding: 4px 10px;"></div>
                </div>`;
        }).join('');

        clozeContent.innerHTML = `
            <div class="cloze-passage">${htmlPassage}</div>
            <div class="cloze-questions">${questionsHtml}</div>
        `;
    } else if (clozeContent) {
        clozeContent.innerHTML = '<p style="color:var(--text-muted)">Không có bài điền từ.</p>';
    }
}

window.checkMCQ = function checkMCQ(btn, correctAns) {
    const container = btn.parentElement;
    const selected = container.querySelector('input[type="radio"]:checked');
    const reveal = container.querySelector('.answer-reveal');

    if (!selected) {
        alert('Vui lòng chọn 1 đáp án!');
        return;
    }

    reveal.className = 'answer-reveal';
    if (selected.value.toUpperCase() === correctAns.toUpperCase()) {
        reveal.textContent = '✓ Chính xác! Đáp án đúng là: ' + correctAns;
        reveal.classList.add('correct');
    } else {
        reveal.textContent = '✗ Chưa đúng. Đáp án đúng là: ' + correctAns;
        reveal.classList.add('wrong');
    }
};

// ===== Tabs =====
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(`${btn.dataset.tab}-content`).classList.add('active');
    });
});
