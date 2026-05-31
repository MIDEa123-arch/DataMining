// ===== DOM Elements =====
const homeView = document.getElementById('home-view');
const learningView = document.getElementById('learning-view');
const topicsContainer = document.getElementById('topics-container');
const btnBackHome = document.getElementById('btn-back-home');
const navLogo = document.getElementById('nav-logo');
const learningTitle = document.getElementById('learning-title');
const topicText = document.getElementById('topic-text');

const btnProcess = document.getElementById('btn-process');
const fileInput = document.getElementById('video-upload');
const loading = document.getElementById('loading');
const playerSection = document.getElementById('player-section');
const questionsSection = document.getElementById('questions-section');
const videoPlayer = document.getElementById('video-player');
const cefrText = document.getElementById('cefr-level-text');
const transcriptBox = document.getElementById('transcript-box');
const mcqContent = document.getElementById('mcq-content');
const vocabContent = document.getElementById('vocab-content');

let transcriptData = [];

// ===== Initialization & Navigation =====
document.addEventListener('DOMContentLoaded', loadHomeView);

navLogo.addEventListener('click', showHome);
btnBackHome.addEventListener('click', showHome);

function showHome() {
    learningView.classList.add('hidden');
    homeView.classList.remove('hidden');
    videoPlayer.pause();
    loadHomeView();
}

function showLearningView(title) {
    homeView.classList.add('hidden');
    learningView.classList.remove('hidden');
    learningTitle.textContent = title || "Đang học...";
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ===== Load Videos from API =====
async function loadHomeView() {
    try {
        const res = await fetch('/api/videos');
        if (!res.ok) throw new Error("Failed to fetch videos");
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
        topicsContainer.innerHTML = '<p style="color:var(--text-muted); text-align:center;">Chưa có video nào. Hãy tải lên một video để bắt đầu!</p>';
        return;
    }

    // Group by topic
    const grouped = {};
    videos.forEach(v => {
        const t = v.topic || "Khác";
        if (!grouped[t]) grouped[t] = [];
        grouped[t].push(v);
    });

    for (const [topic, vids] of Object.entries(grouped)) {
        const section = document.createElement('div');
        section.className = 'topic-section';
        
        const title = document.createElement('h3');
        title.className = 'topic-title';
        title.textContent = topic;
        
        const grid = document.createElement('div');
        grid.className = 'video-cards-grid';
        
        vids.forEach(v => {
            const card = document.createElement('div');
            card.className = 'video-card';
            card.innerHTML = `
                <div class="vc-title">${v.filename}</div>
                <div class="vc-meta">
                    <span class="vc-cefr">${v.cefr_level || "A1"}</span>
                    <button class="btn secondary" style="font-size:0.8rem; padding:4px 10px;">Học ngay</button>
                </div>
            `;
            card.addEventListener('click', () => loadVideoDetails(v.id));
            grid.appendChild(card);
        });
        
        section.appendChild(title);
        section.appendChild(grid);
        topicsContainer.appendChild(section);
    }
}

// ===== Load Specific Video =====
async function loadVideoDetails(id) {
    try {
        const res = await fetch(`/api/videos/${id}`);
        if (!res.ok) throw new Error("Failed to fetch video details");
        const data = await res.json();
        
        setupLearningEnvironment(data);
        showLearningView(data.filename);
        
    } catch (e) {
        alert("Lỗi khi tải chi tiết video: " + e.message);
    }
}

function setupLearningEnvironment(data) {
    // Video
    videoPlayer.src = data.video_url;
    
    // CEFR & Topic
    cefrText.textContent = data.cefr_level || '—';
    if(topicText) topicText.textContent = data.topic || '—';

    // Transcript
    transcriptData = data.transcript || [];
    renderTranscript(transcriptData);

    // Questions
    renderQuestions(data.questions || {});
    
    // Vocabulary
    renderVocabulary(data.vocabulary || []);
}

function renderVocabulary(vocabList) {
    if (vocabContent) {
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
            
            // Highlight stressed syllable if possible, or just show IPA
            // eng-to-ipa puts 'ˈ' before the stressed syllable
            let ipaHtml = v.ipa;
            if (v.has_stress) {
                ipaHtml = ipaHtml.replace(/ˈ/g, '<span style="color:var(--primary); font-weight:bold;">ˈ</span>');
            }
            
            card.innerHTML = `
                <div class="vocab-header">
                    <span class="vocab-word">${v.word}</span>
                    <span class="vocab-cefr">${v.cefr}</span>
                </div>
                <div class="vocab-ipa">${ipaHtml}</div>
                <div class="vocab-meaning">${v.meaning}</div>
            `;
            grid.appendChild(card);
        });
        
        vocabContent.appendChild(grid);
    }
}

// ===== Upload & Process =====
btnProcess.addEventListener('click', async () => {
    if (!fileInput.files.length) {
        alert("Vui lòng chọn file video trước.");
        return;
    }

    loading.classList.remove('hidden');
    btnProcess.disabled = true;
    
    const loadingText = document.getElementById('loading-text');

    try {
        let lastData = null;
        for (let i = 0; i < fileInput.files.length; i++) {
            const file = fileInput.files[i];
            
            if (loadingText) {
                loadingText.textContent = `AI đang phân tích Video ${i + 1}/${fileInput.files.length} (${file.name})... Quá trình này có thể mất vài phút.`;
            }
            
            const formData = new FormData();
            formData.append("file", file);

            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                console.error(`Lỗi xử lý video ${file.name}: ${errData.detail || response.status}`);
                alert(`Lỗi xử lý video ${file.name}: ${errData.detail || response.status}`);
                continue;
            }

            lastData = await response.json();
        }
        
        if (fileInput.files.length === 1 && lastData) {
            // Nếu chỉ up 1 video, học ngay
            setupLearningEnvironment(lastData);
            showLearningView(lastData.filename);
        } else {
            // Nếu up nhiều video, tải lại trang chủ
            alert("Đã phân tích xong toàn bộ video!");
            loadHomeView();
        }

    } catch (err) {
        alert("Lỗi xử lý video: " + err.message);
        console.error(err);
    } finally {
        loading.classList.add('hidden');
        if (loadingText) loadingText.textContent = 'AI đang phân tích... Xin chờ một lát.';
        btnProcess.disabled = false;
        fileInput.value = ''; // Reset file input
    }
});

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

        div.innerHTML = `<span class="timestamp">${formatTime(seg.start)}</span>${seg.text}`;

        div.addEventListener('click', () => {
            videoPlayer.currentTime = parseFloat(seg.start);
            videoPlayer.play();
        });

        transcriptBox.appendChild(div);
    });

    // Full text
    const fullTextEl = document.getElementById('full-transcript-text');
    if (fullTextEl) {
        fullTextEl.textContent = segments.map(s => s.text).join(' ');
    }
}

// Full text toggle
document.getElementById('btn-full-transcript').addEventListener('click', () => {
    document.getElementById('full-transcript-section').classList.toggle('hidden');
});

// ===== Sync transcript highlight =====
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

    // Auto-scroll within the transcript box only
    if (activeEl) {
        const boxRect = transcriptBox.getBoundingClientRect();
        const elRect = activeEl.getBoundingClientRect();
        if (elRect.top < boxRect.top || elRect.bottom > boxRect.bottom) {
            activeEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }
});

// ===== Questions =====
function renderQuestions(qData) {
    mcqContent.innerHTML = '';
    const clozeContent = document.getElementById('cloze-content');
    if(clozeContent) clozeContent.innerHTML = '';

    if (qData.error) {
        mcqContent.innerHTML = `<p style="color:var(--danger)">${qData.error}</p>`;
        return;
    }

    // MCQ
    if (qData.multiple_choice && qData.multiple_choice.length) {
        qData.multiple_choice.forEach((q, i) => {
            const qDiv = document.createElement('div');
            qDiv.className = 'question-item';

            let optionsHtml = '';
            q.options.forEach((opt, j) => {
                const uid = `mcq-${i}-${j}`;
                optionsHtml += `
                <label class="option-label" for="${uid}">
                    <input type="radio" name="mcq-${i}" id="${uid}" value="${opt.charAt(0)}">
                    <span>${opt}</span>
                </label>`;
            });

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

    // CLOZE TEST
    if (clozeContent && qData.cloze_test && qData.cloze_test.passage) {
        const cloze = qData.cloze_test;
        
        // Render Passage
        // Format of blank: ___(1)___
        let htmlPassage = cloze.passage.replace(/___\((\d+)\)___/g, '<span class="cloze-blank">($1)</span>');
        
        let questionsHtml = '';
        if (cloze.questions && cloze.questions.length) {
            cloze.questions.forEach((q, i) => {
                let optionsHtml = '';
                q.options.forEach((opt, j) => {
                    const uid = `cloze-${i}-${j}`;
                    optionsHtml += `
                    <label class="option-label" for="${uid}">
                        <input type="radio" name="cloze-${i}" id="${uid}" value="${opt.charAt(0)}">
                        <span>${opt}</span>
                    </label>`;
                });
                
                questionsHtml += `
                <div class="cloze-q-item">
                    <h4>Câu ${q.number}</h4>
                    <div class="options">${optionsHtml}</div>
                    <button class="btn primary" style="font-size:0.8rem; padding: 4px 10px;" onclick="checkMCQ(this, '${q.answer}')">Kiểm tra</button>
                    <div class="answer-reveal" style="font-size:0.8rem; padding: 4px 10px;"></div>
                </div>`;
            });
        }
        
        clozeContent.innerHTML = `
            <div class="cloze-passage">${htmlPassage}</div>
            <div class="cloze-questions">${questionsHtml}</div>
        `;
    } else if (clozeContent) {
        clozeContent.innerHTML = '<p style="color:var(--text-muted)">Không có bài điền từ.</p>';
    }
}

// ===== Check MCQ (sử dụng chung cho cả Reading và Cloze) =====
function checkMCQ(btn, correctAns) {
    const container = btn.parentElement;
    const selected = container.querySelector('input[type="radio"]:checked');
    const reveal = container.querySelector('.answer-reveal');

    if (!selected) {
        alert("Vui lòng chọn 1 đáp án!");
        return;
    }

    reveal.className = 'answer-reveal';
    if (selected.value.toUpperCase() === correctAns.toUpperCase()) {
        reveal.textContent = "✓ Chính xác! Đáp án đúng là: " + correctAns;
        reveal.classList.add('correct');
    } else {
        reveal.textContent = "✗ Chưa đúng. Đáp án đúng là: " + correctAns;
        reveal.classList.add('wrong');
    }
}

// ===== Tabs =====
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(`${btn.dataset.tab}-content`).classList.add('active');
    });
});
