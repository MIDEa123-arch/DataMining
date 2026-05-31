import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import random
import re
import string

# ======== Model chuyên biệt cho Question Generation ========
QG_MODEL_NAME = "valhalla/t5-base-qg-hl"   # Fine-tuned trên SQuAD cho việc sinh câu hỏi
qg_tokenizer = None
qg_model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_qg_model():
    global qg_tokenizer, qg_model
    if qg_model is None:
        print(f"[QG] Loading {QG_MODEL_NAME} on {device}...")
        qg_tokenizer = AutoTokenizer.from_pretrained(QG_MODEL_NAME)
        qg_model = AutoModelForSeq2SeqLM.from_pretrained(QG_MODEL_NAME).to(device)
        print("[QG] Model loaded successfully.")

def qg_generate(prompt, max_length=72):
    inputs = qg_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    with torch.no_grad():
        outputs = qg_model.generate(
            **inputs, max_length=max_length, num_beams=4, early_stopping=True
        )
    return qg_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# ======== Stopwords ========
STOPWORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
    "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
    "about", "against", "between", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
    "t", "can", "will", "just", "don", "should", "now", "also", "would", "could",
    "like", "even", "well", "way", "get", "got", "much", "many", "really", "one",
    "two", "first", "know", "think", "make", "go", "going", "say", "said", "thing",
    "let", "still", "right", "actually", "something", "lot", "kind", "things",
    "want", "need", "see", "look", "come", "take", "tell", "good",
    "every", "back", "may", "might", "must", "shall", "made", "us",
    "pretty", "enough", "yet", "whether", "rather", "perhaps", "however",
])


# ======== NLP: Trích xuất Answer Candidates ========

def get_sentences(text):
    raw = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in raw if 8 <= len(s.strip().split()) <= 40]

def extract_answer_candidates(sentence):
    """Trích xuất các cụm từ quan trọng từ câu, xếp theo mức độ ưu tiên."""
    candidates = []
    seen = set()
    
    def add(text, ctype, priority):
        key = text.lower().strip()
        if key and key not in seen and len(key) > 1:
            seen.add(key)
            candidates.append({"text": text.strip(), "type": ctype, "priority": priority})
    
    # 1. Số liệu / phần trăm / đơn vị đo lường (ưu tiên CAO)
    numbers = re.findall(
        r'\d+(?:\.\d+)?(?:\s*(?:to|and|or|-)\s*\d+(?:\.\d+)?)?(?:\s*%|\s*percent)?'
        r'(?:\s+(?:liters?|percent|glasses?|water|degrees?))?',
        sentence
    )
    for n in numbers:
        n = n.strip()
        if len(n) > 1:
            add(n, "number", 5)
    
    # 2. Thuật ngữ khoa học / kỹ thuật dài (ưu tiên CAO)
    words = sentence.split()
    for w in words:
        clean = w.strip(string.punctuation)
        cl = clean.lower()
        if len(clean) > 8 and cl not in STOPWORDS:
            add(clean, "technical", 4)
    
    # 3. Cụm danh từ 2-4 từ liên tiếp (ưu tiên TRUNG BÌNH-CAO)
    current = []
    for w in words:
        clean = w.strip(string.punctuation)
        cl = clean.lower()
        if cl and cl not in STOPWORDS and len(cl) > 2 and not cl.isdigit():
            current.append(clean)
        else:
            if 2 <= len(current) <= 4:
                phrase = " ".join(current)
                add(phrase, "phrase", 3)
            current = []
    if 2 <= len(current) <= 4:
        add(" ".join(current), "phrase", 3)
    
    # 4. Từ đơn quan trọng, viết hoa hoặc dài > 5 ký tự (ưu tiên THẤP)
    for w in words:
        clean = w.strip(string.punctuation)
        cl = clean.lower()
        if cl not in STOPWORDS and len(clean) > 4:
            p = 2 if clean[0].isupper() else 1
            add(clean, "noun", p)
    
    candidates.sort(key=lambda x: x["priority"], reverse=True)
    return candidates


# ======== Sinh câu hỏi ========

def generate_question_for_answer(answer_text, sentence):
    """Dùng model QG chuyên biệt để sinh câu hỏi.
    Input format: 'generate question: ... <hl> answer <hl> ... context ...'
    """
    # Đánh dấu đáp án trong câu bằng <hl> token
    marked = sentence.replace(answer_text, f"<hl> {answer_text} <hl>", 1)
    prompt = f"generate question: {marked}"
    question = qg_generate(prompt)
    
    if not question or len(question) < 5:
        return None
    if not question.endswith('?'):
        question = question.rstrip('.!,;') + '?'
    return question


def find_distractors(answer_text, answer_type, all_candidates, count=3):
    """Tìm đáp án sai từ các candidate khác trong bài.
    Ưu tiên cùng loại (type) với đáp án đúng."""
    answer_lower = answer_text.lower()
    
    same_type = [c["text"] for c in all_candidates
                 if c["text"].lower() != answer_lower and c["type"] == answer_type]
    diff_type = [c["text"] for c in all_candidates
                 if c["text"].lower() != answer_lower and c["type"] != answer_type]
    
    # Loại trùng lặp
    seen = {answer_lower}
    pool = []
    for d in same_type + diff_type:
        dl = d.lower()
        if dl not in seen and dl not in answer_lower and answer_lower not in dl:
            seen.add(dl)
            pool.append(d)
    
    if len(pool) >= count:
        selected = random.sample(pool[:max(count * 3, len(pool))], min(count, len(pool)))
    else:
        selected = pool[:]
    
    # Nếu vẫn thiếu distractor → bổ sung generic nhưng hợp lý
    fallbacks = [
        "cellular membrane", "chemical reaction", "neural pathway",
        "molecular bond", "thermal energy", "genetic code",
        "organic compound", "metabolic process", "vascular system",
        "respiratory function", "digestive tract", "immune response",
    ]
    random.shuffle(fallbacks)
    for fb in fallbacks:
        if len(selected) >= count:
            break
        if fb.lower() not in seen:
            selected.append(fb)
            seen.add(fb.lower())
    
    return selected[:count]


# ======== Main Entry Point ========

def generate_questions(text: str):
    """Sinh 10 câu MCQ + 10 câu Fill-in-the-blank chất lượng cao."""
    try:
        load_qg_model()
    except Exception as e:
        return {"error": f"Could not load QG model: {e}"}
    
    sentences = get_sentences(text)
    if not sentences:
        return {"error": "Text quá ngắn để sinh câu hỏi."}
    
    # ---- Bước 1: Trích xuất toàn bộ answer candidates ----
    all_candidates = []
    sent_candidates = {}
    for sent in sentences:
        cands = extract_answer_candidates(sent)
        sent_candidates[sent] = cands
        all_candidates.extend(cands)
    
    mcq_list = []
    fib_list = []
    used_mcq = set()
    used_fib = set()
    
    shuffled = list(sentences)
    random.shuffle(shuffled)
    
    # ---- Bước 2: Sinh MCQ ----
    print("[QG] Generating MCQ questions...")
    for sent in shuffled:
        if len(mcq_list) >= 10:
            break
        
        cands = sent_candidates.get(sent, [])
        if not cands:
            continue
        
        # Chọn candidate tốt nhất chưa dùng, có priority >= 2
        # Lọc ra các ứng viên hợp lệ
        valid_cands = [c for c in cands if c["text"].lower() not in used_mcq and c["priority"] >= 2]
        if not valid_cands:
            continue
            
        # Shuffle các candidate có cùng priority cao nhất để tạo sự đa dạng
        max_prio = max(c["priority"] for c in valid_cands)
        top_cands = [c for c in valid_cands if c["priority"] == max_prio]
        chosen = random.choice(top_cands)
        
        answer = chosen["text"]
        used_mcq.add(answer.lower())
        
        # Sinh câu hỏi bằng model QG
        question = generate_question_for_answer(answer, sent)
        if not question:
            continue
        
        # Tìm distractor
        distractors = find_distractors(answer, chosen["type"], all_candidates, count=3)
        if len(distractors) < 3:
            continue
        
        # Lắp ráp MCQ
        options = [answer] + distractors[:3]
        random.shuffle(options)
        correct_idx = options.index(answer)
        correct_letter = chr(65 + correct_idx)
        labeled = [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)]
        
        mcq_list.append({
            "question": question,
            "options": labeled,
            "answer": correct_letter
        })
        print(f"  MCQ {len(mcq_list)}/10: {question[:70]}")
    
    # ---- Bước 3: Sinh Cloze Test (dạng đề thi THPT) ----
    print("[QG] Generating Cloze Test passage...")
    cloze = generate_cloze_test(sentences, sent_candidates, all_candidates)
    
    print(f"[QG] Done! Generated {len(mcq_list)} MCQ + {len(cloze.get('questions',[]))} Cloze blanks.")
    return {
        "multiple_choice": mcq_list,
        "cloze_test": cloze
    }


def generate_cloze_test(sentences, sent_candidates, all_candidates, num_blanks=10):
    """Sinh đề Cloze Test dạng THPT: 1 đoạn văn dài, 10 chỗ trống đánh số, mỗi chỗ có 4 lựa chọn."""
    
    # Chọn đoạn văn dài nhất có thể (lấy các câu liên tiếp)
    # Ưu tiên lấy đoạn liền mạch, đủ dài để chứa >= 10 từ khó
    best_passage_sents = []
    best_score = 0
    
    for start in range(len(sentences)):
        for end in range(start + 3, min(start + 15, len(sentences) + 1)):
            chunk = sentences[start:end]
            # Đếm số candidate có priority >= 3 trong đoạn này
            score = 0
            for s in chunk:
                cands = sent_candidates.get(s, [])
                score += sum(1 for c in cands if c["priority"] >= 3)
            
            word_count = sum(len(s.split()) for s in chunk)
            if score >= num_blanks and 80 <= word_count <= 350 and score > best_score:
                best_score = score
                best_passage_sents = chunk
    
    # Fallback: nếu không tìm đủ, lấy hết
    if not best_passage_sents:
        best_passage_sents = sentences[:min(12, len(sentences))]
    
    # Sắp xếp candidate theo câu để phân bố đều các chỗ trống
    # Nhóm candidate theo từng câu
    cands_by_sent = []
    for s in best_passage_sents:
        sent_cands = [c for c in sent_candidates.get(s, []) if c["priority"] >= 3]
        if sent_cands:
            # Chọn ứng viên tốt nhất trong câu này
            sent_cands.sort(key=lambda x: x["priority"], reverse=True)
            cands_by_sent.append(sent_cands)
            
    selected = []
    used = set()
    
    # Lặp qua từng câu (vòng tròn) để rải đều chỗ trống
    while len(selected) < num_blanks and cands_by_sent:
        added_in_round = False
        for sent_cands in cands_by_sent:
            if len(selected) >= num_blanks:
                break
                
            # Tìm candidate hợp lệ đầu tiên trong câu này
            for i, c in enumerate(sent_cands):
                key = c["text"].lower()
                if key not in used:
                    used.add(key)
                    selected.append(c)
                    sent_cands.pop(i) # Xóa để không chọn lại
                    added_in_round = True
                    break
        
        # Xóa các câu đã hết candidate
        cands_by_sent = [sc for sc in cands_by_sent if sc]
        
        if not added_in_round:
            break # Không còn candidate nào hợp lệ
            
    # Nếu vẫn chưa đủ num_blanks, cố gắng lấy thêm từ passage_candidates (nếu có)
    if len(selected) < num_blanks:
        passage_candidates = []
        for s in best_passage_sents:
            passage_candidates.extend([c for c in sent_candidates.get(s, []) if c["priority"] >= 3])
        passage_candidates.sort(key=lambda x: x["priority"], reverse=True)
        for c in passage_candidates:
            if len(selected) >= num_blanks:
                break
            key = c["text"].lower()
            if key not in used:
                used.add(key)
                selected.append(c)
    
    # Xây dựng đoạn văn với các chỗ trống được đánh số
    passage = " ".join(best_passage_sents)
    questions = []
    
    # Sắp xếp selected theo vị trí xuất hiện trong passage (từ trái sang phải)
    for c in selected:
        pos = passage.lower().find(c["text"].lower())
        c["position"] = pos if pos >= 0 else 9999
    selected.sort(key=lambda x: x["position"])
    
    # Thay thế từng từ bằng blank có đánh số
    offset = 0
    for idx, c in enumerate(selected):
        answer = c["text"]
        blank_num = idx + 1
        blank_marker = f"___({blank_num})___"
        
        # Tìm vị trí chính xác trong passage (có tính offset)
        pos = passage.find(answer, offset)
        if pos == -1:
            # Thử tìm case-insensitive
            lower_passage = passage.lower()
            lower_answer = answer.lower()
            pos = lower_passage.find(lower_answer, offset)
            if pos >= 0:
                answer = passage[pos:pos + len(answer)]
        
        if pos >= 0:
            passage = passage[:pos] + blank_marker + passage[pos + len(answer):]
            offset = pos + len(blank_marker)
            
            # Sinh distractor cho blank này
            distractors = find_distractors(answer, c["type"], all_candidates, count=3)
            
            options = [answer] + distractors[:3]
            random.shuffle(options)
            correct_idx = options.index(answer)
            correct_letter = chr(65 + correct_idx)
            labeled = [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)]
            
            questions.append({
                "number": blank_num,
                "options": labeled,
                "answer": correct_letter
            })
    
    return {
        "passage": passage,
        "questions": questions
    }
