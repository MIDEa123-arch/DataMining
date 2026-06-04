import re
import random
import string
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# =========================================================
# QUESTION SERVICE - NO TEMPLATE RELAXED
#
# Mục tiêu:
# - Không dùng template sinh câu hỏi kiểu "X is Y -> What is X?"
# - T5 vẫn là model sinh câu hỏi chính
# - spaCy chỉ dùng để chọn answer candidate tốt
# - SBERT dùng để lấy context liên quan
# - Có QA validation nhưng mặc định TẮT vì QA validator thường loại quá nhiều câu
# - Không ép đủ 10 câu, nhưng đã nới lỏng để không bị trả rỗng
# =========================================================

QG_MODEL_NAME = "valhalla/t5-base-qg-hl"
QA_MODEL_NAME = "deepset/roberta-base-squad2"
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SPACY_MODEL_NAME = "en_core_web_sm"

device = "cuda" if torch.cuda.is_available() else "cpu"

qg_tokenizer = None
qg_model = None

qa_pipeline = None
qa_attempted = False

sbert_model = None
sbert_attempted = False

spacy_nlp = None
spacy_attempted = False


# =========================================================
# CONFIG
# =========================================================

MAX_MCQ = 10
MAX_CLOZE_BLANKS = 10

# QUAN TRỌNG:
# True  = lọc rất gắt, dễ không sinh ra câu hỏi
# False = chỉ dùng rule validation nhẹ, dễ có câu hỏi hơn
USE_QA_VALIDATION = False

MAX_SENTENCES_FOR_QG = 120
MAX_CONTEXT_SENTENCES = 5
MAX_CONTEXT_WORDS = 110
SIMILARITY_THRESHOLD = 0.45

# Tránh output toàn một dạng câu hỏi
MAX_SAME_PREFIX = {
    "what is": 3,
    "what are": 3,
    "what does": 3,
    "what do": 3,
    "what can": 3,
    "how": 3,
    "why": 3,
    "when": 3,
    "where": 3,
    "who": 3,
    "which": 3,
    "other": 4,
}


# =========================================================
# GENERAL FILTERS
# =========================================================

STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as",
    "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "can", "will", "just", "should", "now", "also", "would", "could",
    "may", "might", "must", "shall"
}

GENERIC_BAD_WORDS = {
    "basically", "actually", "really", "pretty", "simply", "clearly",
    "firstly", "secondly", "thirdly", "finally", "easily", "equally",
    "closely", "perfectly", "probably", "perhaps", "maybe", "rather",
    "around", "within", "without", "before", "after", "during",
    "particular", "different", "important", "possible", "actual",
    "shown", "said", "saying", "think", "thinking", "going",
    "look", "looks", "make", "makes", "made", "take", "takes",
    "give", "gives", "got", "getting", "come", "comes",
    "example", "thing", "things", "something", "anything",
    "someone", "everyone", "people", "person", "way", "kind",
    "lot", "much", "many", "right", "wrong", "good", "bad",
    "yes", "no", "please", "thanks", "watching", "subscribe",
    "video", "future", "line", "below"
}

BAD_ANSWER_STARTS = {
    "called", "using", "used", "use", "perform", "performs", "performed",
    "determine", "determining", "looking", "look", "looks", "move",
    "moving", "say", "saying", "think", "thinking", "let", "lets",
    "let's", "would", "could", "should", "might", "maybe", "perhaps"
}

BAD_ANSWER_ENDS = {
    "like", "of", "to", "in", "on", "at", "by", "with", "for",
    "whether", "that", "which", "who", "what"
}

ACRONYM_RE = re.compile(r"^[A-Z][A-Z0-9]{1,9}$")


# =========================================================
# LOADERS
# =========================================================

def load_qg_model():
    global qg_tokenizer, qg_model
    if qg_model is None:
        print(f"[QG] Loading {QG_MODEL_NAME} on {device}...")
        qg_tokenizer = AutoTokenizer.from_pretrained(QG_MODEL_NAME)
        qg_model = AutoModelForSeq2SeqLM.from_pretrained(QG_MODEL_NAME).to(device)
        print("[QG] T5 loaded.")


def load_qa_model():
    global qa_pipeline, qa_attempted
    if qa_attempted:
        return qa_pipeline

    qa_attempted = True
    if not USE_QA_VALIDATION:
        qa_pipeline = None
        return None

    try:
        device_id = 0 if device == "cuda" else -1
        print(f"[QG] Loading QA validator {QA_MODEL_NAME} on {device}...")
        qa_pipeline = pipeline(
            "question-answering",
            model=QA_MODEL_NAME,
            tokenizer=QA_MODEL_NAME,
            device=device_id,
        )
        print("[QG] QA validator loaded.")
    except Exception as e:
        print(f"[QG] QA validator unavailable: {e}")
        qa_pipeline = None

    return qa_pipeline


def load_sbert_model():
    global sbert_model, sbert_attempted
    if sbert_attempted:
        return sbert_model

    sbert_attempted = True
    try:
        from sentence_transformers import SentenceTransformer
        print(f"[QG] Loading SBERT {SBERT_MODEL_NAME}...")
        sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
        print("[QG] SBERT loaded.")
    except Exception as e:
        print(f"[QG] SBERT unavailable, fallback TF-IDF: {e}")
        sbert_model = None

    return sbert_model


def load_spacy_model():
    global spacy_nlp, spacy_attempted
    if spacy_attempted:
        return spacy_nlp

    spacy_attempted = True
    try:
        import spacy
        print(f"[QG] Loading spaCy {SPACY_MODEL_NAME}...")
        spacy_nlp = spacy.load(SPACY_MODEL_NAME)
        print("[QG] spaCy loaded.")
    except Exception as e:
        print(f"[QG] spaCy unavailable, fallback regex: {e}")
        spacy_nlp = None

    return spacy_nlp


# =========================================================
# BASIC UTILS
# =========================================================

def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def clean_text(text: str) -> str:
    text = normalize_spaces(text)
    text = text.strip(string.punctuation + "“”‘’`")
    return normalize_spaces(text)


def is_acronym(text: str) -> bool:
    return bool(ACRONYM_RE.fullmatch(text.strip()))


def lexical_tokens(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9]+", text.lower())
    return [t for t in toks if t not in STOPWORDS and t not in GENERIC_BAD_WORDS]


def is_boilerplate_sentence(sentence: str) -> bool:
    s = sentence.lower()
    patterns = [
        "please like",
        "like and subscribe",
        "thanks for watching",
        "thank you for watching",
        "drop us a line",
        "see more videos",
        "if you have any questions",
    ]
    return any(p in s for p in patterns)


def strip_bad_answer_prefix(text: str) -> str:
    text = clean_text(text)

    text = re.sub(
        r"^(called|using|used|use|perform|performs|performed|to perform|"
        r"look for|looking for|determine|determining|move across|"
        r"tasks like|task like|like)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^(tasks?|things?)\s+like\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(the\s+)?presence\s+of\s+", "", text, flags=re.IGNORECASE)

    return clean_text(text)


def question_prefix(question: str) -> str:
    q = normalize_spaces(question).lower()
    if q.startswith("what is"):
        return "what is"
    if q.startswith("what are"):
        return "what are"
    if q.startswith("what does"):
        return "what does"
    if q.startswith("what do"):
        return "what do"
    if q.startswith("what can"):
        return "what can"
    for w in ["how", "why", "when", "where", "who", "which"]:
        if q.startswith(w):
            return w
    return "other"


# =========================================================
# SENTENCE SPLITTING
# =========================================================

def split_sentences(text: str) -> List[str]:
    text = normalize_spaces(text)
    if not text:
        return []

    nlp = load_spacy_model()
    if nlp is not None:
        doc = nlp(text)
        raw = [normalize_spaces(s.text) for s in doc.sents]
    else:
        raw = [normalize_spaces(s) for s in re.split(r"(?<=[.!?])\s+", text)]

    sentences = []
    for s in raw:
        words = s.split()
        if not (6 <= len(words) <= 70):
            continue
        if is_boilerplate_sentence(s):
            continue
        sentences.append(s)

    return sentences[:MAX_SENTENCES_FOR_QG]


# =========================================================
# ANSWER CANDIDATES
# =========================================================

def is_good_answer(text: str, ctype: str = "", span: Any = None, allow_single: bool = False) -> bool:
    text = strip_bad_answer_prefix(text)
    if not text or len(text) < 2:
        return False

    lower = text.lower()
    words = lower.split()
    toks = lexical_tokens(text)

    if len(words) > 8:
        return False
    if lower in STOPWORDS or lower in GENERIC_BAD_WORDS:
        return False
    if not toks:
        return False
    if words[0] in BAD_ANSWER_STARTS:
        return False
    if words[-1] in BAD_ANSWER_ENDS:
        return False
    if len(words) == 1 and lower.endswith("ly"):
        return False

    if ctype.startswith("entity:"):
        return True
    if ctype == "number":
        return True
    if is_acronym(text):
        return True

    if span is not None:
        root_pos = getattr(span.root, "pos_", "")
        if root_pos in {"ADV", "VERB", "AUX", "PRON", "DET", "ADP", "CCONJ", "SCONJ", "INTJ", "PART"}:
            return False
        has_noun = any(getattr(tok, "pos_", "") in {"NOUN", "PROPN", "NUM"} for tok in span)
        if not has_noun:
            return False

    # Ưu tiên cụm 2-7 từ
    if 2 <= len(words) <= 7:
        return True

    # Nới lỏng single noun/entity/term để không bị rỗng
    if len(words) == 1:
        return allow_single and len(lower) >= 4 and lower not in GENERIC_BAD_WORDS

    return False


def candidate_priority(text: str, ctype: str) -> int:
    if ctype.startswith("entity:PERSON"):
        return 10
    if ctype.startswith("entity:ORG") or ctype.startswith("entity:PRODUCT") or ctype.startswith("entity:EVENT"):
        return 9
    if ctype.startswith("entity:"):
        return 8
    if ctype == "number":
        return 8
    if ctype == "noun_chunk":
        return 7
    if ctype == "phrase":
        return 5
    if ctype == "acronym":
        return 6
    if ctype == "single_term":
        return 3
    return 1


def add_candidate(candidates: List[Dict[str, Any]], seen: set, text: str, ctype: str, sent_idx: int, span: Any = None, allow_single: bool = False):
    text = strip_bad_answer_prefix(text)
    key = text.lower()

    if key in seen:
        return
    if not is_good_answer(text, ctype, span, allow_single=allow_single):
        return

    seen.add(key)
    candidates.append({
        "text": text,
        "type": ctype,
        "priority": candidate_priority(text, ctype),
        "sentence_index": sent_idx,
    })


def add_fallback_phrases(sentence: str, sent_idx: int, candidates: List[Dict[str, Any]], seen: set):
    # Fallback tổng quát: cụm từ liên tiếp không phải stopword.
    words = sentence.split()
    current = []

    for w in words:
        clean = clean_text(w)
        low = clean.lower()

        if (
            clean
            and low not in STOPWORDS
            and low not in GENERIC_BAD_WORDS
            and len(low) > 2
            and not re.fullmatch(r"\d+", low)
        ):
            current.append(clean)
        else:
            if 2 <= len(current) <= 6:
                add_candidate(candidates, seen, " ".join(current), "phrase", sent_idx)
            current = []

    if 2 <= len(current) <= 6:
        add_candidate(candidates, seen, " ".join(current), "phrase", sent_idx)


def extract_candidates(sentence: str, sent_idx: int) -> List[Dict[str, Any]]:
    candidates = []
    seen = set()
    nlp = load_spacy_model()

    if nlp is not None:
        doc = nlp(sentence)

        entity_labels = {
            "PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT", "EVENT",
            "WORK_OF_ART", "DATE", "TIME", "MONEY", "PERCENT",
            "QUANTITY", "CARDINAL", "NORP", "LAW"
        }

        for ent in doc.ents:
            if ent.label_ in entity_labels:
                add_candidate(candidates, seen, ent.text, f"entity:{ent.label_}", sent_idx, ent)

        for chunk in doc.noun_chunks:
            text = clean_text(chunk.text)
            text = re.sub(r"^(a|an|the|this|that|these|those|our|your|my|his|her|their)\s+", "", text, flags=re.I)
            add_candidate(candidates, seen, text, "noun_chunk", sent_idx, chunk)

        for tok in doc:
            if tok.pos_ in {"NOUN", "PROPN"} and not tok.is_stop and tok.is_alpha:
                add_candidate(candidates, seen, tok.text, "single_term", sent_idx, tok, allow_single=True)

    # fallback phrase luôn chạy để tránh thiếu candidate
    add_fallback_phrases(sentence, sent_idx, candidates, seen)

    # number patterns tổng quát
    number_patterns = [
        r"\b\d+\s*(?:x|by)\s*\d+\b",
        r"\b\d+(?:\.\d+)?(?:\s*(?:%|percent|percentage))\b",
        r"\b\d+(?:\.\d+)?\s+(?:years?|days?|hours?|minutes?|seconds?|degrees?|pixels?|layers?|blocks?|filters?|columns?|rows?|times)\b",
        r"\b\d+(?:\.\d+)?\b",
    ]
    for pat in number_patterns:
        for match in re.findall(pat, sentence, flags=re.I):
            add_candidate(candidates, seen, match, "number", sent_idx)

    candidates.sort(key=lambda c: (c["priority"], len(c["text"].split())), reverse=True)
    return candidates


def collect_candidates(sentences: List[str]) -> Tuple[List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
    all_items = []
    by_sent = {}

    for i, sent in enumerate(sentences):
        items = extract_candidates(sent, i)
        by_sent[i] = items
        all_items.extend(items)

    unique = []
    seen = set()
    for item in sorted(all_items, key=lambda x: x.get("priority", 0), reverse=True):
        text = strip_bad_answer_prefix(item["text"])
        key = text.lower()
        if key not in seen and is_good_answer(text, item["type"], allow_single=True):
            item = dict(item)
            item["text"] = text
            seen.add(key)
            unique.append(item)

    return unique, by_sent


# =========================================================
# SBERT CONTEXT
# =========================================================

def fallback_similarity_matrix(sentences: List[str]):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(stop_words="english").fit_transform(sentences)
        return (vec @ vec.T).toarray()
    except Exception:
        n = len(sentences)
        return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def build_sentence_similarity_matrix(sentences: List[str]):
    if len(sentences) <= 1:
        return [[1.0]]

    model = load_sbert_model()
    if model is None:
        return fallback_similarity_matrix(sentences)

    try:
        emb = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
        return emb @ emb.T
    except Exception:
        return fallback_similarity_matrix(sentences)


def build_context(sentences: List[str], main_idx: int, sim_matrix) -> str:
    scored = []
    for i in range(len(sentences)):
        if i == main_idx:
            continue
        score = float(sim_matrix[main_idx][i])
        if score >= SIMILARITY_THRESHOLD:
            scored.append((i, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    selected = [main_idx] + [i for i, _ in scored[:MAX_CONTEXT_SENTENCES - 1]]

    for i in (main_idx - 1, main_idx + 1):
        if len(selected) >= 3:
            break
        if 0 <= i < len(sentences) and i not in selected:
            selected.append(i)

    selected = sorted(set(selected))

    keep = []
    total = 0
    for i in selected:
        wc = len(sentences[i].split())
        if keep and total + wc > MAX_CONTEXT_WORDS:
            continue
        keep.append(i)
        total += wc

    return " ".join(sentences[i] for i in keep)


# =========================================================
# T5 QUESTION GENERATION
# =========================================================

def mark_answer(context: str, sentence: str, answer: str) -> Optional[str]:
    answer = clean_text(answer)
    if not answer:
        return None

    pat = re.compile(re.escape(answer), flags=re.I)
    m = pat.search(context)
    if m:
        return context[:m.start()] + f"<hl> {context[m.start():m.end()]} <hl>" + context[m.end():]

    m = pat.search(sentence)
    if m:
        marked_sent = sentence[:m.start()] + f"<hl> {sentence[m.start():m.end()]} <hl>" + sentence[m.end():]
        return context.replace(sentence, marked_sent, 1) if sentence in context else marked_sent

    return None


def t5_generate_question(answer: str, sentence: str, context: str) -> Optional[str]:
    load_qg_model()

    marked = mark_answer(context, sentence, answer)
    if not marked:
        return None

    prompt = f"generate question: {marked}"
    inputs = qg_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)

    with torch.no_grad():
        outputs = qg_model.generate(
            **inputs,
            max_length=72,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    q = qg_tokenizer.decode(outputs[0], skip_special_tokens=True)
    q = normalize_spaces(q)

    if not q or len(q.split()) < 5:
        return None

    if not q.endswith("?"):
        q = q.rstrip(".!,;:") + "?"

    return q


# =========================================================
# VALIDATION
# =========================================================

def answer_overlap(expected: str, predicted: str) -> bool:
    e = clean_text(expected).lower()
    p = clean_text(predicted).lower()

    if not e or not p:
        return False
    if e in p or p in e:
        return True

    et = set(lexical_tokens(e))
    pt = set(lexical_tokens(p))

    if not et or not pt:
        return False

    return len(et & pt) / max(len(et), 1) >= 0.45


def is_good_question(question: str, answer: str) -> bool:
    q = normalize_spaces(question).lower()
    a = clean_text(answer).lower()

    if not q.endswith("?"):
        return False
    if len(q.split()) < 5:
        return False

    # Nếu câu hỏi chứa nguyên đáp án thì bỏ
    if a and a in q:
        return False

    # Chặn một số dạng cực kỳ mơ hồ
    bad_fragments = [
        "what kind of shapes do you think",
        "when you think of",
        "how could",
        "what can we do within",
        "what is the first group",
        "what do we do to the right",
        "how many layers does",
        "how would",
    ]
    if any(x in q for x in bad_fragments):
        return False

    return True


def validate_question(question: str, answer: str, context: str) -> bool:
    if not is_good_question(question, answer):
        return False

    qa = load_qa_model()
    if qa is None:
        return True

    try:
        result = qa(question=question, context=context)
        predicted = result.get("answer", "")
        score = float(result.get("score", 0.0))
    except Exception:
        return True

    if score < 0.08:
        return False

    return answer_overlap(answer, predicted)


# =========================================================
# DISTRACTORS
# =========================================================

def item_category(item: Dict[str, Any]) -> str:
    t = item.get("type", "")
    if t.startswith("entity:"):
        return "entity"
    if t == "number":
        return "number"
    if t == "noun_chunk" or t == "phrase":
        return "phrase"
    if t in {"acronym", "single_term"}:
        return "term"
    return "other"


def option_is_valid(text: str) -> bool:
    return is_good_answer(text, "phrase", allow_single=True)


def sbert_rerank_distractors(answer: str, pool: List[str], count: int = 3) -> List[str]:
    if len(pool) <= count:
        return pool[:count]

    model = load_sbert_model()
    if model is None:
        return pool[:count]

    try:
        texts = [answer] + pool
        emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        sims = emb[0] @ emb[1:].T

        ranked = []
        for d, s in zip(pool, sims):
            s = float(s)
            if 0.03 <= s <= 0.92:
                ranked.append((d, s))

        ranked.sort(key=lambda x: x[1], reverse=True)
        selected = [d for d, _ in ranked[:count]]

        for d in pool:
            if len(selected) >= count:
                break
            if d not in selected:
                selected.append(d)

        return selected[:count]
    except Exception:
        return pool[:count]


def find_distractors(answer_item: Dict[str, Any], all_items: List[Dict[str, Any]], count: int = 3) -> List[str]:
    ans = answer_item["text"]
    ans_low = ans.lower()
    ans_cat = item_category(answer_item)

    pool_same = []
    pool_any = []
    seen = {ans_low}

    sorted_items = sorted(
        all_items,
        key=lambda x: (
            x.get("priority", 0),
            len(x["text"].split())
        ),
        reverse=True,
    )

    for item in sorted_items:
        text = strip_bad_answer_prefix(item["text"])
        low = text.lower()

        if low in seen:
            continue
        if ans_low in low or low in ans_low:
            continue
        if not option_is_valid(text):
            continue

        seen.add(low)
        if item_category(item) == ans_cat:
            pool_same.append(text)
        else:
            pool_any.append(text)

    pool = pool_same + pool_any

    # Không dùng fallback generic ngoài bài, nhưng nới lỏng lấy candidate bất kỳ trong bài.
    return sbert_rerank_distractors(ans, pool, count=count)


# =========================================================
# MCQ GENERATION
# =========================================================

def build_mcq(question: str, answer_item: Dict[str, Any], all_items: List[Dict[str, Any]], context: str) -> Optional[Dict[str, Any]]:
    answer = strip_bad_answer_prefix(answer_item["text"])

    if not validate_question(question, answer, context):
        return None

    distractors = find_distractors(answer_item, all_items, count=3)
    if len(distractors) < 3:
        return None

    options = [answer] + distractors[:3]

    uniq = []
    seen = set()
    for opt in options:
        opt = strip_bad_answer_prefix(opt)
        low = opt.lower()
        if low not in seen:
            seen.add(low)
            uniq.append(opt)

    if len(uniq) < 4:
        return None

    random.shuffle(uniq)
    correct_idx = uniq.index(answer)

    return {
        "question": question,
        "options": [f"{chr(65+i)}. {opt}" for i, opt in enumerate(uniq)],
        "answer": chr(65 + correct_idx),
        "answer_text": answer,
        "context": context,
    }


def generate_mcq_questions(text: str, max_questions: int = MAX_MCQ) -> List[Dict[str, Any]]:
    load_qg_model()

    sentences = split_sentences(text)
    if not sentences:
        return []

    sim_matrix = build_sentence_similarity_matrix(sentences)
    all_items, by_sent = collect_candidates(sentences)

    print(f"[QG] sentences={len(sentences)}, candidates={len(all_items)}")

    results = []
    used_answers = set()
    used_questions = set()
    prefix_counts: Dict[str, int] = {}

    candidates = sorted(
        all_items,
        key=lambda x: (
            len(x["text"].split()) >= 2,
            x.get("priority", 0),
            len(x["text"].split())
        ),
        reverse=True,
    )

    attempts = 0
    rejected = 0

    for cand in candidates:
        if len(results) >= max_questions:
            break

        attempts += 1
        answer = strip_bad_answer_prefix(cand["text"])
        akey = answer.lower()

        if akey in used_answers:
            continue

        sent_idx = cand["sentence_index"]
        if not (0 <= sent_idx < len(sentences)):
            continue

        sentence = sentences[sent_idx]
        context = build_context(sentences, sent_idx, sim_matrix)

        question = t5_generate_question(answer, sentence, context)
        if not question:
            rejected += 1
            continue

        qkey = question.lower()
        if qkey in used_questions:
            rejected += 1
            continue

        pref = question_prefix(question)
        if prefix_counts.get(pref, 0) >= MAX_SAME_PREFIX.get(pref, 3):
            rejected += 1
            continue

        mcq = build_mcq(question, cand, all_items, context)
        if mcq:
            results.append(mcq)
            used_answers.add(akey)
            used_questions.add(qkey)
            prefix_counts[pref] = prefix_counts.get(pref, 0) + 1
            print(f"[QG] MCQ {len(results)}: {question} -> {answer}")
        else:
            rejected += 1

    print(f"[QG] attempts={attempts}, accepted={len(results)}, rejected={rejected}")
    return results


# =========================================================
# CLOZE TEST
# =========================================================

def generate_cloze_test(text: str, num_blanks: int = MAX_CLOZE_BLANKS) -> Dict[str, Any]:
    sentences = split_sentences(text)
    if not sentences:
        return {"passage": "", "questions": []}

    all_items, by_sent = collect_candidates(sentences)

    best = (0, min(8, len(sentences)), -1)

    for start in range(len(sentences)):
        for end in range(start + 3, min(start + 14, len(sentences)) + 1):
            wc = sum(len(sentences[i].split()) for i in range(start, end))
            if not (70 <= wc <= 340):
                continue

            score = 0
            for item in all_items:
                if start <= item["sentence_index"] < end:
                    score += item.get("priority", 0)

            if score > best[2]:
                best = (start, end, score)

    start, end, _ = best
    passage = " ".join(sentences[start:end])

    passage_items = []
    for item in all_items:
        if start <= item["sentence_index"] < end:
            if item["type"] == "single_term":
                continue
            pos = passage.lower().find(item["text"].lower())
            if pos >= 0:
                copied = dict(item)
                copied["_pos"] = pos
                passage_items.append(copied)

    passage_items.sort(key=lambda x: (x["_pos"], -x.get("priority", 0)))

    selected = []
    used = set()
    for item in passage_items:
        if len(selected) >= num_blanks:
            break
        key = item["text"].lower()
        if key not in used:
            used.add(key)
            selected.append(item)

    questions = []
    offset = 0

    for idx, item in enumerate(selected):
        answer = strip_bad_answer_prefix(item["text"])
        pos = passage.lower().find(answer.lower(), offset)
        if pos < 0:
            continue

        real_answer = passage[pos:pos + len(answer)]
        marker = f"___({idx + 1})___"
        passage = passage[:pos] + marker + passage[pos + len(answer):]
        offset = pos + len(marker)

        distractors = find_distractors(item, all_items, count=3)
        if len(distractors) < 3:
            continue

        options = [real_answer] + distractors[:3]
        random.shuffle(options)
        correct_idx = options.index(real_answer)

        questions.append({
            "number": idx + 1,
            "options": [f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)],
            "answer": chr(65 + correct_idx),
            "answer_text": real_answer,
        })

    return {
        "passage": passage,
        "questions": questions,
    }


# =========================================================
# PUBLIC ENTRY POINT
# =========================================================

def generate_questions(text: str) -> Dict[str, Any]:
    text = normalize_spaces(text)
    if not text or len(text.split()) < 40:
        return {"error": "Text quá ngắn để sinh câu hỏi chất lượng."}

    try:
        mcq = generate_mcq_questions(text, MAX_MCQ)
        cloze = generate_cloze_test(text, MAX_CLOZE_BLANKS)

        return {
            "multiple_choice": mcq,
            "cloze_test": cloze,
            "meta": {
                "mcq_count": len(mcq),
                "cloze_count": len(cloze.get("questions", [])),
                "qa_validation": USE_QA_VALIDATION,
                "note": "No template. Relaxed filters so T5 can actually return questions."
            }
        }
    except Exception as e:
        return {"error": f"Could not generate questions: {e}"}


if __name__ == "__main__":
    sample = """
    A convolutional neural network, or CNN, is an area of deep learning that specializes in pattern recognition.
    A filter is basically just a three by three block.
    Pooling combines numeric arrays from filters.
    CNNs can perform object identification in images.
    """
    from pprint import pprint
    pprint(generate_questions(sample))
