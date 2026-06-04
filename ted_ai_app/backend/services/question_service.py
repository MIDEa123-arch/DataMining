import random
import string
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# =========================================================
# QUESTION SERVICE - MODEL BASED / NO REGEX SEMANTIC FILTER
#
# Không dùng regex để "bao quát" semantic.
# Regex không được import.
#
# Dùng:
# - spaCy: tách câu + lấy noun chunk/entity.
# - SBERT: chấm context window + chấm độ khác nhau của options.
# - QG model: sinh câu hỏi từ answer + context.
# - QA model: kiểm tra câu hỏi có trả lời đúng từ context không.
#
# Lưu ý:
# - Không fallback ghép cụm bằng regex.
# - Không ép đủ 10 nếu transcript không đủ chất lượng.
# =========================================================

QG_MODEL_NAME = "mrm8488/t5-base-finetuned-question-generation-ap"
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


MIN_TEXT_WORDS = 40
TARGET_MCQ = 10
MAX_MCQ = 15
MAX_CLOZE_BLANKS = 10

MAX_SENTENCES_FOR_QG = 180
MIN_CONTEXT_WORDS = 7
MAX_CONTEXT_WORDS = 95

NUM_Q_PER_CANDIDATE = 4
MAX_CANDIDATES_TO_TRY = 120

USE_QA_VALIDATION = True
MIN_QA_SCORE = 0.06

# Không dùng regex, nhưng vẫn có guard ngắn để chặn dạng output lỗi quá rõ từ model.
BAD_QUESTION_PHRASES = [
    "which of the following",
    "main idea of the passage",
    "what is the main idea",
    "do not include the answer",
    "correct answer",
    "answer above",
    "what episode",
    "which episode",
    "what is the title",
    "what is the video",
    "what is the line",
    "what is the text",
]

MAX_SAME_PREFIX = {
    "what is": 3,
    "what are": 3,
    "what does": 3,
    "what do": 3,
    "what can": 2,
    "how": 2,
    "why": 3,
    "when": 3,
    "where": 3,
    "who": 4,
    "which": 3,
    "other": 4,
}


# =========================================================
# LOADERS
# =========================================================

def load_qg_model():
    global qg_tokenizer, qg_model
    if qg_model is None:
        print(f"[QG] Loading {QG_MODEL_NAME} on {device}...")
        qg_tokenizer = AutoTokenizer.from_pretrained(QG_MODEL_NAME)
        qg_model = AutoModelForSeq2SeqLM.from_pretrained(QG_MODEL_NAME).to(device)
        qg_model.eval()
        print("[QG] QG model loaded.")


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
        print(f"[QG] SBERT unavailable: {e}")
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
        print(f"[QG] spaCy unavailable: {e}")
        spacy_nlp = None

    return spacy_nlp


# =========================================================
# BASIC UTILS - NO REGEX
# =========================================================

def normalize_spaces(text: str) -> str:
    return " ".join((text or "").split())


def clean_text(text: str) -> str:
    text = normalize_spaces(text)
    return normalize_spaces(text.strip(string.punctuation + "“”‘’`\""))


def word_count(text: str) -> int:
    return len(normalize_spaces(text).split())


def lower_text(text: str) -> str:
    return clean_text(text).lower()


def has_letters(text: str) -> bool:
    return any(ch.isalpha() for ch in text or "")


def question_prefix(question: str) -> str:
    q = normalize_spaces(question).lower()
    for p in ["what is", "what are", "what does", "what do", "what can"]:
        if q.startswith(p):
            return p
    for p in ["how", "why", "when", "where", "who", "which"]:
        if q.startswith(p):
            return p
    return "other"


def clean_transcript_for_questions(text: str) -> str:
    """
    Không dùng regex. Chỉ normalize space.
    Transcript cleaning nâng cao nên làm ở bước ASR/whisper riêng.
    """
    return normalize_spaces(text)


def split_sentences(text: str) -> List[str]:
    text = clean_transcript_for_questions(text)
    if not text:
        return []

    nlp = load_spacy_model()
    if nlp is None:
        print("[QG] spaCy is required for this no-regex version.")
        return []

    doc = nlp(text)
    sentences = []

    for sent in doc.sents:
        s = normalize_spaces(sent.text)
        wc = word_count(s)
        if not (4 <= wc <= 60):
            continue
        if not has_letters(s):
            continue
        sentences.append(s)

    return sentences[:MAX_SENTENCES_FOR_QG]


# =========================================================
# SBERT HELPERS
# =========================================================

def encode_texts(texts: List[str]):
    model = load_sbert_model()
    if model is None:
        return None
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def cosine_sim_text(a: str, b: str) -> float:
    model = load_sbert_model()
    if model is None:
        return 0.0
    emb = model.encode([a, b], convert_to_numpy=True, normalize_embeddings=True)
    return float(emb[0] @ emb[1].T)


def build_sentence_similarity_matrix(sentences: List[str]):
    n = len(sentences)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if n == 1:
        return np.ones((1, 1), dtype=np.float32)

    emb = encode_texts(sentences)
    if emb is None:
        return np.eye(n, dtype=np.float32)

    return emb @ emb.T


def _window_indices(n: int, main_idx: int) -> List[List[int]]:
    windows = [
        [main_idx],
        [main_idx - 1, main_idx],
        [main_idx, main_idx + 1],
        [main_idx - 1, main_idx, main_idx + 1],
        [main_idx - 2, main_idx - 1, main_idx],
        [main_idx, main_idx + 1, main_idx + 2],
        [main_idx - 2, main_idx - 1, main_idx, main_idx + 1],
        [main_idx - 1, main_idx, main_idx + 1, main_idx + 2],
    ]

    valid = []
    seen = set()

    for w in windows:
        w = tuple(sorted(set(i for i in w if 0 <= i < n)))
        if not w or main_idx not in w or w in seen:
            continue
        seen.add(w)
        valid.append(list(w))

    return valid


def context_coherence_score(sentences: List[str], indices: List[int], main_idx: int, sim_matrix, answer: str) -> float:
    context = " ".join(sentences[i] for i in indices)
    wc = word_count(context)
    score = 0.0

    # Context phải chứa answer dạng substring cơ bản.
    if lower_text(answer) in lower_text(context):
        score += 2.0
    else:
        score -= 3.0

    if MIN_CONTEXT_WORDS <= wc <= MAX_CONTEXT_WORDS:
        score += 1.0
    elif wc < MIN_CONTEXT_WORDS:
        score -= 1.0
    elif wc > MAX_CONTEXT_WORDS:
        score -= 2.0

    adjacent = []
    for a, b in zip(indices, indices[1:]):
        adjacent.append(float(sim_matrix[a][b]))

    if adjacent:
        score += 1.2 * (sum(adjacent) / len(adjacent))

    related = []
    for idx in indices:
        if idx != main_idx:
            related.append(float(sim_matrix[main_idx][idx]))

    if related:
        score += 0.8 * (sum(related) / len(related))

    # Dùng spaCy để xem câu chính có pronoun không, thay vì regex.
    nlp = load_spacy_model()
    if nlp is not None:
        doc = nlp(sentences[main_idx])
        has_pronoun = any(tok.pos_ == "PRON" for tok in doc)
        if has_pronoun:
            if main_idx - 1 in indices:
                score += 0.8
            else:
                score -= 0.6

    return float(score)


def build_sbert_window_context(sentences: List[str], main_idx: int, sim_matrix, answer: str) -> Tuple[str, float, List[int]]:
    windows = _window_indices(len(sentences), main_idx)
    scored = []

    for w in windows:
        context = " ".join(sentences[i] for i in w)
        if word_count(context) > MAX_CONTEXT_WORDS:
            continue
        score = context_coherence_score(sentences, w, main_idx, sim_matrix, answer)
        scored.append((score, w, context))

    if not scored:
        return sentences[main_idx], 0.0, [main_idx]

    scored.sort(key=lambda x: x[0], reverse=True)
    score, indices, context = scored[0]
    return context, float(score), indices


# =========================================================
# ANSWER CANDIDATES - spaCy ONLY
# =========================================================

def is_acronym_like(text: str) -> bool:
    text = clean_text(text)
    return 2 <= len(text) <= 10 and text.upper() == text and any(ch.isalpha() for ch in text)


def is_good_answer_spacy(text: str, span: Any = None, ctype: str = "") -> bool:
    text = clean_text(text)
    if not text or not has_letters(text):
        return False

    words = text.split()
    if len(words) > 6:
        return False

    lower = text.lower()
    if lower in {"example", "process", "overview", "text", "video", "passage", "document", "documents"}:
        return False

    if ctype.startswith("entity:"):
        return True

    if is_acronym_like(text):
        return True

    if span is None:
        return False

    root = getattr(span, "root", span)
    root_pos = getattr(root, "pos_", "")

    # Không lấy cụm có root là verb/adv/pron/determiner...
    if root_pos not in {"NOUN", "PROPN", "NUM"}:
        return False

    try:
        toks = list(span)
    except TypeError:
        toks = [span]

    if not any(getattr(tok, "pos_", "") in {"NOUN", "PROPN", "NUM"} for tok in toks):
        return False

    # Không lấy cụm bắt đầu bằng interjection/discourse marker nếu spaCy nhận được.
    if toks and getattr(toks[0], "pos_", "") in {"INTJ", "CCONJ", "SCONJ", "ADP", "DET", "PRON"}:
        return False

    return True


def candidate_priority(text: str, ctype: str) -> int:
    t = text.lower()

    if ctype.startswith("entity:PERSON"):
        return 14
    if ctype.startswith("entity:ORG") or ctype.startswith("entity:PRODUCT") or ctype.startswith("entity:EVENT"):
        return 10
    if ctype.startswith("entity:"):
        return 9
    if ctype == "number":
        return 8
    if ctype == "noun_chunk":
        if any(key in t for key in [
            "house", "wolf", "barn", "farm", "broom", "coop", "burrow",
            "network", "filter", "image", "pixel", "layer", "prediction"
        ]):
            return 10
        return 7
    if ctype == "single_term":
        return 4
    return 1


def candidate_category(item: Dict[str, Any]) -> str:
    t = item.get("type", "")
    text = item.get("text", "").lower()

    if t.startswith("entity:PERSON"):
        return "person"
    if t.startswith("entity:"):
        return "entity"
    if t == "number":
        return "number"

    if any(key in text for key in ["house", "barn", "coop", "burrow", "field", "room", "farm"]):
        return "place_or_structure"
    if any(key in text for key in ["network", "cnn", "filter", "pixel", "image", "layer", "prediction"]):
        return "technical_term"
    if len(text.split()) == 1:
        return "term"
    return "phrase"


def add_candidate(candidates: List[Dict[str, Any]], seen: set, text: str, ctype: str, sent_idx: int, span: Any = None):
    text = clean_text(text)
    key = text.lower()

    if key in seen:
        return
    if not is_good_answer_spacy(text, span=span, ctype=ctype):
        return

    item = {
        "text": text,
        "type": ctype,
        "priority": candidate_priority(text, ctype),
        "sentence_index": sent_idx,
    }
    item["category"] = candidate_category(item)

    seen.add(key)
    candidates.append(item)


def extract_candidates(sentence: str, sent_idx: int) -> List[Dict[str, Any]]:
    nlp = load_spacy_model()
    if nlp is None:
        return []

    doc = nlp(sentence)
    candidates = []
    seen = set()

    entity_labels = {
        "PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT", "EVENT",
        "WORK_OF_ART", "DATE", "TIME", "MONEY", "PERCENT",
        "QUANTITY", "CARDINAL", "NORP", "LAW"
    }

    for ent in doc.ents:
        if ent.label_ in entity_labels:
            add_candidate(candidates, seen, ent.text, f"entity:{ent.label_}", sent_idx, ent)

    for chunk in doc.noun_chunks:
        # Bỏ determiner bằng token spaCy, không regex.
        toks = list(chunk)
        while toks and toks[0].pos_ in {"DET", "PRON"}:
            toks = toks[1:]
        phrase = clean_text(" ".join(tok.text for tok in toks))
        add_candidate(candidates, seen, phrase, "noun_chunk", sent_idx, chunk)

    # Single terms chỉ lấy nếu là PROPN/acronym hoặc technical noun viết hoa.
    for tok in doc:
        if tok.is_stop or not tok.is_alpha:
            continue
        if tok.pos_ == "PROPN" or is_acronym_like(tok.text):
            add_candidate(candidates, seen, tok.text, "single_term", sent_idx, tok)

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

    for item in sorted(all_items, key=lambda x: (x.get("priority", 0), len(x["text"].split())), reverse=True):
        key = item["text"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique, by_sent


# =========================================================
# QUESTION GENERATION
# =========================================================

def clean_generated_question(q: str) -> str:
    q = normalize_spaces(q)
    lower = q.lower()
    for prefix in ["question:", "q:", "question -", "q -"]:
        if lower.startswith(prefix):
            q = q[len(prefix):].strip()
            break

    if "\n" in q:
        q = q.split("\n")[0].strip()

    q = q.strip(" \"'`")
    if q and not q.endswith("?"):
        q = q.rstrip(".!,;:") + "?"

    return normalize_spaces(q)


def generate_question_candidates(answer: str, context: str, num_return_sequences: int = NUM_Q_PER_CANDIDATE) -> List[str]:
    load_qg_model()

    answer = clean_text(answer)
    context = normalize_spaces(context)

    if not answer or not context:
        return []

    prompt = f"answer: {answer} context: {context}"
    inputs = qg_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)

    num_return_sequences = max(1, int(num_return_sequences))
    num_beams = max(4, num_return_sequences * 2)

    with torch.no_grad():
        outputs = qg_model.generate(
            **inputs,
            max_length=64,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    questions = []
    seen = set()

    for out in outputs:
        q = qg_tokenizer.decode(out, skip_special_tokens=True)
        q = clean_generated_question(q)
        if not q:
            continue
        key = q.lower()
        if key not in seen:
            seen.add(key)
            questions.append(q)

    return questions


# =========================================================
# VALIDATION
# =========================================================

def is_good_question(question: str, answer: str) -> bool:
    q = normalize_spaces(question).lower()
    a = clean_text(answer).lower()

    if not q.endswith("?"):
        return False
    if word_count(q) < 5 or word_count(q) > 20:
        return False

    if a and a in q:
        return False

    if " would " in f" {q} ":
        return False

    for bad in BAD_QUESTION_PHRASES:
        if bad in q:
            return False

    return True


def answer_match_score(expected: str, predicted: str) -> float:
    e = clean_text(expected).lower()
    p = clean_text(predicted).lower()

    if not e or not p:
        return 0.0

    if e == p:
        return 1.0
    if e in p or p in e:
        return 0.85

    # Dùng SBERT cho semantic match, không regex token overlap.
    sim = cosine_sim_text(e, p)
    return sim


def qa_validate_question(question: str, answer: str, context: str) -> Tuple[bool, float, str]:
    if not is_good_question(question, answer):
        return False, 0.0, ""

    qa = load_qa_model()
    if qa is None:
        return True, 0.5, ""

    try:
        result = qa(question=question, context=context)
        predicted = result.get("answer", "")
        qa_score = float(result.get("score", 0.0))
    except Exception:
        return True, 0.4, ""

    match_score = answer_match_score(answer, predicted)

    if qa_score < MIN_QA_SCORE:
        return False, qa_score, predicted

    if match_score < 0.72:
        return False, qa_score, predicted

    return True, qa_score * match_score, predicted


# =========================================================
# DISTRACTORS - SBERT DISTINCTNESS
# =========================================================

def option_set_is_good(options: List[str]) -> bool:
    if len(options) != 4:
        return False

    cleaned = [clean_text(o) for o in options]
    lows = [o.lower() for o in cleaned]

    if len(set(lows)) != 4:
        return False

    model = load_sbert_model()
    if model is None:
        # Nếu thiếu SBERT thì chỉ kiểm tra trùng chuỗi.
        return True

    emb = model.encode(cleaned, convert_to_numpy=True, normalize_embeddings=True)
    sim = emb @ emb.T

    # Không để 2 options quá giống nhau.
    for i in range(len(cleaned)):
        for j in range(i + 1, len(cleaned)):
            if float(sim[i][j]) >= 0.80:
                return False

    return True


def find_distractors(answer_item: Dict[str, Any], all_items: List[Dict[str, Any]], question: str = "", count: int = 3) -> List[str]:
    answer = clean_text(answer_item["text"])
    answer_low = answer.lower()
    answer_cat = answer_item.get("category", "")

    pool = []
    seen = {answer_low}

    # Ưu tiên cùng category, nhưng vẫn dùng SBERT distinctness ở bước build.
    sorted_items = sorted(
        all_items,
        key=lambda x: (
            x.get("category") == answer_cat,
            x.get("priority", 0),
            len(x["text"].split())
        ),
        reverse=True,
    )

    for item in sorted_items:
        text = clean_text(item["text"])
        low = text.lower()

        if not text or low in seen:
            continue
        if low in answer_low or answer_low in low:
            continue
        if question and low in question.lower():
            continue

        seen.add(low)
        pool.append(text)

    if not pool:
        return []

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
            # Distractor cần hơi liên quan nhưng không quá giống.
            if 0.05 <= s <= 0.74:
                ranked.append((d, s))

        ranked.sort(key=lambda x: x[1], reverse=True)
        selected = [d for d, _ in ranked[:count]]

        for d in pool:
            if len(selected) >= count:
                break
            if d not in selected:
                trial = [answer] + selected + [d]
                if len(trial) < 4 or option_set_is_good(trial + pool[: max(0, 4 - len(trial))]):
                    selected.append(d)

        return selected[:count]
    except Exception:
        return pool[:count]


def build_mcq(
    question: str,
    answer_item: Dict[str, Any],
    all_items: List[Dict[str, Any]],
    context: str,
    qa_score: float,
    context_score: float,
    context_indices: List[int],
) -> Optional[Dict[str, Any]]:
    question = clean_generated_question(question)
    answer = clean_text(answer_item["text"])

    distractors = find_distractors(answer_item, all_items, question=question, count=3)
    if len(distractors) < 3:
        return None

    options = [answer] + distractors[:3]
    if not option_set_is_good(options):
        return None

    random.shuffle(options)
    correct_idx = options.index(answer)

    return {
        "question": question,
        "options": [f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)],
        "answer": chr(65 + correct_idx),
        "answer_text": answer,
        "context": context,
        "type": "reading",
        "quality": {
            "qa_score": float(qa_score),
            "context_score": float(context_score),
            "answer_priority": int(answer_item.get("priority", 0)),
            "context_indices": context_indices,
        },
    }


def mcq_quality_score(mcq: Dict[str, Any]) -> float:
    q = mcq.get("quality", {})
    return (
        2.0 * float(q.get("qa_score", 0.0))
        + 0.5 * float(q.get("context_score", 0.0))
        + 0.08 * float(q.get("answer_priority", 0.0))
    )


# =========================================================
# GENERATE MCQ
# =========================================================

def candidate_order(all_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        all_items,
        key=lambda x: (
            x.get("priority", 0),
            2 <= len(x["text"].split()) <= 4,
            -len(x["text"].split())
        ),
        reverse=True,
    )


def generate_mcq_questions(text: str, max_questions: int = MAX_MCQ) -> List[Dict[str, Any]]:
    load_qg_model()

    sentences = split_sentences(text)
    if not sentences:
        return []

    if load_spacy_model() is None:
        return []

    all_items, by_sent = collect_candidates(sentences)
    sim_matrix = build_sentence_similarity_matrix(sentences)

    print(f"[QG] sentences={len(sentences)}, candidates={len(all_items)}")

    if len(all_items) < 4:
        return []

    raw_results = []
    used_pairs = set()

    for cand in candidate_order(all_items)[:MAX_CANDIDATES_TO_TRY]:
        if len(raw_results) >= max_questions * 3:
            break

        answer = clean_text(cand["text"])
        sent_idx = cand["sentence_index"]

        if not answer or not (0 <= sent_idx < len(sentences)):
            continue

        context, ctx_score, ctx_indices = build_sbert_window_context(
            sentences=sentences,
            main_idx=sent_idx,
            sim_matrix=sim_matrix,
            answer=answer,
        )

        questions = generate_question_candidates(answer, context, NUM_Q_PER_CANDIDATE)

        for question in questions:
            key = (answer.lower(), question.lower())
            if key in used_pairs:
                continue

            ok, qa_score, predicted = qa_validate_question(question, answer, context)
            if not ok:
                continue

            mcq = build_mcq(
                question=question,
                answer_item=cand,
                all_items=all_items,
                context=context,
                qa_score=qa_score,
                context_score=ctx_score,
                context_indices=ctx_indices,
            )

            if mcq is None:
                continue

            used_pairs.add(key)
            raw_results.append(mcq)
            print(f"[QG] MCQ candidate {len(raw_results)}: {question} -> {answer} | qa={qa_score:.3f}")

    raw_results.sort(key=mcq_quality_score, reverse=True)

    final = []
    used_questions = set()
    used_answers = set()
    prefix_counts: Dict[str, int] = {}

    for mcq in raw_results:
        if len(final) >= max_questions:
            break

        qkey = mcq["question"].lower()
        akey = mcq["answer_text"].lower()
        pref = question_prefix(mcq["question"])

        if qkey in used_questions:
            continue
        if akey in used_answers:
            continue
        if prefix_counts.get(pref, 0) >= MAX_SAME_PREFIX.get(pref, 3):
            continue

        final.append(mcq)
        used_questions.add(qkey)
        used_answers.add(akey)
        prefix_counts[pref] = prefix_counts.get(pref, 0) + 1

    print(f"[QG] final accepted={len(final)}")
    return final


# =========================================================
# CLOZE TEST - spaCy based, no regex
# =========================================================

def generate_cloze_test(text: str, num_blanks: int = MAX_CLOZE_BLANKS) -> Dict[str, Any]:
    sentences = split_sentences(text)
    if not sentences:
        return {"passage": "", "questions": []}

    if load_spacy_model() is None:
        return {"passage": " ".join(sentences[:6]), "questions": []}

    all_items, by_sent = collect_candidates(sentences)
    if not all_items:
        return {"passage": " ".join(sentences[:6]), "questions": []}

    best = (0, min(8, len(sentences)), -1)

    for start in range(len(sentences)):
        for end in range(start + 3, min(start + 12, len(sentences)) + 1):
            wc = sum(word_count(sentences[i]) for i in range(start, end))
            if not (60 <= wc <= 260):
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
            pos = passage.lower().find(item["text"].lower())
            if pos >= 0:
                copied = dict(item)
                copied["_pos"] = pos
                passage_items.append(copied)

    passage_items.sort(key=lambda x: (x["_pos"], -x.get("priority", 0)))

    questions = []
    selected = []
    used = set()

    for item in passage_items:
        if len(selected) >= num_blanks:
            break
        key = item["text"].lower()
        if key not in used:
            used.add(key)
            selected.append(item)

    offset = 0

    for idx, item in enumerate(selected):
        answer = clean_text(item["text"])
        low_passage = passage.lower()
        pos = low_passage.find(answer.lower(), offset)
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
        if not option_set_is_good(options):
            continue

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
# PUBLIC ENTRY
# =========================================================

def generate_questions(text: str) -> Dict[str, Any]:
    text = clean_transcript_for_questions(text)
    if not text or word_count(text) < MIN_TEXT_WORDS:
        return {"error": "Text quá ngắn để sinh câu hỏi chất lượng."}

    try:
        mcq = generate_mcq_questions(text, MAX_MCQ)
        cloze = generate_cloze_test(text, MAX_CLOZE_BLANKS)

        note = (
            "Model-based mode: spaCy extracts candidates; SBERT scores nearby context windows and option distinctness; "
            "QA validates answerability. No regex semantic fallback is used."
        )

        if len(mcq) < TARGET_MCQ:
            note += f" Only {len(mcq)} reading questions passed validation."

        return {
            "multiple_choice": mcq,
            "cloze_test": cloze,
            "meta": {
                "mcq_count": len(mcq),
                "cloze_count": len(cloze.get("questions", [])),
                "target_mcq_count": TARGET_MCQ,
                "max_mcq_count": MAX_MCQ,
                "qa_validation": USE_QA_VALIDATION,
                "note": note,
            }
        }
    except Exception as e:
        return {"error": f"Could not generate questions: {e}"}


if __name__ == "__main__":
    sample = """
    A convolutional neural network, or CNN, is a type of deep learning model.
    CNNs are often used for image recognition. An image is made of pixels.
    A filter looks at small parts of the image and finds patterns.
    Pooling reduces the size of the feature map.
    These steps help the network recognize objects in images.
    """
    from pprint import pprint
    pprint(generate_questions(sample))
