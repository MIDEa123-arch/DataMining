import os
import re
import joblib
import numpy as np
from collections import Counter

import nltk
from textblob import TextBlob

model_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "models",
    "hybrid_bundle_v2.pkl",
)
cefr_model = None

ACADEMIC_WORDS = {
    "analysis", "approach", "area", "assume", "available", "concept", "consistent",
    "context", "create", "data", "derive", "estimate", "evidence", "function",
    "identify", "involve", "issue", "method", "occur", "percent", "process",
    "require", "research", "similar", "structure", "theory", "variable", "authority",
    "benefit", "capacity", "challenge", "conclusion", "definition", "environment",
    "establish", "factor", "interpret", "principle", "procedure", "significant",
    "strategy", "technology", "transformation",
}
CONNECTOR_WORDS = {
    "and", "or", "but", "however", "although", "because", "since", "therefore",
    "moreover", "furthermore", "meanwhile", "instead", "otherwise", "hence", "thus",
    "consequently", "nevertheless", "rather", "still", "whereas", "yet",
}
AUXILIARY_WORDS = {
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "shall", "should", "may", "might", "must", "can", "could",
}
CEFR_ORDER_MAP = {
    "A1": 1.0,
    "A2": 2.0,
    "B1": 3.0,
    "B2": 4.0,
    "C1": 5.0,
    "C2": 6.0,
}


def _ensure_nltk_resources():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/stopwords", "stopwords"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


def _estimate_word_cefr(word: str) -> str:
    length = len(word)
    if length <= 5:
        return "A1"
    if length <= 6:
        return "A2"
    if length <= 8:
        return "B1"
    if length <= 10:
        return "B2"
    if length <= 12:
        return "C1"
    return "C2"


def _syllable_count(word: str) -> int:
    word = word.lower()
    word = re.sub(r"[^a-z]", "", word)
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_char = ""
    for char in word:
        if char in vowels and prev_char not in vowels:
            count += 1
        prev_char = char
    if word.endswith("e") and not word.endswith("le") and count > 1:
        count -= 1
    if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
        count += 1
    return max(count, 1)


def _is_known_word(word: str) -> bool:
    if not word or not word.isalpha():
        return False
    try:
        from nltk.corpus import wordnet
        return len(wordnet.synsets(word.lower())) > 0
    except Exception:
        return False


def load_cefr_model():
    global cefr_model
    if cefr_model is not None:
        return

    _ensure_nltk_resources()

    if not os.path.exists(model_path):
        print(f"Error loading CEFR model: file not found at {model_path}")
        return

    try:
        cefr_model = joblib.load(model_path)
        if not isinstance(cefr_model, dict):
            raise TypeError("CEFR model bundle must be a dict containing pipeline components.")

        expected_keys = [
            "xgb_models", "lr", "scaler",
            "tfidf_word", "tfidf_char",
            "svd_word", "svd_char",
            "encoder", "feature_cols",
        ]
        missing = [k for k in expected_keys if k not in cefr_model]
        if missing:
            raise KeyError(f"CEFR model bundle is missing keys: {missing}")

        print("CEFR model loaded via joblib.")
    except Exception as exc:
        cefr_model = None
        print(f"Error loading CEFR model: {exc}")


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text)


def _pos_tags(text: str) -> list[tuple[str, str]]:
    try:
        return TextBlob(text).tags
    except Exception:
        return []


def _readability_scores(words: list[str], sentences: list[str]) -> tuple[float, float]:
    word_count = len(words)
    sentence_count = max(len(sentences), 1)
    total_syllables = sum(_syllable_count(word) for word in words)
    avg_sentence_length = word_count / sentence_count if sentence_count else 0.0
    avg_syllables_per_word = total_syllables / word_count if word_count else 0.0
    flesch = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
    kincaid = 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59
    return flesch, kincaid


def extract_features(text: str) -> np.ndarray:
    if cefr_model is None:
        load_cefr_model()
    if cefr_model is None:
        raise RuntimeError("CEFR model is not loaded; cannot extract features.")

    feature_cols = cefr_model["feature_cols"]
    text = (text or "").strip()
    sentences = _split_sentences(text)
    if not sentences and text:
        sentences = [text]

    words = _word_tokens(text)
    lower_words = [w.lower() for w in words if w.isalpha()]
    word_count = len(words)
    sentence_count = len(sentences) or 1
    avg_sentence_length = word_count / sentence_count if sentence_count else 0.0
    avg_word_length = float(np.mean([len(w) for w in words])) if words else 0.0
    unique_words = len(set(lower_words))
    lexical_diversity = unique_words / word_count if word_count else 0.0

    stopwords = set()
    try:
        stopwords = set(nltk.corpus.stopwords.words("english"))
    except Exception:
        stopwords = {
            "a", "an", "the", "and", "or", "but", "if", "while", "with",
            "at", "by", "for", "from", "in", "into", "of", "on", "to", "up", "as"
        }

    content_words = [w for w in lower_words if w not in stopwords]
    content_word_count = len(content_words)
    content_word_ratio = content_word_count / word_count if word_count else 0.0
    content_unique = len(set(content_words))
    content_lexical_diversity = content_unique / content_word_count if content_word_count else 0.0

    cefr_labels = [_estimate_word_cefr(w) for w in lower_words]
    cefr_counts = Counter(cefr_labels)

    pct_scale = 100.0
    pct_A1 = (cefr_counts["A1"] / word_count * pct_scale) if word_count else 0.0
    pct_A2 = (cefr_counts["A2"] / word_count * pct_scale) if word_count else 0.0
    pct_B1 = (cefr_counts["B1"] / word_count * pct_scale) if word_count else 0.0
    pct_B2 = (cefr_counts["B2"] / word_count * pct_scale) if word_count else 0.0
    pct_C1 = (cefr_counts["C1"] / word_count * pct_scale) if word_count else 0.0
    pct_C2 = (cefr_counts["C2"] / word_count * pct_scale) if word_count else 0.0
    pct_A1_A2 = pct_A1 + pct_A2
    pct_B1_B2 = pct_B1 + pct_B2
    pct_C1_C2 = pct_C1 + pct_C2
    pct_B1_plus = pct_B1 + pct_B2 + pct_C1 + pct_C2
    pct_B2_plus = pct_B2 + pct_C1 + pct_C2

    cefr_scores = [CEFR_ORDER_MAP[label] for label in cefr_labels] if cefr_labels else [0.0]
    avg_cefr_score = float(np.mean(cefr_scores)) if cefr_scores else 0.0
    max_cefr_score = float(max(cefr_scores)) if cefr_scores else 0.0

    known_word_count = sum(1 for w in lower_words if _is_known_word(w))
    unknown_word_ratio = 1.0 - known_word_count / word_count if word_count else 0.0
    known_content_word_ratio = (
        sum(1 for w in content_words if _is_known_word(w)) / content_word_count
    ) if content_word_count else 0.0

    long_word_count = sum(1 for w in lower_words if len(w) > 6)
    long_word_ratio = long_word_count / word_count if word_count else 0.0
    content_long_word_ratio = (
        sum(1 for w in content_words if len(w) > 6) / content_word_count
    ) if content_word_count else 0.0
    academic_word_ratio = (
        sum(1 for w in content_words if w in ACADEMIC_WORDS) / content_word_count
    ) if content_word_count else 0.0

    complex_sentence_ratio = sum(1 for s in sentences if len(_word_tokens(s)) > 20) / sentence_count
    connector_ratio = sum(1 for w in lower_words if w in CONNECTOR_WORDS) / word_count if word_count else 0.0

    pos_tagged = _pos_tags(text)
    pos_counts = Counter()
    for token, tag in pos_tagged:
        pos = tag.upper()
        if pos.startswith("NNP"):
            pos_counts["PROPN"] += 1
        if pos.startswith("NN"):
            pos_counts["NOUN"] += 1
        if pos.startswith("VB"):
            pos_counts["VERB"] += 1
        if pos.startswith("JJ"):
            pos_counts["ADJ"] += 1
        if pos.startswith("RB"):
            pos_counts["ADV"] += 1
        if pos in {"IN", "TO"}:
            pos_counts["ADP"] += 1
        if pos in {"PRP", "PRP$", "WP", "WP$"}:
            pos_counts["PRON"] += 1
        if pos == "MD" or token.lower() in AUXILIARY_WORDS:
            pos_counts["AUX"] += 1

    pos_noun_ratio = pos_counts["NOUN"] / word_count if word_count else 0.0
    pos_verb_ratio = pos_counts["VERB"] / word_count if word_count else 0.0
    pos_adj_ratio = pos_counts["ADJ"] / word_count if word_count else 0.0
    pos_adv_ratio = pos_counts["ADV"] / word_count if word_count else 0.0
    pos_propn_ratio = pos_counts["PROPN"] / word_count if word_count else 0.0
    pos_adp_ratio = pos_counts["ADP"] / word_count if word_count else 0.0
    pos_aux_ratio = pos_counts["AUX"] / word_count if word_count else 0.0
    pos_pron_ratio = pos_counts["PRON"] / word_count if word_count else 0.0

    flesch_reading_ease, flesch_kincaid_grade = _readability_scores(lower_words, sentences)

    feature_dict = {
        "word_count": float(word_count),
        "sentence_count": float(sentence_count),
        "avg_sentence_length": float(avg_sentence_length),
        "avg_word_length": float(avg_word_length),
        "unique_words": float(unique_words),
        "lexical_diversity": float(lexical_diversity),
        "content_word_count": float(content_word_count),
        "content_word_ratio": float(content_word_ratio),
        "content_lexical_diversity": float(content_lexical_diversity),
        "pct_A1": float(pct_A1),
        "pct_A2": float(pct_A2),
        "pct_B1": float(pct_B1),
        "pct_B2": float(pct_B2),
        "pct_C1": float(pct_C1),
        "pct_C2": float(pct_C2),
        "pct_A1_A2": float(pct_A1_A2),
        "pct_B1_B2": float(pct_B1_B2),
        "pct_C1_C2": float(pct_C1_C2),
        "pct_B1_plus": float(pct_B1_plus),
        "pct_B2_plus": float(pct_B2_plus),
        "avg_cefr_score": float(avg_cefr_score),
        "max_cefr_score": float(max_cefr_score),
        "unknown_word_ratio": float(unknown_word_ratio),
        "proper_noun_ratio": float(pos_propn_ratio),
        "known_content_word_ratio": float(known_content_word_ratio),
        "long_word_ratio": float(long_word_ratio),
        "content_long_word_ratio": float(content_long_word_ratio),
        "academic_word_ratio": float(academic_word_ratio),
        "complex_sentence_ratio": float(complex_sentence_ratio),
        "connector_ratio": float(connector_ratio),
        "flesch_reading_ease": float(flesch_reading_ease),
        "flesch_kincaid_grade": float(flesch_kincaid_grade),
        "pos_noun_ratio": float(pos_noun_ratio),
        "pos_verb_ratio": float(pos_verb_ratio),
        "pos_adj_ratio": float(pos_adj_ratio),
        "pos_adv_ratio": float(pos_adv_ratio),
        "pos_propn_ratio": float(pos_propn_ratio),
        "pos_adp_ratio": float(pos_adp_ratio),
        "pos_aux_ratio": float(pos_aux_ratio),
        "pos_pron_ratio": float(pos_pron_ratio),
    }

    feature_vector = np.asarray([feature_dict[col] for col in feature_cols], dtype=np.float32)
    if feature_vector.shape[0] != len(feature_cols):
        raise ValueError(
            f"Feature vector length {feature_vector.shape[0]} does not match expected {len(feature_cols)} feature columns."
        )
    return feature_vector


def predict_cefr(text: str) -> str:
    if cefr_model is None:
        load_cefr_model()
    if cefr_model is None:
        raise RuntimeError("CEFR model is not loaded; cannot predict CEFR.")
    if not isinstance(cefr_model, dict):
        raise TypeError("Loaded CEFR model bundle is not a valid dict pipeline.")

    feature_vector = extract_features(text)
    feature_cols = cefr_model["feature_cols"]
    if feature_vector.ndim != 1 or feature_vector.shape[0] != len(feature_cols):
        raise ValueError(
            f"Mismatched feature dimensions: extracted {feature_vector.shape[0]} features, expected {len(feature_cols)}."
        )

    X_feats = cefr_model["scaler"].transform(feature_vector.reshape(1, -1))
    X_word = cefr_model["svd_word"].transform(
        cefr_model["tfidf_word"].transform([text])
    )
    X_char = cefr_model["svd_char"].transform(
        cefr_model["tfidf_char"].transform([text])
    )
    X = np.concatenate([X_feats, X_word, X_char], axis=1)

    print("Feature shape:", X.shape)
    print("Expected features:", len(feature_cols))

    model_probs = [m.predict_proba(X) for m in cefr_model["xgb_models"]]
    model_probs.append(cefr_model["lr"].predict_proba(X))
    probs = np.mean(model_probs, axis=0)[0]

    print("Prediction probabilities:", probs)

    label_index = int(np.argmax(probs))
    label = cefr_model["encoder"].inverse_transform([label_index])[0]
    return str(label)
