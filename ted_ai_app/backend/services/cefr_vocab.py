import csv
import os
import re


CEFR_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")
CEFR_RANK = {level: index for index, level in enumerate(CEFR_LEVELS)}
CEFR_WEIGHTS = {
    "A1": 1.0,
    "A2": 1.2,
    "B1": 2.5,
    "B2": 5.0,
    "C1": 8.0,
    "C2": 15.0,
}

_VOCAB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "models",
    "word_list_cefr.csv",
)
_cefr_vocab = None


def normalize_vocab_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def load_cefr_vocab() -> dict:
    global _cefr_vocab

    if _cefr_vocab is not None:
        return _cefr_vocab

    vocab = {}
    if not os.path.exists(_VOCAB_PATH):
        print("Warning: CEFR vocabulary file not found.")
        _cefr_vocab = vocab
        return _cefr_vocab

    try:
        with open(_VOCAB_PATH, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                word = normalize_vocab_key(row.get("headword", ""))
                level = row.get("CEFR", "").strip().upper()
                if word and level in CEFR_RANK:
                    existing = vocab.get(word)
                    if existing is None or CEFR_RANK[level] < CEFR_RANK[existing]:
                        vocab[word] = level
    except Exception as e:
        print(f"Error loading CEFR vocabulary: {e}")

    _cefr_vocab = vocab
    return _cefr_vocab


def lookup_word_cefr(word: str):
    return load_cefr_vocab().get(normalize_vocab_key(word))


def estimate_by_length(word: str) -> str:
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


def estimate_word_cefr(word: str) -> str:
    return lookup_word_cefr(word) or estimate_by_length(word)


def estimate_text_cefr(text: str) -> str:
    exact_level = lookup_word_cefr(text)
    if exact_level:
        return exact_level

    words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?", text)
    if not words:
        return "A1"

    levels = [estimate_word_cefr(word) for word in words]
    return max(levels, key=lambda level: CEFR_RANK[level])


def cefr_weight(level: str) -> float:
    return CEFR_WEIGHTS.get(level, 1.0)
