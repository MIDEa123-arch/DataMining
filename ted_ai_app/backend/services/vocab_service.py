from collections import Counter

import eng_to_ipa as ipa
import nltk
from deep_translator import GoogleTranslator
from textblob import TextBlob

from services.cefr_vocab import CEFR_RANK, cefr_weight, estimate_word_cefr

try:
    nltk.data.find("corpora/brown")
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("brown", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("wordnet", quiet=True)

lemmatizer = nltk.stem.WordNetLemmatizer()


def _wordnet_pos(tag: str):
    if tag.startswith("JJ"):
        return "a"
    if tag.startswith("NN"):
        return "n"
    return "n"


def _is_real_dictionary_word(word: str) -> bool:
    try:
        from nltk.corpus import wordnet

        return bool(wordnet.synsets(word))
    except Exception:
        return True


def _target_cefr_weight(word_cefr: str, target_cefr: str | None) -> float:
    if not target_cefr or target_cefr not in CEFR_RANK or word_cefr not in CEFR_RANK:
        return cefr_weight(word_cefr)

    distance = abs(CEFR_RANK[word_cefr] - CEFR_RANK[target_cefr])
    if distance == 0:
        return 5.0
    if distance == 1:
        return 2.2
    if distance == 2:
        return 0.9
    return 0.35


def _cefr_distance(word_cefr: str, target_cefr: str | None) -> int:
    if not target_cefr or target_cefr not in CEFR_RANK or word_cefr not in CEFR_RANK:
        return 0
    return abs(CEFR_RANK[word_cefr] - CEFR_RANK[target_cefr])


def extract_vocabulary(text: str, top_n: int = 20, target_cefr: str | None = None) -> list:
    """
    Extract top keywords from nouns/adjectives and prefer words near the video's CEFR level.
    """
    try:
        blob = TextBlob(text)

        valid_words = []
        for word, tag in blob.tags:
            lowered = word.lower()
            if len(lowered) < 4 or any(c.isdigit() for c in lowered) or not lowered.isalpha():
                continue
            if tag.startswith("NNP"):
                continue
            if tag.startswith("NN") or tag.startswith("JJ"):
                lemma = lemmatizer.lemmatize(lowered, _wordnet_pos(tag))
                if not _is_real_dictionary_word(lemma):
                    continue
                valid_words.append(lemma)

        freq = Counter(valid_words)

        word_scores = []
        for word, count in freq.items():
            cefr_level = estimate_word_cefr(word)
            final_score = count * _target_cefr_weight(cefr_level, target_cefr)
            word_scores.append((final_score, count, word, cefr_level))

        word_scores.sort(
            key=lambda item: (
                item[0],
                item[1],
                -_cefr_distance(item[3], target_cefr),
            ),
            reverse=True,
        )

        vocab_list = []
        translator = GoogleTranslator(source="en", target="vi")

        for score, count, word, cefr in word_scores:
            if len(vocab_list) >= top_n:
                break

            ipa_text = ipa.convert(word)
            if ipa_text.endswith("*"):
                ipa_text = ipa_text[:-1]

            has_stress = "Ëˆ" in ipa_text or "ˈ" in ipa_text

            try:
                meaning = translator.translate(word)
            except Exception:
                meaning = "Không dịch được"

            vocab_list.append(
                {
                    "word": word,
                    "ipa": f"/{ipa_text}/",
                    "has_stress": has_stress,
                    "cefr": cefr,
                    "meaning": meaning,
                }
            )

        return vocab_list
    except Exception as e:
        print(f"Error extracting vocabulary: {e}")
        return []
