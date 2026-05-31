import re
import nltk
from textblob import TextBlob
from collections import Counter
import eng_to_ipa as ipa
from deep_translator import GoogleTranslator

# Tải tài nguyên NLTK cần thiết cho TextBlob
try:
    nltk.data.find('corpora/brown')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('brown', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('punkt_tab', quiet=True)

def estimate_word_cefr(word: str) -> str:
    """
    Estimate CEFR level based on word length and character complexity as a heuristic.
    """
    length = len(word)
    if length <= 5:
        return "A1"
    elif length <= 6:
        return "A2"
    elif length <= 8:
        return "B1"
    elif length <= 10:
        return "B2"
    elif length <= 12:
        return "C1"
    else:
        return "C2"

def extract_vocabulary(text: str, top_n: int = 20) -> list:
    """
    Extract top keywords combining POS Tagging (Nouns/Adjectives) and word difficulty.
    """
    try:
        blob = TextBlob(text)
        
        # Chỉ lấy Danh từ (NN) và Tính từ (JJ) vì chúng chứa ý nghĩa chủ đề
        valid_words = []
        for word, tag in blob.tags:
            w = word.lower()
            # Bỏ qua từ quá ngắn, hoặc chứa số, hoặc các dấu câu
            if len(w) < 4 or any(c.isdigit() for c in w) or not w.isalpha():
                continue
            if tag.startswith('NN') or tag.startswith('JJ'):
                valid_words.append(w)
                
        # Tính tần suất xuất hiện
        freq = Counter(valid_words)
        
        word_scores = []
        for word, count in freq.items():
            cefr_level = estimate_word_cefr(word)
            
            # Trọng số ưu tiên từ khó (C2, C1)
            cefr_weight = {
                "A1": 1.0, 
                "A2": 1.2, 
                "B1": 2.5, 
                "B2": 5.0, 
                "C1": 8.0, 
                "C2": 15.0
            }.get(cefr_level, 1.0)
            
            final_score = count * cefr_weight
            word_scores.append((final_score, word, cefr_level))
            
        # Sắp xếp theo final_score giảm dần
        word_scores.sort(key=lambda x: x[0], reverse=True)
        
        vocab_list = []
        translator = GoogleTranslator(source='en', target='vi')
        
        for score, word, cefr in word_scores:
            if len(vocab_list) >= top_n:
                break
                
            # Lấy IPA và nhấn âm
            ipa_text = ipa.convert(word)
            if ipa_text.endswith("*"): 
                ipa_text = ipa_text[:-1]
                
            has_stress = 'ˈ' in ipa_text
            
            # Dịch nghĩa
            try:
                meaning = translator.translate(word)
            except:
                meaning = "Không dịch được"
                
            vocab_list.append({
                "word": word,
                "ipa": f"/{ipa_text}/",
                "has_stress": has_stress,
                "cefr": cefr,
                "meaning": meaning
            })
            
        return vocab_list
    except Exception as e:
        print(f"Error extracting vocabulary: {e}")
        return []
