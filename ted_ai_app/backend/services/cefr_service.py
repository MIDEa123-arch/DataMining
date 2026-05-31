import os
import joblib

model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "hybrid_bundle_v2.pkl")
cefr_model = None

def load_cefr_model():
    global cefr_model
    if cefr_model is None:
        try:
            if os.path.exists(model_path):
                cefr_model = joblib.load(model_path)
                print("CEFR model loaded via joblib.")
            else:
                print("Warning: CEFR model file not found.")
        except Exception as e:
            print(f"Error loading CEFR model: {e}")

def extract_features(text: str):
    # Dựa vào mô hình hybrid của bạn (TF-IDF + XGBoost)
    # File joblib thường chứa toàn bộ pipeline (vectorizer + classifier)
    # Nên ở đây ta chỉ cần trả về text gốc trong list
    return [text]

def predict_cefr(text: str) -> str:
    load_cefr_model()
    
    # Hàm fallback dựa trên độ dài và độ phức tạp
    def fallback_predict(txt):
        words = txt.split()
        length = len(words)
        unique_words = len(set(words))
        if length == 0: return "A1"
        lexical_richness = unique_words / length
        
        if length < 100 or lexical_richness < 0.4: return "A2"
        elif length < 300: return "B1"
        elif length < 600: return "B2"
        elif length < 1000: return "C1"
        else: return "C2"

    if cefr_model is None or isinstance(cefr_model, dict):
        # Nếu chưa có pipeline rút trích đặc trưng chính xác từ user, dùng fallback
        return fallback_predict(text)
        
    try:
        features = extract_features(text)
        prediction = cefr_model.predict([features])[0]
        return str(prediction)
    except Exception as e:
        print(f"Error predicting CEFR: {e}")
        return fallback_predict(text)
