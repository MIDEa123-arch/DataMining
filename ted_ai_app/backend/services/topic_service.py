import os
import joblib

model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "yahoo_10topics_tfidf_linearsvc.joblib")
labels_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "labels.txt")

topic_model = None
topic_labels = []

TOPIC_MAP = {
    "Computers & Internet": "Technology",
    "Education & Reference": "Education",
    "Business & Finance": "Business",
    "Health": "Health",
    "Science & Mathematics": "Science",
    "Sports": "Sports",
    "Entertainment & Music": "Entertainment",
    "Politics & Government": "Politics",
    "Family & Relationships": "Society",
    "Society & Culture": "Society",
}

def load_topic_model():
    global topic_model, topic_labels
    
    if not topic_labels:
        try:
            if os.path.exists(labels_path):
                with open(labels_path, "r", encoding="utf-8") as f:
                    topic_labels = [line.strip() for line in f if line.strip()]
            else:
                print("Warning: Topic labels file not found.")
        except Exception as e:
            print(f"Error loading topic labels: {e}")

    if topic_model is None:
        try:
            if os.path.exists(model_path):
                topic_model = joblib.load(model_path)
                print("Topic model loaded via joblib.")
            else:
                print("Warning: Topic model file not found.")
        except Exception as e:
            print(f"Error loading topic model: {e}")

def predict_topic(text: str) -> str:
    load_topic_model()
    
    if topic_model is None:
        raise RuntimeError("Topic model is not loaded; cannot predict topic.")
        
    try:
        prediction_idx = topic_model.predict([text])[0]
        
        if isinstance(prediction_idx, str):
            raw_label = prediction_idx
        elif 0 <= int(prediction_idx) < len(topic_labels):
            raw_label = topic_labels[int(prediction_idx)]
        else:
            raw_label = str(prediction_idx)
        return TOPIC_MAP.get(raw_label, raw_label)
    except Exception as e:
        raise RuntimeError(f"Error predicting topic: {e}") from e
