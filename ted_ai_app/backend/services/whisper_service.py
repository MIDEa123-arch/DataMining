import whisper
import os

model = None

def load_model():
    global model
    if model is None:
        print("Loading Whisper model (base)...")
        # Sử dụng model 'base' cho tốc độ. Có thể đổi sang 'small', 'medium'
        model = whisper.load_model("base")
        print("Whisper model loaded.")

def transcribe_video(video_path: str):
    load_model()
    print(f"Transcribing {video_path}...")
    result = model.transcribe(video_path)
    
    segments = []
    full_text = result["text"]
    
    for segment in result["segments"]:
        segments.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"].strip()
        })
        
    return {
        "full_text": full_text.strip(),
        "segments": segments
    }
