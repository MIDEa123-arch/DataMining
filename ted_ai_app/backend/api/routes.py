from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
from services.whisper_service import transcribe_video
from services.cefr_service import predict_cefr
from services.question_service import generate_questions
from services.topic_service import predict_topic
from services.vocab_service import extract_vocabulary
import database

router = APIRouter()

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

class TextRequest(BaseModel):
    text: str

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # 1. Trích xuất text bằng Whisper
    transcript_data = transcribe_video(file_path)
    full_text = transcript_data["full_text"]
    
    # 2. Đánh giá CEFR
    cefr_level = predict_cefr(full_text)
    
    # 3. Phân loại Topic
    topic = predict_topic(full_text)
    
    # 4. Gen câu hỏi bằng LLM
    questions = generate_questions(full_text)
    
    # 5. Rút trích từ vựng
    vocabulary = extract_vocabulary(full_text, top_n=15)
    
    video_url = f"/uploads/{file.filename}"
    
    record = {
        "filename": file.filename,
        "cefr_level": cefr_level,
        "topic": topic,
        "transcript": transcript_data["segments"],
        "questions": questions,
        "vocabulary": vocabulary,
        "video_url": video_url,
        "full_text": full_text
    }
    
    # Lưu vào database
    video_id = database.save_video_record(record)
    
    return record

@router.get("/videos")
async def get_videos():
    return database.get_all_videos()

@router.get("/videos/{video_id}")
async def get_video_by_id(video_id: str):
    video = database.get_video_by_id(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video

@router.post("/generate_questions")
async def get_questions(req: TextRequest):
    questions = generate_questions(req.text)
    return {"questions": questions}

