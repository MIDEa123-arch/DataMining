import os
import shutil

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

import database
from services.cefr_service import predict_cefr
from services.question_service import generate_questions
from services.topic_service import predict_topic
from services.vocab_service import extract_vocabulary
from services.whisper_service import transcribe_video

router = APIRouter()

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


class TextRequest(BaseModel):
    text: str


def _process_uploaded_video(file_path: str, filename: str):
    transcript_data = transcribe_video(file_path)
    full_text = transcript_data["full_text"]
    cefr_level = predict_cefr(full_text)
    topic = predict_topic(full_text)
    questions = generate_questions(full_text)
    vocabulary = extract_vocabulary(full_text, top_n=15, target_cefr=cefr_level)

    record = {
        "filename": filename,
        "cefr_level": cefr_level,
        "topic": topic,
        "transcript": transcript_data["segments"],
        "questions": questions,
        "vocabulary": vocabulary,
        "video_url": f"/uploads/{filename}",
        "full_text": full_text,
    }

    database.save_video_record(record)
    return record


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    filename = os.path.basename(file.filename or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return _process_uploaded_video(file_path, filename)


@router.get("/videos")
async def get_videos():
    return database.get_all_videos()


@router.get("/videos/{video_id}")
async def get_video_by_id(video_id: str):
    video = database.get_video_by_id(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video


@router.delete("/videos/{video_id}")
async def delete_video(video_id: str):
    video = database.delete_video_record(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    filename = os.path.basename(video.get("filename", ""))
    if filename:
        file_path = os.path.abspath(os.path.join(UPLOAD_DIR, filename))
        upload_root = os.path.abspath(UPLOAD_DIR)
        if file_path.startswith(upload_root + os.sep) and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Deleted record but could not delete file: {exc}",
                )

    return {"deleted": True, "id": video_id}


@router.post("/generate_questions")
async def get_questions(req: TextRequest):
    questions = generate_questions(req.text)
    return {"questions": questions}
