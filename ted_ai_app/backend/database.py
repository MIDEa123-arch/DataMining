import os
import json
import uuid
from typing import Dict, Any

DB_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "db.json")

def _ensure_db_exists():
    data_dir = os.path.dirname(DB_FILE)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)

def read_db():
    _ensure_db_exists()
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading DB: {e}")
        return []

def write_db(data):
    _ensure_db_exists()
    try:
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error writing DB: {e}")

def save_video_record(record: Dict[str, Any]) -> str:
    """Save a new video record and return its ID. Overwrites if filename exists."""
    db = read_db()
    
    # Check if filename already exists to avoid duplicates
    filename = record.get("filename")
    for existing in db:
        if existing.get("filename") == filename:
            # Keep the old ID but update the content
            record["id"] = existing["id"]
            existing.update(record)
            write_db(db)
            return existing["id"]
            
    video_id = str(uuid.uuid4())
    record["id"] = video_id
    db.append(record)
    write_db(db)
    return video_id

def get_all_videos():
    """Return all videos (metadata only to save bandwidth)."""
    db = read_db()
    # Strip heavy fields like transcript/questions for list view
    return [
        {
            "id": v.get("id"),
            "filename": v.get("filename"),
            "video_url": v.get("video_url"),
            "cefr_level": v.get("cefr_level"),
            "topic": v.get("topic", "Technology")
        } for v in db
    ]

def get_video_by_id(video_id: str):
    """Get full details of a specific video."""
    db = read_db()
    for v in db:
        if v.get("id") == video_id:
            return v
    return None
