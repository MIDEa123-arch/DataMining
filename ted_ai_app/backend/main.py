import os
import shutil
import imageio_ffmpeg

# Whisper gọi "ffmpeg" chứ không gọi đường dẫn đầy đủ, mà imageio_ffmpeg lại tải file tên "ffmpeg-win64-v4.2.2.exe"
ffmpeg_exe_path = imageio_ffmpeg.get_ffmpeg_exe()
local_ffmpeg_path = os.path.join(os.path.dirname(__file__), "ffmpeg.exe")
if not os.path.exists(local_ffmpeg_path):
    try:
        shutil.copy(ffmpeg_exe_path, local_ffmpeg_path)
    except:
        pass

# Thêm thư mục backend vào PATH để subprocess tìm thấy ffmpeg.exe
os.environ["PATH"] = os.path.dirname(__file__) + os.pathsep + os.environ.get("PATH", "")
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

app = FastAPI(title="TED AI Video App")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")

# Serve frontend static files
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

# Serve uploaded files
uploads_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
os.makedirs(uploads_dir, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")

app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
