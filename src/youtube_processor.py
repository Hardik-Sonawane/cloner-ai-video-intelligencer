from pathlib import Path
import shutil
import imageio_ffmpeg
from yt_dlp import YoutubeDL

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def download_youtube_video(url):
    ffmpeg_exe = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    
    ydl_opts = {
        "outtmpl": str(UPLOAD_DIR / "%(title)s.%(ext)s"),
        "format": "bestaudio/best[ext=mp4]/best",
        "ffmpeg_location": ffmpeg_exe,
        "nocheckcertificate": True,
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "ios", "mweb"]
            }
        }
    }
    
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_path = ydl.prepare_filename(info)
        
    return video_path
