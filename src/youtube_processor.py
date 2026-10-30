import os
from pathlib import Path
import shutil
import imageio_ffmpeg
from yt_dlp import YoutubeDL

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def cleanup_stale_parts():
    """Remove any orphaned .part files that cause HTTP 416 errors."""
    try:
        for part_file in UPLOAD_DIR.glob("*.part"):
            try:
                part_file.unlink(missing_ok=True)
            except Exception:
                pass
    except Exception:
        pass


def download_youtube_video(url: str) -> str:
    """
    Downloads YouTube video or audio stream with automatic retry and error recovery.
    Returns the resolved local file path.
    """
    cleanup_stale_parts()

    ffmpeg_exe = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()

    client_configurations = [
        ["android", "web", "mweb"],
        ["android_creator", "web"],
        ["tv", "web"],
        None,  # default clients
    ]

    last_error = None

    for clients in client_configurations:
        ydl_opts = {
            "outtmpl": str(UPLOAD_DIR / "%(title)s.%(ext)s"),
            # Prefer crisp audio / lightweight 720p stream to ensure fast download & avoid timeouts
            "format": "bestaudio[ext=m4a]/bestaudio/best[ext=mp4][height<=720]/best",
            "ffmpeg_location": ffmpeg_exe,
            "nocheckcertificate": True,
            "overwrites": True,
            "continuedl": False,
            "quiet": True,
            "no_warnings": True,
            "ignoreerrors": False,
        }

        if clients:
            ydl_opts["extractor_args"] = {"youtube": {"player_client": clients}}

        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if not info:
                    raise RuntimeError("Could not retrieve video metadata.")

                # Resolve downloaded filename
                expected_path = Path(ydl.prepare_filename(info))
                if expected_path.exists():
                    return str(expected_path)

                # Check if file downloaded with another extension (e.g. .m4a, .webm, .mp4)
                stem = expected_path.stem
                for candidate in UPLOAD_DIR.glob(f"{stem}.*"):
                    if candidate.suffix.lower() not in [".part", ".ytdl"] and candidate.is_file():
                        return str(candidate)

                return str(expected_path)

        except Exception as e:
            last_error = e
            cleanup_stale_parts()
            continue

    raise RuntimeError(f"Failed to download YouTube video after multiple strategies: {last_error}")
