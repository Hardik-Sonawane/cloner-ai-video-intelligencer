from pathlib import Path
import shutil
import ffmpeg
import imageio_ffmpeg

TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

FFMPEG_EXE = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()


def extract_audio(video_path):

    video_path = Path(video_path)

    audio_path = TEMP_DIR / f"{video_path.stem}.mp3"

    (
        ffmpeg
        .input(str(video_path))
        .output(
            str(audio_path),
            acodec="libmp3lame",
            audio_bitrate="64k",
            ac=1,
            ar=16000
        )
        .overwrite_output()
        .run(cmd=FFMPEG_EXE, quiet=True)
    )

    return str(audio_path)
