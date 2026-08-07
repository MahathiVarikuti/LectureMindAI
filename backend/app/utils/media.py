"""File type validation and media utilities."""

import subprocess
from pathlib import Path

PDF_EXTENSIONS = {".pdf"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".avi", ".mov", ".flv", ".wmv"}

ALLOWED_EXTENSIONS = PDF_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def detect_modality(filename: str) -> str | None:
    ext = Path(filename).suffix.lower()
    if ext in PDF_EXTENSIONS:
        return "pdf"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return None


def get_media_duration(file_path: str) -> float:
    """Get duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            file_path,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return float(result.stdout.strip())


def extract_audio_from_video(video_path: str, output_path: str) -> str:
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            output_path,
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr}")
    return output_path


def sample_video_frames(video_path: str, output_dir: str, interval: float) -> list[dict]:
    """Extract frames at given interval; returns [{path, timestamp}]."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pattern = str(out / "frame_%04d.jpg")

    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vf",
            f"fps=1/{interval}",
            "-q:v",
            "2",
            pattern,
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame sampling failed: {result.stderr}")

    duration = get_media_duration(video_path)
    frames: list[dict] = []
    for i, fp in enumerate(sorted(out.glob("frame_*.jpg"))):
        ts = i * interval
        if ts > duration:
            ts = duration
        frames.append({"path": str(fp), "timestamp": ts})
    return frames
