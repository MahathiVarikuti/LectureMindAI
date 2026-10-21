"""File-type validation and ffmpeg/ffprobe helpers. All subprocess calls have timeouts and
turn failures into AppError so clients get a meaningful message, never a stack trace."""
import re
import subprocess
from pathlib import Path

from app.errors import AppError

PDF_EXTENSIONS = {".pdf"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".avi", ".mov", ".flv", ".wmv"}
ALLOWED_EXTENSIONS = PDF_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def safe_filename(name: str) -> str:
    """Strip any path and replace characters that are unsafe on Windows/Linux file systems."""
    base = re.split(r"[\\/]", name or "")[-1].strip()
    base = re.sub(r"[^\w.\- ()]", "_", base)[:150]
    return base or "upload"


def detect_modality(filename: str) -> str | None:
    """'pdf' | 'audio' | 'video' from the extension (the client MIME type is not trusted)."""
    ext = Path(filename).suffix.lower()
    if ext in PDF_EXTENSIONS:
        return "pdf"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return None


def looks_like_pdf(path) -> bool:
    with open(path, "rb") as f:
        return f.read(1024).lstrip().startswith(b"%PDF-")


def _run(cmd, timeout, what):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise AppError(500, f"{cmd[0]} is not installed on the server (needed for {what}).") from exc
    except subprocess.TimeoutExpired as exc:
        raise AppError(504, f"{what} timed out.") from exc


def get_media_duration(path: str) -> float:
    """Duration in seconds via ffprobe; also proves the file is real, decodable media."""
    result = _run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        30, "reading media duration",
    )
    try:
        if result.returncode != 0:
            raise ValueError
        return float(result.stdout.strip())
    except ValueError:
        raise AppError(400, "The file is not valid audio/video media.") from None


def extract_audio(video_path: str, output_path: str) -> str:
    """Video -> 16 kHz mono WAV, the format Whisper wants."""
    result = _run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le", "-ar", "16000",
         "-ac", "1", str(output_path)],
        900, "audio extraction",
    )
    if result.returncode != 0:
        raise AppError(400, "Could not extract an audio track from this video.")
    return output_path


def sample_frames(video_path: str, output_dir: str, interval: float, max_frames: int) -> list[dict]:
    """One frame every `interval` seconds -> [{path, timestamp}]. Frames are downscaled
    (CLIP resizes to 224px anyway) so temporary disk use stays small."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = _run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vf", f"fps=1/{interval},scale=-2:360",
         "-frames:v", str(max_frames), "-q:v", "3", str(out / "frame_%05d.jpg")],
        900, "frame sampling",
    )
    if result.returncode != 0:
        raise AppError(400, "Could not sample frames from this video.")
    # With fps=1/N the k-th output frame (0-based) is taken at ~k*N seconds.
    return [{"path": str(p), "timestamp": i * interval}
            for i, p in enumerate(sorted(out.glob("frame_*.jpg")))]
