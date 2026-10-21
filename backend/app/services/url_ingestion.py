"""URL download via yt-dlp."""

import tempfile
import uuid
from pathlib import Path

import yt_dlp

from app.services.ingestion import ingest_audio, ingest_video
from app.utils.media import detect_modality


def download_url(url: str) -> tuple[str, str]:
    """Download media from URL; returns (file_path, filename)."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="lecturemind_"))
    out_template = str(tmp_dir / "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "best",
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
        "max_filesize": 500 * 1024 * 1024,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        if not Path(filename).exists():
            # Sometimes extension differs
            candidates = list(tmp_dir.glob("*"))
            if not candidates:
                raise RuntimeError("Download failed: no file produced")
            filename = str(candidates[0])
        title = info.get("title", "download")
        ext = Path(filename).suffix or ".mp4"
        safe_name = f"{title}{ext}"[:200]
        return filename, safe_name


def ingest_url(session_id: str, url: str) -> dict:
    file_path, filename = download_url(url)
    try:
        modality = detect_modality(filename)
        if modality == "video":
            return ingest_video(session_id, file_path, filename)
        if modality == "audio":
            return ingest_audio(session_id, file_path, filename)
        # Default to video pipeline for unknown ext
        return ingest_video(session_id, file_path, filename)
    finally:
        try:
            Path(file_path).unlink(missing_ok=True)
            parent = Path(file_path).parent
            if parent.name.startswith("lecturemind_"):
                import shutil
                shutil.rmtree(parent, ignore_errors=True)
        except Exception:
            pass
