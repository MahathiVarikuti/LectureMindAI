"""Ingest audio/video from a URL via yt-dlp.

Intended for the user's own recordings or openly-licensed content (the UI says so); this
is not a general-purpose downloader. The URL is checked to be a public http(s) address
(no localhost / private networks) before anything is fetched, and the media is downloaded
to a temp directory that is always removed afterwards.
"""
import ipaddress
import shutil
import socket
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse

from app.config import settings
from app.errors import AppError
from app.services import ingestion
from app.utils.media import safe_filename


def validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise AppError(400, "Only http(s) URLs are supported.")
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(parsed.hostname, None)}
    except socket.gaierror as exc:
        raise AppError(400, "Could not resolve that URL's host.") from exc
    if not all(ipaddress.ip_address(a).is_global for a in addresses):
        raise AppError(400, "URLs pointing at private or local addresses are not allowed.")


def _download(url: str, tmp: Path, want_video: bool) -> tuple[Path, str]:
    import yt_dlp
    from yt_dlp.utils import DownloadError

    opts = {
        "format": "best[height<=480]/best" if want_video else "bestaudio/best",
        "outtmpl": str(tmp / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,
        "retries": 2,
        "max_filesize": settings.max_media_size_bytes,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            # Probe first: reject live streams and over-long media without downloading them.
            info = ydl.extract_info(url, download=False)
            if info.get("is_live"):
                raise AppError(400, "Live streams are not supported.")
            duration = info.get("duration")
            if duration and duration > settings.max_media_duration_seconds:
                raise AppError(413, f"Media is {duration / 60:.0f} min long; the limit is "
                                    f"{settings.max_media_duration_minutes} min.")
            ydl.extract_info(url, download=True)
    except DownloadError as exc:
        raise AppError(400, f"Could not download that URL: {str(exc)[:200]}") from exc
    files = [p for p in tmp.iterdir() if p.is_file() and not p.name.endswith((".part", ".ytdl"))]
    if not files:
        raise AppError(400, "Nothing was downloaded (the file may exceed the size limit).")
    title = (info.get("title") or files[0].stem)
    return files[0], safe_filename(f"{title}{files[0].suffix}")


def ingest_url(session_id: str, url: str, include_frames: bool = False) -> dict:
    ingestion.check_can_accept(session_id)
    validate_public_url(url)
    tmp = Path(tempfile.mkdtemp(prefix="lecturemind_url_"))
    try:
        path, filename = _download(url, tmp, want_video=include_frames)
        ingestion.check_storage(session_id, 0)
        document_id = uuid.uuid4().hex
        # Temp file is deleted afterwards, so the document has no stored file / storage cost.
        if include_frames:
            return ingestion.ingest_video(session_id, document_id, str(path), filename)
        return ingestion.ingest_audio(session_id, document_id, str(path), filename)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
