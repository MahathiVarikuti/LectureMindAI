import hashlib
import os
import uuid

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.config import settings
from app.db import sqlite
from app.dependencies import validate_session_id
from app.errors import AppError
from app.limiter import limiter, upload_limit
from app.models import UploadUrlRequest
from app.services import ingestion
from app.services.url_ingestion import ingest_url
from app.utils.media import detect_modality, looks_like_pdf, safe_filename

router = APIRouter()
BLOCK = 1024 * 1024


async def _save_stream(file: UploadFile, dest: str, max_bytes: int, session_id: str):
    """Stream to disk in 1 MB blocks (never the whole file in RAM), enforcing the size and
    storage limits as bytes arrive so an oversized upload is rejected early."""
    digest, size = hashlib.sha256(), 0
    used = sqlite.total_storage(session_id)
    with open(dest, "wb") as out:
        while block := await file.read(BLOCK):
            size += len(block)
            if size > max_bytes:
                raise AppError(413, f"File is larger than the {max_bytes // (1024 * 1024)} MB limit.")
            if used + size > settings.max_storage_per_session_bytes:
                raise AppError(413, f"Storage limit reached ({settings.max_storage_per_session_mb} MB per session).")
            digest.update(block)
            out.write(block)
    return digest.hexdigest(), size


@router.post("/upload")
@limiter.limit(upload_limit)
async def upload(request: Request, file: UploadFile = File(...), session_id: str = Form(...)):
    session_id = validate_session_id(session_id)
    filename = safe_filename(file.filename or "")
    modality = detect_modality(filename)  # by extension: client MIME type is not trusted
    if modality is None:
        raise AppError(400, "Unsupported file type. Upload a PDF, audio (mp3/wav/m4a/...) or video (mp4/webm/mkv/...).")
    ingestion.check_can_accept(session_id)

    document_id = uuid.uuid4().hex
    session_dir = settings.upload_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    dest = str(session_dir / f"{document_id}_{filename}")
    max_bytes = settings.max_pdf_size_bytes if modality == "pdf" else settings.max_media_size_bytes

    try:
        file_hash, _size = await _save_stream(file, dest, max_bytes, session_id)
        if modality == "pdf" and not looks_like_pdf(dest):
            raise AppError(400, "The file is not a valid PDF.")
        existing = sqlite.find_by_hash(session_id, file_hash)
        if existing:
            os.remove(dest)
            return {"document_id": existing["document_id"], "filename": existing["filename"],
                    "chunks": existing["chunk_count"], "modality": existing["modality"],
                    "duplicate": True, "message": "This file was already uploaded."}
        ingest = {"pdf": ingestion.ingest_pdf, "audio": ingestion.ingest_audio,
                  "video": ingestion.ingest_video}[modality]
        return await run_in_threadpool(ingest, session_id, document_id, dest, filename, file_hash, dest)
    except BaseException:
        if os.path.exists(dest):
            os.remove(dest)
        raise


@router.post("/upload-url")
@limiter.limit(upload_limit)
async def upload_url(request: Request, body: UploadUrlRequest):
    session_id = validate_session_id(body.session_id)
    return await run_in_threadpool(ingest_url, session_id, body.url.strip(), body.include_frames)
