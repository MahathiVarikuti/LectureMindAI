"""Ingestion pipelines: PDF, audio, video (transcript + frames). All limits are checked
before the expensive work (transcription / embedding) starts.

Callers pass a path to a file already stored on disk (or in a temp dir for URLs). On any
failure the vectors written so far are purged so a half-ingested document never lingers.
"""
import shutil
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

from app.config import settings
from app.db import sqlite
from app.db.chroma import image_collection, text_collection
from app.dependencies import get_retriever
from app.errors import AppError
from app.services.embedding_service import embed_documents
from app.services.pdf_service import extract_text_from_pdf
from app.utils.chunker import chunk_pages, chunk_segments
from app.utils.media import extract_audio, get_media_duration, sample_frames

ADD_BATCH = 500
_slots = threading.BoundedSemaphore(settings.max_concurrent_ingestions)


@contextmanager
def ingestion_slot():
    """Models are memory-hungry; run only a few ingestions at once (serialised by default)."""
    if not _slots.acquire(timeout=120):
        raise AppError(503, "The server is busy processing other uploads. Please retry shortly.")
    try:
        yield
    finally:
        _slots.release()


def check_can_accept(session_id: str) -> None:
    if sqlite.count_documents(session_id) >= settings.max_documents_per_session:
        raise AppError(413, f"Document limit reached ({settings.max_documents_per_session} per session).")


def check_storage(session_id: str, incoming_bytes: int) -> None:
    if sqlite.total_storage(session_id) + incoming_bytes > settings.max_storage_per_session_bytes:
        raise AppError(413, f"Storage limit reached ({settings.max_storage_per_session_mb} MB per session).")


def check_duration(path: str) -> float:
    duration = get_media_duration(path)
    if duration > settings.max_media_duration_seconds:
        raise AppError(
            413, f"Media is {duration / 60:.0f} min long; the limit is {settings.max_media_duration_minutes} min."
        )
    return duration


def purge_vectors(document_id: str) -> None:
    for collection in (text_collection(), image_collection()):
        collection.delete(where={"document_id": document_id})
    get_retriever().invalidate()


def reconcile_orphans(session_id: str | None = None) -> None:
    """Remove vectors whose document ownership row no longer exists."""
    known = sqlite.document_ids(session_id)
    for collection in (text_collection(), image_collection()):
        data = collection.get(include=["metadatas"])
        orphan_ids = [
            vector_id for vector_id, meta in zip(data["ids"], data.get("metadatas") or [])
            if (not session_id or (meta or {}).get("session_id") == session_id)
            and (meta or {}).get("document_id") not in known
        ]
        if orphan_ids:
            collection.delete(ids=orphan_ids)
    get_retriever().invalidate()


def _store_text(session_id, document_id, filename, modality, chunks) -> int:
    """chunks: [{text, page?, timestamp_start?, timestamp_end?}] -> text_chunks collection."""
    if not chunks:
        return 0
    uploaded_at = sqlite.now_iso()
    texts = [c["text"] for c in chunks]
    embeddings = embed_documents(texts)
    collection = text_collection()
    for start in range(0, len(chunks), ADD_BATCH):
        ids, metas = [], []
        for i in range(start, min(start + ADD_BATCH, len(chunks))):
            c = chunks[i]
            meta = {
                "session_id": session_id, "document_id": document_id, "filename": filename,
                "modality": modality, "chunk_id": i, "uploaded_at": uploaded_at,
            }
            # Optional fields are omitted (not None): Chroma rejects None metadata values.
            for key in ("page", "timestamp_start", "timestamp_end"):
                if c.get(key) is not None:
                    meta[key] = c[key]
            ids.append(f"{document_id}_{i}")
            metas.append(meta)
        collection.add(
            ids=ids, documents=texts[start:start + ADD_BATCH],
            embeddings=embeddings[start:start + ADD_BATCH].tolist(), metadatas=metas,
        )
    return len(chunks)


def _store_frames(session_id, document_id, filename, frames) -> int:
    if not frames:
        return 0
    from app.services.clip_service import embed_images

    uploaded_at = sqlite.now_iso()
    collection = image_collection()
    for start in range(0, len(frames), ADD_BATCH):
        batch = frames[start:start + ADD_BATCH]
        collection.add(
            ids=[f"{document_id}_frame_{start + i}" for i in range(len(batch))],
            embeddings=embed_images([f["path"] for f in batch]),
            metadatas=[{
                "session_id": session_id, "document_id": document_id, "filename": filename,
                "modality": "video_frame", "timestamp": float(f["timestamp"]),
                "uploaded_at": uploaded_at,
            } for f in batch],
        )
    return len(frames)


def _result(document_id, filename, chunks, modality, message):
    return {"document_id": document_id, "filename": filename, "chunks": chunks,
            "modality": modality, "message": message}


def ingest_pdf(session_id, document_id, path, filename, file_hash=None, stored_path=None):
    size = Path(path).stat().st_size
    try:
        with ingestion_slot():
            chunks = chunk_pages(extract_text_from_pdf(pdf_path=str(path)))
            if not chunks:
                raise AppError(422, "No extractable text found. The PDF may be scanned images (needs OCR).")
            count = _store_text(session_id, document_id, filename, "pdf", chunks)
        sqlite.add_document(document_id, session_id, filename, "pdf", stored_path, size, count, file_hash)
    except BaseException:
        purge_vectors(document_id)
        raise
    get_retriever().invalidate()
    return _result(document_id, filename, count, "pdf", f"PDF ingested ({count} chunks).")


def _transcribe_to_chunks(path):
    from app.services.transcription import transcribe_audio

    return chunk_segments(transcribe_audio(path))


def ingest_audio(session_id, document_id, path, filename, file_hash=None, stored_path=None):
    size = Path(path).stat().st_size
    check_duration(path)  # fail fast, before the expensive transcription
    try:
        with ingestion_slot():
            chunks = _transcribe_to_chunks(path)
            if not chunks:
                raise AppError(422, "No speech could be transcribed from this audio.")
            count = _store_text(session_id, document_id, filename, "audio", chunks)
        sqlite.add_document(document_id, session_id, filename, "audio", stored_path, size, count, file_hash)
    except BaseException:
        purge_vectors(document_id)
        raise
    get_retriever().invalidate()
    return _result(document_id, filename, count, "audio", f"Audio ingested ({count} transcript chunks).")


def ingest_video(session_id, document_id, path, filename, file_hash=None, stored_path=None,
                 with_frames=True):
    """Video = audio track -> transcript pipeline, plus sampled frames -> CLIP -> image_frames."""
    size = Path(path).stat().st_size
    duration = check_duration(path)
    work = Path(tempfile.mkdtemp(prefix="lecturemind_"))
    try:
        with ingestion_slot():
            audio_path = extract_audio(path, str(work / "audio.wav"))
            chunks = _transcribe_to_chunks(audio_path)
            text_count = _store_text(session_id, document_id, filename, "video_transcript", chunks)
            frame_count = 0
            if with_frames:
                max_frames = int(duration // settings.frame_sample_interval) + 2
                frames = sample_frames(path, str(work / "frames"), settings.frame_sample_interval, max_frames)
                frame_count = _store_frames(session_id, document_id, filename, frames)
            if text_count + frame_count == 0:
                raise AppError(422, "No speech or frames could be extracted from this video.")
        sqlite.add_document(document_id, session_id, filename, "video", stored_path, size,
                            text_count + frame_count, file_hash)
    except BaseException:
        purge_vectors(document_id)
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
    get_retriever().invalidate()
    return _result(document_id, filename, text_count + frame_count, "video",
                   f"Video ingested ({text_count} transcript chunks, {frame_count} frames).")
