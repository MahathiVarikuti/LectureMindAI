"""Document ingestion pipelines."""

import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app import database as db
from app.config import settings
from app.services.clip_embeddings import embed_image
from app.services.embeddings import embed_texts
from app.services.transcription import transcribe_audio
from app.services.vector_store import add_image_frames, add_text_chunks
from app.utils.chunking import chunk_segments, chunk_text
from app.utils.media import (
    extract_audio_from_video,
    get_media_duration,
    sample_video_frames,
)
from app.utils.pdf_extractor import extract_pdf_pages


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_dir(session_id: str) -> Path:
    d = settings.upload_path / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def validate_upload_limits(session_id: str, file_size: int, is_pdf: bool) -> None:
    if db.count_documents(session_id) >= settings.max_documents_per_session:
        raise ValueError("Maximum documents per session exceeded")
    if db.total_storage(session_id) + file_size > settings.max_storage_per_session_bytes:
        raise ValueError("Maximum storage per session exceeded")
    if is_pdf and file_size > settings.max_pdf_size_bytes:
        raise ValueError("PDF file too large")


def _store_text_chunks(
    session_id: str,
    document_id: str,
    filename: str,
    modality: str,
    chunks: list[dict],
) -> int:
    if not chunks:
        return 0
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    ids = []
    documents = []
    metadatas = []
    uploaded_at = _iso_now()
    for i, chunk in enumerate(chunks):
        ids.append(f"{document_id}_text_{i}")
        documents.append(chunk["text"])
        meta: dict = {
            "session_id": session_id,
            "document_id": document_id,
            "filename": filename,
            "modality": modality,
            "chunk_id": i,
            "uploaded_at": uploaded_at,
        }
        if chunk.get("page") is not None:
            meta["page"] = int(chunk["page"])
        if chunk.get("timestamp_start") is not None:
            meta["timestamp_start"] = float(chunk["timestamp_start"])
        if chunk.get("timestamp_end") is not None:
            meta["timestamp_end"] = float(chunk["timestamp_end"])
        metadatas.append(meta)
    add_text_chunks(ids, embeddings, documents, metadatas)
    return len(chunks)


def ingest_pdf(session_id: str, file_path: str, filename: str) -> dict:
    document_id = str(uuid.uuid4())
    file_size = Path(file_path).stat().st_size
    validate_upload_limits(session_id, file_size, is_pdf=True)

    dest = _session_dir(session_id) / f"{document_id}_{filename}"
    shutil.copy2(file_path, dest)

    pages = extract_pdf_pages(str(dest))
    chunks: list[dict] = []
    for page in pages:
        for text in chunk_text(page["text"]):
            chunks.append({"text": text, "page": page["page"]})

    count = _store_text_chunks(session_id, document_id, filename, "pdf", chunks)
    db.add_document(document_id, session_id, filename, "pdf", str(dest), file_size, count)
    return {
        "document_id": document_id,
        "filename": filename,
        "chunks": count,
        "modality": "pdf",
        "message": f"PDF ingested with {count} chunks",
    }


def ingest_audio(session_id: str, file_path: str, filename: str, modality: str = "audio") -> dict:
    document_id = str(uuid.uuid4())
    file_size = Path(file_path).stat().st_size
    validate_upload_limits(session_id, file_size, is_pdf=False)

    duration = get_media_duration(file_path)
    if duration > settings.max_media_duration_seconds:
        raise ValueError("Media duration exceeds limit")

    dest = _session_dir(session_id) / f"{document_id}_{filename}"
    shutil.copy2(file_path, dest)

    segments = transcribe_audio(str(dest))
    chunks = chunk_segments(segments)
    count = _store_text_chunks(session_id, document_id, filename, modality, chunks)
    db.add_document(document_id, session_id, filename, modality, str(dest), file_size, count)
    return {
        "document_id": document_id,
        "filename": filename,
        "chunks": count,
        "modality": modality,
        "message": f"Audio ingested with {count} chunks",
    }


def ingest_video(session_id: str, file_path: str, filename: str) -> dict:
    document_id = str(uuid.uuid4())
    file_size = Path(file_path).stat().st_size
    validate_upload_limits(session_id, file_size, is_pdf=False)

    duration = get_media_duration(file_path)
    if duration > settings.max_media_duration_seconds:
        raise ValueError("Media duration exceeds limit")

    dest = _session_dir(session_id) / f"{document_id}_{filename}"
    shutil.copy2(file_path, dest)

    work_dir = _session_dir(session_id) / f"{document_id}_work"
    work_dir.mkdir(exist_ok=True)
    audio_path = str(work_dir / "audio.wav")
    extract_audio_from_video(str(dest), audio_path)

    segments = transcribe_audio(audio_path)
    chunks = chunk_segments(segments)
    transcript_modality = "video_transcript"
    text_count = _store_text_chunks(
        session_id, document_id, filename, transcript_modality, chunks
    )

    frames_dir = str(work_dir / "frames")
    frames = sample_video_frames(str(dest), frames_dir, settings.frame_sample_interval)
    uploaded_at = _iso_now()
    if frames:
        frame_ids = []
        frame_embeddings = []
        frame_metas = []
        for i, frame in enumerate(frames):
            frame_ids.append(f"{document_id}_frame_{i}")
            frame_embeddings.append(embed_image(frame["path"]))
            frame_metas.append(
                {
                    "session_id": session_id,
                    "document_id": document_id,
                    "filename": filename,
                    "modality": "video_frame",
                    "timestamp": frame["timestamp"],
                    "uploaded_at": uploaded_at,
                }
            )
        add_image_frames(frame_ids, frame_embeddings, frame_metas)

    total_chunks = text_count + len(frames)
    db.add_document(
        document_id, session_id, filename, "video", str(dest), file_size, total_chunks
    )

    try:
        shutil.rmtree(work_dir)
    except Exception:
        pass

    return {
        "document_id": document_id,
        "filename": filename,
        "chunks": total_chunks,
        "modality": "video",
        "message": f"Video ingested: {text_count} transcript chunks, {len(frames)} frames",
    }
