"""FastAPI route handlers."""

import shutil
import tempfile
import uuid
from pathlib import Path

from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from slowapi import Limiter
from slowapi.util import get_remote_address

from app import database as db
from app.config import settings
from app.models import (
    DocumentInfo,
    QueryRequest,
    QueryResponse,
    UploadResponse,
    UploadUrlRequest,
)
from app.services.ingestion import ingest_audio, ingest_pdf, ingest_video
from app.services.query_service import run_query
from app.services.url_ingestion import ingest_url
from app.services.vector_store import delete_document_vectors
from app.utils.media import ALLOWED_EXTENSIONS, detect_modality

router = APIRouter()


def _key_func(request: Request) -> str:
    sid = request.query_params.get("session_id")
    if sid:
        return sid
    form_data = getattr(request.state, "_form_data", None)
    if isinstance(form_data, dict) and form_data.get("session_id"):
        return str(form_data["session_id"])
    body = getattr(request.state, "_body_json", None)
    if body and isinstance(body, dict) and body.get("session_id"):
        return str(body["session_id"])
    return get_remote_address(request)


limiter = Limiter(key_func=_key_func)


def _validate_session_id(session_id: str) -> None:
    try:
        value = UUID(str(session_id))
    except (TypeError, ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid session_id") from None
    if value.version != 4:
        raise HTTPException(status_code=400, detail="session_id must be a UUID v4")


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/upload", response_model=UploadResponse)
@limiter.limit(f"{settings.rate_limit_upload}/minute")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = Form(...),
):
    _validate_session_id(session_id)
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    modality = detect_modality(file.filename)
    if not modality:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    content = await file.read()
    if modality == "pdf" and len(content) > settings.max_pdf_size_bytes:
        raise HTTPException(status_code=413, detail="PDF file too large")

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        if modality == "pdf":
            result = ingest_pdf(session_id, tmp_path, file.filename)
        elif modality == "audio":
            result = ingest_audio(session_id, tmp_path, file.filename)
        else:
            result = ingest_video(session_id, tmp_path, file.filename)
        return UploadResponse(**result)
    except ValueError as e:
        msg = str(e)
        if "too large" in msg.lower() or "exceeded" in msg.lower():
            raise HTTPException(status_code=413, detail=msg)
        raise HTTPException(status_code=400, detail=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/upload-url", response_model=UploadResponse)
@limiter.limit(f"{settings.rate_limit_upload}/minute")
async def upload_url(request: Request, body: UploadUrlRequest):
    _validate_session_id(body.session_id)
    try:
        result = ingest_url(body.session_id, body.url)
        return UploadResponse(**result)
    except ValueError as e:
        msg = str(e)
        if "exceeded" in msg.lower():
            raise HTTPException(status_code=413, detail=msg)
        raise HTTPException(status_code=400, detail=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL ingestion failed: {e}")


@router.post("/query", response_model=QueryResponse)
@limiter.limit(f"{settings.rate_limit_query}/minute")
async def query(request: Request, body: QueryRequest):
    _validate_session_id(body.session_id)
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        return await run_query(body.query, body.session_id, body.selected_documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents(session_id: str = Query(...)):
    _validate_session_id(session_id)
    docs = db.get_documents(session_id)
    return [DocumentInfo(**d) for d in docs]


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    session_id: str = Query(...),
):
    _validate_session_id(session_id)
    doc = db.delete_document(document_id, session_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    delete_document_vectors(document_id, session_id)
    file_path = Path(doc["file_path"])
    if file_path.exists():
        file_path.unlink()

    work_dir = file_path.parent / f"{document_id}_work"
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)

    return {"message": "Document deleted", "document_id": document_id}


@router.get("/messages")
async def get_chat_history(session_id: str = Query(...)):
    _validate_session_id(session_id)
    return db.get_messages(session_id)
