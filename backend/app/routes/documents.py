from pathlib import Path

from fastapi import APIRouter, Header

from app.config import settings
from app.db import sqlite
from app.db.chroma import image_collection, text_collection
from app.dependencies import get_retriever, resolve_session_id, validate_session_id
from app.errors import AppError

router = APIRouter()


@router.get("/documents")
def list_documents(session_id: str | None = None, x_session_id: str | None = Header(None)):
    """Only this session's documents (WHERE session_id = ?)."""
    return sqlite.list_documents(resolve_session_id(session_id, x_session_id))


@router.delete("/documents/{document_id}")
def delete_document(
    document_id: str,
    session_id: str | None = None,
    x_session_id: str | None = Header(None),
):
    session_id = resolve_session_id(session_id, x_session_id)
    doc = sqlite.get_document(session_id, document_id)
    if doc is None:
        # Same 404 whether the id is unknown or belongs to another session: no existence leak.
        raise AppError(404, "Document not found.")

    where = {"$and": [{"document_id": document_id}, {"session_id": session_id}]}
    for collection in (text_collection(), image_collection()):
        collection.delete(where=where)
    get_retriever().invalidate()

    if doc["file_path"]:
        path = Path(doc["file_path"]).resolve()
        # Only ever delete inside this session's own upload folder.
        if (settings.upload_dir / session_id).resolve() in path.parents and path.exists():
            path.unlink()
    sqlite.delete_document(session_id, document_id)
    return {"message": "Document deleted.", "document_id": document_id, "filename": doc["filename"]}
