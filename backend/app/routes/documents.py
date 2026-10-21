import os

from fastapi import APIRouter, HTTPException

from app.db.chroma import collection
from app.dependencies import retriever
from app.routes.upload import UPLOAD_DIR

router = APIRouter()


@router.get("/documents")
def get_documents():
    results = collection.get(include=["metadatas"])

    documents = {}
    for meta in results.get("metadatas") or []:
        if not meta or not meta.get("document_id"):
            continue
        document_id = meta["document_id"]
        if document_id not in documents:
            documents[document_id] = {
                "document_id": document_id,
                "filename": meta.get("filename"),
                "source_type": meta.get("source_type", "pdf"),
                "chunk_count": 0,
                "uploaded_at": meta.get("uploaded_at"),
            }
        documents[document_id]["chunk_count"] += 1

    return list(documents.values())


@router.delete("/documents/{document_id}")
def delete_document(document_id: str):
    results = collection.get(where={"document_id": document_id}, include=["metadatas"])
    ids = results["ids"]
    if not ids:
        raise HTTPException(status_code=404, detail="Document not found")

    meta = results["metadatas"][0]
    filename = meta["filename"]
    # Files are stored as "<document_id>_<filename>"; older uploads lack the field.
    stored_filename = meta.get("stored_filename") or f"{document_id}_{filename}"
    file_path = os.path.join(UPLOAD_DIR, os.path.basename(stored_filename))
    if os.path.exists(file_path):
        os.remove(file_path)

    collection.delete(ids=ids)
    retriever.invalidate()

    return {
        "message": "Document deleted successfully",
        "document_id": document_id,
        "filename": filename,
        "deleted_chunks": len(ids),
    }
