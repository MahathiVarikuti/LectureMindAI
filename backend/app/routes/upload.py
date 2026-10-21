import hashlib
import os
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from app import config
from app.db.chroma import collection
from app.dependencies import retriever
from app.services.embedding_service import embed_documents
from app.services.pdf_service import extract_text_from_pdf
from app.utils.chunker import chunk_text

router = APIRouter()

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads")
ADD_BATCH_SIZE = 1000

os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    document_id = str(uuid4())
    stored_filename = f"{document_id}_{os.path.basename(file.filename)}"
    file_path = os.path.join(UPLOAD_DIR, stored_filename)

    # Stream to disk in small blocks so a huge upload can't exhaust RAM.
    digest, size = hashlib.sha256(), 0
    max_bytes = config.MAX_PDF_SIZE_MB * 1024 * 1024
    try:
        with open(file_path, "wb") as f:
            while block := await file.read(1024 * 1024):
                size += len(block)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"PDF is larger than {config.MAX_PDF_SIZE_MB} MB.",
                    )
                digest.update(block)
                f.write(block)
    except BaseException:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    file_hash = digest.hexdigest()

    existing = collection.get(where={"file_hash": file_hash}, limit=1, include=["metadatas"])
    if existing["ids"]:
        os.remove(file_path)
        meta = existing["metadatas"][0]
        return {
            "filename": meta["filename"],
            "document_id": meta["document_id"],
            "chunks": 0,
            "duplicate": True,
            "message": "This file was already uploaded.",
        }

    uploaded_at = datetime.now(timezone.utc).isoformat()
    all_chunks, all_metadata = [], []
    for page in extract_text_from_pdf(pdf_path=file_path, document_id=document_id, filename=stored_filename):
        for chunk in chunk_text(page["text"]):
            all_metadata.append({
                "filename": file.filename,
                "stored_filename": stored_filename,
                "file_hash": file_hash,
                "page": page["page"],
                "chunk_index": len(all_chunks),
                "source_type": "pdf",
                "document_id": document_id,
                "uploaded_at": uploaded_at,
            })
            all_chunks.append(chunk)

    if not all_chunks:
        os.remove(file_path)
        raise HTTPException(
            status_code=422,
            detail="No extractable text found. The PDF may be scanned images (needs OCR).",
        )

    embeddings = embed_documents(all_chunks)
    ids = [f"{document_id}_{i}" for i in range(len(all_chunks))]
    for start in range(0, len(ids), ADD_BATCH_SIZE):
        end = start + ADD_BATCH_SIZE
        collection.add(
            ids=ids[start:end],
            documents=all_chunks[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=all_metadata[start:end],
        )
    retriever.invalidate()

    return {
        "filename": file.filename,
        "chunks": len(all_chunks),
        "message": "File uploaded successfully!",
        "document_id": document_id,
        "duplicate": False,
    }
