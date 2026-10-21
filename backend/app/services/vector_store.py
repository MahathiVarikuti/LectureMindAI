"""ChromaDB vector store with dual collections."""

from datetime import datetime, timezone
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings

_client: chromadb.ClientAPI | None = None
TEXT_COLLECTION = "text_chunks"
IMAGE_COLLECTION = "image_frames"


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def get_text_collection():
    return get_client().get_or_create_collection(
        name=TEXT_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def get_image_collection():
    return get_client().get_or_create_collection(
        name=IMAGE_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_text_chunks(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    col = get_text_collection()
    col.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def add_image_frames(
    ids: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any]],
) -> None:
    col = get_image_collection()
    col.add(ids=ids, embeddings=embeddings, metadatas=metadatas)


def query_text_chunks(
    query_embedding: list[float],
    session_id: str,
    document_ids: list[str] | None = None,
    n_results: int = 5,
) -> dict:
    col = get_text_collection()
    where: dict[str, Any] = {"session_id": session_id}
    if document_ids:
        where = {"$and": [{"session_id": session_id}, {"document_id": {"$in": document_ids}}]}
    return col.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )


def query_image_frames(
    query_embedding: list[float],
    session_id: str,
    document_ids: list[str] | None = None,
    n_results: int = 3,
) -> dict:
    col = get_image_collection()
    where: dict[str, Any] = {"session_id": session_id}
    if document_ids:
        where = {"$and": [{"session_id": session_id}, {"document_id": {"$in": document_ids}}]}
    return col.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["metadatas", "distances"],
    )


def delete_document_vectors(document_id: str, session_id: str) -> None:
    for col_fn in (get_text_collection, get_image_collection):
        col = col_fn()
        try:
            results = col.get(
                where={"$and": [{"session_id": session_id}, {"document_id": document_id}]}
            )
            if results and results.get("ids"):
                col.delete(ids=results["ids"])
        except Exception:
            pass
