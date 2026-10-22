"""Persistent ChromaDB with two deliberately separate collections.

text_chunks  - PDF text and audio/video transcripts (SentenceTransformers embeddings)
image_frames - sampled video frames (CLIP embeddings)

They are separate because the two embedding models live in different vector spaces
(and have different dimensions): mixing them in one collection would make distances
meaningless. Chroma metadata values must be str/int/float/bool, so optional fields
(page, timestamps) are simply omitted when they do not apply rather than set to None.
"""
from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=1)
def get_client():
    import chromadb

    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(settings.chroma_dir))


def text_collection():
    # Cosine space: embeddings are L2-normalised, so distance = 1 - cosine similarity.
    return get_client().get_or_create_collection(
        name=settings.text_collection, metadata={"hnsw:space": "cosine"}
    )


def image_collection():
    return get_client().get_or_create_collection(
        name=settings.image_collection, metadata={"hnsw:space": "cosine"}
    )


def build_where(session_id=None, document_ids=None):
    """Chroma `where` filter enforcing session ownership (and optional document scope)."""
    clauses = []
    if session_id:
        clauses.append({"session_id": session_id})
    if document_ids:
        clauses.append({"document_id": {"$in": list(document_ids)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}
