"""Visual retrieval: match a text query against sampled video frames (CLIP).

Scores from CLIP (text->image cosine, typically 0.2-0.35) and from the text embedder
(cosine, typically 0.3-0.8) are not on comparable scales, so we never rank them in one
list. Each collection is queried independently and thresholded on its own scale; the
frames are then shown to the LLM and the user as a separate, labelled section.
"""
from app.config import settings
from app.db.chroma import build_where, image_collection


def retrieve_frames(query, session_id, document_ids=None, embed_text=None):
    if not session_id:
        return []
    collection = image_collection()
    where = build_where(session_id, document_ids)
    # Skip loading CLIP at all when this session has no video frames.
    if not collection.get(where=where, limit=1)["ids"]:
        return []
    if embed_text is None:
        from app.services.clip_service import embed_text
    res = collection.query(
        query_embeddings=[embed_text(query)],
        n_results=settings.max_image_hits,
        where=where,
        include=["metadatas", "distances"],
    )
    hits = []
    for cid, meta, dist in zip(res["ids"][0], res["metadatas"][0], res["distances"][0]):
        score = 1 - dist
        if score >= settings.image_min_score:
            hits.append({"id": cid, "metadata": meta or {}, "score": score, "text": ""})
    return hits
