"""RAG query pipeline with cross-modal retrieval."""

from app import database as db
from app.models import QueryResponse, SourceCitation
from app.services.clip_embeddings import embed_text_clip
from app.services.embeddings import embed_query
from app.services.ollama import generate_answer, rewrite_query
from app.services.vector_store import query_image_frames, query_text_chunks

NOT_FOUND_MSG = (
    "I could not find relevant information in your uploaded material to answer this question."
)
RELEVANCE_THRESHOLD = 0.65  # cosine distance; lower is better for Chroma


def _normalize_scores(items: list[dict]) -> list[dict]:
    """Min-max normalize distances to relevance scores (higher = better)."""
    if not items:
        return []
    distances = [it["distance"] for it in items]
    min_d, max_d = min(distances), max(distances)
    span = max_d - min_d if max_d > min_d else 1.0
    for it in items:
        it["relevance"] = 1.0 - ((it["distance"] - min_d) / span)
    return items


def _parse_text_results(raw: dict) -> list[dict]:
    items: list[dict] = []
    if not raw or not raw.get("ids") or not raw["ids"][0]:
        return items
    for i, doc_id in enumerate(raw["ids"][0]):
        meta = raw["metadatas"][0][i]
        items.append(
            {
                "type": "text",
                "document_id": meta.get("document_id", ""),
                "filename": meta.get("filename", ""),
                "modality": meta.get("modality", ""),
                "page": meta.get("page"),
                "timestamp_start": meta.get("timestamp_start"),
                "timestamp_end": meta.get("timestamp_end"),
                "snippet": raw["documents"][0][i][:300] if raw.get("documents") else "",
                "distance": raw["distances"][0][i],
            }
        )
    return items


def _parse_image_results(raw: dict) -> list[dict]:
    items: list[dict] = []
    if not raw or not raw.get("ids") or not raw["ids"][0]:
        return items
    for i, _ in enumerate(raw["ids"][0]):
        meta = raw["metadatas"][0][i]
        items.append(
            {
                "type": "image",
                "document_id": meta.get("document_id", ""),
                "filename": meta.get("filename", ""),
                "modality": meta.get("modality", "video_frame"),
                "timestamp": meta.get("timestamp"),
                "snippet": f"[Video frame at {meta.get('timestamp', 0):.1f}s]",
                "distance": raw["distances"][0][i],
            }
        )
    return items


def _format_source_label(item: dict) -> str:
    fname = item.get("filename", "unknown")
    if item.get("page") is not None:
        return f"{fname} (page {item['page']})"
    if item.get("timestamp_start") is not None:
        ts = item["timestamp_start"]
        te = item.get("timestamp_end")
        if te is not None:
            return f"{fname} ({ts:.1f}s - {te:.1f}s)"
        return f"{fname} ({ts:.1f}s)"
    if item.get("timestamp") is not None:
        return f"{fname} (frame at {item['timestamp']:.1f}s)"
    return fname


def _build_context(text_items: list[dict], image_items: list[dict]) -> str:
    parts: list[str] = []
    if text_items:
        parts.append("=== Text Sources ===")
        for it in text_items:
            parts.append(f"[{_format_source_label(it)}]\n{it['snippet']}")
    if image_items:
        parts.append("=== Visual Sources ===")
        for it in image_items:
            parts.append(f"[{_format_source_label(it)}]\n{it['snippet']}")
    return "\n\n".join(parts)


async def run_query(
    query: str,
    session_id: str,
    selected_documents: list[str] | None = None,
) -> QueryResponse:
    history = db.get_messages(session_id, limit=10)
    rewritten = query
    if history:
        try:
            rewritten = await rewrite_query(query, history)
        except Exception:
            rewritten = query

    text_emb = embed_query(rewritten)
    text_raw = query_text_chunks(text_emb, session_id, selected_documents, n_results=5)
    text_items = _normalize_scores(_parse_text_results(text_raw))

    # Always query both collections; low relevance naturally excludes irrelevant frames
    clip_emb = embed_text_clip(rewritten)
    image_raw = query_image_frames(clip_emb, session_id, selected_documents, n_results=3)
    image_items = _normalize_scores(_parse_image_results(image_raw))

    text_items.sort(key=lambda x: x["relevance"], reverse=True)
    image_items.sort(key=lambda x: x["relevance"], reverse=True)

    # Filter by raw distance threshold
    text_items = [t for t in text_items if t["distance"] < RELEVANCE_THRESHOLD]
    image_items = [i for i in image_items if i["distance"] < RELEVANCE_THRESHOLD]

    if not text_items and not image_items:
        db.add_message(session_id, "user", query)
        db.add_message(session_id, "assistant", NOT_FOUND_MSG)
        return QueryResponse(answer=NOT_FOUND_MSG, sources=[], used_documents=[], context="")

    context = _build_context(text_items, image_items)
    try:
        answer = await generate_answer(rewritten, context)
    except Exception as e:
        answer = f"Error generating answer: {e}"

    sources: list[SourceCitation] = []
    used_docs: set[str] = set()
    for it in text_items + image_items:
        used_docs.add(it["document_id"])
        sources.append(
            SourceCitation(
                document_id=it["document_id"],
                filename=it["filename"],
                modality=it["modality"],
                page=it.get("page"),
                timestamp_start=it.get("timestamp_start"),
                timestamp_end=it.get("timestamp_end"),
                timestamp=it.get("timestamp"),
                snippet=it.get("snippet", ""),
            )
        )

    db.add_message(session_id, "user", query)
    db.add_message(session_id, "assistant", answer)

    return QueryResponse(
        answer=answer,
        sources=sources,
        used_documents=list(used_docs),
        context=context,
    )
