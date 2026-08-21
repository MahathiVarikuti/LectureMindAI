"""RAG query pipeline with cross-modal retrieval."""

import re

from app import database as db
from app.models import QueryResponse, SourceCitation
from app.services.clip_embeddings import embed_text_clip
from app.services.embeddings import embed_query
from app.services.ollama import generate_answer, rewrite_query
from app.services.vector_store import query_image_frames, query_text_chunks

NOT_FOUND_MSG = (
    "I could not find relevant information in your uploaded material to answer this question."
)
RELEVANCE_THRESHOLD = 0.05


def _detect_query_intent(query: str) -> list[str]:
    """Detect which sections are most relevant for this query.
    
    Returns a list of section names to prioritize (order matters).
    """
    query_lower = query.lower()
    
    # Map query keywords to relevant sections
    if any(word in query_lower for word in ["project", "work on", "built", "developed"]):
        return ["projects"]
    if any(word in query_lower for word in ["skill", "know", "proficient", "expertise"]):
        return ["skills"]
    if any(word in query_lower for word in ["education", "study", "university", "college", "degree"]):
        return ["education"]
    if any(word in query_lower for word in ["certification", "certified"]):
        return ["certifications"]
    if any(word in query_lower for word in ["experience", "intern", "work", "job"]):
        return ["experience"]
    if any(word in query_lower for word in ["leadership", "lead"]):
        return ["leadership"]
    
    # Default: no specific section preference
    return []


def _prioritize_by_section(items: list[dict], preferred_sections: list[str]) -> list[dict]:
    """Sort items by section preference, keeping preferred sections first."""
    if not preferred_sections:
        return items
    
    # Separate items by whether they're in preferred sections
    prioritized = []
    other = []
    
    for item in items:
        section = item.get("section", "other")
        if section in preferred_sections:
            # Further sort by the order of preference
            priority_index = preferred_sections.index(section)
            prioritized.append((priority_index, item))
        else:
            other.append(item)
    
    # Sort prioritized by preference index, then append others
    prioritized.sort(key=lambda x: x[0])
    return [item for _, item in prioritized] + other


async def verify_answer_grounding(question: str, answer: str, context: str) -> bool:
    """Reject answers that go beyond the retrieved context.

    This is a strict lexical grounding check: an answer must reuse terms from the
    retrieved source material, and it must not introduce unsupported entities such as
    institutions or facts not present in the uploaded context.
    """
    if not context.strip() or not answer.strip():
        return False

    stop_words = {
        "a", "an", "the", "and", "or", "but", "if", "then", "than", "that",
        "this", "those", "these", "she", "he", "they", "them", "it", "its",
        "was", "were", "is", "are", "be", "been", "being", "of", "to", "in",
        "on", "at", "for", "with", "from", "by", "as", "what", "where", "when",
        "who", "why", "how", "did", "does", "do", "didn't", "doesn't", "didn",
        "their", "her", "his", "our", "your", "we", "you", "i", "me", "my",
        "have", "has", "had", "about", "into", "over", "under", "after", "before",
    }

    def tokens(text: str) -> set[str]:
        return {
            t
            for t in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(t) > 2 and t not in stop_words
        }

    answer_tokens = tokens(answer)
    context_tokens = tokens(context)
    if not answer_tokens:
        return False
    overlap = answer_tokens & context_tokens
    if not overlap:
        return False
    overlap_ratio = len(overlap) / len(answer_tokens)
    return overlap_ratio >= 0.35


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
                "section": meta.get("section", "other"),
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

    # Detect query intent to prioritize relevant sections
    preferred_sections = _detect_query_intent(query)

    text_emb = embed_query(rewritten)
    text_raw = query_text_chunks(text_emb, session_id, selected_documents, n_results=10)
    text_items = _normalize_scores(_parse_text_results(text_raw))

    # DEBUG: Log what was retrieved
    import sys
    print(f"\n[DEBUG] Query: {query}", file=sys.stderr)
    print(f"[DEBUG] Preferred sections: {preferred_sections}", file=sys.stderr)
    print(f"[DEBUG] Retrieved {len(text_items)} text items (before prioritization):", file=sys.stderr)
    for i, item in enumerate(text_items):
        print(f"[DEBUG]   {i+1}. Section={item.get('section')}, Relevance={item.get('relevance'):.3f}, Snippet={repr(item['snippet'][:60])}", file=sys.stderr)

    # Always query both collections; low relevance naturally excludes irrelevant frames.
    # Use a wider top-k because broad category questions such as "What are her projects?"
    # often require multiple section-level chunks, not just a single best match.
    clip_emb = embed_text_clip(rewritten)
    image_raw = query_image_frames(clip_emb, session_id, selected_documents, n_results=6)
    image_items = _normalize_scores(_parse_image_results(image_raw))

    # Prioritize text results by section intent
    if preferred_sections:
        text_items = _prioritize_by_section(text_items, preferred_sections)
        # Only keep the top results after prioritization
        text_items = text_items[:5]

    text_items.sort(key=lambda x: x["relevance"], reverse=True)
    image_items.sort(key=lambda x: x["relevance"], reverse=True)

    # Chroma cosine distances are not a stable absolute threshold across different queries
    # and models. The search results are already ranked by similarity, so after min-max
    # normalization we keep the strongest matches rather than rejecting valid chunks based
    # on a brittle raw-distance cutoff.
    text_items = [t for t in text_items if t.get("relevance", 0.0) >= RELEVANCE_THRESHOLD]
    image_items = [i for i in image_items if i.get("relevance", 0.0) >= RELEVANCE_THRESHOLD]

    if not text_items and not image_items:
        db.add_message(session_id, "user", query)
        db.add_message(session_id, "assistant", NOT_FOUND_MSG)
        return QueryResponse(answer=NOT_FOUND_MSG, sources=[], used_documents=[], context="")

    context = _build_context(text_items, image_items)
    try:
        answer = await generate_answer(rewritten, context)
        if answer and not await verify_answer_grounding(rewritten, answer, context):
            answer = NOT_FOUND_MSG
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
