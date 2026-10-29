"""The full question-answering pipeline, shared by the API and the evaluation harness.

    query -> (rewrite if follow-up) -> hybrid retrieve -> rerank
          -> relevance threshold (abstain early, no LLM call)
          -> grounded generation with citations
          -> NLI grounding check (flag / abstain on hallucination)
"""
import re
import time

from app import config
from app.services.frame_retrieval import retrieve_frames
from app.services.llm_service import (
    format_context,
    format_timestamp,
    generate_answer,
    is_not_found,
    rewrite_query,
)
from app.services.retrieval_service import select_context


def _source(chunk):
    meta = chunk.get("metadata") or {}
    return {
        "chunk_id": chunk["id"],
        "document_id": meta.get("document_id"),
        "filename": meta.get("filename"),
        "modality": meta.get("modality"),
        "page": meta.get("page"),
        "timestamp_start": meta.get("timestamp_start"),
        "timestamp_end": meta.get("timestamp_end"),
        "timestamp": meta.get("timestamp"),
        "text": chunk["text"],
        "score": round(float(chunk["score"]), 4),
    }


_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def align_citations(answer, sources):
    """Keep only the text sources the answer actually cites, renumbered 1..n in order of first use.

    The LLM cites passages as [1], [2]...; listing sources it never used makes the
    citations look wrong. Video frames are always kept (they are visual evidence the
    LLM cannot quote). If the answer cites nothing valid, return everything unchanged.
    """
    order = []
    for match in _CITATION.finditer(answer):
        for n in (int(x) for x in match.group(1).split(",")):
            if 1 <= n <= len(sources) and n not in order:
                order.append(n)
    if not order:
        return answer, sources
    order += [i for i, s in enumerate(sources, start=1) if s.get("modality") == "video_frame" and i not in order]
    new_number = {old: new for new, old in enumerate(order, start=1)}

    def rewrite(match):
        numbers = [new_number[n] for n in (int(x) for x in match.group(1).split(",")) if n in new_number]
        return "[" + ", ".join(str(n) for n in numbers) + "]" if numbers else ""

    return _CITATION.sub(rewrite, answer), [sources[n - 1] for n in order]


def answer_question(
    query,
    retriever,
    chat_history=None,
    document_ids=None,
    model=None,
    relevance_threshold=None,
    grounding=None,
    rewrite=True,
    session_id=None,
):
    timings = {}
    t0 = time.perf_counter()

    search_query = rewrite_query(chat_history or [], query, model=model) if rewrite else query
    timings["rewrite"] = time.perf_counter() - t0

    t = time.perf_counter()
    candidates = retriever.retrieve(search_query, document_ids=document_ids, session_id=session_id)
    if relevance_threshold is None:
        relevance_threshold = getattr(retriever, "relevance_threshold", None)
    chunks, abstain_reason = select_context(
        candidates, threshold=relevance_threshold,
        per_chunk_threshold=getattr(retriever, "use_reranker", True),
    )
    # Visual matches come from the separate image collection (session-scoped; skipped
    # entirely when the session has no video frames).
    frames = retrieve_frames(search_query, session_id, document_ids) if session_id else []
    timings["retrieval"] = time.perf_counter() - t

    result = {
        "query": query,
        "rewritten_query": search_query,
        "answer": config.NOT_FOUND_ANSWER,
        "abstained": True,
        "abstain_reason": abstain_reason,
        "top_score": round(max(float(c["score"]) for c in candidates), 4) if candidates else 0.0,
        "sources": [_source(c) for c in chunks] + [_source(f) for f in frames],
        "context": "",
        "grounding": None,
        "flagged": False,
        "timings": timings,
    }
    if abstain_reason:
        if frames:
            # No text matched, but some frames look relevant. Frames carry no text the
            # LLM could ground an answer in, so list them directly without an LLM call.
            listing = "\n".join(
                f"- {f['metadata'].get('filename', 'video')} at "
                f"{format_timestamp(f['metadata'].get('timestamp'))}" for f in frames
            )
            result.update(
                answer="No transcript or text matched, but these video moments look visually related:\n"
                       + listing,
                abstained=False, abstain_reason=None,
            )
        timings["total"] = time.perf_counter() - t0
        return result

    t = time.perf_counter()
    result["context"] = format_context(chunks, frames)
    answer = generate_answer(search_query, chunks, model=model, frames=frames)
    timings["generation"] = time.perf_counter() - t
    result["answer"] = answer

    if is_not_found(answer):
        result["abstain_reason"] = "llm_not_found"
        timings["total"] = time.perf_counter() - t0
        return result
    result["abstained"] = False
    result["answer"], result["sources"] = align_citations(answer, result["sources"])

    use_grounding = config.USE_GROUNDING_CHECK if grounding is None else grounding
    if use_grounding:
        from app.services.grounding_service import check_grounding

        t = time.perf_counter()
        report = check_grounding(answer, [c["text"] for c in chunks])
        timings["grounding"] = time.perf_counter() - t
        result["grounding"] = report
        result["flagged"] = report["groundedness"] < config.GROUNDEDNESS_THRESHOLD

    timings["total"] = time.perf_counter() - t0
    return result
