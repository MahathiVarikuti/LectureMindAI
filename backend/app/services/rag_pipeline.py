"""The full question-answering pipeline, shared by the API and the evaluation harness.

    query -> (rewrite if follow-up) -> hybrid retrieve -> rerank
          -> relevance threshold (abstain early, no LLM call)
          -> grounded generation with citations
          -> NLI grounding check (flag / abstain on hallucination)
"""
import time

from app import config
from app.services.grounding_service import check_grounding
from app.services.llm_service import generate_answer, is_not_found, rewrite_query
from app.services.retrieval_service import select_context


def _source(chunk):
    meta = chunk.get("metadata") or {}
    return {
        "chunk_id": chunk["id"],
        "document_id": meta.get("document_id"),
        "filename": meta.get("filename"),
        "page": meta.get("page"),
        "text": chunk["text"],
        "score": round(float(chunk["score"]), 4),
    }


def answer_question(
    query,
    retriever,
    chat_history=None,
    document_ids=None,
    model=None,
    relevance_threshold=None,
    grounding=None,
    rewrite=True,
):
    timings = {}
    t0 = time.perf_counter()

    search_query = rewrite_query(chat_history or [], query, model=model) if rewrite else query
    timings["rewrite"] = time.perf_counter() - t0

    t = time.perf_counter()
    candidates = retriever.retrieve(search_query, document_ids=document_ids)
    chunks, abstain_reason = select_context(candidates, threshold=relevance_threshold)
    timings["retrieval"] = time.perf_counter() - t

    result = {
        "query": query,
        "rewritten_query": search_query,
        "answer": config.NOT_FOUND_ANSWER,
        "abstained": True,
        "abstain_reason": abstain_reason,
        "top_score": round(float(candidates[0]["score"]), 4) if candidates else 0.0,
        "sources": [_source(c) for c in chunks],
        "grounding": None,
        "flagged": False,
        "timings": timings,
    }
    if abstain_reason:
        timings["total"] = time.perf_counter() - t0
        return result

    t = time.perf_counter()
    answer = generate_answer(search_query, chunks, model=model)
    timings["generation"] = time.perf_counter() - t
    result["answer"] = answer

    if is_not_found(answer):
        result["abstain_reason"] = "llm_not_found"
        timings["total"] = time.perf_counter() - t0
        return result
    result["abstained"] = False

    use_grounding = config.USE_GROUNDING_CHECK if grounding is None else grounding
    if use_grounding:
        t = time.perf_counter()
        report = check_grounding(answer, [c["text"] for c in chunks])
        timings["grounding"] = time.perf_counter() - t
        result["grounding"] = report
        result["flagged"] = report["groundedness"] < config.GROUNDEDNESS_THRESHOLD

    timings["total"] = time.perf_counter() - t0
    return result
