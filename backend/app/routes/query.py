import requests
from fastapi import APIRouter, Header, Request
from fastapi.concurrency import run_in_threadpool

from app import config
from app.db import sqlite
from app.dependencies import get_retriever, resolve_session_id, validate_session_id
from app.errors import AppError
from app.limiter import limiter, query_limit
from app.models import QueryRequest
from app.services.chat_memory import append_turn, clear_session, get_full_history, get_history
from app.services.rag_pipeline import answer_question

router = APIRouter()


def _run_query(body: QueryRequest, session_id: str, document_ids):
    return answer_question(
        body.query,
        get_retriever(),
        chat_history=get_history(session_id),
        document_ids=document_ids,
        model=body.model,
        session_id=session_id,
    )


@router.post("/query")
@limiter.limit(query_limit)
async def query_rag(request: Request, body: QueryRequest):
    session_id = validate_session_id(body.session_id)
    if body.model and body.model not in config.ALLOWED_MODELS:
        raise AppError(400, "The requested language model is not allowed.")
    if not body.query.strip():
        raise AppError(400, "Query is empty.")

    document_ids = None
    if body.selected_documents:
        owned = {d["document_id"] for d in sqlite.list_documents(session_id)}
        document_ids = [d for d in body.selected_documents if d in owned]
        if not document_ids:
            raise AppError(404, "Document not found.")

    try:
        result = await run_in_threadpool(_run_query, body, session_id, document_ids)
    except requests.Timeout as exc:
        raise AppError(504, "The language model timed out. Try again or use a smaller model.") from exc
    except requests.RequestException as exc:
        raise AppError(503, "The language model (Ollama) is unavailable. Is it running?") from exc

    append_turn(session_id, body.query, result["answer"])
    sources = result["sources"]
    return {
        **result,
        "used_documents": sorted({s["filename"] for s in sources if s["filename"]}),
    }


@router.get("/history")
def history(session_id: str | None = None, x_session_id: str | None = Header(None)):
    return {"messages": get_full_history(resolve_session_id(session_id, x_session_id))}


@router.delete("/history")
def reset_history(session_id: str | None = None, x_session_id: str | None = Header(None)):
    clear_session(resolve_session_id(session_id, x_session_id))
    return {"message": "Conversation cleared"}
