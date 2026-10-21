from typing import List, Optional

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.dependencies import retriever
from app.services.chat_memory import append_turn, clear_session, get_history
from app.services.rag_pipeline import answer_question

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    session_id: str
    selected_documents: List[str] = Field(default_factory=list)
    model: Optional[str] = None  # override the Ollama model, e.g. "qwen2.5:7b"


@router.post("/query")
def query_rag(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query is empty.")
    try:
        result = answer_question(
            request.query,
            retriever,
            chat_history=get_history(request.session_id),
            document_ids=request.selected_documents or None,
            model=request.model,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=503, detail=f"LLM backend unavailable: {exc}") from exc

    append_turn(request.session_id, request.query, result["answer"])

    sources = result["sources"]
    return {
        **result,
        # kept for older clients
        "context": [s["text"] for s in sources],
        "used_documents": sorted({s["filename"] for s in sources if s["filename"]}),
    }


@router.delete("/session/{session_id}")
def reset_session(session_id: str):
    clear_session(session_id)
    return {"message": "Session cleared", "session_id": session_id}
