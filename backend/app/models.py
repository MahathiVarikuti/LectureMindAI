"""Pydantic request models. session_id is validated in the routes (400, not 422)."""
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(max_length=2000)
    session_id: str
    selected_documents: list[str] | None = None
    model: str | None = None  # optionally override an allowlisted Ollama model
