"""Pydantic request/response models."""

from pydantic import BaseModel, Field


class UploadUrlRequest(BaseModel):
    url: str
    session_id: str


class QueryRequest(BaseModel):
    query: str
    session_id: str
    selected_documents: list[str] | None = None


class SourceCitation(BaseModel):
    document_id: str
    filename: str
    modality: str
    page: int | None = None
    timestamp_start: float | None = None
    timestamp_end: float | None = None
    timestamp: float | None = None
    snippet: str = ""


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    used_documents: list[str]
    context: str = ""


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks: int
    modality: str
    message: str


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    modality: str
    chunk_count: int
    uploaded_at: str


class HealthResponse(BaseModel):
    status: str = "ok"
