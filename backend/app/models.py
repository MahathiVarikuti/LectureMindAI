"""Pydantic request/response models."""

import uuid

from pydantic import BaseModel, Field, field_validator


def _validate_session_uuid(value: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("session_id must be a valid UUID v4") from exc
    if parsed.version != 4:
        raise ValueError("session_id must be a valid UUID v4")
    return str(parsed)


class UploadUrlRequest(BaseModel):
    url: str
    session_id: str

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        return _validate_session_uuid(value)


class QueryRequest(BaseModel):
    query: str
    session_id: str
    selected_documents: list[str] | None = None

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        return _validate_session_uuid(value)


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
