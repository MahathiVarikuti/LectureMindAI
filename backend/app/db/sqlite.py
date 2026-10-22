"""SQLite persistence for chat history and document ownership.

Every table carries `session_id` and every query below filters on it, so one
session can never read or modify another session's rows. The session id is a
random UUID v4 held in the browser: lightweight isolation, NOT authentication
(see docs/security.md).
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from app.config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    modality TEXT NOT NULL,
    file_path TEXT,
    file_size INTEGER NOT NULL DEFAULT 0,
    file_hash TEXT,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    uploaded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_session ON documents(session_id);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    path = settings.sqlite_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(_SCHEMA)


@contextmanager
def connect():
    # One short-lived connection per call: sqlite3 connections are not safe to share
    # across FastAPI's worker threads, and per-call connections are cheap.
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.sqlite_path), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        yield conn
        conn.commit()
    finally:
        conn.close()


# --- chat messages ----------------------------------------------------------
def add_message(session_id: str, role: str, content: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now_iso()),
        )
        # Keep the table bounded: drop the oldest rows beyond the per-session cap.
        conn.execute(
            """DELETE FROM messages WHERE session_id = ? AND id NOT IN (
                   SELECT id FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?)""",
            (session_id, session_id, settings.max_stored_messages_per_session),
        )


def get_messages(session_id: str, limit: int | None = None) -> list[dict]:
    """Oldest-first messages; `limit` keeps only the most recent N."""
    with connect() as conn:
        if limit:
            rows = conn.execute(
                "SELECT role, content, created_at FROM messages WHERE session_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()[::-1]
        else:
            rows = conn.execute(
                "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
    return [dict(r) for r in rows]


def clear_messages(session_id: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))


# --- documents --------------------------------------------------------------
def add_document(document_id, session_id, filename, modality, file_path, file_size,
                 chunk_count, file_hash=None) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO documents (document_id, session_id, filename, modality, file_path, "
            "file_size, file_hash, chunk_count, uploaded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (document_id, session_id, filename, modality, file_path, file_size, file_hash,
             chunk_count, now_iso()),
        )


def list_documents(session_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT document_id, filename, modality, chunk_count, uploaded_at FROM documents "
            "WHERE session_id = ? ORDER BY uploaded_at DESC",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_document(session_id: str, document_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE session_id = ? AND document_id = ?",
            (session_id, document_id),
        ).fetchone()
    return dict(row) if row else None


def find_by_hash(session_id: str, file_hash: str) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE session_id = ? AND file_hash = ?",
            (session_id, file_hash),
        ).fetchone()
    return dict(row) if row else None


def delete_document(session_id: str, document_id: str) -> None:
    with connect() as conn:
        conn.execute(
            "DELETE FROM documents WHERE session_id = ? AND document_id = ?",
            (session_id, document_id),
        )


def count_documents(session_id: str) -> int:
    with connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM documents WHERE session_id = ?", (session_id,)
        ).fetchone()[0]


def total_storage(session_id: str) -> int:
    with connect() as conn:
        return conn.execute(
            "SELECT COALESCE(SUM(file_size), 0) FROM documents WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
