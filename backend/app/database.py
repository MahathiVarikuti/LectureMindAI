"""SQLite database for chat history."""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings


def _ensure_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(
            """
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
                file_path TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                uploaded_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_documents_session ON documents(session_id);
            """
        )
        conn.commit()
    finally:
        conn.close()


@contextmanager
def get_connection():
    db_path = settings.sqlite_db_path
    _ensure_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def add_message(session_id: str, role: str, content: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now),
        )


def get_messages(session_id: str, limit: int | None = None) -> list[dict]:
    with get_connection() as conn:
        if limit:
            rows = conn.execute(
                "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
            return [dict(r) for r in reversed(rows)]
        rows = conn.execute(
            "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def add_document(
    document_id: str,
    session_id: str,
    filename: str,
    modality: str,
    file_path: str,
    file_size: int,
    chunk_count: int,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO documents
               (document_id, session_id, filename, modality, file_path, file_size, chunk_count, uploaded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (document_id, session_id, filename, modality, file_path, file_size, chunk_count, now),
        )


def get_documents(session_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT document_id, filename, modality, chunk_count, uploaded_at
               FROM documents WHERE session_id = ? ORDER BY uploaded_at DESC""",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_document(document_id: str, session_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE document_id = ? AND session_id = ?",
            (document_id, session_id),
        ).fetchone()
        return dict(row) if row else None


def delete_document(document_id: str, session_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE document_id = ? AND session_id = ?",
            (document_id, session_id),
        ).fetchone()
        if not row:
            return None
        conn.execute(
            "DELETE FROM documents WHERE document_id = ? AND session_id = ?",
            (document_id, session_id),
        )
        return dict(row)


def count_documents(session_id: str) -> int:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM documents WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row["cnt"]


def total_storage(session_id: str) -> int:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(file_size), 0) as total FROM documents WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row["total"]
