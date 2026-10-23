"""Chat memory backed by SQLite (survives restarts and page reloads).

The full conversation is persisted; only the most recent MAX_HISTORY_MESSAGES are
handed to the LLM for query rewriting.
"""
from app import config
from app.db import sqlite


def get_history(session_id):
    rows = sqlite.get_messages(session_id, limit=config.MAX_HISTORY_MESSAGES)
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def get_full_history(session_id):
    return sqlite.get_messages(session_id)


def append_turn(session_id, question, answer, sources=None):
    sqlite.add_message(session_id, "user", question)
    sqlite.add_message(session_id, "assistant", answer, sources=sources)


def clear_session(session_id):
    sqlite.clear_messages(session_id)
