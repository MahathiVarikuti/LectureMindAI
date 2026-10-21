from app import config

# session_id -> [{"role": "user" | "assistant", "content": str}, ...]
# In-memory only: history is lost when the server restarts.
chat_sessions = {}


def get_history(session_id):
    return chat_sessions.get(session_id, [])


def append_turn(session_id, question, answer):
    history = chat_sessions.setdefault(session_id, [])
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    chat_sessions[session_id] = history[-config.MAX_HISTORY_MESSAGES:]


def clear_session(session_id):
    chat_sessions.pop(session_id, None)
