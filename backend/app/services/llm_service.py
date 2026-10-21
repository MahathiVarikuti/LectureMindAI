import re

import requests

from app import config

SYSTEM_PROMPT = f"""You are LectureMind, a study assistant that answers questions about the student's uploaded lecture material.

Rules:
1. Use ONLY the numbered context passages. Do not use outside knowledge, even if you know the answer.
2. Cite the passage numbers you used in square brackets after each claim, e.g. "Gradient descent minimises the loss [2]."
3. If the passages do not contain the answer, reply with exactly: "{config.NOT_FOUND_ANSWER}"
4. If the passages only partly answer the question, answer that part and say what is missing.
5. Be concise and clear. Do not invent formulas, numbers, names or examples that are not in the passages."""

REWRITE_PROMPT = """Rewrite the user's latest question so it can be understood without the conversation.
Replace pronouns like "it", "this", "they" with what they refer to. Keep it short.
If the question is already self-contained, return it unchanged.
Output ONLY the rewritten question, nothing else.

Example:
Conversation:
user: What is a convolutional neural network?
assistant: A convolutional neural network is a network that uses convolution layers to process images.
Latest question: How is it trained?
Rewritten question: How is a convolutional neural network trained?

Conversation:
{history}

Latest question: {question}
Rewritten question:"""

# Words that usually mean the question depends on earlier turns.
_FOLLOW_UP = re.compile(
    r"\b(it|its|this|that|these|those|they|them|their|he|she|his|her|above|previous|"
    r"former|latter|same|more|else|also|again|another|example|elaborate|explain further)\b",
    re.IGNORECASE,
)


def chat(messages, model=None, temperature=None, num_ctx=None, timeout=300):
    response = requests.post(
        f"{config.OLLAMA_URL}/api/chat",
        json={
            "model": model or config.LLM_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": config.LLM_TEMPERATURE if temperature is None else temperature,
                "num_ctx": num_ctx or config.LLM_NUM_CTX,
            },
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def format_context(chunks):
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata") or {}
        source = meta.get("filename", "unknown")
        if meta.get("page"):
            source += f", page {meta['page']}"
        parts.append(f"[{i}] ({source})\n{chunk['text']}")
    return "\n\n".join(parts)


def generate_answer(query, chunks, model=None):
    user = f"Context passages:\n\n{format_context(chunks)}\n\nQuestion: {query}"
    return chat(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        model=model,
    )


def is_not_found(answer):
    normalized = answer.lower()
    return "could not find that information" in normalized or "not found in the uploaded" in normalized


def needs_rewrite(chat_history, query):
    if not chat_history:
        return False
    return len(query.split()) <= 4 or bool(_FOLLOW_UP.search(query))


def _clean_rewrite(text, original):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return original
    line = re.sub(r"^(rewritten|standalone)?\s*question\s*:\s*", "", lines[0], flags=re.IGNORECASE)
    line = line.strip().strip('"').strip("'").strip()
    # Small models sometimes answer the question instead of rewriting it.
    if not line or len(line) > max(3 * len(original), 200):
        return original
    return line


def rewrite_query(chat_history, current_query, model=None):
    if not needs_rewrite(chat_history, current_query):
        return current_query
    history = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in chat_history[-4:])
    prompt = REWRITE_PROMPT.format(history=history, question=current_query)
    try:
        return _clean_rewrite(chat([{"role": "user", "content": prompt}], model=model), current_query)
    except requests.RequestException:
        return current_query
