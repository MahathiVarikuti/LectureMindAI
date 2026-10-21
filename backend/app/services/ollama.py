"""Ollama LLM integration."""

import httpx

from app.config import settings


async def _ollama_generate(prompt: str, system: str | None = None) -> str:
    payload: dict = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system
    try:
        async with httpx.AsyncClient(timeout=settings.ollama_timeout) as client:
            resp = await client.post(f"{settings.ollama_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
    except httpx.TimeoutException:
        raise RuntimeError("LLM request timed out")
    except httpx.HTTPError as e:
        raise RuntimeError(f"LLM request failed: {e}")


async def rewrite_query(query: str, history: list[dict]) -> str:
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in history[-8:]
    )
    system = (
        "You rewrite follow-up questions into standalone questions using chat history. "
        "Return ONLY the rewritten question, nothing else."
    )
    prompt = f"Chat history:\n{history_text}\n\nFollow-up question: {query}\n\nStandalone question:"
    return await _ollama_generate(prompt, system=system)


async def generate_answer(query: str, context: str) -> str:
    system = (
        "You are a helpful assistant answering questions about lecture materials. "
        "Use ONLY the provided context. If the context does not contain enough "
        "information to answer, say you cannot find the answer in the uploaded material. "
        "Carefully combine facts from all relevant context passages, and do simple "
        "arithmetic when the question asks for a total or marks division. Do not say "
        "information is missing when it is explicitly present in the context. "
        "Cite sources by filename and page/timestamp when relevant."
    )
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    return await _ollama_generate(prompt, system=system)
