"""Test setup: isolated temp storage, no real models (fake embedder / LLM / Whisper / CLIP)."""
import hashlib
import os
import re
import tempfile
import uuid
from pathlib import Path

TEST_DIR = Path(tempfile.mkdtemp(prefix="lecturemind_test_"))
os.environ.update({
    "UPLOAD_DIR": str(TEST_DIR / "uploads"),
    "CHROMA_DIR": str(TEST_DIR / "chroma"),
    "SQLITE_PATH": str(TEST_DIR / "test.db"),
    "CORS_ORIGINS": "http://localhost:3000",
    "RATE_LIMIT_QUERY": "1000",
    "RATE_LIMIT_UPLOAD": "1000",
    "RELEVANCE_THRESHOLD": "0.2",
})

import fitz  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


class FakeEncoder:
    """Bag-of-words hashing embedder: texts sharing words are similar, unrelated texts are not."""

    def encode(self, texts, normalize_embeddings=True, **_kwargs):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        out = np.zeros((len(items), 384), dtype="float32")
        for row, text in enumerate(items):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 384] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        out = out / np.where(norms == 0, 1, norms)
        return out[0] if single else out


@pytest.fixture(autouse=True)
def fake_models(monkeypatch):
    from app.services import embedding_service

    monkeypatch.setattr(embedding_service, "get_model", lambda name=None: FakeEncoder())


@pytest.fixture
def llm(monkeypatch):
    """Replace the Ollama call; records every prompt so tests can assert on plumbing."""
    calls = []

    def fake_chat(messages, **kwargs):
        calls.append(messages)
        return "The answer is grounded in the lecture [1]."

    monkeypatch.setattr("app.services.llm_service.chat", fake_chat)
    return calls


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def session_a():
    return str(uuid.uuid4())


@pytest.fixture
def session_b():
    return str(uuid.uuid4())


def make_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(50, 50, 550, 780), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def pdf_bytes():
    return make_pdf(
        "Gradient descent is an optimisation algorithm that minimises a loss function by "
        "repeatedly stepping in the direction of the negative gradient. The learning rate "
        "controls the step size of gradient descent."
    )


def upload_pdf(client, session_id, data, name="notes.pdf"):
    return client.post("/upload", data={"session_id": session_id},
                       files={"file": (name, data, "application/pdf")})
