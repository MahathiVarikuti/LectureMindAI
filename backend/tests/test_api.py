"""Backend tests for LectureMindAI."""

import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

# Use temp dirs for test isolation
TEST_DIR = tempfile.mkdtemp(prefix="lecturemind_test_")
os.environ["UPLOAD_DIR"] = str(Path(TEST_DIR) / "uploads")
os.environ["CHROMA_DIR"] = str(Path(TEST_DIR) / "chroma")
os.environ["SQLITE_PATH"] = str(Path(TEST_DIR) / "test.db")
os.environ["CORS_ORIGINS"] = "http://localhost:3000"

from app.main import app  # noqa: E402
from app.utils.chunking import chunk_text  # noqa: E402


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def session_a():
    return str(uuid.uuid4())


@pytest.fixture
def session_b():
    return str(uuid.uuid4())


class TestChunking:
    def test_empty_text(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text(self):
        assert chunk_text("hello") == ["hello"]

    def test_overlap(self):
        text = "a" * 2500
        chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200)
        assert len(chunks) >= 2
        assert all(len(c) <= 1000 for c in chunks)


class TestHealth:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestUploadValidation:
    def test_unsupported_file_type(self, client, session_a):
        resp = client.post(
            "/upload",
            data={"session_id": session_a},
            files={"file": ("test.exe", b"data", "application/octet-stream")},
        )
        assert resp.status_code == 400

    def test_invalid_session(self, client):
        resp = client.post(
            "/upload",
            data={"session_id": "short"},
            files={"file": ("test.pdf", b"%PDF", "application/pdf")},
        )
        assert resp.status_code == 400


class TestSessionIsolation:
    def test_session_b_cannot_list_session_a_docs(self, client, session_a, session_b):
        from app import database as db

        doc_id = str(uuid.uuid4())
        db.add_document(
            doc_id, session_a, "secret.pdf", "pdf", "/tmp/secret.pdf", 100, 5
        )

        resp_a = client.get("/documents", params={"session_id": session_a})
        resp_b = client.get("/documents", params={"session_id": session_b})

        assert resp_a.status_code == 200
        assert len(resp_a.json()) >= 1
        assert resp_b.status_code == 200
        assert len(resp_b.json()) == 0

    def test_session_b_cannot_delete_session_a_doc(self, client, session_a, session_b):
        from app import database as db

        doc_id = str(uuid.uuid4())
        db.add_document(
            doc_id, session_a, "secret.pdf", "pdf", "/tmp/secret.pdf", 100, 5
        )

        resp = client.delete(
            f"/documents/{doc_id}", params={"session_id": session_b}
        )
        assert resp.status_code == 404

        # Verify still exists for session A
        resp_a = client.get("/documents", params={"session_id": session_a})
        ids = [d["document_id"] for d in resp_a.json()]
        assert doc_id in ids


class TestEmptyContextGuard:
    @patch("app.services.query_service.embed_query", return_value=[0.0] * 384)
    @patch("app.services.query_service.embed_text_clip", return_value=[0.0] * 512)
    @patch("app.services.query_service.query_text_chunks")
    @patch("app.services.query_service.query_image_frames")
    @patch("app.services.query_service.generate_answer", new_callable=AsyncMock)
    def test_no_relevant_chunks_skips_llm(
        self, mock_llm, mock_img, mock_text, mock_clip, mock_embed, client, session_a
    ):
        mock_text.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        mock_img.return_value = {
            "ids": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        resp = client.post(
            "/query",
            json={"query": "What is quantum gravity?", "session_id": session_a},
        )
        assert resp.status_code == 200
        assert "could not find" in resp.json()["answer"].lower()
        mock_llm.assert_not_called()


class TestDeletion:
    def test_delete_removes_file(self, client, session_a):
        from app import database as db

        upload_dir = Path(os.environ["UPLOAD_DIR"]) / session_a
        upload_dir.mkdir(parents=True, exist_ok=True)
        doc_id = str(uuid.uuid4())
        file_path = upload_dir / f"{doc_id}_test.pdf"
        file_path.write_text("test content")

        db.add_document(
            doc_id, session_a, "test.pdf", "pdf", str(file_path), 12, 1
        )

        resp = client.delete(
            f"/documents/{doc_id}", params={"session_id": session_a}
        )
        assert resp.status_code == 200
        assert not file_path.exists()

        resp_list = client.get("/documents", params={"session_id": session_a})
        ids = [d["document_id"] for d in resp_list.json()]
        assert doc_id not in ids
