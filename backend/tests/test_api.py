"""API-level tests. The LLM, embedder, Whisper and CLIP are faked: we test the plumbing
(limits, scoping, context assembly, error mapping), never the exact wording of an answer."""
import os
import uuid
from pathlib import Path

import pytest
import requests

from app.config import settings
from app.db import sqlite
from app.db.chroma import image_collection, text_collection
from tests.conftest import make_pdf, upload_pdf


def query(client, sid, text, **extra):
    return client.post("/query", json={"query": text, "session_id": sid, **extra})


class TestBasics:
    def test_health(self, client):
        assert client.get("/health").json() == {"status": "ok"}

    def test_cors_restricted_to_configured_origin(self, client):
        ok = client.options("/health", headers={"Origin": "http://localhost:3000",
                                                "Access-Control-Request-Method": "GET"})
        bad = client.options("/health", headers={"Origin": "http://evil.example",
                                                 "Access-Control-Request-Method": "GET"})
        assert ok.headers.get("access-control-allow-origin") == "http://localhost:3000"
        assert "access-control-allow-origin" not in bad.headers

    def test_invalid_session_is_400(self, client, pdf_bytes):
        r = upload_pdf(client, "123e4567-e89b-12d3-a456-426614174000", pdf_bytes)
        assert r.status_code == 400 and "UUID v4" in r.json()["detail"]
        assert client.get("/documents", params={"session_id": "nope"}).status_code == 400


class TestUploadValidation:
    def test_unsupported_type(self, client, session_a):
        r = client.post("/upload", data={"session_id": session_a},
                        files={"file": ("virus.exe", b"MZ", "application/pdf")})
        assert r.status_code == 400

    def test_fake_pdf_rejected_by_content(self, client, session_a):
        r = upload_pdf(client, session_a, b"this is not a pdf")
        assert r.status_code == 400
        assert not list((settings.upload_dir / session_a).glob("*"))  # nothing left on disk

    def test_pdf_too_large_is_413(self, client, session_a, pdf_bytes, monkeypatch):
        monkeypatch.setattr(settings, "max_pdf_size_mb", 0)
        assert upload_pdf(client, session_a, pdf_bytes).status_code == 413
        assert not list((settings.upload_dir / session_a).glob("*"))

    def test_document_count_limit(self, client, session_a, monkeypatch):
        monkeypatch.setattr(settings, "max_documents_per_session", 1)
        assert upload_pdf(client, session_a, make_pdf("First document about databases and indexing.")).status_code == 200
        r = upload_pdf(client, session_a, make_pdf("Second document about networking and routing."), "b.pdf")
        assert r.status_code == 413

    def test_storage_limit(self, client, session_a, pdf_bytes, monkeypatch):
        monkeypatch.setattr(settings, "max_storage_per_session_mb", 0)
        assert upload_pdf(client, session_a, pdf_bytes).status_code == 413

    def test_duplicate_upload_is_detected(self, client, session_a, pdf_bytes):
        first = upload_pdf(client, session_a, pdf_bytes).json()
        second = upload_pdf(client, session_a, pdf_bytes).json()
        assert second["duplicate"] and second["document_id"] == first["document_id"]
        assert len(client.get("/documents", params={"session_id": session_a}).json()) == 1

    def test_audio_duration_checked_before_transcription(self, client, session_a, monkeypatch):
        called = []
        monkeypatch.setattr("app.services.ingestion.get_media_duration", lambda p: 3600.0)
        monkeypatch.setattr("app.services.transcription.transcribe_audio", lambda p: called.append(p))
        r = client.post("/upload", data={"session_id": session_a},
                        files={"file": ("lecture.mp3", b"fake-audio-bytes", "audio/mpeg")})
        assert r.status_code == 413 and "limit" in r.json()["detail"]
        assert called == []  # expensive step never ran
        assert not list((settings.upload_dir / session_a).glob("*"))


class TestPdfRoundTrip:
    def test_upload_then_query_cites_page(self, client, session_a, pdf_bytes, llm):
        up = upload_pdf(client, session_a, pdf_bytes)
        assert up.status_code == 200
        body = up.json()
        assert body["modality"] == "pdf" and body["chunks"] >= 1 and body["document_id"]

        r = query(client, session_a, "What does the learning rate control in gradient descent?")
        assert r.status_code == 200
        data = r.json()
        assert data["sources"][0]["page"] == 1
        assert data["sources"][0]["modality"] == "pdf"
        assert data["used_documents"] == ["notes.pdf"]
        assert "gradient descent" in data["context"].lower()
        assert len(llm) == 1  # one LLM call (no history -> no rewrite call)
        assert "Context passages" in llm[0][-1]["content"]

    def test_metadata_schema(self, client, session_a, pdf_bytes):
        doc_id = upload_pdf(client, session_a, pdf_bytes).json()["document_id"]
        metas = text_collection().get(where={"document_id": doc_id}, include=["metadatas"])["metadatas"]
        for meta in metas:
            assert {"session_id", "document_id", "filename", "modality", "chunk_id", "uploaded_at"} <= set(meta)
            assert meta["modality"] == "pdf" and meta["session_id"] == session_a and meta["page"] == 1
            assert None not in meta.values()


class TestAudioVideoUrl:
    @pytest.fixture
    def fake_media(self, monkeypatch, tmp_path):
        monkeypatch.setattr("app.services.ingestion.get_media_duration", lambda p: 120.0)
        segs = [{"text": "Photosynthesis converts light energy into chemical energy in chloroplasts.",
                 "start": 12.0, "end": 19.5},
                {"text": "The Calvin cycle then fixes carbon dioxide into sugars.", "start": 19.5, "end": 27.0}]
        monkeypatch.setattr("app.services.transcription.transcribe_audio", lambda p: segs)

        def fake_extract(video, out):
            Path(out).write_bytes(b"wav")
            return out

        def fake_frames(video, out_dir, interval, max_frames):
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            frames = []
            for i in range(3):
                p = Path(out_dir) / f"frame_{i}.jpg"
                p.write_bytes(b"jpg")
                frames.append({"path": str(p), "timestamp": i * interval})
            return frames

        monkeypatch.setattr("app.services.ingestion.extract_audio", fake_extract)
        monkeypatch.setattr("app.services.ingestion.sample_frames", fake_frames)
        # CLIP: every frame -> e0; a query mentioning "diagram" -> e0 (match), else e1 (no match)
        def unit(i):
            v = [0.0] * 512
            v[i] = 1.0
            return v

        monkeypatch.setattr("app.services.clip_service.embed_images", lambda paths: [unit(0) for _ in paths])
        monkeypatch.setattr("app.services.clip_service.embed_text",
                            lambda text: unit(0) if "diagram" in text.lower() else unit(1))

    def test_audio_round_trip_with_timestamps(self, client, session_a, llm, fake_media):
        r = client.post("/upload", data={"session_id": session_a},
                        files={"file": ("bio.mp3", b"audio-bytes", "audio/mpeg")})
        assert r.status_code == 200 and r.json()["modality"] == "audio"
        data = query(client, session_a, "What does photosynthesis convert into chemical energy?").json()
        src = data["sources"][0]
        assert src["modality"] == "audio" and src["timestamp_start"] == 12.0 and src["timestamp_end"] == 27.0
        assert "00:12" in data["context"]

    def test_video_indexes_transcript_and_frames(self, client, session_a, llm, fake_media):
        r = client.post("/upload", data={"session_id": session_a},
                        files={"file": ("bio.mp4", b"video-bytes", "video/mp4")})
        assert r.status_code == 200 and r.json()["modality"] == "video"
        doc_id = r.json()["document_id"]
        assert text_collection().get(where={"document_id": doc_id})["ids"]
        assert len(image_collection().get(where={"document_id": doc_id})["ids"]) == 3

        text_meta = text_collection().get(where={"document_id": doc_id}, include=["metadatas"])["metadatas"][0]
        assert text_meta["modality"] == "video_transcript"

        # cross-modal: one query returns text chunks AND frames, in separate labelled sections
        data = query(client, session_a, "Show the photosynthesis diagram and explain what it converts").json()
        modalities = {s["modality"] for s in data["sources"]}
        assert {"video_transcript", "video_frame"} <= modalities
        assert "Visually related video frames" in data["context"]

    def test_frames_only_match_answers_without_llm(self, client, session_a, llm, fake_media):
        client.post("/upload", data={"session_id": session_a},
                    files={"file": ("bio.mp4", b"video-bytes", "video/mp4")})
        data = query(client, session_a, "diagram").json()
        assert data["sources"] and all(s["modality"] == "video_frame" for s in data["sources"])
        assert llm == []

    def test_url_ingestion_and_cleanup(self, client, session_a, llm, fake_media, monkeypatch):
        made = []

        def fake_download(url, tmp, want_video):
            made.append(tmp)
            f = tmp / "clip.m4a"
            f.write_bytes(b"audio")
            return f, "Remote lecture.m4a"

        monkeypatch.setattr("app.services.url_ingestion._download", fake_download)
        monkeypatch.setattr("app.services.url_ingestion.validate_public_url", lambda u: None)
        r = client.post("/upload-url", json={"url": "https://example.com/v", "session_id": session_a})
        assert r.status_code == 200 and r.json()["modality"] == "audio"
        assert made and not made[0].exists()  # temp dir removed

    @pytest.mark.parametrize("url", ["http://127.0.0.1/secret", "http://localhost:8000/x", "ftp://example.com/a"])
    def test_url_rejects_local_and_non_http(self, client, session_a, url):
        r = client.post("/upload-url", json={"url": url, "session_id": session_a})
        assert r.status_code == 400


class TestSessionIsolation:
    def test_session_b_cannot_see_query_or_delete_session_a(self, client, session_a, session_b, pdf_bytes, llm):
        doc_id = upload_pdf(client, session_a, pdf_bytes).json()["document_id"]

        assert len(client.get("/documents", params={"session_id": session_a}).json()) == 1
        assert client.get("/documents", params={"session_id": session_b}).json() == []

        # B cannot delete A's document (same 404 as a nonexistent id)
        missing = client.delete(f"/documents/{uuid.uuid4().hex}", params={"session_id": session_b})
        denied = client.delete(f"/documents/{doc_id}", params={"session_id": session_b})
        assert denied.status_code == 404 and denied.json() == missing.json()
        assert len(client.get("/documents", params={"session_id": session_a}).json()) == 1

        # B's queries never retrieve A's content
        r = query(client, session_b, "What does the learning rate control in gradient descent?").json()
        assert r["sources"] == [] and "could not find" in r["answer"].lower()
        assert llm == []
        # ...even when B names A's document explicitly
        assert query(client, session_b, "gradient descent", selected_documents=[doc_id]).status_code == 404

    def test_chat_history_is_per_session(self, client, session_a, session_b, pdf_bytes, llm):
        upload_pdf(client, session_a, pdf_bytes)
        query(client, session_a, "What is gradient descent?")
        assert len(client.get("/history", params={"session_id": session_a}).json()["messages"]) == 2
        assert client.get("/history", params={"session_id": session_b}).json()["messages"] == []


class TestDeletion:
    def test_delete_removes_vectors_file_and_row(self, client, session_a, pdf_bytes):
        doc_id = upload_pdf(client, session_a, pdf_bytes).json()["document_id"]
        stored = sqlite.get_document(session_a, doc_id)["file_path"]
        assert os.path.exists(stored)
        assert text_collection().get(where={"document_id": doc_id})["ids"]

        r = client.delete(f"/documents/{doc_id}", params={"session_id": session_a})
        assert r.status_code == 200
        assert not os.path.exists(stored)  # the actual file is gone
        assert text_collection().get(where={"document_id": doc_id})["ids"] == []
        assert image_collection().get(where={"document_id": doc_id})["ids"] == []
        assert client.get("/documents", params={"session_id": session_a}).json() == []


class TestEmptyContextGuard:
    def test_irrelevant_query_skips_llm(self, client, session_a, pdf_bytes, llm):
        upload_pdf(client, session_a, pdf_bytes)
        r = query(client, session_a, "Who painted Mona Lisa?")
        assert r.status_code == 200
        assert "could not find" in r.json()["answer"].lower()
        assert llm == []

    def test_empty_query_is_400(self, client, session_a):
        assert query(client, session_a, "   ").status_code == 400


class TestLlmPlumbing:
    def test_followup_rewrites_with_history(self, client, session_a, pdf_bytes, llm):
        upload_pdf(client, session_a, pdf_bytes)
        query(client, session_a, "What is gradient descent?")
        before = len(llm)
        query(client, session_a, "What does it use the learning rate for?")
        # rewrite call + answer call; the rewrite prompt contains the earlier turn
        assert len(llm) - before == 2
        assert "What is gradient descent?" in llm[before][-1]["content"]

    def test_history_persists_in_sqlite_and_can_be_cleared(self, client, session_a, pdf_bytes, llm):
        upload_pdf(client, session_a, pdf_bytes)
        query(client, session_a, "What is gradient descent?")
        assert [m["role"] for m in sqlite.get_messages(session_a)] == ["user", "assistant"]
        client.delete("/history", params={"session_id": session_a})
        assert sqlite.get_messages(session_a) == []

    def test_llm_down_is_503_and_timeout_is_504(self, client, session_a, pdf_bytes, monkeypatch):
        upload_pdf(client, session_a, pdf_bytes)

        def boom(exc):
            def raiser(*a, **k):
                raise exc
            return raiser

        monkeypatch.setattr("app.services.llm_service.chat", boom(requests.ConnectionError("down")))
        assert query(client, session_a, "What is gradient descent?").status_code == 503
        monkeypatch.setattr("app.services.llm_service.chat", boom(requests.Timeout("slow")))
        assert query(client, session_a, "What is gradient descent?").status_code == 504


class TestRateLimit:
    def test_query_rate_limit_returns_429(self, client, session_a, monkeypatch):
        from app.limiter import limiter

        limiter.reset()
        monkeypatch.setattr(settings, "rate_limit_query", 2)
        codes = [query(client, session_a, "hello there").status_code for _ in range(4)]
        assert 429 in codes and codes[0] != 429
        monkeypatch.setattr(settings, "rate_limit_query", 1000)
        limiter.reset()
