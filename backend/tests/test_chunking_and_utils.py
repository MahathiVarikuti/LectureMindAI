import pytest

from app.dependencies import validate_session_id
from app.errors import AppError
from app.utils.chunker import chunk_pages, chunk_segments, chunk_text
from app.utils.media import detect_modality, safe_filename


class TestChunking:
    def test_text_overlap_and_size(self):
        text = "First paragraph. " * 20 + "\n\n" + "Second paragraph. " * 20
        chunks = chunk_text(text, chunk_size=120, overlap=30, min_chars=1)
        assert len(chunks) > 1
        assert all(len(c) <= 120 for c in chunks)
        assert any(set(chunks[i].split()) & set(chunks[i + 1].split()) for i in range(len(chunks) - 1))

    def test_empty_text(self):
        assert chunk_text("   ") == []

    def test_pdf_heading_carries_across_pages(self):
        chunks = chunk_pages([
            {"page": 1, "text": "Gradient descent\nThe optimiser updates weights."},
            {"page": 2, "text": "It moves opposite the gradient direction."},
        ], min_chars=1)
        assert chunks[0]["text"].startswith("Gradient descent\n")
        assert chunks[1]["text"].startswith("Gradient descent\n")

    def test_segments_keep_timestamps_and_overlap(self):
        segs = [{"text": f"sentence number {i} here.", "start": float(i * 5), "end": float(i * 5 + 4)}
                for i in range(12)]
        chunks = chunk_segments(segs, chunk_size=80, overlap=30)
        assert len(chunks) > 1
        assert chunks[0]["timestamp_start"] == 0.0
        assert chunks[-1]["timestamp_end"] == 59.0
        for c in chunks:
            assert c["timestamp_start"] < c["timestamp_end"]
        # the next chunk restarts inside the previous one (overlap)
        assert chunks[1]["timestamp_start"] < chunks[0]["timestamp_end"]


class TestValidation:
    def test_session_id_must_be_uuid_v4(self):
        assert validate_session_id("123e4567-e89b-42d3-a456-426614174000")
        for bad in ("abc", "123e4567-e89b-12d3-a456-426614174000", ""):
            with pytest.raises(AppError) as exc:
                validate_session_id(bad)
            assert exc.value.status_code == 400

    def test_modality_by_extension(self):
        assert detect_modality("a.PDF") == "pdf"
        assert detect_modality("a.mp3") == "audio"
        assert detect_modality("a.mkv") == "video"
        assert detect_modality("a.exe") is None

    def test_safe_filename_strips_paths(self):
        assert "/" not in safe_filename("../../etc/passwd")
        assert safe_filename("..\\..\\x.pdf") == "x.pdf"
        assert safe_filename("") == "upload"


class TestSectionAwareChunking:
    """Regression for the resume case: sections and entries must not be cut or merged."""

    @staticmethod
    def _chunks():
        from pathlib import Path
        return chunk_text(Path(__file__).with_name("resume_sample.txt").read_text(encoding="utf-8"))

    def test_each_project_is_its_own_chunk_with_heading(self):
        projects = [c for c in self._chunks() if c.startswith("PROJECTS")]
        assert len(projects) == 3
        for name in ("Data Quality Intelligence Platform", "FaceFindr", "LectureMind AI"):
            assert sum(name in c for c in projects) == 1

    def test_sections_are_not_merged(self):
        chunks = self._chunks()
        cert = next(c for c in chunks if c.startswith("CERTIFICATIONS"))
        assert "Web Development (IBM)" in cert and "LEADERSHIP" not in cert
        edu = next(c for c in chunks if c.startswith("EDUCATION"))
        assert "EXPERIENCE" not in edu

    def test_no_chunk_starts_mid_bullet(self):
        for c in self._chunks():
            body = c.split("\n", 1)[1] if "\n" in c else c
            assert not body.lstrip().startswith("●")

    def test_plain_prose_without_headings_is_unchanged(self):
        text = "Gradient descent minimises a loss. " * 60
        chunks = chunk_text(text, chunk_size=300, overlap=50, min_chars=1)
        assert len(chunks) > 1 and all(len(c) <= 300 for c in chunks)
