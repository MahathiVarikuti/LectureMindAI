import unittest

from app.services.rag_pipeline import _source
from app.services.retrieval_service import select_context
from app.utils.chunker import chunk_text


class RagPrimitiveTests(unittest.TestCase):
    def test_chunking_preserves_text_and_overlap(self):
        text = "First paragraph. " * 20 + "\n\n" + "Second paragraph. " * 20
        chunks = chunk_text(text, chunk_size=120, overlap=30, min_chars=1)

        self.assertGreater(len(chunks), 1)
        self.assertIn("First paragraph.", chunks[0])
        self.assertIn("Second paragraph.", chunks[-1])
        self.assertTrue(any(set(chunks[i].split()) & set(chunks[i + 1].split())
                            for i in range(len(chunks) - 1)))

    def test_relevance_threshold_abstains_without_candidates(self):
        kept, reason = select_context([], threshold=0.125)

        self.assertEqual(kept, [])
        self.assertEqual(reason, "no_documents")

    def test_relevance_threshold_keeps_only_relative_hits(self):
        candidates = [
            {"id": "doc-a_0", "text": "supported", "score": 0.9},
            {"id": "doc-a_1", "text": "weak", "score": 0.04},
            {"id": "doc-a_2", "text": "supported too", "score": 0.5},
        ]

        kept, reason = select_context(candidates, top_k=3, threshold=0.1,
                                       relative_cutoff=0.5)

        self.assertIsNone(reason)
        self.assertEqual([item["id"] for item in kept], ["doc-a_0", "doc-a_2"])

    def test_source_exposes_stable_chunk_id_and_document_metadata(self):
        source = _source({
            "id": "document-123_7",
            "text": "A passage",
            "score": 0.87654,
            "metadata": {"document_id": "document-123", "filename": "notes.pdf", "page": 3},
        })

        self.assertEqual(source["chunk_id"], "document-123_7")
        self.assertEqual(source["document_id"], "document-123")
        self.assertEqual(source["score"], 0.8765)


if __name__ == "__main__":
    unittest.main()
