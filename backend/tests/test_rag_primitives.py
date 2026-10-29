import unittest

from app import config
from app.services.llm_service import is_not_found
from app.services.rag_pipeline import _source
from app.services.retrieval_service import Retriever, select_context
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

    def test_sparse_only_hit_cannot_satisfy_abstain_threshold(self):
        kept, reason = select_context([
            {"id": "sparse", "score": 0.99, "sparse_only": True},
            {"id": "dense", "score": 0.1, "sparse_only": False},
        ], threshold=0.2)
        self.assertEqual(kept, [])
        self.assertEqual(reason, "low_relevance")

    def test_top_score_is_best_candidate(self):
        class FakeRetriever:
            relevance_threshold = 0.1

            def retrieve(self, *args, **kwargs):
                return [
                    {"id": "first", "text": "first", "score": 0.2, "metadata": {}},
                    {"id": "best", "text": "best", "score": 0.8, "metadata": {}},
                ]

        from app.services.rag_pipeline import answer_question
        result = answer_question("question", FakeRetriever(), rewrite=False, session_id=None)
        self.assertEqual(result["top_score"], 0.8)

    def test_not_found_requires_exact_sentence(self):
        self.assertTrue(is_not_found(" I COULD NOT FIND THAT INFORMATION IN THE UPLOADED MATERIAL. [1] "))
        self.assertFalse(is_not_found("I could not find that information in the uploaded material, but here is more."))


if __name__ == "__main__":
    unittest.main()


class DenseContextTests(unittest.TestCase):
    def test_dense_mode_keeps_low_scoring_chunks_when_best_passes(self):
        candidates = [{"id": f"c{i}", "text": "t", "score": s} for i, s in enumerate([0.31, 0.22, 0.18, 0.12])]
        kept, reason = select_context(candidates, top_k=4, threshold=0.25, per_chunk_threshold=False)
        self.assertIsNone(reason)
        self.assertEqual(len(kept), 4)

    def test_reranker_mode_still_drops_chunks_below_threshold(self):
        candidates = [{"id": "a", "text": "t", "score": 0.9}, {"id": "b", "text": "t", "score": 0.05}]
        kept, _ = select_context(candidates, top_k=4, threshold=0.125, per_chunk_threshold=True)
        self.assertEqual([c["id"] for c in kept], ["a"])


class AlignCitationsTests(unittest.TestCase):
    SOURCES = [{"chunk_id": c} for c in "abcd"]

    def test_keeps_only_cited_sources_and_renumbers(self):
        from app.services.rag_pipeline import align_citations

        answer, sources = align_citations("Python [3]. SQL [1, 3].", self.SOURCES)
        self.assertEqual(answer, "Python [1]. SQL [2, 1].")
        self.assertEqual([s["chunk_id"] for s in sources], ["c", "a"])

    def test_no_valid_citation_keeps_all_sources(self):
        from app.services.rag_pipeline import align_citations

        answer, sources = align_citations("No markers here, or bad [9].", self.SOURCES)
        self.assertEqual(sources, self.SOURCES)
        self.assertEqual(answer, "No markers here, or bad [9].")

    def test_out_of_range_marker_is_dropped_when_others_are_valid(self):
        from app.services.rag_pipeline import align_citations

        answer, sources = align_citations("A [2] B [9].", self.SOURCES)
        self.assertEqual(answer, "A [1] B .")
        self.assertEqual([s["chunk_id"] for s in sources], ["b"])

    def test_video_frames_are_always_kept(self):
        from app.services.rag_pipeline import align_citations

        sources = [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "f", "modality": "video_frame"}]
        answer, kept = align_citations("Only the second [2].", sources)
        self.assertEqual([s["chunk_id"] for s in kept], ["b", "f"])
        self.assertEqual(answer, "Only the second [1].")

