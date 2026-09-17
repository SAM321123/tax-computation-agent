"""Tests for Layer 1 (article chunking) and the retrieval/citation-verification
guardrail that Layer 3 depends on."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.ingestion.law_ingest import build_index, resolve_effective_chunks, chunk_by_article
from app.rag.classifier import RawClassification, verify_citation
from app.rag.retriever import HybridRetriever
from app.rag.vector_store import SqliteVectorStore, build_vector_database


class TestArticleChunking(unittest.TestCase):
    def test_splits_on_article_headings(self):
        text = (
            "Article 1\nFirst article body.\n\n"
            "Article 2\nSecond article body, longer.\n"
        )
        chunks = chunk_by_article(text, law_name="Test Law", decision_no="TL1", effective_date="", source_path="x")
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].article, "1")
        self.assertIn("First article body", chunks[0].text)
        self.assertEqual(chunks[1].article, "2")

    def test_falls_back_to_whole_document_without_headings(self):
        text = "No article headings here, just prose guidance."
        chunks = chunk_by_article(text, law_name="Guide", decision_no="G1", effective_date="", source_path="x")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].article, "")


class TestSupersession(unittest.TestCase):
    def test_superseded_chunk_is_dropped(self):
        chunks = build_index()
        # Manually mark one chunk as superseded by another to test the filter logic.
        chunks[1].supersedes = chunks[0].chunk_id
        effective = resolve_effective_chunks(chunks)
        self.assertNotIn(chunks[0], effective)
        self.assertIn(chunks[1], effective)


class TestHybridRetrieval(unittest.TestCase):
    def setUp(self):
        self.chunks = resolve_effective_chunks(build_index())
        self.retriever = HybridRetriever(self.chunks)

    def test_finds_entertainment_article_for_entertainment_query(self):
        results = self.retriever.search("entertainment expenditure client dinners", top_k=3)
        self.assertTrue(any(r.chunk.article == "32" for r in results))

    def test_finds_small_business_relief_for_revenue_threshold_query(self):
        results = self.retriever.search("small business relief revenue threshold 3 million", top_k=3)
        self.assertTrue(any("Small Business Relief" in r.chunk.text or r.chunk.article == "2" for r in results))


class TestPersistentVectorStore(unittest.TestCase):
    def test_round_trip_preserves_chunks_and_search(self):
        chunks = resolve_effective_chunks(build_index())
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "laws.sqlite3"
            build_vector_database(chunks, path, source_dir=Path("sample_data/law_corpus"))
            store = SqliteVectorStore.load(path)
            retriever = HybridRetriever(store.chunks, embedder=store.embedder, matrix=store.matrix)
            results = retriever.search("entertainment client dinners", top_k=3)

        self.assertEqual(len(store.chunks), len(chunks))
        self.assertTrue(results)
        self.assertTrue(any("entertainment" in result.chunk.text.lower() for result in results))


class TestCitationVerification(unittest.TestCase):
    def setUp(self):
        self.chunks = resolve_effective_chunks(build_index())
        self.retriever = HybridRetriever(self.chunks)

    def test_citation_present_in_context_verifies(self):
        context = self.retriever.search("entertainment", top_k=3)
        raw = RawClassification(
            treatment="partially_deductible",
            disallowed_fraction=0.5,
            relied_on_chunk_id=context[0].chunk.chunk_id,
            confidence=0.9,
            reasoning="test",
        )
        self.assertIsNotNone(verify_citation(raw, context))

    def test_fabricated_citation_is_rejected(self):
        context = self.retriever.search("entertainment", top_k=3)
        raw = RawClassification(
            treatment="non_deductible",
            disallowed_fraction=1.0,
            relied_on_chunk_id="Some_Fake_Decision::Art999",
            confidence=0.95,
            reasoning="hallucinated",
        )
        self.assertIsNone(verify_citation(raw, context))


if __name__ == "__main__":
    unittest.main()
