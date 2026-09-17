"""
Layer 3 (retrieval half) — hybrid search over the law corpus.

Legal text has exact terms of art ("Qualifying Free Zone Person", "Article
30") that dense embeddings alone under-match, and small corpora make a
learned dense retriever overkill. So this hybrid combines:

  1. TF-IDF cosine similarity (`TfidfEmbedder`) — the pluggable "dense"
     channel. Swap this class for a real embedding model (OpenAI
     text-embedding-3-large, or a local sentence-transformers model) by
     implementing the same `.fit(texts)` / `.embed(texts)` interface —
     nothing else in this file or in classifier.py needs to change.
  2. Exact keyword / phrase overlap — catches precise term-of-art and
     Article-number matches that a pure vector search can miss.

Production should add a cross-encoder reranker on the merged top-k before
handing results to the LLM classifier; the `rerank` hook below is where
that plugs in.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.ingestion.law_ingest import LawChunk


class TfidfEmbedder:
    """Default, offline embedding backend. Same interface a real embedding
    model should implement: `.fit(texts)` once, then `.embed(texts) -> sparse matrix`."""

    def __init__(self) -> None:
        # The FTA folder contains long guides and user manuals. A bounded
        # vocabulary keeps the local store predictable while exact legal terms
        # are still covered by the hybrid keyword channel below.
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=100_000,
            sublinear_tf=True,
        )
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)
        self._fitted = True

    def embed(self, texts: list[str]):
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.fit() must be called before embed().")
        # Keep the matrix sparse. A real FTA corpus is much larger than the
        # synthetic fixture and densifying it wastes memory during indexing.
        return self.vectorizer.transform(texts)


def _keyword_score(query: str, text: str) -> float:
    """Cheap exact-overlap signal: fraction of the query's significant words
    (len > 3, to skip stopwords like 'the'/'and') that appear verbatim in the
    chunk. Catches Article-number and term-of-art matches TF-IDF can blur."""
    q_words = {w for w in re.findall(r"[a-zA-Z0-9']+", query.lower()) if len(w) > 3}
    if not q_words:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for w in q_words if w in text_lower)
    return hits / len(q_words)


@dataclass
class RetrievedChunk:
    chunk: LawChunk
    score: float


class HybridRetriever:
    def __init__(
        self,
        chunks: list[LawChunk],
        embedder: TfidfEmbedder | None = None,
        vector_weight: float = 0.6,
        matrix=None,
    ):
        self.chunks = chunks
        self.embedder = embedder or TfidfEmbedder()
        self.vector_weight = vector_weight
        if not chunks:
            self._matrix = np.empty((0, 0))
        elif matrix is not None:
            self._matrix = matrix
        else:
            texts = [c.text for c in chunks]
            self.embedder.fit(texts)
            self._matrix = self.embedder.embed(texts)

    @classmethod
    def from_vector_database(cls, path):
        """Load chunks and their fitted vector matrix without re-embedding."""
        from app.rag.vector_store import SqliteVectorStore

        store = SqliteVectorStore.load(path)
        return cls(store.chunks, embedder=store.embedder, matrix=store.matrix)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not self.chunks:
            return []
        query_vec = self.embedder.embed([query])
        vector_scores = cosine_similarity(query_vec, self._matrix)[0]
        keyword_scores = np.array([_keyword_score(query, c.text) for c in self.chunks])

        combined = self.vector_weight * vector_scores + (1 - self.vector_weight) * keyword_scores
        ranked_idx = np.argsort(-combined)[:top_k]
        return [RetrievedChunk(chunk=self.chunks[i], score=float(combined[i])) for i in ranked_idx if combined[i] > 0]

    def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Hook for a cross-encoder reranker. Identity function until one is wired in —
        production should replace this with a real cross-encoder (e.g. Cohere rerank,
        or a local ms-marco cross-encoder) scoring (query, chunk.text) pairs directly."""
        return candidates
