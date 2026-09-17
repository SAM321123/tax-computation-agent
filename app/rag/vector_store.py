"""Persistent local vector database for the law RAG pipeline.

This store deliberately has no service dependency: SQLite stores the citable law
chunks, the fitted TF-IDF model, and the sparse vector matrix. It is therefore
usable by the API, CLI, and tests while retaining a clean migration seam to
Postgres/pgvector or another hosted vector database later.
"""
from __future__ import annotations

import hashlib
import io
import json
import pickle
import sqlite3
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from app.ingestion.law_ingest import LawChunk
from app.rag.retriever import TfidfEmbedder


SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    ordinal INTEGER PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS vector_matrix (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    payload BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS embedding_model (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    payload BLOB NOT NULL
);
"""


def source_fingerprint(source_dir: Path) -> str:
    """Return a cheap fingerprint that invalidates the DB when source files change."""
    source_dir = Path(source_dir)
    entries: list[tuple[str, int, int]] = []
    for path in sorted(p for p in source_dir.rglob("*") if p.is_file()):
        stat = path.stat()
        entries.append((str(path.relative_to(source_dir)), stat.st_size, stat.st_mtime_ns))
    return hashlib.sha256(json.dumps(entries, ensure_ascii=True).encode()).hexdigest()


def _compressed_numpy_payload(matrix: csr_matrix) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        data=matrix.data.astype(np.float32),
        indices=matrix.indices.astype(np.int32),
        indptr=matrix.indptr.astype(np.int32),
        shape=np.asarray(matrix.shape, dtype=np.int64),
    )
    return zlib.compress(buffer.getvalue(), level=6)


def _matrix_from_payload(payload: bytes) -> csr_matrix:
    with np.load(io.BytesIO(zlib.decompress(payload))) as arrays:
        shape = tuple(int(value) for value in arrays["shape"])
        return csr_matrix(
            (arrays["data"], arrays["indices"], arrays["indptr"]),
            shape=shape,
        )


class SqliteVectorStore:
    """A small persistent vector DB with metadata and sparse vector search data."""

    def __init__(self, path: Path, chunks: list[LawChunk], embedder: TfidfEmbedder, matrix: csr_matrix, metadata: dict[str, str]):
        self.path = path
        self.chunks = chunks
        self.embedder = embedder
        self.matrix = matrix
        self.metadata = metadata

    @classmethod
    def create(cls, chunks: list[LawChunk], path: Path, source_dir: Path | None = None) -> "SqliteVectorStore":
        path = Path(path)
        source_dir = Path(source_dir) if source_dir is not None else None
        if not chunks:
            raise ValueError("Cannot build a vector database with no law chunks.")

        embedder = TfidfEmbedder()
        embedder.fit([chunk.text for chunk in chunks])
        matrix = embedder.embed([chunk.text for chunk in chunks]).tocsr().astype(np.float32)
        metadata = {
            "schema_version": "1",
            "chunk_count": str(len(chunks)),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_dir": str(source_dir) if source_dir else "",
            "source_fingerprint": source_fingerprint(source_dir) if source_dir else "",
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        if temporary_path.exists():
            temporary_path.unlink()
        connection = sqlite3.connect(temporary_path)
        try:
            connection.executescript(SCHEMA)
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                metadata.items(),
            )
            connection.executemany(
                "INSERT INTO chunks(ordinal, chunk_id, payload) VALUES (?, ?, ?)",
                [
                    (ordinal, chunk.chunk_id, json.dumps(chunk.__dict__, ensure_ascii=False))
                    for ordinal, chunk in enumerate(chunks)
                ],
            )
            connection.execute(
                "INSERT INTO vector_matrix(id, payload) VALUES (1, ?)",
                (sqlite3.Binary(_compressed_numpy_payload(matrix)),),
            )
            connection.execute(
                "INSERT INTO embedding_model(id, payload) VALUES (1, ?)",
                (sqlite3.Binary(zlib.compress(pickle.dumps(embedder.vectorizer, protocol=5))),),
            )
            connection.commit()
        finally:
            connection.close()
        try:
            temporary_path.replace(path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        return cls(path, chunks, embedder, matrix, metadata)

    @classmethod
    def load(cls, path: Path) -> "SqliteVectorStore":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        connection = sqlite3.connect(path)
        try:
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            rows = connection.execute("SELECT payload FROM chunks ORDER BY ordinal").fetchall()
            matrix_payload = connection.execute("SELECT payload FROM vector_matrix WHERE id = 1").fetchone()
            model_payload = connection.execute("SELECT payload FROM embedding_model WHERE id = 1").fetchone()
        finally:
            connection.close()

        if not rows or not matrix_payload or not model_payload:
            raise ValueError(f"Vector database is incomplete: {path}")
        chunks = [LawChunk(**json.loads(row[0])) for row in rows]
        embedder = TfidfEmbedder()
        embedder.vectorizer = pickle.loads(zlib.decompress(model_payload[0]))
        embedder._fitted = True
        matrix = _matrix_from_payload(matrix_payload[0])
        if len(chunks) != matrix.shape[0]:
            raise ValueError(f"Vector database chunk/vector count mismatch: {path}")
        return cls(path, chunks, embedder, matrix, metadata)


def build_vector_database(chunks: list[LawChunk], path: Path, source_dir: Path | None = None) -> SqliteVectorStore:
    """Build and persist a vector database from already chunked law documents."""
    return SqliteVectorStore.create(chunks, path, source_dir=source_dir)


def ensure_vector_database(
    chunks: list[LawChunk],
    path: Path,
    source_dir: Path,
    rebuild: bool = False,
) -> SqliteVectorStore:
    """Load a current DB or build it when missing/stale/corrupt."""
    path = Path(path)
    source_dir = Path(source_dir)
    fingerprint = source_fingerprint(source_dir)
    if not rebuild and path.exists():
        try:
            store = SqliteVectorStore.load(path)
            if (
                store.metadata.get("source_fingerprint") == fingerprint
                and store.metadata.get("chunk_count") == str(len(chunks))
            ):
                return store
        except (OSError, sqlite3.Error, ValueError, pickle.PickleError, EOFError, zlib.error):
            pass
    return build_vector_database(chunks, path, source_dir=source_dir)
