"""
Layer 1 — turn the UAE tax law corpus into citable, versioned chunks.

Real UAE tax law documents (Federal Decree-Laws, Cabinet Decisions,
Ministerial Decisions, FTA guides/clarifications) don't chunk like generic
text: a citation is only useful if it points at the specific Article it
came from, and later Cabinet/Ministerial Decisions routinely amend
specific Articles of an earlier law. So this module chunks on Article
boundaries (never a fixed token window) and tracks a `supersedes` link so
retrieval can resolve to the currently-effective text.

Point LAW_SOURCE_DIR at the real ~700-document corpus to use this for
real; sample_data/law_corpus/ has a handful of synthetic documents in the
same shape so the pipeline is runnable here without the real files.
"""
from __future__ import annotations

import json
import os
import re
import argparse
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - optional fallback
    PdfReader = None

try:
    import pdfplumber
except ImportError:  # pragma: no cover - degrades gracefully if not installed
    pdfplumber = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_LAW_SOURCE_DIR = PROJECT_ROOT / "sample_data" / "law_corpus"
FTA_LAW_SOURCE_DIR = Path(
    os.environ.get(
        "FTA_LAW_SOURCE_DIR",
        r"D:\project\UAE-Portal\Info\FTA LAW GUIDE\FTA LAW GUIDE",
    )
)
LAW_SOURCE_DIR = SAMPLE_LAW_SOURCE_DIR
INDEX_PATH = PROJECT_ROOT / "data" / "law_corpus" / "index.json"
FTA_INDEX_PATH = Path(
    os.environ.get(
        "FTA_LAW_INDEX_PATH",
        str(PROJECT_ROOT / "data" / "law_corpus" / "fta_law_guide_index.json"),
    )
)
FTA_VECTOR_DB_PATH = Path(
    os.environ.get(
        "FTA_VECTOR_DB_PATH",
        str(PROJECT_ROOT / "data" / "law_corpus" / "fta_law_guide.sqlite3"),
    )
)

SUPPORTED_LAW_EXTENSIONS = {".pdf", ".txt", ".md", ".json", ".xml", ".html", ".htm"}

# Matches "Article 12", "Article 12 bis", "Article 12(3)" at the start of a line —
# the anchor the chunker splits on. Real documents will need a second pattern for
# Arabic-numeral-only headings and for Cabinet Decision "Clause" numbering; both are
# cheap additions to ARTICLE_HEADING_RE once the real corpus's conventions are known.
ARTICLE_HEADING_RE = re.compile(
    r"^\s*Article\s+(?:No\.?\s*)?\(?([0-9]+[A-Za-z\-]*)\)?\b",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class LawChunk:
    chunk_id: str
    law_name: str
    decision_no: str
    article: str
    effective_date: str
    text: str
    source_path: str
    supersedes: str = ""  # chunk_id of an earlier Article this one amends/replaces, if any

    def citation_str(self) -> str:
        parts = [self.law_name]
        if self.article:
            parts.append(f"Art. {self.article}")
        return ", ".join(parts)


def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        if PdfReader is not None:
            reader = PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            if text.strip():
                return text
        if pdfplumber is None:
            raise RuntimeError(
                f"No PDF text extractor available for {path}. Scanned decisions will also "
                "need an OCR fallback (Tesseract / a cloud document-intelligence API) — "
                "text extraction alone will return an empty/garbled string for those."
            )
        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def configured_law_source_dir() -> Path:
    """Return the configured corpus, preferring the real FTA folder when present.

    ``build_index()`` intentionally keeps the small sample corpus as its default
    for fast unit tests. The application pipeline uses this function, so a local
    checkout automatically picks up the real FTA guide folder without changing
    test fixtures. Set ``FTA_LAW_SOURCE_DIR`` to use a different corpus.
    """
    configured = os.environ.get("FTA_LAW_SOURCE_DIR")
    if configured:
        return Path(configured)
    if FTA_LAW_SOURCE_DIR.exists():
        return FTA_LAW_SOURCE_DIR
    return SAMPLE_LAW_SOURCE_DIR


def default_index_path(source_dir: Path) -> Path:
    """Choose a separate generated JSON index for the real corpus."""
    try:
        if source_dir.resolve() == SAMPLE_LAW_SOURCE_DIR.resolve():
            return INDEX_PATH
    except OSError:
        pass
    return FTA_INDEX_PATH


def default_vector_db_path(source_dir: Path) -> Path:
    """Choose the persistent vector database for a law source directory."""
    try:
        if source_dir.resolve() == SAMPLE_LAW_SOURCE_DIR.resolve():
            return PROJECT_ROOT / "data" / "law_corpus" / "sample_law_vectors.sqlite3"
    except OSError:
        pass
    return FTA_VECTOR_DB_PATH


def _safe_id(value: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "document"
    digest = hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{readable}__{digest}"


def chunk_by_article(text: str, law_name: str, decision_no: str, effective_date: str, source_path: str) -> list[LawChunk]:
    """Split a document's text into one chunk per Article. Falls back to a single
    whole-document chunk if no Article headings are found (covers short Cabinet/
    Ministerial Decisions that amend a single provision without their own numbering)."""
    matches = list(ARTICLE_HEADING_RE.finditer(text))
    chunks: list[LawChunk] = []

    if not matches:
        chunk_id = f"{_safe_id(source_path)}::whole"
        return [
            LawChunk(
                chunk_id=chunk_id,
                law_name=law_name,
                decision_no=decision_no,
                article="",
                effective_date=effective_date,
                text=text.strip(),
                source_path=source_path,
            )
        ]

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        article_no = m.group(1)
        body = text[start:end].strip()
        # TOCs and repeated article labels can occur in one PDF, so the
        # occurrence number is part of the storage key. The article itself is
        # preserved separately for the human-readable citation.
        chunk_id = f"{_safe_id(source_path)}::Art{article_no}::part{i + 1}"
        chunks.append(
            LawChunk(
                chunk_id=chunk_id,
                law_name=law_name,
                decision_no=decision_no,
                article=article_no,
                effective_date=effective_date,
                text=body,
                source_path=source_path,
            )
        )
    return chunks


def load_manifest(source_dir: Path) -> list[dict]:
    """manifest.json alongside the source documents carries the metadata that isn't
    reliably extractable from the PDF text itself: decision number, effective date,
    and what (if anything) a document supersedes. In production this is populated
    from FTA's own publication index rather than hand-maintained."""
    manifest_path = source_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    return json.loads(manifest_path.read_text())


def _default_metadata(path: Path) -> dict[str, str]:
    # The real FTA directory has no manifest. The file name is still a stable,
    # auditable document identifier and is preferable to inventing a legal
    # decision number or effective date.
    title = path.stem.replace("_", " ").strip()
    return {"law_name": title, "decision_no": title, "effective_date": ""}


def build_index(source_dir: Path = LAW_SOURCE_DIR, index_path: Path = INDEX_PATH) -> list[LawChunk]:
    manifest = {m["file"]: m for m in load_manifest(source_dir)}
    all_chunks: list[LawChunk] = []

    paths = sorted(
        path for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_LAW_EXTENSIONS
    )
    for path in paths:
        if path.name == "manifest.json":
            continue
        meta = {**_default_metadata(path), **manifest.get(path.name, {})}
        try:
            text = extract_text(path)
        except Exception as exc:
            # Keep the document in the manifest/index so an OCR gap is visible
            # and does not silently remove a law from the knowledge base.
            text = f"[Text extraction failed for {path.name}: {exc}]"
        if not text.strip():
            text = f"[No extractable text in {path.name}; OCR is required for retrieval.]"
        chunks = chunk_by_article(
            text,
            law_name=meta.get("law_name", path.stem),
            decision_no=meta.get("decision_no", ""),
            effective_date=meta.get("effective_date", ""),
            source_path=str(path),
        )
        for c in chunks:
            c.supersedes = meta.get("supersedes", "")
        all_chunks.extend(chunks)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps([asdict(c) for c in all_chunks], indent=2))
    return all_chunks


def load_index(index_path: Path = INDEX_PATH) -> list[LawChunk]:
    if not index_path.exists():
        return build_index()
    raw = json.loads(index_path.read_text())
    return [LawChunk(**r) for r in raw]


def resolve_effective_chunks(chunks: list[LawChunk]) -> list[LawChunk]:
    """Drop any chunk that a later chunk's `supersedes` points at, so retrieval
    only ever surfaces the currently-effective text of an Article."""
    superseded_ids = {c.supersedes for c in chunks if c.supersedes}
    return [c for c in chunks if c.chunk_id not in superseded_ids]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the UAE law chunk index and vector database.")
    parser.add_argument("--source-dir", type=Path, default=configured_law_source_dir())
    parser.add_argument("--index-path", type=Path)
    parser.add_argument("--vector-db", type=Path)
    args = parser.parse_args()

    source_dir = args.source_dir
    index_path = args.index_path or default_index_path(source_dir)
    vector_db_path = args.vector_db or default_vector_db_path(source_dir)
    chunks = build_index(source_dir, index_path)
    from app.rag.vector_store import build_vector_database

    build_vector_database(chunks, vector_db_path, source_dir=source_dir)
    print(f"Indexed {len(chunks)} chunks from {source_dir}")
    print(f"Vector database: {vector_db_path}")
    document_count = len({c.source_path for c in chunks})
    print(f"Documents represented: {document_count}")
    for c in chunks[:5]:
        print(f"  {c.chunk_id}  [{c.citation_str()}]  ({len(c.text)} chars)")
    if len(chunks) > 5:
        print(f"  ... {len(chunks) - 5} more chunks")
