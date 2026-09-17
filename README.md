# UAE Tax Computation Agent — Reference Implementation

This is a runnable scaffold for the architecture in the plan: a dynamic document
understanding layer accepts client files without assuming one fixed workbook
schema; the tax computation remains grounded in the versioned UAE Corporate Tax
law corpus, with citations verified against retrieved source text before they
are shown.

It runs end-to-end on the sample data in `sample_data/` with **no external
API keys or network access** — retrieval uses a local TF-IDF index and
classification uses a deterministic heuristic classifier. Both are behind
clean interfaces so you can swap in real embeddings (OpenAI / a local
sentence-transformers model) and a real LLM (OpenAI / Anthropic) without
touching the rest of the pipeline — see "Swapping in production components"
below.

## Why it's built this way

Tax computation is arithmetic under a law-derived policy, not a
language-generation task. If an LLM regenerates the computation from scratch
on every run, you get no audit trail, no reproducibility, and citations that
are *plausible* rather than *verified*. So the design keeps a hard line:

- **Always resolves statutory facts from the FTA corpus:** rate bands, Small
  Business Relief threshold, entertainment disallowance, interest limitation,
  tax-loss cap, and Free Zone treatment are extracted by
  `app/rag/law_policy.py` and returned with the source citation. If a fact
  cannot be verified, the computation fails closed for review; no stale legal
  number is embedded in `app/rules_engine/ct_engine.py`.
- **Goes through RAG + LLM, always with a verified citation:** "is this
  specific cost wholly and exclusively for the business", "does this
  transaction match a Free Zone Qualifying Income category". These are
  `app/rag/classifier.py`, and every citation it returns is checked against
  the chunks actually retrieved — an unverifiable citation is dropped, not
  shown.
- **Never auto-finalized:** every `ComputationResult` carries a
  `review_queue` of items below a confidence threshold. The pipeline does
  not produce a "filed" number — it produces a number with a full trail
  back to either a rule citation or a flagged-for-review classification.

## Project layout

```
app/
  ingestion/        Layer 1 — parse UAE law PDFs/text into Article-level,
                     citable chunks with amendment/version metadata.
  intake/            Layer 2 — profile and extract arbitrary client documents
                     (TB, financial statements, working files, PDF, DOCX,
                     CSV, XLSX) into AI-ready normalized rows.
  rules_engine/      Arithmetic mechanics for the CT computation.
  rag/               Law-policy extraction plus hybrid retrieval and a
                     citation-verified classifier for ambiguous line items.
  orchestration/     Wires the layers into one pipeline with an audit trail.
  api/               FastAPI app (source only — see note below).
tests/               unittest suite: rules engine golden cases, trial
                     balance parsing, retrieval, citation verification,
                     one full end-to-end run.
sample_data/         A synthetic trial balance and a handful of synthetic
                     law articles standing in for the real 700-document
                     corpus, so the pipeline is runnable and testable here.
```

## Running it

```bash
# from uae-tax-agent/
python3 -m app.ingestion.law_ingest          # builds data/law_corpus/index.json
python3 -m tests.run_all                      # runs the whole test suite
python3 demo.py                               # end-to-end demo on sample data
```

### React UI

The `frontend/` directory contains a React/Vite interface for uploading any
combination of TBs, financial statements, working files, PDFs, DOCX, CSV, and
XLSX files, then viewing the generated computation with citations and a review
queue.

```bash
# terminal 1 — API
uvicorn app.api.main:app --reload

# terminal 2 — UI
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. During local development, Vite proxies the UI's
`POST /compute` request to the FastAPI service on port 8000, avoiding browser
localhost/CORS issues. Set `VITE_API_BASE_URL` when the API runs on another
host in a deployed environment.

The local intake fallback discovers tables and amount-bearing text without
requiring exact `Account`/`Amount` headers. The production seam is
`app/intake/dynamic_documents.py`: connect a document-intelligence or LLM
agent there to return normalized rows, confidence, and provenance. Ambiguous
files should become review items rather than being forced through guessed
logic.

## Swapping in production components

Every external dependency is behind an interface, deliberately, so the
rules engine and audit trail never change when you upgrade the AI parts:

- **Embeddings** (`app/rag/retriever.py`, class `TfidfEmbedder`): implement
  the same `.embed(texts) -> np.ndarray` interface with `text-embedding-3-large`
  (OpenAI) or a local `bge-large` (sentence-transformers) and pass it into
`HybridRetriever(embedder=...)`. Keep the TF-IDF/keyword channel alongside
  it — legal text has exact terms of art ("Qualifying Free Zone Person",
  "Article 21") that dense embeddings alone under-match.
- **LLM classification** (`app/rag/classifier.py`, class `LLMClient`):
  implement `.classify(item, context_chunks) -> ClassificationResult` against
  the OpenAI or Anthropic API using structured/JSON output — never freeform
  prose — and keep the existing `verify_citations()` gate in front of it.
- **Storage** (`app/rag/vector_store.py`): the default is a persistent,
  SQLite-backed vector database containing the citable chunks, sparse TF-IDF
  vectors, and fitted model. It can be replaced with PostgreSQL + pgvector
  behind the same retriever contract.
- **Real law corpus:** the application automatically uses
  `D:\project\UAE-Portal\Info\FTA LAW GUIDE\FTA LAW GUIDE` when it exists.
  Set `FTA_LAW_SOURCE_DIR` to point at another corpus. The ingestion walks all
  subdirectories and indexes PDF, TXT, Markdown, JSON, XML, and HTML files.
  Build it explicitly with:

  ```bash
  python -m app.ingestion.law_ingest --source-dir "D:\project\UAE-Portal\Info\FTA LAW GUIDE\FTA LAW GUIDE"
  ```

  This writes `data/law_corpus/fta_law_guide.sqlite3` and a JSON audit index.
  The pipeline checks a source-file fingerprint on every run and rebuilds the
  vector database only when the law folder changes. PDFs with no text remain
  visible in the index and are marked as requiring OCR instead of being
  silently discarded.

## What's intentionally *not* built here

This is still a reference core, not the production product: no auth, durable
client-document storage, async job queue, OCR provider, or external LLM adapter.
Those services wrap the same normalized document contract and audit trail.
