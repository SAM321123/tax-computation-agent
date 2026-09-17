"""
Layer 3 (judgment half) — classify ambiguous line items with a verified citation.

This is the only place in the pipeline where an LLM's output can become an
Adjustment, and it is deliberately narrow:

  1. Retrieve candidate law chunks for the line item (HybridRetriever).
  2. Ask the LLM to classify against *only* those chunks, returning
     structured JSON (never freeform prose) that names which chunk it
     relied on.
  3. `verify_citation` checks that the cited chunk_id was actually in the
     retrieved set before anything downstream ever sees it. A citation
     that doesn't verify is dropped — the item is flagged for manual
     review instead of shown with a plausible-looking but ungrounded
     citation.

`LLMClient` is an interface; `HeuristicLLMClient` is a deterministic,
offline stand-in so the pipeline is fully runnable and testable without an
API key. Swap in a real OpenAI/Anthropic client behind the same interface
(`.classify(item_description, context_chunks) -> RawClassification`) —
structured/function-calling output only, so `verify_citation` always has
a well-formed chunk_id to check.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.ingestion.law_ingest import LawChunk
from app.rag.retriever import HybridRetriever, RetrievedChunk
from app.rules_engine.models import Adjustment, Citation, CitationSource


@dataclass
class RawClassification:
    """What an LLM call is expected to return: structured, and naming the
    exact chunk it relied on — never a citation it recalled from memory."""

    treatment: str          # "deductible" | "non_deductible" | "partially_deductible" | "exempt" | "unclear"
    disallowed_fraction: float  # 0..1 — fraction of the amount to add back, if partially/non-deductible
    relied_on_chunk_id: str
    confidence: float       # 0..1, the model's own estimate
    reasoning: str


class LLMClient(ABC):
    @abstractmethod
    def classify(self, item_description: str, context: list[RetrievedChunk]) -> RawClassification: ...


class HeuristicLLMClient(LLMClient):
    """Deterministic stand-in for an LLM call: keyword-matches the item
    description against the retrieved chunks and returns a plausible
    classification, always citing the top-scoring retrieved chunk. This
    exists so the pipeline runs end-to-end offline — replace with a real
    LLM client for production judgment quality, not for the interface
    shape, which stays the same."""

    NON_DEDUCTIBLE_HINTS = ("gift", "hospitality", "client entertain", "entertain")
    QUALIFYING_ACTIVITY_HINTS = ("manufactur", "logistics", "headquarter", "treasury", "reinsurance", "fund manage")

    def classify(self, item_description: str, context: list[RetrievedChunk]) -> RawClassification:
        if not context:
            return RawClassification(
                treatment="unclear",
                disallowed_fraction=0.0,
                relied_on_chunk_id="",
                confidence=0.0,
                reasoning="No relevant law chunks retrieved for this item.",
            )
        top = context[0]
        # Only match hints against the item's own description — matching against the
        # retrieved chunk's text too would let an unrelated retrieved chunk (e.g. an
        # entertainment clarification surfacing for an "Office Rent" query on a small
        # corpus) leak a false positive into an item it was never about.
        item_lower = item_description.lower()

        if any(h in item_lower for h in self.NON_DEDUCTIBLE_HINTS):
            return RawClassification(
                treatment="partially_deductible",
                disallowed_fraction=0.5,
                relied_on_chunk_id=top.chunk.chunk_id,
                confidence=min(0.9, 0.5 + top.score),
                reasoning="Item description matches entertainment/hospitality-type costs "
                "addressed in the retrieved guidance.",
            )
        if any(h in item_lower for h in self.QUALIFYING_ACTIVITY_HINTS):
            return RawClassification(
                treatment="exempt",
                disallowed_fraction=0.0,
                relied_on_chunk_id=top.chunk.chunk_id,
                confidence=min(0.85, 0.45 + top.score),
                reasoning="Item description matches a listed Qualifying Activity.",
            )
        return RawClassification(
            treatment="unclear",
            disallowed_fraction=0.0,
            relied_on_chunk_id=top.chunk.chunk_id,
            confidence=min(0.4, top.score),
            reasoning="No confident keyword match against retrieved guidance — route to "
            "manual review.",
        )


def verify_citation(raw: RawClassification, context: list[RetrievedChunk]) -> LawChunk | None:
    """The single highest-leverage guardrail in this module: only accept a
    citation whose chunk_id was actually present in what was retrieved and
    shown to the model. This can't catch a wrong *interpretation* of a real
    Article, but it eliminates hallucinated Article numbers outright."""
    for rc in context:
        if rc.chunk.chunk_id == raw.relied_on_chunk_id:
            return rc.chunk
    return None


def classify_line_item(
    account_name: str,
    description: str,
    amount: float,
    retriever: HybridRetriever,
    llm: LLMClient,
    top_k: int = 3,
) -> Adjustment:
    query = f"{account_name}. {description}".strip()
    context = retriever.search(query, top_k=top_k)
    context = retriever.rerank(query, context)

    raw = llm.classify(query, context)
    verified_chunk = verify_citation(raw, context)

    if verified_chunk is None:
        return Adjustment(
            label=f"Unresolved classification — {account_name}",
            amount=0.0,
            citation=Citation(law_name="(no verified citation)", article=""),
            citation_source=CitationSource.UNVERIFIED,
            confidence=0.0,
            line_ref=account_name,
            notes=raw.reasoning if raw.relied_on_chunk_id else "No candidate law chunk retrieved.",
        )

    if raw.treatment in ("non_deductible", "partially_deductible"):
        addback = round(amount * (1.0 if raw.treatment == "non_deductible" else raw.disallowed_fraction), 2)
        adj_amount = addback
        label = f"RAG-classified disallowance — {account_name}"
    elif raw.treatment == "exempt":
        adj_amount = -abs(amount)
        label = f"RAG-classified exemption — {account_name}"
    else:
        adj_amount = 0.0
        label = f"Flagged for manual review — {account_name}"

    return Adjustment(
        label=label,
        amount=adj_amount,
        citation=Citation(
            law_name=verified_chunk.law_name,
            article=verified_chunk.article,
            decision_no=verified_chunk.decision_no,
            effective_date=verified_chunk.effective_date,
        ),
        citation_source=CitationSource.RAG_CLASSIFIED,
        confidence=raw.confidence,
        line_ref=account_name,
        notes=raw.reasoning,
    )
