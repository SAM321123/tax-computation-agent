"""
Wires Layers 1-4 into one pipeline with a full audit trail.

Order of operations:
  1. Load the law corpus (Layer 1) and build the retriever.
  2. Parse the client trial balance into LedgerLines (Layer 2).
  3. Resolve statutory facts from the current law corpus and run arithmetic mechanics.
  4. Send every still-UNMAPPED line through the RAG classifier (Layer 3),
     merging its Adjustments into the same ComputationResult — so a line
     the keyword mapper couldn't place still ends up taxed correctly
     instead of silently dropped.
  5. Return the ComputationResult: taxable income, tax due, every
     Adjustment with its citation, and a review_queue of anything below
     the confidence threshold.

This function is intentionally the *only* place that talks to all four
layers — everything else stays composable and independently testable.
"""
from __future__ import annotations

import sqlite3
import zlib
from pathlib import Path

from app.ingestion.law_ingest import (
    build_index,
    configured_law_source_dir,
    default_index_path,
    default_vector_db_path,
    resolve_effective_chunks,
)
from app.intake.trial_balance import parse_trial_balance, summarize
from app.rag.classifier import HeuristicLLMClient, LLMClient, classify_line_item
from app.rag.law_policy import LawPolicyResolver
from app.rag.retriever import HybridRetriever
from app.rag.vector_store import SqliteVectorStore, ensure_vector_database, source_fingerprint
from app.rules_engine import ct_engine
from app.rules_engine.models import ComputationResult, LineCategory, LineItemMapping


def _tax_treatment(adjustment) -> str:
    if adjustment is None or adjustment.amount == 0:
        return "Included in accounting profit; no adjustment"
    if adjustment.amount > 0:
        return "Add-back to taxable income"
    return "Deduction from taxable income"


def _mapping_for_line(line, adjustment=None, method="law_document") -> LineItemMapping:
    if adjustment is None:
        return LineItemMapping(
            account_name=line.account_name,
            amount=line.amount,
            mapped_category=line.category.value,
            mapping_status="mapped",
            mapping_method=method,
            tax_treatment=_tax_treatment(None),
            source_file=line.source_file,
            source_sheet=line.source_sheet,
            source_row=line.source_row,
        )

    needs_review = adjustment.needs_review
    if adjustment.citation_source.value == "law_document":
        status = "review_required" if needs_review else "mapped"
        category = line.category.value
        mapping_method = "law_document"
    elif adjustment.citation_source.value == "rag_classified" and not needs_review:
        status = "mapped"
        category = "ai_exempt_income" if adjustment.amount < 0 else "ai_non_deductible_expense"
        mapping_method = "ai_verified"
    else:
        status = "review_required"
        category = "other_expense"
        mapping_method = "ai_review"
    return LineItemMapping(
        account_name=line.account_name,
        amount=line.amount,
        mapped_category=category,
        mapping_status=status,
        mapping_method=mapping_method,
        tax_treatment=_tax_treatment(adjustment),
        citation=adjustment.citation,
        citation_source=adjustment.citation_source,
        confidence=adjustment.confidence,
        needs_review=needs_review,
        notes=adjustment.notes,
        source_file=line.source_file,
        source_sheet=line.source_sheet,
        source_row=line.source_row,
    )


def run_pipeline(
    trial_balance_path: Path | None,
    elections: ct_engine.Elections,
    llm: LLMClient | None = None,
    related_party_accounts: set[str] | None = None,
    ledger=None,
    law_source_dir: Path | None = None,
    vector_db_path: Path | None = None,
    rebuild_law_index: bool = False,
) -> ComputationResult:
    source_dir = Path(law_source_dir) if law_source_dir else configured_law_source_dir()
    index_path = default_index_path(source_dir)
    vector_db = Path(vector_db_path) if vector_db_path else default_vector_db_path(source_dir)

    # Loading a current vector DB is fast. Only extract the PDFs again when the
    # corpus fingerprint changes or a caller explicitly requests a rebuild.
    retriever = None
    if not rebuild_law_index and vector_db.exists():
        try:
            store = SqliteVectorStore.load(vector_db)
            if store.metadata.get("source_fingerprint") == source_fingerprint(source_dir):
                retriever = HybridRetriever(store.chunks, embedder=store.embedder, matrix=store.matrix)
        except (OSError, ValueError, EOFError, ImportError, sqlite3.Error, zlib.error):
            retriever = None

    if retriever is None:
        chunks = resolve_effective_chunks(build_index(source_dir, index_path))
        store = ensure_vector_database(chunks, vector_db, source_dir, rebuild=True)
        retriever = HybridRetriever(store.chunks, embedder=store.embedder, matrix=store.matrix)

    # Resolve every statutory fact from the current indexed FTA corpus before
    # doing any arithmetic. There is intentionally no stale legal fallback in
    # the rules engine.
    policy = LawPolicyResolver(retriever).resolve()

    llm = llm or HeuristicLLMClient()

    ledger = ledger or parse_trial_balance(trial_balance_path, related_party_accounts)
    totals = summarize(ledger)

    result = ct_engine.run(
        ledger=ledger,
        accounting_profit=totals["accounting_profit"],
        revenue=totals["revenue"],
        ebitda=totals["ebitda"],
        elections=elections,
        policy=policy,
    )

    # Small Business Relief short-circuits everything else — no line-level
    # classification needed if the whole computation is already Nil.
    if result.small_business_relief_applied:
        result.line_items = [
            _mapping_for_line(line) if line.category != LineCategory.UNMAPPED else LineItemMapping(
                account_name=line.account_name,
                amount=line.amount,
                mapped_category="requires_review",
                mapping_status="review_required",
                mapping_method="ai_review",
                tax_treatment="Small Business Relief applied; line-level review skipped",
                confidence=0.0,
                needs_review=True,
                notes="Line-level classification was skipped because Small Business Relief reduced taxable income to Nil.",
                source_file=line.source_file,
                source_sheet=line.source_sheet,
                source_row=line.source_row,
            )
            for line in ledger
        ]
        return result

    law_adjustments = {
        adjustment.line_ref: adjustment
        for adjustment in result.adjustments
        if adjustment.line_ref
    }
    for line in ledger:
        if line.category != LineCategory.UNMAPPED:
            result.line_items.append(_mapping_for_line(line, law_adjustments.get(line.account_name)))
            continue
        adj = classify_line_item(
            account_name=line.account_name,
            description=line.description,
            amount=line.amount,
            retriever=retriever,
            llm=llm,
        )
        result.add(adj)
        result.line_items.append(_mapping_for_line(line, adj, method="ai_verified"))

    # Re-derive taxable income / tax due now that ambiguous line-item
    # classifications (which arithmetic mechanics cannot infer) are included.
    result.taxable_income = round(result.accounting_profit + result.total_adjustments, 2)
    result.tax_due, result.free_zone_split = ct_engine.compute_tax_due(result.taxable_income, elections, policy)

    return result


def print_report(result: ComputationResult) -> None:
    print("=" * 72)
    print("UAE CORPORATE TAX COMPUTATION")
    print("=" * 72)
    print(f"Accounting profit before tax:          {result.accounting_profit:>15,.2f}")
    print("-" * 72)
    for adj in result.adjustments:
        flag = "  [REVIEW]" if adj.needs_review else ""
        print(f"  {adj.label:<48} {adj.amount:>15,.2f}{flag}")
        print(f"      cite: {adj.citation}  (source={adj.citation_source.value}, confidence={adj.confidence:.2f})")
    print("-" * 72)
    print(f"Taxable income:                         {result.taxable_income:>15,.2f}")
    if result.free_zone_split:
        fz = result.free_zone_split
        print(f"  Free Zone split — qualifying (0%):    {fz['qualifying_income']:>15,.2f}")
        print(f"  Free Zone split — non-qualifying (9%): {fz['non_qualifying_income']:>15,.2f}")
    print(f"Corporate Tax due:                      {result.tax_due:>15,.2f}")
    print(f"Small Business Relief applied:          {result.small_business_relief_applied}")
    print("=" * 72)
    if result.review_queue:
        print(f"\n{len(result.review_queue)} item(s) flagged for human review before filing:")
        for adj in result.review_queue:
            print(f"  - {adj.label} (confidence {adj.confidence:.2f}): {adj.notes}")
    else:
        print("\nNo items flagged for review.")
