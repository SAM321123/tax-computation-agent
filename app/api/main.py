"""
FastAPI app — source only in this environment (no outbound access to PyPI
here to install `fastapi`/`uvicorn`). Install and run with:

    pip install fastapi uvicorn python-multipart
    uvicorn app.api.main:app --reload

Deliberately thin: every endpoint just calls into orchestration/pipeline.py,
which is the layer that's actually tested. This file's only job is HTTP
plumbing — file upload handling and response shaping — so it's the one
part of the codebase you'd expect to rewrite if you moved to a different
framework or added auth/async job handling (Phase 5 in the architecture
plan) without touching the pipeline itself.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.orchestration.pipeline import run_pipeline
from app.intake.dynamic_documents import DocumentIngestionError, ingest_documents
from app.rules_engine.ct_engine import Elections
from app.rules_engine.models import Adjustment, ComputationResult, LineItemMapping

app = FastAPI(title="UAE Tax Computation Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ElectionsIn(BaseModel):
    small_business_relief: bool = False
    is_qualifying_free_zone_person: bool = False
    qualifying_free_zone_income_ratio: float = 0.0
    prior_year_tax_losses: float = 0.0
    is_in_scope_mne_group: bool = False


class AdjustmentOut(BaseModel):
    label: str
    amount: float
    citation: str
    citation_source: str
    confidence: float
    needs_review: bool
    notes: str
    source_file: str
    source_sheet: str
    source_row: int | None

    @classmethod
    def from_domain(cls, a: Adjustment) -> "AdjustmentOut":
        return cls(
            label=a.label,
            amount=a.amount,
            citation=str(a.citation),
            citation_source=a.citation_source.value,
            confidence=a.confidence,
            needs_review=a.needs_review,
            notes=a.notes,
            source_file=a.source_file,
            source_sheet=a.source_sheet,
            source_row=a.source_row,
        )


class LineItemMappingOut(BaseModel):
    account_name: str
    amount: float
    mapped_category: str
    mapping_status: str
    mapping_method: str
    tax_treatment: str
    citation: str
    citation_source: str
    confidence: float
    needs_review: bool
    notes: str
    source_file: str
    source_sheet: str
    source_row: int | None

    @classmethod
    def from_domain(cls, item: LineItemMapping) -> "LineItemMappingOut":
        return cls(
            account_name=item.account_name,
            amount=item.amount,
            mapped_category=item.mapped_category,
            mapping_status=item.mapping_status,
            mapping_method=item.mapping_method,
            tax_treatment=item.tax_treatment,
            citation=str(item.citation) if item.citation else "",
            citation_source=item.citation_source.value if item.citation_source else "",
            confidence=item.confidence,
            needs_review=item.needs_review,
            notes=item.notes,
            source_file=item.source_file,
            source_sheet=item.source_sheet,
            source_row=item.source_row,
        )


class ComputationOut(BaseModel):
    accounting_profit: float
    taxable_income: float
    tax_due: float
    small_business_relief_applied: bool
    free_zone_split: dict | None
    adjustments: list[AdjustmentOut]
    review_queue: list[AdjustmentOut]
    line_items: list[LineItemMappingOut]
    input_documents: list[dict]
    ingestion_summary: list[dict]

    @classmethod
    def from_domain(cls, r: ComputationResult, input_documents: list[dict] | None = None, ingestion_summary: list[dict] | None = None) -> "ComputationOut":
        return cls(
            accounting_profit=r.accounting_profit,
            taxable_income=r.taxable_income,
            tax_due=r.tax_due,
            small_business_relief_applied=r.small_business_relief_applied,
            free_zone_split=r.free_zone_split,
            adjustments=[AdjustmentOut.from_domain(a) for a in r.adjustments],
            review_queue=[AdjustmentOut.from_domain(a) for a in r.review_queue],
            line_items=[LineItemMappingOut.from_domain(item) for item in r.line_items],
            input_documents=input_documents or [],
            ingestion_summary=ingestion_summary or [],
        )


@app.post("/compute", response_model=ComputationOut)
async def compute(
    input_documents: list[UploadFile] = File(...),
    small_business_relief: bool = False,
    is_qualifying_free_zone_person: bool = False,
    qualifying_free_zone_income_ratio: float = 0.0,
    prior_year_tax_losses: float = 0.0,
    is_in_scope_mne_group: bool = False,
) -> ComputationOut:
    """Accept arbitrary client documents and run dynamic intake over them.

    The local fallback discovers tabular or amount-bearing rows. Production
    deployments should replace that fallback with an AI document-understanding
    adapter that returns the same normalized ledger contract.
    """
    uploaded_documents: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        saved_paths: list[Path] = []
        for document in input_documents:
            safe_name = Path(document.filename or "supporting_document").name
            document_path = Path(tmp) / safe_name
            with document_path.open("wb") as f:
                shutil.copyfileobj(document.file, f)
            saved_paths.append(document_path)
            uploaded_documents.append(
                {
                    "filename": document_path.name,
                    "content_type": document.content_type or "application/octet-stream",
                    "size_bytes": document_path.stat().st_size,
                    "role": "input_document",
                }
            )

        try:
            ingestion = ingest_documents(saved_paths)
        except DocumentIngestionError as exc:
            raise HTTPException(
                status_code=422,
                detail={"message": str(exc), "documents": [profile.as_dict() for profile in exc.profiles]},
            ) from exc

        elections = Elections(
            small_business_relief=small_business_relief,
            is_qualifying_free_zone_person=is_qualifying_free_zone_person,
            qualifying_free_zone_income_ratio=qualifying_free_zone_income_ratio,
            prior_year_tax_losses=prior_year_tax_losses,
            is_in_scope_mne_group=is_in_scope_mne_group,
        )
        try:
            result = run_pipeline(None, elections, ledger=ingestion.ledger)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ComputationOut.from_domain(result, uploaded_documents, [profile.as_dict() for profile in ingestion.profiles])


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
