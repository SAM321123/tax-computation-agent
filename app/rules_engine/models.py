"""
Shared data model for the pipeline.

Everything downstream of client-file intake — the rules engine, the RAG
classifier, and the final report — reads and writes these types. Keeping
one schema here (rather than passing raw dicts between layers) is what
makes the audit trail possible: every number in the final computation can
be traced back to a LedgerLine and an Adjustment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class LineCategory(str, Enum):
    """Standard categories a trial balance line gets mapped into."""

    REVENUE = "revenue"
    COST_OF_SALES = "cost_of_sales"
    STAFF_COST = "staff_cost"
    ENTERTAINMENT = "entertainment"
    INTEREST_EXPENSE = "interest_expense"
    DEPRECIATION = "depreciation"
    DONATION = "donation"
    FINE_PENALTY = "fine_penalty"
    DIVIDEND_INCOME = "dividend_income"
    CAPITAL_GAIN = "capital_gain"
    RELATED_PARTY = "related_party"
    OTHER_EXPENSE = "other_expense"
    OTHER_INCOME = "other_income"
    UNMAPPED = "unmapped"


@dataclass
class LedgerLine:
    """One normalized line from a client's trial balance / working file."""

    account_name: str
    amount: float  # positive = income/credit-natured per accounting sign; see intake layer
    category: LineCategory = LineCategory.UNMAPPED
    description: str = ""  # free text used for RAG classification when category is ambiguous
    is_related_party: bool = False
    source_file: str = ""
    source_sheet: str = ""
    source_row: Optional[int] = None


class CitationSource(str, Enum):
    LAW_DOCUMENT = "law_document"      # computation fact resolved from indexed law text
    RAG_CLASSIFIED = "rag_classified"  # LLM judgment, backed by a verified citation
    UNVERIFIED = "unverified"          # LLM proposed a citation that failed verification


@dataclass
class Citation:
    law_name: str
    article: str
    decision_no: str = ""
    effective_date: str = ""

    def __str__(self) -> str:
        parts = [self.law_name]
        if self.article:
            parts.append(f"Art. {self.article}")
        if self.decision_no:
            parts.append(f"({self.decision_no})")
        return ", ".join(parts)


@dataclass
class Adjustment:
    """One add-back / deduction / exemption applied to accounting profit."""

    label: str
    amount: float  # signed: positive = added back to taxable income, negative = deducted
    citation: Citation
    citation_source: CitationSource
    confidence: float = 1.0  # corpus-resolved facts default to 1.0; uncertain cases carry a lower score
    line_ref: Optional[str] = None  # account_name of the LedgerLine this came from
    notes: str = ""
    source_file: str = ""
    source_sheet: str = ""
    source_row: Optional[int] = None

    @property
    def needs_review(self) -> bool:
        return self.citation_source in (CitationSource.LAW_DOCUMENT, CitationSource.RAG_CLASSIFIED, CitationSource.UNVERIFIED) and self.confidence < 0.75


@dataclass
class LineItemMapping:
    """Auditable mapping for every source line, including lines with no tax adjustment."""

    account_name: str
    amount: float
    mapped_category: str
    mapping_status: str  # mapped | review_required
    mapping_method: str  # law_document | ai_verified | ai_review
    tax_treatment: str
    citation: Optional[Citation] = None
    citation_source: Optional[CitationSource] = None
    confidence: float = 1.0
    needs_review: bool = False
    notes: str = ""
    source_file: str = ""
    source_sheet: str = ""
    source_row: Optional[int] = None


@dataclass
class ComputationResult:
    accounting_profit: float
    adjustments: list[Adjustment] = field(default_factory=list)
    taxable_income: float = 0.0
    tax_due: float = 0.0
    small_business_relief_applied: bool = False
    free_zone_split: Optional[dict] = None
    review_queue: list[Adjustment] = field(default_factory=list)
    line_items: list[LineItemMapping] = field(default_factory=list)

    def add(self, adjustment: Adjustment) -> None:
        self.adjustments.append(adjustment)
        if adjustment.needs_review:
            self.review_queue.append(adjustment)

    @property
    def total_adjustments(self) -> float:
        return sum(a.amount for a in self.adjustments)
