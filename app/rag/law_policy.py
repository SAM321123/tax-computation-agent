"""Resolve computation facts from the indexed law corpus.

The computation engine owns arithmetic and control flow only. Statutory rates,
thresholds, caps, and disallowance percentages are read from retrieved law
chunks at runtime and returned with the exact source citation. If the corpus
does not contain an unambiguous fact, computation stops rather than silently
falling back to a stale number embedded in Python.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.ingestion.law_ingest import LawChunk
from app.rag.retriever import HybridRetriever, RetrievedChunk
from app.rules_engine.models import Citation


class LawPolicyResolutionError(ValueError):
    """Raised when a required legal fact cannot be verified from the corpus."""


@dataclass(frozen=True)
class LawFact:
    name: str
    value: float | None
    citation: Citation
    evidence: str


@dataclass(frozen=True)
class LawPolicy:
    zero_rate_threshold: float
    standard_rate: float
    small_business_relief_revenue_cap: float
    entertainment_disallowance_rate: float
    interest_cap_rate_of_ebitda: float
    interest_de_minimis: float
    loss_relief_offset_cap: float
    citations: dict[str, Citation]

    def citation(self, fact_name: str) -> Citation:
        return self.citations[fact_name]


def _date_key(chunk: LawChunk) -> tuple[int, int, int]:
    """Prefer the newest source when the corpus contains amended duplicates."""
    raw = " ".join((chunk.effective_date, chunk.decision_no, chunk.law_name, chunk.source_path))
    candidates: list[tuple[int, int, int]] = []
    for day, month, year in re.findall(r"\b(\d{1,2})[-_/](\d{1,2})[-_/](20\d{2})\b", raw):
        candidates.append((int(year), int(month), int(day)))
    for year in re.findall(r"\b(20\d{2})\b", raw):
        candidates.append((int(year), 0, 0))
    return max(candidates, default=(0, 0, 0))


class LawPolicyResolver:
    """Extract the current calculation facts from retrieved legal text."""

    _QUERIES = {
        "zero_rate_threshold": "Cabinet Decision income subject Corporate Tax at zero percent amount not exceeding",
        "standard_rate": "Cabinet Decision income subject Corporate Tax at nine percent amount exceeds",
        "small_business": "Small Business Relief Taxable Person Revenue threshold relevant Tax Period",
        "entertainment": "Entertainment Expenditure allowed to deduct percentage customers suppliers business partners",
        "interest_cap": "General Interest Deduction Limitation Net Interest Expenditure accounting EBITDA deductible percentage",
        "interest_minimum": "De Minimis Net Interest Expenditure limitation does not apply amount Minister",
        "loss_relief": "Tax Loss Relief amount used reduce Taxable Income cannot exceed percentage",
    }

    def __init__(self, retriever: HybridRetriever, top_k: int = 12):
        self.retriever = retriever
        self.top_k = top_k

    def _chunks(self, query: str, article: str | None = None) -> list[RetrievedChunk]:
        candidates = self.retriever.search(query, top_k=self.top_k)
        if article is not None:
            candidates = [candidate for candidate in candidates if candidate.chunk.article == article]
        return sorted(candidates, key=lambda candidate: (_date_key(candidate.chunk), candidate.score), reverse=True)

    @staticmethod
    def _fact(name: str, candidate: RetrievedChunk, value: float, evidence: str) -> LawFact:
        chunk = candidate.chunk
        return LawFact(
            name=name,
            value=value,
            citation=Citation(
                law_name=chunk.law_name,
                article=chunk.article,
                decision_no=chunk.decision_no,
                effective_date=chunk.effective_date,
            ),
            evidence=evidence,
        )

    def _required(self, name: str, query: str, article: str | None, pattern: str, flags: int = re.IGNORECASE) -> LawFact:
        for candidate in self._chunks(query, article):
            match = re.search(pattern, candidate.chunk.text.replace("\uFFFD", ""), flags)
            if match:
                return self._fact(name, candidate, float(match.group(1).replace(",", "")), match.group(0))
        raise LawPolicyResolutionError(
            f"Required legal fact '{name}' could not be verified from the indexed law corpus. "
            "Refresh the FTA law corpus or route this computation to review."
        )

    def _citation_required(self, name: str, query: str, article: str | None = None) -> Citation:
        candidates = self._chunks(query, article)
        if not candidates:
            raise LawPolicyResolutionError(
                f"Required legal source '{name}' could not be retrieved from the indexed law corpus. "
                "Refresh the FTA law corpus or route this computation to review."
            )
        chunk = candidates[0].chunk
        return Citation(
            law_name=chunk.law_name,
            article=chunk.article,
            decision_no=chunk.decision_no,
            effective_date=chunk.effective_date,
        )

    def resolve(self) -> LawPolicy:
        threshold_fact = self._required(
            "zero_rate_threshold",
            self._QUERIES["zero_rate_threshold"],
            article="2",
            pattern=r"not\s+exceeding\s*\(?([0-9][0-9,]*)\)?",
        )
        standard_fact = None
        for candidate in self._chunks(self._QUERIES["standard_rate"], article="3"):
            text = candidate.chunk.text.replace("\uFFFD", "")
            match = re.search(
                r"(?:^|\n)\s*b\.\s*([0-9]+(?:\.[0-9]+)?)\s*%.*?Taxable\s+Income.*?exceeds",
                text,
                re.IGNORECASE | re.DOTALL,
            )
            if match:
                standard_fact = self._fact(
                    "standard_rate",
                    candidate,
                    float(match.group(1)) / 100,
                    match.group(0),
                )
                break
        if standard_fact is None:
            raise LawPolicyResolutionError(
                "Required standard Corporate Tax rate could not be verified from Article 3(b) in the indexed law corpus."
            )

        small_business = self._required(
            "small_business_relief_revenue_cap",
            self._QUERIES["small_business"],
            article="2",
            pattern=r"Revenue\s+threshold.*?AED\s*([0-9][0-9,]*)",
            flags=re.IGNORECASE | re.DOTALL,
        )
        entertainment = self._required(
            "entertainment_allowed_rate",
            self._QUERIES["entertainment"],
            article="32",
            pattern=r"allowed\s+to\s+deduct\s*([0-9]+(?:\.[0-9]+)?)\s*%",
        )
        interest_cap = self._required(
            "interest_cap_rate_of_ebitda",
            self._QUERIES["interest_cap"],
            article="30",
            pattern=r"deductible\s+up\s+to\s*([0-9]+(?:\.[0-9]+)?)\s*%",
        )
        interest_minimum = self._required(
            "interest_de_minimis",
            self._QUERIES["interest_minimum"],
            article="8",
            pattern=r"does\s+not\s+exceed\s+AED\s*([0-9][0-9,]*)",
        )
        loss_relief = self._required(
            "loss_relief_offset_cap",
            self._QUERIES["loss_relief"],
            article="37",
            pattern=r"cannot\s+exceed\s*([0-9]+(?:\.[0-9]+)?)\s*%",
        )

        citations = {
            "zero_rate_threshold": threshold_fact.citation,
            "standard_rate": standard_fact.citation,
            "small_business_relief_revenue_cap": small_business.citation,
            "entertainment_disallowance_rate": entertainment.citation,
            "non_deductible_expenditure": self._citation_required(
                "non_deductible_expenditure",
                "Non-deductible Expenditure donations fines penalties Article 33",
                article="33",
            ),
            "interest_cap_rate_of_ebitda": interest_cap.citation,
            "interest_de_minimis": interest_minimum.citation,
            "loss_relief_offset_cap": loss_relief.citation,
            "exempt_income_dividend": self._citation_required(
                "exempt_income_dividend",
                "Article 22 dividends distributions received resident juridical person exempt income",
                article="22",
            ),
            "exempt_income_participation": self._citation_required(
                "exempt_income_participation",
                "Article 23 participation exemption qualifying interest",
                article="23",
            ),
            "free_zone": self._citation_required(
                "free_zone",
                "Article 18 Qualifying Free Zone Person qualifying income",
                article="18",
            ),
            "mne": self._citation_required(
                "mne",
                "Domestic Minimum Top-up Tax Federal Decree-Law No. 60 of 2023",
            ),
        }
        return LawPolicy(
            zero_rate_threshold=threshold_fact.value,
            standard_rate=standard_fact.value,
            small_business_relief_revenue_cap=small_business.value,
            entertainment_disallowance_rate=1.0 - float(entertainment.value) / 100,
            interest_cap_rate_of_ebitda=float(interest_cap.value) / 100,
            interest_de_minimis=interest_minimum.value,
            loss_relief_offset_cap=float(loss_relief.value) / 100,
            citations=citations,
        )
