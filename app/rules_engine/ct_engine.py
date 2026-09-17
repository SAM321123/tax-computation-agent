"""UAE Corporate Tax computation mechanics.

This module owns arithmetic and control flow only. Statutory thresholds, rates,
caps, and disallowance percentages are supplied by ``LawPolicy``, which is
resolved from the indexed FTA law corpus at runtime. If the corpus cannot
verify a required fact, the pipeline fails closed instead of using a stale
legal constant embedded in this module.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.rag.law_policy import LawPolicy
from app.rules_engine.models import Adjustment, CitationSource, ComputationResult, LedgerLine, LineCategory


@dataclass
class Elections:
    """Choices the taxpayer/agent makes that change which rules apply."""

    small_business_relief: bool = False
    is_qualifying_free_zone_person: bool = False
    qualifying_free_zone_income_ratio: float = 0.0
    prior_year_tax_losses: float = 0.0
    is_in_scope_mne_group: bool = False


def small_business_relief_eligible(revenue: float, elections: Elections, policy: LawPolicy) -> bool:
    return elections.small_business_relief and revenue <= policy.small_business_relief_revenue_cap


def entertainment_addback(ledger: list[LedgerLine], policy: LawPolicy) -> list[Adjustment]:
    out = []
    for line in ledger:
        if line.category == LineCategory.ENTERTAINMENT and line.amount:
            out.append(Adjustment(
                label=f"Entertainment disallowance — {line.account_name}",
                amount=round(line.amount * policy.entertainment_disallowance_rate, 2),
                citation=policy.citation("entertainment_disallowance_rate"),
                citation_source=CitationSource.LAW_DOCUMENT,
                line_ref=line.account_name,
                source_file=line.source_file,
                source_sheet=line.source_sheet,
                source_row=line.source_row,
            ))
    return out


def fines_and_donations_addback(ledger: list[LedgerLine], policy: LawPolicy) -> list[Adjustment]:
    out = []
    for line in ledger:
        if line.category == LineCategory.FINE_PENALTY and line.amount:
            out.append(Adjustment(
                label=f"Fine/penalty disallowance — {line.account_name}",
                amount=line.amount,
                citation=policy.citation("non_deductible_expenditure"),
                citation_source=CitationSource.LAW_DOCUMENT,
                line_ref=line.account_name,
                source_file=line.source_file,
                source_sheet=line.source_sheet,
                source_row=line.source_row,
            ))
        elif line.category == LineCategory.DONATION and line.amount:
            out.append(Adjustment(
                label=f"Donation disallowance (QPBE status unconfirmed) — {line.account_name}",
                amount=line.amount,
                citation=policy.citation("non_deductible_expenditure"),
                citation_source=CitationSource.LAW_DOCUMENT,
                confidence=0.5,
                line_ref=line.account_name,
                notes="Deductible only if the recipient is an approved Qualifying Public Benefit Entity. Confirm recipient status via the RAG classifier or manual lookup before treating as an exempt donation.",
                source_file=line.source_file,
                source_sheet=line.source_sheet,
                source_row=line.source_row,
            ))
    return out


def interest_cap_addback(ledger: list[LedgerLine], ebitda: float, policy: LawPolicy) -> list[Adjustment]:
    net_interest = sum(l.amount for l in ledger if l.category == LineCategory.INTEREST_EXPENSE)
    if net_interest <= policy.interest_de_minimis:
        return []
    cap = max(ebitda, 0.0) * policy.interest_cap_rate_of_ebitda
    disallowed = max(net_interest - cap, 0.0)
    if disallowed <= 0:
        return []
    return [Adjustment(
        label="General interest deduction limitation",
        amount=round(disallowed, 2),
        citation=policy.citation("interest_cap_rate_of_ebitda"),
        citation_source=CitationSource.LAW_DOCUMENT,
        notes=f"Net interest {net_interest:,.2f} vs. law-corpus-derived EBITDA cap {cap:,.2f}.",
    )]


def exempt_income_deduction(ledger: list[LedgerLine], policy: LawPolicy) -> list[Adjustment]:
    out = []
    for line in ledger:
        if line.category in (LineCategory.DIVIDEND_INCOME, LineCategory.CAPITAL_GAIN) and line.amount:
            fact = "exempt_income_dividend" if line.category == LineCategory.DIVIDEND_INCOME else "exempt_income_participation"
            out.append(Adjustment(
                label=f"Exempt income deduction — {line.account_name}",
                amount=-abs(line.amount),
                citation=policy.citation(fact),
                citation_source=CitationSource.LAW_DOCUMENT,
                confidence=0.6,
                line_ref=line.account_name,
                notes="Participation-exemption ownership/holding-period conditions not yet verified — route to RAG classifier or manual review before relying on this exemption.",
                source_file=line.source_file,
                source_sheet=line.source_sheet,
                source_row=line.source_row,
            ))
    return out


def free_zone_split(taxable_income: float, elections: Elections, policy: LawPolicy) -> tuple[float, dict | None]:
    if not elections.is_qualifying_free_zone_person:
        return taxable_income, None
    ratio = max(0.0, min(1.0, elections.qualifying_free_zone_income_ratio))
    qualifying = taxable_income * ratio
    non_qualifying = taxable_income - qualifying
    return non_qualifying, {
        "qualifying_income": round(qualifying, 2),
        "non_qualifying_income": round(non_qualifying, 2),
        "qualifying_rate": 0.0,
        "non_qualifying_rate": policy.standard_rate,
        "citation": str(policy.citation("free_zone")),
    }


def apply_loss_relief(taxable_income: float, elections: Elections, policy: LawPolicy) -> tuple[float, Adjustment | None]:
    if taxable_income <= 0 or elections.prior_year_tax_losses <= 0:
        return taxable_income, None
    cap = taxable_income * policy.loss_relief_offset_cap
    used = min(elections.prior_year_tax_losses, cap)
    if used <= 0:
        return taxable_income, None
    return taxable_income - used, Adjustment(
        label="Tax loss relief applied",
        amount=-round(used, 2),
        citation=policy.citation("loss_relief_offset_cap"),
        citation_source=CitationSource.LAW_DOCUMENT,
        notes=f"Capped at the law-corpus-derived percentage ({cap:,.2f}); {elections.prior_year_tax_losses - used:,.2f} carried forward.",
    )


def compute_tax_due(taxable_income: float, elections: Elections, policy: LawPolicy) -> tuple[float, dict | None]:
    non_qz_income, fz_info = free_zone_split(taxable_income, elections, policy)
    if non_qz_income <= policy.zero_rate_threshold:
        return 0.0, fz_info
    return round((non_qz_income - policy.zero_rate_threshold) * policy.standard_rate, 2), fz_info


def run(
    ledger: list[LedgerLine],
    accounting_profit: float,
    revenue: float,
    ebitda: float,
    elections: Elections,
    policy: LawPolicy,
) -> ComputationResult:
    result = ComputationResult(accounting_profit=accounting_profit)

    if small_business_relief_eligible(revenue, elections, policy):
        result.small_business_relief_applied = True
        result.add(Adjustment(
            label="Small Business Relief — taxable income treated as Nil",
            amount=-accounting_profit,
            citation=policy.citation("small_business_relief_revenue_cap"),
            citation_source=CitationSource.LAW_DOCUMENT,
            notes=f"Revenue {revenue:,.2f} <= the law-corpus-derived revenue threshold of {policy.small_business_relief_revenue_cap:,.0f} AED.",
        ))
        result.taxable_income = 0.0
        result.tax_due = 0.0
        return result

    for adjustment in entertainment_addback(ledger, policy):
        result.add(adjustment)
    for adjustment in fines_and_donations_addback(ledger, policy):
        result.add(adjustment)
    for adjustment in interest_cap_addback(ledger, ebitda, policy):
        result.add(adjustment)
    for adjustment in exempt_income_deduction(ledger, policy):
        result.add(adjustment)

    pre_loss_income = accounting_profit + result.total_adjustments
    post_loss_income, loss_adjustment = apply_loss_relief(pre_loss_income, elections, policy)
    if loss_adjustment:
        result.add(loss_adjustment)

    result.taxable_income = round(post_loss_income, 2)
    result.tax_due, result.free_zone_split = compute_tax_due(result.taxable_income, elections, policy)

    if elections.is_in_scope_mne_group:
        result.add(Adjustment(
            label="Flagged: in-scope MNE group — DMTT applicability not computed by this engine",
            amount=0.0,
            citation=policy.citation("mne"),
            citation_source=CitationSource.LAW_DOCUMENT,
            confidence=0.0,
            notes="Route to specialist review — this computation module does not calculate the applicable minimum-tax regime.",
        ))

    return result
