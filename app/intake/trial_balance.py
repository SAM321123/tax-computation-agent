"""
Layer 2 — parse a client trial balance into the standard LedgerLine schema.

Every client names accounts differently ("Misc Exp - Entertainment", "T&E",
"Client Hospitality"). `CATEGORY_KEYWORDS` is a first-pass, rule-based
mapper; anything it can't confidently place is left as `LineCategory.UNMAPPED`
with the raw account name carried through as `description`, so the RAG
classifier (app/rag/classifier.py) has something concrete to search
against instead of silently mis-categorizing it. In production this
keyword pass is the fast path and a human confirms/corrects the mapping
once per client — reused on every subsequent period for that client.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.rules_engine.models import LedgerLine, LineCategory

# Ordered so more specific phrases are checked before generic ones
# (e.g. "entertainment" before a generic "expense" fallback would ever exist).
CATEGORY_KEYWORDS: list[tuple[LineCategory, tuple[str, ...]]] = [
    (LineCategory.ENTERTAINMENT, ("entertain", "hospitality", "client gift", "client dinner", "staff party client")),
    (LineCategory.FINE_PENALTY, ("fine", "penalty", "penalties")),
    (LineCategory.DONATION, ("donation", "charity", "csr contribution")),
    (LineCategory.INTEREST_EXPENSE, ("interest expense", "interest paid", "finance cost", "loan interest")),
    (LineCategory.DEPRECIATION, ("depreciation", "amortisation", "amortization")),
    (LineCategory.DIVIDEND_INCOME, ("dividend income", "dividend received")),
    (LineCategory.CAPITAL_GAIN, ("gain on disposal", "gain on sale of investment", "capital gain")),
    (LineCategory.STAFF_COST, ("salary", "salaries", "wages", "staff cost", "payroll")),
    (LineCategory.COST_OF_SALES, ("cost of sales", "cost of goods sold", "cogs", "direct cost")),
    (LineCategory.REVENUE, ("revenue", "sales income", "service income", "turnover")),
]


def categorize(account_name: str) -> LineCategory:
    name = account_name.lower()
    for category, keywords in CATEGORY_KEYWORDS:
        if any(k in name for k in keywords):
            return category
    return LineCategory.UNMAPPED


def parse_trial_balance(path: Path, related_party_accounts: set[str] | None = None) -> list[LedgerLine]:
    """Expects an .xlsx with columns: Account, Amount (a signed net P&L
    contribution — positive for income-natured lines, positive for expense
    amounts as well, since the engine treats each LedgerLine.amount as a
    plain magnitude and applies sign logic per rule, not per input row).
    An optional Description column feeds the RAG classifier when present."""
    df = pd.read_excel(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"account", "amount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Trial balance {path} is missing required column(s): {missing}")

    related_party_accounts = related_party_accounts or set()
    lines: list[LedgerLine] = []
    for i, row in df.iterrows():
        account = str(row["account"]).strip()
        if not account or account.lower() == "nan":
            continue
        amount = float(row["amount"])
        description = str(row.get("description", "")).strip()
        if description.lower() == "nan":
            description = ""
        lines.append(
            LedgerLine(
                account_name=account,
                amount=amount,
                category=categorize(account),
                description=description or account,
                is_related_party=account in related_party_accounts,
                source_file=str(path),
                source_row=int(i) + 2,  # +2: 1-indexed plus header row
            )
        )
    return lines


def summarize(lines: list[LedgerLine]) -> dict:
    """Accounting-profit-relevant aggregates the rules engine needs as inputs:
    net profit, revenue, and a simplified EBITDA (profit + interest + depreciation)."""
    revenue = sum(l.amount for l in lines if l.category == LineCategory.REVENUE)
    other_income = sum(l.amount for l in lines if l.category in (LineCategory.OTHER_INCOME, LineCategory.DIVIDEND_INCOME, LineCategory.CAPITAL_GAIN))
    expenses = sum(
        l.amount
        for l in lines
        if l.category
        not in (
            LineCategory.REVENUE,
            LineCategory.OTHER_INCOME,
            LineCategory.DIVIDEND_INCOME,
            LineCategory.CAPITAL_GAIN,
        )
    )
    accounting_profit = revenue + other_income - expenses
    interest = sum(l.amount for l in lines if l.category == LineCategory.INTEREST_EXPENSE)
    depreciation = sum(l.amount for l in lines if l.category == LineCategory.DEPRECIATION)
    ebitda = accounting_profit + interest + depreciation
    return {
        "revenue": revenue,
        "accounting_profit": accounting_profit,
        "ebitda": ebitda,
    }
