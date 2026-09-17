"""Generates sample_data/trial_balance_sample.xlsx — a synthetic SME trial
balance covering both coded-rule categories (entertainment, interest,
fines, dividends) and UNMAPPED lines meant to exercise the RAG classifier
(client hospitality described in free text, a manufacturing services fee
that should read as Free-Zone Qualifying Income)."""
import pandas as pd
from pathlib import Path

rows = [
    ("Sales Revenue", 4_500_000, "Revenue from consulting services"),
    ("Cost of Sales", 1_800_000, "Direct project delivery costs"),
    ("Staff Salaries", 950_000, "Employee salaries and wages"),
    ("Entertainment - Client Dinners", 60_000, "Client dinners and hospitality"),
    ("Depreciation - Office Equipment", 40_000, "Depreciation on fixed assets"),
    ("Interest Expense - Term Loan", 25_000, "Interest on bank term loan"),
    ("Traffic Fines", 3_500, "Fines for company vehicle traffic violations"),
    ("Donation - Local Charity", 15_000, "Cash donation to a community charity"),
    ("Dividend Income - Subsidiary", 120_000, "Dividend received from UAE subsidiary"),
    ("Client Gift Hampers", 8_000, "Festive season gift hampers sent to key clients"),
    ("Manufacturing Services Fee Income", 300_000, "Fee income from contract manufacturing services provided to a Free Zone customer"),
    ("Office Rent", 180_000, "Annual office rent"),
    ("Professional Fees", 55_000, "Legal and audit fees"),
]

df = pd.DataFrame(rows, columns=["Account", "Amount", "Description"])
out = Path(__file__).parent / "trial_balance_sample.xlsx"
df.to_excel(out, index=False)
print(f"Wrote {out} ({len(df)} rows)")
