"""Runs the full pipeline on sample_data/trial_balance_sample.xlsx and prints
the computation with its audit trail. `python3 demo.py` from the project root."""
from pathlib import Path

from app.orchestration.pipeline import print_report, run_pipeline
from app.rules_engine.ct_engine import Elections

if __name__ == "__main__":
    trial_balance = Path(__file__).parent / "sample_data" / "trial_balance_sample.xlsx"
    elections = Elections(
        small_business_relief=False,  # revenue is above the AED 3m cap in the sample data
        is_qualifying_free_zone_person=False,
        prior_year_tax_losses=0.0,
    )
    result = run_pipeline(trial_balance, elections)
    print_report(result)
