"""Tests for Layer 2 — trial balance parsing and account-name categorization."""
import unittest
from pathlib import Path

from app.intake.trial_balance import categorize, parse_trial_balance, summarize
from app.rules_engine.models import LineCategory

SAMPLE = Path(__file__).resolve().parents[1] / "sample_data" / "trial_balance_sample.xlsx"


class TestCategorize(unittest.TestCase):
    def test_entertainment_keywords(self):
        self.assertEqual(categorize("Entertainment - Client Dinners"), LineCategory.ENTERTAINMENT)
        self.assertEqual(categorize("Client Gift Hampers"), LineCategory.ENTERTAINMENT)

    def test_unmapped_when_no_keyword_matches(self):
        self.assertEqual(categorize("Manufacturing Services Fee Income"), LineCategory.UNMAPPED)

    def test_fines_and_donations(self):
        self.assertEqual(categorize("Traffic Fines"), LineCategory.FINE_PENALTY)
        self.assertEqual(categorize("Donation - Local Charity"), LineCategory.DONATION)


class TestParseTrialBalance(unittest.TestCase):
    def test_parses_sample_file(self):
        lines = parse_trial_balance(SAMPLE)
        self.assertEqual(len(lines), 13)
        names = {l.account_name for l in lines}
        self.assertIn("Sales Revenue", names)

    def test_unmapped_lines_carry_description_for_rag(self):
        lines = parse_trial_balance(SAMPLE)
        manufacturing = next(l for l in lines if "Manufacturing" in l.account_name)
        self.assertEqual(manufacturing.category, LineCategory.UNMAPPED)
        self.assertTrue(manufacturing.description)  # must be non-empty for classify_line_item to work

    def test_missing_required_column_raises(self):
        import pandas as pd
        import tempfile

        # Use a closed path so the test works on Windows, where openpyxl cannot
        # replace a file that NamedTemporaryFile still has open.
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "missing_amount.xlsx"
            pd.DataFrame({"Account": ["X"]}).to_excel(path, index=False)
            with self.assertRaises(ValueError):
                parse_trial_balance(path)


class TestSummarize(unittest.TestCase):
    def test_totals_are_consistent(self):
        lines = parse_trial_balance(SAMPLE)
        totals = summarize(lines)
        self.assertGreater(totals["revenue"], 0)
        self.assertIsInstance(totals["accounting_profit"], float)


if __name__ == "__main__":
    unittest.main()
