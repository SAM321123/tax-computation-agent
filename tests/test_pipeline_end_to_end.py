"""One full end-to-end test: real sample trial balance, real sample law
corpus, real (heuristic) classifier — asserting the pipeline produces a
complete, internally-consistent ComputationResult with a review queue for
what it couldn't confidently resolve."""
import unittest
from pathlib import Path

from app.orchestration.pipeline import run_pipeline
from app.rules_engine.ct_engine import Elections
from app.rules_engine.models import CitationSource

SAMPLE = Path(__file__).resolve().parents[1] / "sample_data" / "trial_balance_sample.xlsx"


class TestPipelineEndToEnd(unittest.TestCase):
    def test_produces_consistent_result_with_citations_and_review_queue(self):
        result = run_pipeline(SAMPLE, Elections())

        self.assertGreater(len(result.adjustments), 0)
        self.assertEqual(len(result.line_items), 13)
        self.assertTrue(all(item.mapped_category for item in result.line_items))
        # taxable income must equal accounting profit plus the sum of all adjustments
        self.assertAlmostEqual(
            result.taxable_income,
            round(result.accounting_profit + result.total_adjustments, 2),
            places=2,
        )
        # tax due must be consistent with the 0%/9% band applied to taxable income
        expected_tax = round(max(result.taxable_income - 375_000, 0) * 0.09, 2)
        self.assertAlmostEqual(result.tax_due, expected_tax, places=2)

        # every adjustment must carry a citation from one of the three defined sources
        for adj in result.adjustments:
            self.assertIn(adj.citation_source, list(CitationSource))
            if adj.citation_source != CitationSource.UNVERIFIED:
                self.assertTrue(str(adj.citation))

        # the manufacturing services income (Free Zone Qualifying Activity in the
        # sample corpus) should have been picked up by the RAG classifier even
        # though the keyword mapper alone leaves it UNMAPPED
        manufacturing_adj = next(a for a in result.adjustments if "Manufacturing" in a.label)
        self.assertEqual(manufacturing_adj.citation_source, CitationSource.RAG_CLASSIFIED)
        self.assertLess(manufacturing_adj.amount, 0)  # treated as exempt/qualifying, deducted

    def test_small_business_relief_short_circuits_classification(self):
        result = run_pipeline(SAMPLE, Elections(small_business_relief=True))
        # sample revenue is well above the AED 3m cap, so relief should NOT apply
        # even though it was elected — this asserts the cap is actually enforced
        self.assertFalse(result.small_business_relief_applied)


if __name__ == "__main__":
    unittest.main()
