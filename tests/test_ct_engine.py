"""Golden-case tests for computation mechanics using a test law policy."""
import unittest

from app.rag.law_policy import LawPolicy
from app.rules_engine import ct_engine
from app.rules_engine.models import Citation, LedgerLine, LineCategory


_TEST_ARTICLES = {
    "entertainment_disallowance_rate": "32(2)",
}

POLICY = LawPolicy(
    zero_rate_threshold=375_000.0,
    standard_rate=0.09,
    small_business_relief_revenue_cap=3_000_000.0,
    entertainment_disallowance_rate=0.50,
    interest_cap_rate_of_ebitda=0.30,
    interest_de_minimis=12_000_000.0,
    loss_relief_offset_cap=0.75,
    citations={name: Citation("Test law corpus", _TEST_ARTICLES.get(name, name)) for name in (
        "zero_rate_threshold", "standard_rate", "small_business_relief_revenue_cap",
        "entertainment_disallowance_rate", "non_deductible_expenditure",
        "interest_cap_rate_of_ebitda", "interest_de_minimis", "loss_relief_offset_cap",
        "exempt_income_dividend", "exempt_income_participation", "free_zone", "mne",
    )},
)


class TestRateBands(unittest.TestCase):
    def test_zero_rate_below_threshold(self):
        tax, _ = ct_engine.compute_tax_due(300_000, ct_engine.Elections(), POLICY)
        self.assertEqual(tax, 0.0)

    def test_nine_percent_above_threshold(self):
        tax, _ = ct_engine.compute_tax_due(1_375_000, ct_engine.Elections(), POLICY)
        # (1,375,000 - 375,000) * 9% = 90,000
        self.assertEqual(tax, 90_000.0)

    def test_exactly_at_threshold_is_zero(self):
        tax, _ = ct_engine.compute_tax_due(375_000, ct_engine.Elections(), POLICY)
        self.assertEqual(tax, 0.0)


class TestSmallBusinessRelief(unittest.TestCase):
    def test_eligible_when_elected_and_under_cap(self):
        elections = ct_engine.Elections(small_business_relief=True)
        self.assertTrue(ct_engine.small_business_relief_eligible(2_000_000, elections, POLICY))

    def test_not_eligible_over_cap_even_if_elected(self):
        elections = ct_engine.Elections(small_business_relief=True)
        self.assertFalse(ct_engine.small_business_relief_eligible(3_000_001, elections, POLICY))

    def test_not_eligible_without_election(self):
        elections = ct_engine.Elections(small_business_relief=False)
        self.assertFalse(ct_engine.small_business_relief_eligible(1_000_000, elections, POLICY))

    def test_run_zeroes_out_full_computation(self):
        ledger = [LedgerLine("Entertainment", 100_000, LineCategory.ENTERTAINMENT)]
        elections = ct_engine.Elections(small_business_relief=True)
        result = ct_engine.run(ledger, accounting_profit=500_000, revenue=1_000_000, ebitda=500_000, elections=elections, policy=POLICY)
        self.assertTrue(result.small_business_relief_applied)
        self.assertEqual(result.taxable_income, 0.0)
        self.assertEqual(result.tax_due, 0.0)


class TestEntertainmentAddback(unittest.TestCase):
    def test_fifty_percent_disallowed(self):
        ledger = [LedgerLine("Client Dinners", 40_000, LineCategory.ENTERTAINMENT)]
        adjustments = ct_engine.entertainment_addback(ledger, POLICY)
        self.assertEqual(len(adjustments), 1)
        self.assertEqual(adjustments[0].amount, 20_000.0)
        self.assertEqual(adjustments[0].citation.article, "32(2)")


class TestInterestCap(unittest.TestCase):
    def test_no_addback_below_de_minimis(self):
        ledger = [LedgerLine("Interest", 5_000_000, LineCategory.INTEREST_EXPENSE)]
        self.assertEqual(ct_engine.interest_cap_addback(ledger, ebitda=1_000_000, policy=POLICY), [])

    def test_addback_above_cap_and_de_minimis(self):
        # net interest 20,000,000 > 12,000,000 de minimis; 30% of EBITDA(10,000,000) = 3,000,000
        ledger = [LedgerLine("Interest", 20_000_000, LineCategory.INTEREST_EXPENSE)]
        adjustments = ct_engine.interest_cap_addback(ledger, ebitda=10_000_000, policy=POLICY)
        self.assertEqual(len(adjustments), 1)
        self.assertEqual(adjustments[0].amount, 17_000_000.0)

    def test_no_addback_when_under_ebitda_cap_but_over_de_minimis(self):
        # net interest 13,000,000 > de minimis, but 30% of EBITDA(100,000,000) = 30,000,000 > interest
        ledger = [LedgerLine("Interest", 13_000_000, LineCategory.INTEREST_EXPENSE)]
        self.assertEqual(ct_engine.interest_cap_addback(ledger, ebitda=100_000_000, policy=POLICY), [])


class TestLossRelief(unittest.TestCase):
    def test_capped_at_75_percent(self):
        elections = ct_engine.Elections(prior_year_tax_losses=1_000_000)
        new_income, adj = ct_engine.apply_loss_relief(400_000, elections, POLICY)
        # cap = 400,000 * 0.75 = 300,000; only 300,000 of the 1,000,000 loss is used
        self.assertEqual(new_income, 100_000.0)
        self.assertEqual(adj.amount, -300_000.0)

    def test_no_relief_without_losses(self):
        elections = ct_engine.Elections(prior_year_tax_losses=0)
        new_income, adj = ct_engine.apply_loss_relief(400_000, elections, POLICY)
        self.assertEqual(new_income, 400_000.0)
        self.assertIsNone(adj)


class TestFreeZoneSplit(unittest.TestCase):
    def test_non_qfzp_unaffected(self):
        income, info = ct_engine.free_zone_split(1_000_000, ct_engine.Elections(), POLICY)
        self.assertEqual(income, 1_000_000.0)
        self.assertIsNone(info)

    def test_qfzp_splits_by_ratio(self):
        elections = ct_engine.Elections(is_qualifying_free_zone_person=True, qualifying_free_zone_income_ratio=0.7)
        non_qualifying, info = ct_engine.free_zone_split(1_000_000, elections, POLICY)
        self.assertEqual(non_qualifying, 300_000.0)
        self.assertEqual(info["qualifying_income"], 700_000.0)


class TestFullRun(unittest.TestCase):
    def test_end_to_end_common_case(self):
        ledger = [
            LedgerLine("Entertainment", 40_000, LineCategory.ENTERTAINMENT),
            LedgerLine("Traffic Fine", 5_000, LineCategory.FINE_PENALTY),
            LedgerLine("Dividend received", 50_000, LineCategory.DIVIDEND_INCOME),
        ]
        elections = ct_engine.Elections()
        result = ct_engine.run(ledger, accounting_profit=1_000_000, revenue=5_000_000, ebitda=1_100_000, elections=elections, policy=POLICY)
        # +20,000 (entertainment) +5,000 (fine) -50,000 (dividend exempt) = -25,000
        self.assertEqual(result.taxable_income, 975_000.0)
        self.assertEqual(result.tax_due, round((975_000 - 375_000) * 0.09, 2))
        self.assertFalse(result.small_business_relief_applied)


if __name__ == "__main__":
    unittest.main()
