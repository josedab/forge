"""Tests for privacy-preserving feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.privacy import (
    DPHistogram,
    DPMean,
    DPSum,
    KAnonymizer,
    PrivacyBudget,
    PrivacyBudgetExhausted,
)


class TestPrivacyBudget:
    def test_initial_state(self):
        b = PrivacyBudget(total_epsilon=1.0)
        assert b.remaining_epsilon == 1.0
        assert not b.is_exhausted

    def test_spend(self):
        b = PrivacyBudget(total_epsilon=1.0)
        b.spend(0.3, operation="op1")
        assert b.remaining_epsilon == pytest.approx(0.7)

    def test_exhaust(self):
        b = PrivacyBudget(total_epsilon=0.5)
        b.spend(0.5, operation="op1")
        assert b.is_exhausted

    def test_overspend(self):
        b = PrivacyBudget(total_epsilon=0.5)
        with pytest.raises(PrivacyBudgetExhausted, match="remaining"):
            b.spend(0.6, operation="op1")

    def test_delta_overspend(self):
        b = PrivacyBudget(total_epsilon=10.0, total_delta=1e-5)
        with pytest.raises(PrivacyBudgetExhausted, match="δ"):
            b.spend(0.1, delta=1e-4, operation="op1")

    def test_summary(self):
        b = PrivacyBudget(total_epsilon=1.0)
        b.spend(0.3, operation="op1")
        s = b.summary()
        assert s["spent_epsilon"] == pytest.approx(0.3)
        assert s["num_operations"] == 1


class TestDPMean:
    def test_basic(self):
        rng = np.random.RandomState(42)
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dp = DPMean(epsilon=10.0, clip_low=0.0, clip_high=10.0)
        result = dp.compute(data, rng=rng)
        assert abs(result - 3.0) < 1.0  # Should be close with high epsilon

    def test_with_budget(self):
        budget = PrivacyBudget(total_epsilon=1.0)
        dp = DPMean(epsilon=0.5, budget=budget)
        dp.compute(np.array([1.0, 2.0, 3.0]))
        assert budget.remaining_epsilon == pytest.approx(0.5)

    def test_budget_exhaustion(self):
        budget = PrivacyBudget(total_epsilon=0.3)
        dp = DPMean(epsilon=0.5, budget=budget)
        with pytest.raises(PrivacyBudgetExhausted):
            dp.compute(np.array([1.0, 2.0]))

    def test_clipping(self):
        rng = np.random.RandomState(42)
        data = np.array([-100.0, 500.0])
        dp = DPMean(epsilon=100.0, clip_low=0.0, clip_high=1.0)
        result = dp.compute(data, rng=rng)
        assert 0.0 <= result <= 1.5  # Should be near 0.5 after clipping

    def test_empty_data(self):
        dp = DPMean(epsilon=0.1)
        assert dp.compute(np.array([])) == 0.0

    def test_invalid_epsilon(self):
        with pytest.raises(ValueError, match="positive"):
            DPMean(epsilon=0)


class TestDPHistogram:
    def test_basic(self):
        rng = np.random.RandomState(42)
        data = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        dp = DPHistogram(epsilon=10.0, bins=5, clip_low=0.0, clip_high=1.0)
        counts, edges = dp.compute(data, rng=rng)
        assert len(counts) == 5
        assert len(edges) == 6
        assert all(c >= 0 for c in counts)

    def test_with_budget(self):
        budget = PrivacyBudget(total_epsilon=1.0)
        dp = DPHistogram(epsilon=0.3, budget=budget)
        dp.compute(np.array([0.1, 0.5, 0.9]))
        assert budget.remaining_epsilon == pytest.approx(0.7)

    def test_noisy(self):
        rng = np.random.RandomState(42)
        data = np.zeros(100)
        dp = DPHistogram(epsilon=0.01, bins=5, clip_low=0.0, clip_high=1.0)
        counts, _ = dp.compute(data, rng=rng)
        # All data in first bin, but noise should spread
        assert counts[0] > 0

    def test_invalid_epsilon(self):
        with pytest.raises(ValueError):
            DPHistogram(epsilon=-1)


class TestDPSum:
    def test_basic(self):
        rng = np.random.RandomState(42)
        data = np.array([1.0, 2.0, 3.0])
        dp = DPSum(epsilon=10.0, clip_low=0.0, clip_high=5.0)
        result = dp.compute(data, rng=rng)
        assert abs(result - 6.0) < 2.0

    def test_with_budget(self):
        budget = PrivacyBudget(total_epsilon=1.0)
        dp = DPSum(epsilon=0.2, budget=budget)
        dp.compute(np.array([1.0, 2.0]))
        assert budget.remaining_epsilon == pytest.approx(0.8)

    def test_invalid_epsilon(self):
        with pytest.raises(ValueError):
            DPSum(epsilon=0)


class TestKAnonymizer:
    def test_check_satisfied(self):
        df = pd.DataFrame({"age": [25, 25, 25, 30, 30, 30], "zip": [1, 1, 1, 2, 2, 2]})
        anon = KAnonymizer(k=3, quasi_identifiers=["age", "zip"])
        result = anon.check(df)
        assert result["is_satisfied"]
        assert result["min_group_size"] == 3

    def test_check_violated(self):
        df = pd.DataFrame({"age": [25, 25, 30], "zip": [1, 1, 2]})
        anon = KAnonymizer(k=3, quasi_identifiers=["age", "zip"])
        result = anon.check(df)
        assert not result["is_satisfied"]
        assert result["violating_groups"] >= 1

    def test_suppress(self):
        df = pd.DataFrame({"age": [25, 25, 25, 30], "zip": [1, 1, 1, 2]})
        anon = KAnonymizer(k=3, quasi_identifiers=["age", "zip"])
        result = anon.suppress(df)
        # Last row (group size 1) should be suppressed
        assert pd.isna(result.loc[3, "age"])

    def test_no_quasi_identifiers(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        anon = KAnonymizer(k=2)
        assert anon.check(df)["is_satisfied"]
        assert len(anon.suppress(df)) == 3

    def test_generalize_numeric(self):
        df = pd.DataFrame({"age": [22, 25, 33, 45, 55, 60]})
        anon = KAnonymizer(k=2)
        result = anon.generalize_numeric(df, "age", bins=3)
        assert result["age"].nunique() <= 3

    def test_generalize_missing_column(self):
        df = pd.DataFrame({"a": [1, 2]})
        anon = KAnonymizer(k=2)
        with pytest.raises(ValueError, match="not found"):
            anon.generalize_numeric(df, "missing")

    def test_invalid_k(self):
        with pytest.raises(ValueError, match="k must"):
            KAnonymizer(k=1)
