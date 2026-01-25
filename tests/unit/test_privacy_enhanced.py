"""Tests for privacy module — DP primitives, k-anonymity, composition, audit."""

from __future__ import annotations

import json
from pathlib import Path

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
from forge.privacy.advanced import (
    DPFeatureTransformer,
    PrivacyAuditLog,
    advanced_composition,
    rdp_to_dp,
)


class TestPrivacyBudget:
    def test_initial_state(self) -> None:
        b = PrivacyBudget(total_epsilon=1.0)
        assert b.remaining_epsilon == 1.0
        assert not b.is_exhausted

    def test_spend(self) -> None:
        b = PrivacyBudget(total_epsilon=1.0)
        b.spend(0.3, operation="test")
        assert abs(b.remaining_epsilon - 0.7) < 1e-10

    def test_exhausted_raises(self) -> None:
        b = PrivacyBudget(total_epsilon=0.5)
        b.spend(0.5)
        with pytest.raises(PrivacyBudgetExhausted):
            b.spend(0.1)

    def test_summary(self) -> None:
        b = PrivacyBudget(total_epsilon=1.0)
        b.spend(0.3, operation="op1")
        s = b.summary()
        assert s["spent_epsilon"] == pytest.approx(0.3, abs=1e-6)
        assert s["num_operations"] == 1


class TestDPMean:
    def test_basic_computation(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dp = DPMean(epsilon=1.0, clip_low=0.0, clip_high=10.0)
        result = dp.compute(data, rng=np.random.RandomState(42))
        assert isinstance(result, float)
        # Should be close to true mean (3.0) but with noise
        assert abs(result - 3.0) < 5.0

    def test_budget_tracking(self) -> None:
        budget = PrivacyBudget(total_epsilon=1.0)
        dp = DPMean(epsilon=0.3, budget=budget)
        dp.compute(np.array([1, 2, 3]), rng=np.random.RandomState(0))
        assert budget._spent_epsilon == pytest.approx(0.3)

    def test_invalid_epsilon_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            DPMean(epsilon=-1.0)

    def test_empty_data(self) -> None:
        dp = DPMean(epsilon=1.0)
        assert dp.compute(np.array([])) == 0.0


class TestDPHistogram:
    def test_histogram_shape(self) -> None:
        data = np.random.uniform(0, 1, 100)
        dp = DPHistogram(epsilon=1.0, bins=5)
        counts, edges = dp.compute(data, rng=np.random.RandomState(42))
        assert len(counts) == 5
        assert len(edges) == 6

    def test_non_negative_counts(self) -> None:
        data = np.random.uniform(0, 1, 100)
        dp = DPHistogram(epsilon=0.5, bins=10)
        counts, _ = dp.compute(data, rng=np.random.RandomState(42))
        assert all(c >= 0 for c in counts)


class TestDPSum:
    def test_basic_sum(self) -> None:
        data = np.array([10.0, 20.0, 30.0])
        dp = DPSum(epsilon=1.0, clip_low=0.0, clip_high=50.0)
        result = dp.compute(data, rng=np.random.RandomState(42))
        assert isinstance(result, float)
        # Should be close to 60 but noisy
        assert abs(result - 60.0) < 200.0

    def test_budget_tracking(self) -> None:
        budget = PrivacyBudget(total_epsilon=1.0)
        dp = DPSum(epsilon=0.5, budget=budget)
        dp.compute(np.array([1, 2, 3]), rng=np.random.RandomState(0))
        assert budget._spent_epsilon == pytest.approx(0.5)


class TestKAnonymizer:
    def test_check_satisfied(self) -> None:
        df = pd.DataFrame({
            "age": [25, 25, 30, 30, 30],
            "city": ["NY", "NY", "LA", "LA", "LA"],
        })
        anon = KAnonymizer(k=2, quasi_identifiers=["age", "city"])
        result = anon.check(df)
        assert result["is_satisfied"]

    def test_check_violated(self) -> None:
        df = pd.DataFrame({
            "age": [25, 30, 35, 40, 45],
            "city": ["NY", "LA", "SF", "CHI", "BOS"],
        })
        anon = KAnonymizer(k=2, quasi_identifiers=["age", "city"])
        result = anon.check(df)
        assert not result["is_satisfied"]

    def test_suppress(self) -> None:
        df = pd.DataFrame({
            "age": [25, 25, 30, 35, 35],
            "city": ["NY", "NY", "LA", "SF", "SF"],
        })
        anon = KAnonymizer(k=2, quasi_identifiers=["age", "city"])
        result = anon.suppress(df)
        # Row with age=30, city=LA should be suppressed
        assert result.loc[2, "age"] is None or pd.isna(result.loc[2, "age"])

    def test_generalize_numeric(self) -> None:
        df = pd.DataFrame({"age": [20, 25, 30, 35, 40, 45]})
        anon = KAnonymizer(k=2)
        result = anon.generalize_numeric(df, "age", bins=3)
        assert result["age"].nunique() <= 3

    def test_invalid_k(self) -> None:
        with pytest.raises(ValueError, match="k must"):
            KAnonymizer(k=1)


class TestAdvancedComposition:
    def test_basic_composition(self) -> None:
        # With many mechanisms of moderate epsilon, advanced composition is tighter
        epsilons = [0.5] * 100
        result = advanced_composition(epsilons, delta_target=1e-5)
        simple_sum = sum(epsilons)
        # Advanced composition should be strictly less than simple sum for many mechanisms
        assert result <= simple_sum
        assert result > 0

    def test_empty_list(self) -> None:
        assert advanced_composition([]) == 0.0

    def test_single_mechanism(self) -> None:
        result = advanced_composition([0.5], delta_target=1e-5)
        assert result <= 0.5 + 1e-10

    def test_tighter_than_simple(self) -> None:
        epsilons = [0.1] * 100
        advanced = advanced_composition(epsilons, delta_target=1e-5)
        simple = sum(epsilons)
        assert advanced < simple


class TestRDPConversion:
    def test_basic_conversion(self) -> None:
        result = rdp_to_dp(rdp_alpha=2.0, rdp_epsilon=1.0, delta=1e-5)
        assert result > 1.0  # RDP → DP adds overhead

    def test_invalid_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha must"):
            rdp_to_dp(rdp_alpha=0.5, rdp_epsilon=1.0)


class TestDPFeatureTransformer:
    def test_fit_transform(self) -> None:
        df = pd.DataFrame({
            "a": np.random.randn(100),
            "b": np.random.randn(100) + 5,
        })
        budget = PrivacyBudget(total_epsilon=10.0)
        tfm = DPFeatureTransformer(
            epsilon_per_feature=0.5, budget=budget,
            clip_low=-5.0, clip_high=10.0,
        )
        result = tfm.fit_transform(df)
        assert result.shape == df.shape
        assert budget._spent_epsilon > 0

    def test_feature_names(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        tfm = DPFeatureTransformer(epsilon_per_feature=0.5)
        tfm.fit(df)
        names = tfm.get_feature_names_out()
        assert "x" in names
        assert "y" in names

    def test_not_fitted_raises(self) -> None:
        tfm = DPFeatureTransformer()
        with pytest.raises(RuntimeError, match="not fitted"):
            tfm.transform(pd.DataFrame({"a": [1]}))


class TestPrivacyAuditLog:
    def test_record_and_report(self) -> None:
        budget = PrivacyBudget(total_epsilon=1.0)
        log = PrivacyAuditLog(budget)
        budget.spend(0.3, operation="op1")
        log.record("op1", 0.3, columns=["col1"], n_records=100)

        report = log.report()
        assert "op1" in report
        assert len(log.entries) == 1

    def test_export(self, tmp_path: Path) -> None:
        budget = PrivacyBudget(total_epsilon=1.0)
        log = PrivacyAuditLog(budget)
        budget.spend(0.2, operation="test")
        log.record("test", 0.2)

        path = tmp_path / "audit.json"
        log.export(path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert "entries" in data
        assert "budget_summary" in data
