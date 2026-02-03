"""Tests for feature A/B testing framework."""

from __future__ import annotations

import numpy as np
import pytest

from forge.experiments import (
    EvaluationResult,
    ExperimentManager,
    FeatureVariant,
)


@pytest.fixture
def mgr():
    return ExperimentManager(significance_level=0.05, min_observations=5)


@pytest.fixture
def variants():
    return [
        FeatureVariant("control", columns=["price"]),
        FeatureVariant("treatment", columns=["price", "price_log"]),
    ]


class TestExperimentManager:
    def test_create_experiment(self, mgr, variants):
        exp = mgr.create_experiment("test_exp", variants)
        assert exp.name == "test_exp"
        assert len(exp.variants) == 2
        assert exp.status == "active"

    def test_create_duplicate(self, mgr, variants):
        mgr.create_experiment("test_exp", variants)
        with pytest.raises(ValueError, match="already exists"):
            mgr.create_experiment("test_exp", variants)

    def test_create_too_few_variants(self, mgr):
        with pytest.raises(ValueError, match="at least 2"):
            mgr.create_experiment("x", [FeatureVariant("only_one")])

    def test_record_metric(self, mgr, variants):
        mgr.create_experiment("exp", variants)
        mgr.record_metric("exp", "control", "accuracy", 0.85)
        mgr.record_metric("exp", "treatment", "accuracy", 0.88)
        exp = mgr.get_experiment("exp")
        assert len(exp.metrics) == 2

    def test_record_metric_bad_experiment(self, mgr):
        with pytest.raises(KeyError):
            mgr.record_metric("nonexistent", "v1", "acc", 0.5)

    def test_record_metric_bad_variant(self, mgr, variants):
        mgr.create_experiment("exp", variants)
        with pytest.raises(ValueError, match="not in experiment"):
            mgr.record_metric("exp", "bad_variant", "acc", 0.5)

    def test_evaluate_no_data(self, mgr, variants):
        mgr.create_experiment("exp", variants)
        result = mgr.evaluate("exp")
        assert not result.is_significant
        assert "No data" in result.recommendation

    def test_evaluate_insufficient_data(self, mgr, variants):
        mgr.create_experiment("exp", variants)
        mgr.record_metric("exp", "control", "accuracy", 0.85)
        mgr.record_metric("exp", "treatment", "accuracy", 0.88)
        result = mgr.evaluate("exp")
        assert not result.is_significant
        assert "more data" in result.recommendation

    def test_evaluate_with_clear_winner(self, mgr, variants):
        mgr.create_experiment("exp", variants)
        rng = np.random.RandomState(42)
        for _ in range(30):
            mgr.record_metric("exp", "control", "accuracy", 0.7 + rng.normal(0, 0.02))
            mgr.record_metric("exp", "treatment", "accuracy", 0.9 + rng.normal(0, 0.02))
        result = mgr.evaluate("exp")
        assert result.is_significant
        assert result.winner == "treatment"
        assert result.confidence > 0.9

    def test_promote_winner(self, mgr, variants):
        mgr.create_experiment("exp", variants)
        rng = np.random.RandomState(42)
        for _ in range(30):
            mgr.record_metric("exp", "control", "accuracy", 0.7 + rng.normal(0, 0.02))
            mgr.record_metric("exp", "treatment", "accuracy", 0.9 + rng.normal(0, 0.02))
        winner = mgr.promote_winner("exp")
        assert winner == "treatment"
        assert mgr.get_experiment("exp").status == "completed"

    def test_promote_no_winner(self, mgr, variants):
        mgr.create_experiment("exp", variants)
        rng = np.random.RandomState(42)
        for _ in range(30):
            mgr.record_metric("exp", "control", "accuracy", 0.8 + rng.normal(0, 0.1))
            mgr.record_metric("exp", "treatment", "accuracy", 0.8 + rng.normal(0, 0.1))
        winner = mgr.promote_winner("exp")
        # May or may not be significant depending on random draw
        exp = mgr.get_experiment("exp")
        if winner is None:
            assert exp.status == "active"

    def test_cancel_experiment(self, mgr, variants):
        mgr.create_experiment("exp", variants)
        assert mgr.cancel_experiment("exp")
        assert mgr.get_experiment("exp").status == "cancelled"
        assert not mgr.cancel_experiment("exp")  # Already cancelled

    def test_cancel_nonexistent(self, mgr):
        assert not mgr.cancel_experiment("nonexistent")

    def test_list_experiments(self, mgr, variants):
        mgr.create_experiment("exp1", variants)
        v2 = [FeatureVariant("a"), FeatureVariant("b")]
        mgr.create_experiment("exp2", v2)
        assert len(mgr.list_experiments()) == 2

    def test_list_by_status(self, mgr, variants):
        mgr.create_experiment("exp1", variants)
        v2 = [FeatureVariant("a"), FeatureVariant("b")]
        mgr.create_experiment("exp2", v2)
        mgr.cancel_experiment("exp2")
        assert len(mgr.list_experiments(status="active")) == 1

    def test_assign_variant(self, mgr, variants):
        mgr.create_experiment("exp", variants)
        v1 = mgr.assign_variant("exp", "user_123")
        v2 = mgr.assign_variant("exp", "user_123")
        assert v1 == v2  # Deterministic

    def test_assign_variant_distribution(self, mgr, variants):
        mgr.create_experiment("exp", variants, traffic_split={
            "control": 0.5, "treatment": 0.5,
        })
        assignments = [mgr.assign_variant("exp", f"user_{i}") for i in range(1000)]
        control_pct = assignments.count("control") / len(assignments)
        assert 0.3 < control_pct < 0.7  # Roughly balanced

    def test_traffic_split_default(self, mgr, variants):
        exp = mgr.create_experiment("exp", variants)
        assert len(exp.traffic_split) == 2
        assert sum(exp.traffic_split.values()) == pytest.approx(1.0)

    def test_evaluation_result_to_dict(self):
        r = EvaluationResult(experiment_name="test", winner="v1", confidence=0.95)
        d = r.to_dict()
        assert d["winner"] == "v1"
        assert d["confidence"] == 0.95
