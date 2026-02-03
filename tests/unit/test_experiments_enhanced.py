"""Tests for experiments module — A/B testing, guardrails, sequential testing."""

from __future__ import annotations

import numpy as np
import pytest

from forge.experiments import (
    EvaluationResult,
    ExperimentManager,
    FeatureVariant,
)
from forge.experiments.guardrails import (
    GuardrailExperimentManager,
    GuardrailMetric,
    SequentialTestResult,
)


@pytest.fixture
def variants() -> list[FeatureVariant]:
    return [
        FeatureVariant("control", columns=["price"]),
        FeatureVariant("treatment", columns=["price", "price_log"]),
    ]


@pytest.fixture
def mgr() -> ExperimentManager:
    return ExperimentManager(significance_level=0.05, min_observations=5)


class TestExperimentManager:
    def test_create_experiment(self, mgr: ExperimentManager, variants: list[FeatureVariant]) -> None:
        exp = mgr.create_experiment("test", variants)
        assert exp.name == "test"
        assert len(exp.variants) == 2
        assert exp.status == "active"

    def test_create_duplicate_raises(self, mgr: ExperimentManager, variants: list[FeatureVariant]) -> None:
        mgr.create_experiment("test", variants)
        with pytest.raises(ValueError, match="already exists"):
            mgr.create_experiment("test", variants)

    def test_single_variant_raises(self, mgr: ExperimentManager) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            mgr.create_experiment("test", [FeatureVariant("solo")])

    def test_record_metric(self, mgr: ExperimentManager, variants: list[FeatureVariant]) -> None:
        mgr.create_experiment("test", variants)
        mgr.record_metric("test", "control", "accuracy", 0.85)
        exp = mgr.get_experiment("test")
        assert exp is not None
        assert len(exp.metrics) == 1

    def test_record_wrong_variant_raises(self, mgr: ExperimentManager, variants: list[FeatureVariant]) -> None:
        mgr.create_experiment("test", variants)
        with pytest.raises(ValueError, match="not in experiment"):
            mgr.record_metric("test", "nonexistent", "acc", 0.5)

    def test_evaluate_no_data(self, mgr: ExperimentManager, variants: list[FeatureVariant]) -> None:
        mgr.create_experiment("test", variants)
        result = mgr.evaluate("test")
        assert result.winner is None

    def test_evaluate_with_data(self, mgr: ExperimentManager, variants: list[FeatureVariant]) -> None:
        mgr.create_experiment("test", variants)
        rng = np.random.RandomState(42)
        for _ in range(20):
            mgr.record_metric("test", "control", "accuracy", 0.80 + rng.normal(0, 0.02))
            mgr.record_metric("test", "treatment", "accuracy", 0.85 + rng.normal(0, 0.02))

        result = mgr.evaluate("test")
        assert isinstance(result, EvaluationResult)
        assert len(result.variant_scores) == 2

    def test_promote_winner(self, mgr: ExperimentManager, variants: list[FeatureVariant]) -> None:
        mgr.create_experiment("test", variants)
        rng = np.random.RandomState(42)
        for _ in range(30):
            mgr.record_metric("test", "control", "accuracy", 0.70 + rng.normal(0, 0.01))
            mgr.record_metric("test", "treatment", "accuracy", 0.90 + rng.normal(0, 0.01))

        winner = mgr.promote_winner("test")
        assert winner == "treatment"
        exp = mgr.get_experiment("test")
        assert exp is not None
        assert exp.status == "completed"

    def test_cancel_experiment(self, mgr: ExperimentManager, variants: list[FeatureVariant]) -> None:
        mgr.create_experiment("test", variants)
        assert mgr.cancel_experiment("test")
        exp = mgr.get_experiment("test")
        assert exp is not None
        assert exp.status == "cancelled"

    def test_list_experiments(self, mgr: ExperimentManager, variants: list[FeatureVariant]) -> None:
        mgr.create_experiment("e1", variants)
        mgr.create_experiment("e2", list(variants))
        assert len(mgr.list_experiments()) == 2
        assert len(mgr.list_experiments(status="active")) == 2

    def test_assign_variant_deterministic(self, mgr: ExperimentManager, variants: list[FeatureVariant]) -> None:
        mgr.create_experiment("test", variants)
        v1 = mgr.assign_variant("test", "user123")
        v2 = mgr.assign_variant("test", "user123")
        assert v1 == v2

    def test_traffic_split(self, mgr: ExperimentManager, variants: list[FeatureVariant]) -> None:
        mgr.create_experiment("test", variants)
        exp = mgr.get_experiment("test")
        assert exp is not None
        assert abs(sum(exp.traffic_split.values()) - 1.0) < 0.01


class TestGuardrailExperimentManager:
    def test_guardrails_pass(self, variants: list[FeatureVariant]) -> None:
        mgr = GuardrailExperimentManager(
            guardrails=[GuardrailMetric("latency", max_value=100)],
        )
        mgr.create_experiment("test", variants)
        mgr.record_metric("test", "control", "latency", 50)
        mgr.record_metric("test", "treatment", "latency", 60)

        result = mgr.check_guardrails("test")
        assert result.passed

    def test_guardrails_fail(self, variants: list[FeatureVariant]) -> None:
        mgr = GuardrailExperimentManager(
            guardrails=[GuardrailMetric("latency", max_value=100)],
        )
        mgr.create_experiment("test", variants)
        mgr.record_metric("test", "treatment", "latency", 150)

        result = mgr.check_guardrails("test")
        assert not result.passed
        assert len(result.violated_guardrails) > 0

    def test_relative_guardrail(self, variants: list[FeatureVariant]) -> None:
        mgr = GuardrailExperimentManager(
            guardrails=[GuardrailMetric("error_rate", relative_threshold=0.1)],
            control_variant="control",
        )
        mgr.create_experiment("test", variants)
        for _ in range(10):
            mgr.record_metric("test", "control", "error_rate", 0.05)
            mgr.record_metric("test", "treatment", "error_rate", 0.10)

        result = mgr.check_guardrails("test")
        assert not result.passed  # 100% increase > 10% threshold

    def test_sequential_test(self, variants: list[FeatureVariant]) -> None:
        mgr = GuardrailExperimentManager(max_looks=5)
        mgr.create_experiment("test", variants)

        rng = np.random.RandomState(42)
        for _ in range(30):
            mgr.record_metric("test", "control", "accuracy", 0.70 + rng.normal(0, 0.01))
            mgr.record_metric("test", "treatment", "accuracy", 0.90 + rng.normal(0, 0.01))

        result = mgr.sequential_test("test", "accuracy")
        assert isinstance(result, SequentialTestResult)
        assert result.current_n > 0

    def test_multi_metric_evaluation(self, variants: list[FeatureVariant]) -> None:
        mgr = GuardrailExperimentManager()
        mgr.create_experiment("test", variants)

        rng = np.random.RandomState(42)
        for _ in range(20):
            mgr.record_metric("test", "control", "accuracy", 0.80 + rng.normal(0, 0.01))
            mgr.record_metric("test", "treatment", "accuracy", 0.85 + rng.normal(0, 0.01))
            mgr.record_metric("test", "control", "auc", 0.82 + rng.normal(0, 0.01))
            mgr.record_metric("test", "treatment", "auc", 0.87 + rng.normal(0, 0.01))

        result = mgr.evaluate_multi_metric(
            "test", ["accuracy", "auc"], weights={"accuracy": 0.5, "auc": 0.5}
        )
        assert result.winner == "treatment"

    def test_no_guardrail_experiment(self, variants: list[FeatureVariant]) -> None:
        mgr = GuardrailExperimentManager(guardrails=[])
        mgr.create_experiment("test", variants)
        result = mgr.check_guardrails("test")
        assert result.passed
