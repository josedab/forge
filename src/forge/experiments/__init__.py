"""Feature A/B testing framework.

Enables data-driven feature engineering decisions through experiments
with traffic splits, metric collection, statistical significance testing,
and automatic winner promotion.

Example:
    >>> from forge.experiments import ExperimentManager, FeatureVariant
    >>> mgr = ExperimentManager()
    >>> exp = mgr.create_experiment("price_features",
    ...     variants=[
    ...         FeatureVariant("v1", columns=["price_log"]),
    ...         FeatureVariant("v2", columns=["price_log", "price_squared"]),
    ...     ])
    >>> mgr.record_metric("price_features", "v1", "accuracy", 0.85)
    >>> mgr.record_metric("price_features", "v2", "accuracy", 0.88)
    >>> result = mgr.evaluate("price_features")
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureVariant:
    """A feature variant in an experiment."""

    name: str
    columns: list[str] = field(default_factory=list)
    description: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricObservation:
    """A single metric observation for a variant."""

    variant_name: str
    metric_name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Experiment:
    """A feature A/B experiment."""

    name: str
    variants: list[FeatureVariant]
    status: str = "active"  # active, completed, cancelled
    created_at: datetime = field(default_factory=datetime.now)
    traffic_split: dict[str, float] = field(default_factory=dict)
    metrics: list[MetricObservation] = field(default_factory=list)
    winner: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.traffic_split:
            n = len(self.variants)
            self.traffic_split = {v.name: 1.0 / n for v in self.variants}


@dataclass
class EvaluationResult:
    """Result of experiment evaluation."""

    experiment_name: str
    winner: str | None = None
    is_significant: bool = False
    confidence: float = 0.0
    variant_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "winner": self.winner,
            "is_significant": self.is_significant,
            "confidence": round(self.confidence, 4),
            "variant_scores": self.variant_scores,
            "recommendation": self.recommendation,
        }


class ExperimentManager:
    """Manages feature A/B testing experiments.

    Parameters:
        significance_level: P-value threshold for statistical significance.
        min_observations: Minimum observations per variant before evaluation.
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        min_observations: int = 5,
    ) -> None:
        self.significance_level = significance_level
        self.min_observations = min_observations
        self._experiments: dict[str, Experiment] = {}

    def create_experiment(
        self,
        name: str,
        variants: list[FeatureVariant],
        traffic_split: dict[str, float] | None = None,
        description: str = "",
    ) -> Experiment:
        """Create a new experiment.

        Args:
            name: Experiment name.
            variants: List of feature variants.
            traffic_split: Optional traffic allocation per variant.
            description: Experiment description.

        Returns:
            Created Experiment.

        Raises:
            ValueError: If <2 variants or experiment exists.
        """
        if len(variants) < 2:
            raise ValueError("Experiments need at least 2 variants")

        if name in self._experiments:
            raise ValueError(f"Experiment '{name}' already exists")

        exp = Experiment(
            name=name,
            variants=variants,
            traffic_split=traffic_split or {},
            description=description,
        )
        self._experiments[name] = exp
        return exp

    def record_metric(
        self,
        experiment_name: str,
        variant_name: str,
        metric_name: str,
        value: float,
    ) -> None:
        """Record a metric observation for a variant.

        Args:
            experiment_name: Experiment to record for.
            variant_name: Which variant.
            metric_name: Metric name (e.g., 'accuracy', 'auc').
            value: Metric value.

        Raises:
            KeyError: If experiment not found.
            ValueError: If variant not in experiment.
        """
        exp = self._experiments.get(experiment_name)
        if exp is None:
            raise KeyError(f"Experiment '{experiment_name}' not found")

        variant_names = {v.name for v in exp.variants}
        if variant_name not in variant_names:
            raise ValueError(f"Variant '{variant_name}' not in experiment")

        exp.metrics.append(MetricObservation(
            variant_name=variant_name,
            metric_name=metric_name,
            value=value,
        ))

    def evaluate(
        self,
        experiment_name: str,
        metric_name: str = "accuracy",
    ) -> EvaluationResult:
        """Evaluate experiment results with statistical testing.

        Uses Welch's t-test to compare variant means.

        Args:
            experiment_name: Experiment to evaluate.
            metric_name: Which metric to compare.

        Returns:
            EvaluationResult with winner and significance.
        """
        exp = self._experiments.get(experiment_name)
        if exp is None:
            raise KeyError(f"Experiment '{experiment_name}' not found")

        # Gather values per variant
        variant_values: dict[str, list[float]] = {}
        for obs in exp.metrics:
            if obs.metric_name == metric_name:
                variant_values.setdefault(obs.variant_name, []).append(obs.value)

        variant_scores: dict[str, dict[str, float]] = {}
        for vname, values in variant_values.items():
            variant_scores[vname] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "n": len(values),
            }

        # Find best variant by mean
        if not variant_scores:
            return EvaluationResult(
                experiment_name=experiment_name,
                recommendation="No data available for evaluation",
            )

        best = max(variant_scores.items(), key=lambda x: x[1]["mean"])
        best_name = best[0]

        # Check sufficient observations
        all_sufficient = all(
            s["n"] >= self.min_observations for s in variant_scores.values()
        )

        # Simple significance test (Welch's t-test between best and second-best)
        is_significant = False
        confidence = 0.0

        sorted_variants = sorted(
            variant_scores.items(), key=lambda x: -x[1]["mean"]
        )

        if len(sorted_variants) >= 2 and all_sufficient:
            a_vals = variant_values[sorted_variants[0][0]]
            b_vals = variant_values[sorted_variants[1][0]]
            t_stat, p_value = self._welch_ttest(a_vals, b_vals)
            is_significant = p_value < self.significance_level
            confidence = 1.0 - p_value

        # Build recommendation
        if is_significant:
            rec = f"Promote '{best_name}' — statistically significant (p<{self.significance_level})"
        elif not all_sufficient:
            rec = f"Need more data — min {self.min_observations} observations per variant"
        else:
            rec = f"'{best_name}' leads but not statistically significant yet"

        return EvaluationResult(
            experiment_name=experiment_name,
            winner=best_name if is_significant else None,
            is_significant=is_significant,
            confidence=confidence,
            variant_scores=variant_scores,
            recommendation=rec,
        )

    def promote_winner(self, experiment_name: str, metric_name: str = "accuracy") -> str | None:
        """Evaluate and promote the winner, closing the experiment.

        Returns:
            Winner variant name, or None if not significant.
        """
        result = self.evaluate(experiment_name, metric_name)
        exp = self._experiments[experiment_name]

        if result.winner:
            exp.winner = result.winner
            exp.status = "completed"
            return result.winner

        return None

    def cancel_experiment(self, name: str) -> bool:
        """Cancel an experiment."""
        exp = self._experiments.get(name)
        if exp and exp.status == "active":
            exp.status = "cancelled"
            return True
        return False

    def get_experiment(self, name: str) -> Experiment | None:
        """Get experiment by name."""
        return self._experiments.get(name)

    def list_experiments(self, status: str | None = None) -> list[Experiment]:
        """List experiments, optionally filtered by status."""
        if status:
            return [e for e in self._experiments.values() if e.status == status]
        return list(self._experiments.values())

    def assign_variant(self, experiment_name: str, entity_id: str) -> str:
        """Deterministically assign an entity to a variant.

        Uses hash-based assignment for consistent splits.

        Args:
            experiment_name: Experiment name.
            entity_id: Entity identifier (user ID, etc.).

        Returns:
            Assigned variant name.
        """
        exp = self._experiments.get(experiment_name)
        if exp is None:
            raise KeyError(f"Experiment '{experiment_name}' not found")

        hash_val = hash(f"{experiment_name}:{entity_id}") % 10000 / 10000

        cumulative = 0.0
        for variant in exp.variants:
            cumulative += exp.traffic_split.get(variant.name, 0)
            if hash_val < cumulative:
                return variant.name

        return exp.variants[-1].name

    @staticmethod
    def _welch_ttest(a: list[float], b: list[float]) -> tuple[float, float]:
        """Welch's t-test for unequal variances."""
        n1, n2 = len(a), len(b)
        if n1 < 2 or n2 < 2:
            return 0.0, 1.0

        m1, m2 = np.mean(a), np.mean(b)
        v1 = np.var(a, ddof=1)
        v2 = np.var(b, ddof=1)

        se = math.sqrt(v1 / n1 + v2 / n2) if (v1 / n1 + v2 / n2) > 0 else 1e-10
        t_stat = (m1 - m2) / se

        # Degrees of freedom (Welch-Satterthwaite)
        num = (v1 / n1 + v2 / n2) ** 2
        den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        df = num / den if den > 0 else 1

        # Approximate p-value using normal distribution for simplicity
        from scipy import stats
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        return float(t_stat), float(p_value)
