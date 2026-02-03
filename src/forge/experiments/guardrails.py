"""Advanced experiment features: guardrail metrics and sequential testing.

Extends the base ExperimentManager with:
- Guardrail metrics that can automatically stop experiments
- Sequential testing with early stopping
- Multi-metric evaluation
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from forge.experiments import (
    EvaluationResult,
    ExperimentManager,
)

logger = logging.getLogger(__name__)


@dataclass
class GuardrailMetric:
    """A metric that must not degrade beyond a threshold.

    If a guardrail is violated, the experiment should be stopped.

    Attributes:
        metric_name: Name of the metric to monitor.
        min_value: Minimum acceptable value (None = no lower bound).
        max_value: Maximum acceptable value (None = no upper bound).
        relative_threshold: Max allowed relative degradation vs control (0.05 = 5%).
    """

    metric_name: str
    min_value: float | None = None
    max_value: float | None = None
    relative_threshold: float | None = None


@dataclass
class GuardrailCheckResult:
    """Result of a guardrail check."""

    passed: bool
    violated_guardrails: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SequentialTestResult:
    """Result of a sequential test at a checkpoint."""

    can_stop: bool
    winner: str | None = None
    confidence: float = 0.0
    current_n: int = 0
    boundary_crossed: bool = False
    spending_used: float = 0.0


class GuardrailExperimentManager(ExperimentManager):
    """Extended experiment manager with guardrail and sequential testing support.

    Args:
        significance_level: P-value threshold for significance.
        min_observations: Minimum observations per variant.
        guardrails: List of guardrail metrics to enforce.
        control_variant: Name of the control variant for relative comparisons.
        max_looks: Maximum number of sequential test checkpoints (for alpha spending).

    Example:
        >>> mgr = GuardrailExperimentManager(
        ...     guardrails=[GuardrailMetric("latency_p99", max_value=100)],
        ...     control_variant="baseline",
        ... )
        >>> exp = mgr.create_experiment("test", variants=[...])
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        min_observations: int = 5,
        guardrails: list[GuardrailMetric] | None = None,
        control_variant: str | None = None,
        max_looks: int = 10,
    ) -> None:
        super().__init__(significance_level, min_observations)
        self.guardrails = guardrails or []
        self.control_variant = control_variant
        self.max_looks = max_looks
        self._look_count: dict[str, int] = {}

    def check_guardrails(
        self, experiment_name: str
    ) -> GuardrailCheckResult:
        """Check all guardrail metrics for an experiment.

        Args:
            experiment_name: Experiment to check.

        Returns:
            GuardrailCheckResult indicating pass/fail.
        """
        exp = self._experiments.get(experiment_name)
        if exp is None:
            raise KeyError(f"Experiment '{experiment_name}' not found")

        violations: list[str] = []
        details: dict[str, Any] = {}

        for guardrail in self.guardrails:
            # Collect values per variant for this metric
            variant_values: dict[str, list[float]] = {}
            for obs in exp.metrics:
                if obs.metric_name == guardrail.metric_name:
                    variant_values.setdefault(obs.variant_name, []).append(obs.value)

            for vname, values in variant_values.items():
                mean_val = float(np.mean(values))

                # Absolute bounds
                if guardrail.min_value is not None and mean_val < guardrail.min_value:
                    violations.append(
                        f"{vname}: {guardrail.metric_name} mean={mean_val:.4f} "
                        f"< min={guardrail.min_value}"
                    )

                if guardrail.max_value is not None and mean_val > guardrail.max_value:
                    violations.append(
                        f"{vname}: {guardrail.metric_name} mean={mean_val:.4f} "
                        f"> max={guardrail.max_value}"
                    )

                # Relative comparison to control
                if (
                    guardrail.relative_threshold is not None
                    and self.control_variant
                    and self.control_variant in variant_values
                    and vname != self.control_variant
                ):
                    control_mean = float(np.mean(variant_values[self.control_variant]))
                    if control_mean != 0:
                        relative_change = (mean_val - control_mean) / abs(control_mean)
                        if abs(relative_change) > guardrail.relative_threshold:
                            violations.append(
                                f"{vname}: {guardrail.metric_name} changed "
                                f"{relative_change:.1%} vs control "
                                f"(threshold: {guardrail.relative_threshold:.1%})"
                            )

                details[f"{vname}_{guardrail.metric_name}"] = mean_val

        return GuardrailCheckResult(
            passed=len(violations) == 0,
            violated_guardrails=violations,
            details=details,
        )

    def sequential_test(
        self,
        experiment_name: str,
        metric_name: str = "accuracy",
    ) -> SequentialTestResult:
        """Perform a sequential test with alpha spending (O'Brien-Fleming-like).

        Allows early stopping when there's strong evidence, while controlling
        the overall false positive rate.

        Args:
            experiment_name: Experiment name.
            metric_name: Metric to test.

        Returns:
            SequentialTestResult indicating whether to stop.
        """
        self._look_count.setdefault(experiment_name, 0)
        self._look_count[experiment_name] += 1
        current_look = self._look_count[experiment_name]

        # O'Brien-Fleming-like alpha spending: more stringent early
        alpha_spent = self._alpha_spending(
            current_look, self.max_looks, self.significance_level
        )

        # Get standard evaluation
        eval_result = self.evaluate(experiment_name, metric_name)

        # Compute p-value using Welch's t-test on top two
        exp = self._experiments[experiment_name]
        variant_values: dict[str, list[float]] = {}
        for obs in exp.metrics:
            if obs.metric_name == metric_name:
                variant_values.setdefault(obs.variant_name, []).append(obs.value)

        p_value = 1.0
        if len(variant_values) >= 2:
            sorted_vars = sorted(
                variant_values.items(),
                key=lambda x: np.mean(x[1]),
                reverse=True,
            )
            a_vals = sorted_vars[0][1]
            b_vals = sorted_vars[1][1]
            if len(a_vals) >= 2 and len(b_vals) >= 2:
                _, p_value = self._welch_ttest(a_vals, b_vals)

        boundary_crossed = p_value < alpha_spent
        total_n = sum(len(v) for v in variant_values.values())

        return SequentialTestResult(
            can_stop=boundary_crossed,
            winner=eval_result.winner if boundary_crossed else None,
            confidence=1.0 - p_value,
            current_n=total_n,
            boundary_crossed=boundary_crossed,
            spending_used=alpha_spent,
        )

    def evaluate_multi_metric(
        self,
        experiment_name: str,
        metrics: list[str],
        weights: dict[str, float] | None = None,
    ) -> EvaluationResult:
        """Evaluate experiment across multiple metrics with optional weights.

        Args:
            experiment_name: Experiment name.
            metrics: List of metric names to evaluate.
            weights: Optional weights per metric (default: equal).

        Returns:
            EvaluationResult with composite scoring.
        """
        if weights is None:
            weights = {m: 1.0 / len(metrics) for m in metrics}

        composite_scores: dict[str, float] = {}

        for metric in metrics:
            result = self.evaluate(experiment_name, metric)
            for vname, scores in result.variant_scores.items():
                weighted_mean = scores.get("mean", 0.0) * weights.get(metric, 0.0)
                composite_scores[vname] = composite_scores.get(vname, 0.0) + weighted_mean

        if not composite_scores:
            return EvaluationResult(
                experiment_name=experiment_name,
                recommendation="No data for multi-metric evaluation",
            )

        best_name = max(composite_scores, key=lambda k: composite_scores[k])

        return EvaluationResult(
            experiment_name=experiment_name,
            winner=best_name,
            variant_scores={
                v: {"composite_score": s} for v, s in composite_scores.items()
            },
            recommendation=f"Best variant: '{best_name}' (composite score: {composite_scores[best_name]:.4f})",
        )

    @staticmethod
    def _alpha_spending(
        current_look: int, max_looks: int, alpha: float
    ) -> float:
        """O'Brien-Fleming-like alpha spending function.

        More conservative early, more liberal later.
        """
        t = current_look / max(max_looks, 1)
        # Approximate O'Brien-Fleming boundary
        from scipy import stats
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        boundary_z = z_alpha / math.sqrt(t) if t > 0 else float("inf")
        return float(2 * (1 - stats.norm.cdf(boundary_z)))
