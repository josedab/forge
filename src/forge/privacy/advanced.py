"""Advanced privacy features: composition theorems and audit logging.

Extends the privacy module with:
- Advanced composition theorem (tighter epsilon accounting)
- Privacy audit log for compliance reporting
- DP feature transformer (sklearn-compatible)
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from forge.privacy import DPMean, PrivacyBudget

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """A single entry in the privacy audit log."""

    timestamp: float
    operation: str
    epsilon_spent: float
    delta_spent: float
    columns_accessed: list[str]
    n_records: int
    cumulative_epsilon: float
    cumulative_delta: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "operation": self.operation,
            "epsilon_spent": self.epsilon_spent,
            "delta_spent": self.delta_spent,
            "columns_accessed": self.columns_accessed,
            "n_records": self.n_records,
            "cumulative_epsilon": self.cumulative_epsilon,
            "cumulative_delta": self.cumulative_delta,
        }


class PrivacyAuditLog:
    """Audit log for privacy operations.

    Records all privacy-consuming operations with timestamps,
    enabling compliance reporting and budget verification.

    Args:
        budget: Privacy budget to track.
    """

    def __init__(self, budget: PrivacyBudget) -> None:
        self.budget = budget
        self._entries: list[AuditEntry] = []

    def record(
        self,
        operation: str,
        epsilon: float,
        delta: float = 0.0,
        columns: list[str] | None = None,
        n_records: int = 0,
    ) -> None:
        """Record a privacy operation."""
        self._entries.append(AuditEntry(
            timestamp=time.time(),
            operation=operation,
            epsilon_spent=epsilon,
            delta_spent=delta,
            columns_accessed=columns or [],
            n_records=n_records,
            cumulative_epsilon=self.budget._spent_epsilon,
            cumulative_delta=self.budget._spent_delta,
        ))

    @property
    def entries(self) -> list[AuditEntry]:
        return list(self._entries)

    def export(self, path: str | Path) -> None:
        """Export audit log to JSON file."""
        data = {
            "budget_summary": self.budget.summary(),
            "entries": [e.to_dict() for e in self._entries],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def report(self) -> str:
        """Generate a human-readable audit report."""
        lines = [
            "Privacy Audit Report",
            "=" * 40,
            f"Total Budget: ε={self.budget.total_epsilon}, δ={self.budget.total_delta}",
            f"Spent: ε={self.budget._spent_epsilon:.6f}, δ={self.budget._spent_delta:.8f}",
            f"Remaining: ε={self.budget.remaining_epsilon:.6f}",
            f"Operations: {len(self._entries)}",
            "",
        ]
        for e in self._entries:
            lines.append(
                f"  [{e.operation}] ε={e.epsilon_spent:.4f}, "
                f"cols={e.columns_accessed}, n={e.n_records}"
            )
        return "\n".join(lines)


def advanced_composition(
    epsilons: list[float],
    delta_target: float = 1e-5,
) -> float:
    """Compute total epsilon under advanced composition theorem.

    Given k mechanisms each with epsilon_i, the advanced composition
    gives a tighter bound than simple summation:

    ε_total = sqrt(2 * k * ln(1/δ')) * max(ε_i) + k * max(ε_i) * (e^max(ε_i) - 1)

    For simplicity, this uses the heterogeneous version:
    ε_total = sqrt(2 * sum(ε_i²) * ln(1/δ))

    Args:
        epsilons: List of per-mechanism epsilon values.
        delta_target: Target delta for the composition.

    Returns:
        Total composed epsilon (tighter than sum).
    """
    if not epsilons:
        return 0.0

    k = len(epsilons)
    simple_sum = sum(epsilons)

    # Advanced composition (heterogeneous)
    sum_sq = sum(e ** 2 for e in epsilons)
    if delta_target > 0:
        advanced = math.sqrt(2 * sum_sq * math.log(1.0 / delta_target))
    else:
        advanced = simple_sum

    return min(simple_sum, advanced)


def rdp_to_dp(
    rdp_alpha: float,
    rdp_epsilon: float,
    delta: float = 1e-5,
) -> float:
    """Convert Rényi DP (RDP) to (ε, δ)-DP.

    Uses the standard conversion: ε = rdp_epsilon + log(1/δ) / (α - 1)

    Args:
        rdp_alpha: Rényi divergence order (α > 1).
        rdp_epsilon: RDP epsilon at order α.
        delta: Target delta.

    Returns:
        (ε, δ)-DP epsilon.
    """
    if rdp_alpha <= 1:
        raise ValueError("RDP alpha must be > 1")
    return rdp_epsilon + math.log(1.0 / delta) / (rdp_alpha - 1)


class DPFeatureTransformer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """sklearn-compatible transformer that applies DP aggregations.

    Computes differentially-private statistics (mean, std) from training
    data and uses them to normalize features with privacy guarantees.

    Args:
        epsilon_per_feature: Privacy budget per feature statistic.
        clip_low: Lower clipping bound.
        clip_high: Upper clipping bound.
        budget: Optional shared privacy budget.
        audit_log: Optional audit log.

    Example:
        >>> budget = PrivacyBudget(total_epsilon=1.0)
        >>> transformer = DPFeatureTransformer(epsilon_per_feature=0.1, budget=budget)
        >>> X_private = transformer.fit_transform(X)
    """

    def __init__(
        self,
        epsilon_per_feature: float = 0.1,
        clip_low: float = -10.0,
        clip_high: float = 10.0,
        budget: PrivacyBudget | None = None,
        audit_log: PrivacyAuditLog | None = None,
    ) -> None:
        self.epsilon_per_feature = epsilon_per_feature
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.budget = budget
        self.audit_log = audit_log
        self._means: dict[str, float] = {}
        self._stds: dict[str, float] = {}
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: Any = None) -> DPFeatureTransformer:
        """Compute DP statistics from training data.

        Args:
            X: Training data.
            y: Ignored.

        Returns:
            Self for chaining.
        """
        rng = np.random.RandomState(42)

        for col in X.select_dtypes(include=[np.number]).columns:
            dp_mean_calc = DPMean(
                epsilon=self.epsilon_per_feature,
                clip_low=self.clip_low,
                clip_high=self.clip_high,
                budget=self.budget,
            )
            self._means[col] = dp_mean_calc.compute(X[col], rng=rng)

            # DP std: compute mean of squared values, then derive std
            dp_sq_mean = DPMean(
                epsilon=self.epsilon_per_feature,
                clip_low=self.clip_low ** 2,
                clip_high=self.clip_high ** 2,
                budget=self.budget,
            )
            sq_mean = dp_sq_mean.compute(X[col] ** 2, rng=rng)
            variance = max(0, sq_mean - self._means[col] ** 2)
            self._stds[col] = math.sqrt(variance) if variance > 0 else 1.0

            if self.audit_log:
                self.audit_log.record(
                    operation="dp_normalize_fit",
                    epsilon=self.epsilon_per_feature * 2,
                    columns=[col],
                    n_records=len(X),
                )

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Normalize using DP statistics (no additional privacy cost).

        Args:
            X: Data to transform.

        Returns:
            Normalized DataFrame.
        """
        if not self._is_fitted:
            raise RuntimeError("DPFeatureTransformer not fitted")

        result = X.copy()
        for col in self._means:
            if col in result.columns:
                std = self._stds[col] if self._stds[col] > 0 else 1.0
                result[col] = (result[col] - self._means[col]) / std

        return result

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Return output feature names."""
        return list(self._means.keys())
