"""Privacy-preserving feature engineering primitives.

Implements differential privacy mechanisms, k-anonymity, and privacy
budget tracking for GDPR/CCPA-compliant feature pipelines.

Example:
    >>> from forge.privacy import PrivacyBudget, DPMean, DPHistogram, KAnonymizer
    >>> budget = PrivacyBudget(total_epsilon=1.0)
    >>> dp_mean = DPMean(epsilon=0.5, budget=budget)
    >>> private_mean = dp_mean.compute(data["salary"])
    >>> print(f"Remaining budget: {budget.remaining}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PrivacyBudgetExhausted(Exception):
    """Raised when the privacy budget is fully consumed."""


@dataclass
class PrivacyBudget:
    """Tracks cumulative privacy spend (ε, δ) across operations.

    Parameters:
        total_epsilon: Total allowed epsilon budget.
        total_delta: Total allowed delta budget.
    """

    total_epsilon: float = 1.0
    total_delta: float = 1e-5
    _spent_epsilon: float = 0.0
    _spent_delta: float = 0.0
    _history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def remaining_epsilon(self) -> float:
        return max(0.0, self.total_epsilon - self._spent_epsilon)

    @property
    def remaining_delta(self) -> float:
        return max(0.0, self.total_delta - self._spent_delta)

    @property
    def is_exhausted(self) -> bool:
        return self._spent_epsilon >= self.total_epsilon

    def spend(self, epsilon: float, delta: float = 0.0, operation: str = "") -> None:
        """Consume budget for an operation.

        Raises:
            PrivacyBudgetExhausted: If insufficient budget.
        """
        if epsilon > self.remaining_epsilon + 1e-12:
            raise PrivacyBudgetExhausted(
                f"Need ε={epsilon}, only {self.remaining_epsilon:.4f} remaining"
            )
        if delta > self.remaining_delta + 1e-12:
            raise PrivacyBudgetExhausted(
                f"Need δ={delta}, only {self.remaining_delta:.6f} remaining"
            )
        self._spent_epsilon += epsilon
        self._spent_delta += delta
        self._history.append({
            "operation": operation,
            "epsilon": epsilon,
            "delta": delta,
        })

    def summary(self) -> dict[str, Any]:
        return {
            "total_epsilon": self.total_epsilon,
            "spent_epsilon": round(self._spent_epsilon, 6),
            "remaining_epsilon": round(self.remaining_epsilon, 6),
            "total_delta": self.total_delta,
            "spent_delta": round(self._spent_delta, 10),
            "remaining_delta": round(self.remaining_delta, 10),
            "num_operations": len(self._history),
        }


class DPMean:
    """Differentially-private mean computation using Laplace mechanism.

    Parameters:
        epsilon: Privacy parameter for this operation.
        sensitivity: Data sensitivity bound (max - min of data range).
            If None, estimated from data clipping bounds.
        clip_low: Lower clipping bound for input values.
        clip_high: Upper clipping bound for input values.
        budget: Optional shared budget tracker.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        sensitivity: float | None = None,
        clip_low: float = 0.0,
        clip_high: float = 1.0,
        budget: PrivacyBudget | None = None,
    ) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.budget = budget

    def compute(
        self,
        data: np.ndarray | pd.Series,
        rng: np.random.RandomState | None = None,
    ) -> float:
        """Compute DP mean of the data.

        Args:
            data: Input data array.
            rng: Random state for reproducibility.

        Returns:
            Differentially private mean.
        """
        if self.budget is not None:
            self.budget.spend(self.epsilon, operation="dp_mean")

        rng = rng or np.random.RandomState()
        arr = np.asarray(data, dtype=float)
        arr = np.clip(arr, self.clip_low, self.clip_high)
        n = len(arr)
        if n == 0:
            return 0.0

        true_mean = float(np.mean(arr))
        sens = self.sensitivity or (self.clip_high - self.clip_low) / n
        noise = rng.laplace(0, sens / self.epsilon)
        return true_mean + noise


class DPHistogram:
    """Differentially-private histogram using Laplace mechanism.

    Parameters:
        epsilon: Privacy parameter.
        bins: Number of histogram bins.
        clip_low: Lower data bound.
        clip_high: Upper data bound.
        budget: Optional shared budget tracker.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        bins: int = 10,
        clip_low: float = 0.0,
        clip_high: float = 1.0,
        budget: PrivacyBudget | None = None,
    ) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = epsilon
        self.bins = bins
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.budget = budget

    def compute(
        self,
        data: np.ndarray | pd.Series,
        rng: np.random.RandomState | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute DP histogram.

        Args:
            data: Input data array.
            rng: Random state for reproducibility.

        Returns:
            Tuple of (noisy_counts, bin_edges).
        """
        if self.budget is not None:
            self.budget.spend(self.epsilon, operation="dp_histogram")

        rng = rng or np.random.RandomState()
        arr = np.asarray(data, dtype=float)
        arr = np.clip(arr, self.clip_low, self.clip_high)

        counts, edges = np.histogram(arr, bins=self.bins, range=(self.clip_low, self.clip_high))
        # Sensitivity is 1 (each person can change one bin by 1)
        noise = rng.laplace(0, 1.0 / self.epsilon, size=len(counts))
        noisy_counts = np.maximum(0, counts + noise)  # Floor at 0
        return noisy_counts, edges


class DPSum:
    """Differentially-private sum using Laplace mechanism.

    Parameters:
        epsilon: Privacy parameter.
        clip_low: Lower clipping bound.
        clip_high: Upper clipping bound.
        budget: Optional shared budget tracker.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        clip_low: float = 0.0,
        clip_high: float = 1.0,
        budget: PrivacyBudget | None = None,
    ) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = epsilon
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.budget = budget

    def compute(
        self,
        data: np.ndarray | pd.Series,
        rng: np.random.RandomState | None = None,
    ) -> float:
        """Compute DP sum of the data."""
        if self.budget is not None:
            self.budget.spend(self.epsilon, operation="dp_sum")

        rng = rng or np.random.RandomState()
        arr = np.asarray(data, dtype=float)
        arr = np.clip(arr, self.clip_low, self.clip_high)
        true_sum = float(np.sum(arr))
        sensitivity = self.clip_high - self.clip_low
        noise = rng.laplace(0, sensitivity / self.epsilon)
        return true_sum + noise


class KAnonymizer:
    """K-Anonymity enforcement via generalization and suppression.

    Groups quasi-identifier columns such that each combination appears
    at least k times. Rare groups are suppressed (set to None).

    Parameters:
        k: Minimum group size.
        quasi_identifiers: Columns to anonymize.
    """

    def __init__(self, k: int = 5, quasi_identifiers: list[str] | None = None) -> None:
        if k < 2:
            raise ValueError("k must be >= 2")
        self.k = k
        self.quasi_identifiers = quasi_identifiers or []

    def check(self, df: pd.DataFrame) -> dict[str, Any]:
        """Check if DataFrame satisfies k-anonymity.

        Returns:
            Dict with is_satisfied, min_group_size, num_violating_groups.
        """
        if not self.quasi_identifiers:
            return {"is_satisfied": True, "min_group_size": len(df), "violating_groups": 0}

        cols = [c for c in self.quasi_identifiers if c in df.columns]
        if not cols:
            return {"is_satisfied": True, "min_group_size": len(df), "violating_groups": 0}

        group_sizes = df.groupby(cols, dropna=False).size()
        min_size = int(group_sizes.min())
        violating = int((group_sizes < self.k).sum())
        return {
            "is_satisfied": min_size >= self.k,
            "min_group_size": min_size,
            "violating_groups": violating,
        }

    def suppress(self, df: pd.DataFrame) -> pd.DataFrame:
        """Suppress rows in groups smaller than k.

        Sets quasi-identifier values to None for groups < k.

        Returns:
            DataFrame with suppressed values.
        """
        if not self.quasi_identifiers:
            return df.copy()

        cols = [c for c in self.quasi_identifiers if c in df.columns]
        if not cols:
            return df.copy()

        result = df.copy()
        group_sizes = result.groupby(cols, dropna=False).transform("size")
        # Use the first column's group size (they're all the same per row)
        mask = group_sizes.iloc[:, 0] < self.k if isinstance(group_sizes, pd.DataFrame) else group_sizes < self.k
        for col in cols:
            result.loc[mask, col] = None
        return result

    def generalize_numeric(
        self,
        df: pd.DataFrame,
        column: str,
        bins: int = 5,
    ) -> pd.DataFrame:
        """Generalize a numeric column into bins to increase anonymity.

        Args:
            df: Input DataFrame.
            column: Column to generalize.
            bins: Number of bins.

        Returns:
            DataFrame with generalized column.
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")

        result = df.copy()
        result[column] = pd.cut(result[column], bins=bins, labels=False)
        return result
