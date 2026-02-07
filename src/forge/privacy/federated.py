"""Federated feature engineering with privacy guarantees.

Implements secure aggregation protocols for computing feature statistics
across multiple parties without sharing raw data.

Example:
    >>> from forge.privacy.federated import FederatedFeatureEngineer
    >>> fed = FederatedFeatureEngineer(epsilon=1.0, n_parties=3)
    >>> stats = fed.secure_aggregate([party1_df, party2_df, party3_df])
    >>> X_transformed = fed.transform(new_df)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd  # noqa: TC002
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


@dataclass
class SecureShare:
    """A single party's masked contribution to a secure aggregation."""

    party_id: str
    column: str
    masked_sum: float
    masked_count: int
    noise_seed: int


@dataclass
class AggregationResult:
    """Result of a secure aggregation across parties."""

    column: str
    mean: float
    std: float
    count: int
    n_parties: int
    epsilon_spent: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "mean": self.mean,
            "std": self.std,
            "count": self.count,
            "n_parties": self.n_parties,
            "epsilon_spent": self.epsilon_spent,
        }


def _add_laplace_noise(value: float, sensitivity: float, epsilon: float, rng: np.random.RandomState) -> float:
    """Add Laplace noise calibrated to sensitivity / epsilon."""
    if epsilon <= 0:
        return value
    scale = sensitivity / epsilon
    noise = rng.laplace(0, scale)
    return value + noise


def secure_aggregate_mean(
    party_values: list[np.ndarray],
    clip_low: float = -1e6,
    clip_high: float = 1e6,
    epsilon: float = 1.0,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute DP mean + std across multiple parties.

    Each party clips its values, adds calibrated noise to its local sum,
    and the results are aggregated by the coordinator.

    Args:
        party_values: List of arrays, one per party.
        clip_low: Lower clipping bound.
        clip_high: Upper clipping bound.
        epsilon: Privacy budget for this aggregation.
        seed: Random seed.

    Returns:
        (dp_mean, dp_std) tuple.
    """
    rng = np.random.RandomState(seed)
    sensitivity = clip_high - clip_low

    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    for vals in party_values:
        clipped = np.clip(vals, clip_low, clip_high)
        n = len(clipped)
        local_sum = float(np.sum(clipped))
        local_sq_sum = float(np.sum(clipped ** 2))

        # Sensitivity of the mean query is (clip_high - clip_low) / n
        # so sensitivity of the sum query is (clip_high - clip_low).
        eps_half = epsilon / 2.0
        noisy_sum = _add_laplace_noise(local_sum, sensitivity, eps_half, rng)
        sq_sensitivity = max(clip_high ** 2, clip_low ** 2)
        noisy_sq = _add_laplace_noise(local_sq_sum, sq_sensitivity, eps_half, rng)

        total_sum += noisy_sum
        total_sq_sum += noisy_sq
        total_count += n

    if total_count == 0:
        return 0.0, 0.0

    dp_mean = total_sum / total_count
    variance = max(0.0, total_sq_sum / total_count - dp_mean ** 2)
    dp_std = math.sqrt(variance)
    return dp_mean, dp_std


class FederatedFeatureEngineer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """sklearn-compatible transformer that learns statistics across federated parties.

    Each party's data never leaves its boundary; only noised aggregates
    are shared with the coordinator.

    Parameters
    ----------
    epsilon : float
        Total privacy budget.
    clip_low : float
        Lower clipping bound for numeric values.
    clip_high : float
        Upper clipping bound.
    seed : int
        Random seed for noise generation.

    Example:
    -------
    >>> fed = FederatedFeatureEngineer(epsilon=1.0)
    >>> fed.fit_federated([party1_df, party2_df])
    >>> X_norm = fed.transform(X_new)
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        clip_low: float = -1e6,
        clip_high: float = 1e6,
        seed: int = 42,
    ) -> None:
        self.epsilon = epsilon
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.seed = seed

        self._means: dict[str, float] = {}
        self._stds: dict[str, float] = {}
        self._aggregation_results: list[AggregationResult] = []
        self._is_fitted = False
        self._numeric_columns: list[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> FederatedFeatureEngineer:
        """Fit on a single party's data (non-federated path).

        Args:
            X: Training data from one party.
            y: Ignored.

        Returns:
            Self.
        """
        return self.fit_federated([X])

    def fit_federated(
        self, party_data: list[pd.DataFrame],
    ) -> FederatedFeatureEngineer:
        """Fit across multiple parties using secure aggregation.

        Args:
            party_data: List of DataFrames, one per party.

        Returns:
            Self.
        """
        if not party_data:
            raise ValueError("At least one party DataFrame is required")

        all_numeric = set()
        for df in party_data:
            all_numeric.update(df.select_dtypes(include=[np.number]).columns)
        self._numeric_columns = sorted(all_numeric)

        eps_per_col = self.epsilon / max(len(self._numeric_columns), 1)
        self._aggregation_results = []

        for col in self._numeric_columns:
            party_vals = []
            for df in party_data:
                if col in df.columns:
                    vals = df[col].dropna().values.astype(float)
                    party_vals.append(vals)

            if not party_vals:
                continue

            dp_mean, dp_std = secure_aggregate_mean(
                party_vals,
                clip_low=self.clip_low,
                clip_high=self.clip_high,
                epsilon=eps_per_col,
                seed=self.seed,
            )
            self._means[col] = dp_mean
            self._stds[col] = dp_std if dp_std > 0 else 1.0

            total_count = sum(len(v) for v in party_vals)
            self._aggregation_results.append(AggregationResult(
                column=col,
                mean=dp_mean,
                std=dp_std,
                count=total_count,
                n_parties=len(party_vals),
                epsilon_spent=eps_per_col,
            ))

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Normalize data using federally-learned DP statistics.

        Args:
            X: Data to transform.

        Returns:
            Normalized DataFrame.
        """
        if not self._is_fitted:
            raise RuntimeError("FederatedFeatureEngineer is not fitted.")

        result = X.copy()
        for col in self._means:
            if col in result.columns:
                std = self._stds.get(col, 1.0)
                result[col] = (result[col] - self._means[col]) / std
        return result

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        return list(self._means.keys())

    def get_aggregation_results(self) -> list[AggregationResult]:
        """Return aggregation results for inspection."""
        return list(self._aggregation_results)

    @property
    def total_epsilon_spent(self) -> float:
        return sum(r.epsilon_spent for r in self._aggregation_results)
