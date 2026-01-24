"""Incremental/online feature generators for streaming data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from forge.exceptions import NotFittedError, ValidationError
from forge.streaming.window import SlidingWindow

if TYPE_CHECKING:
    from typing_extensions import Self


class StreamingFeatureGenerator(BaseEstimator, TransformerMixin, ABC):  # type: ignore[misc]
    """Abstract base for streaming feature generators.

    Streaming generators support both batch (fit/transform) and
    incremental (partial_fit/partial_transform) operation modes.

    Subclasses must implement partial_fit and partial_transform
    for single-event processing. Batch methods delegate to these.
    """

    def __init__(self) -> None:
        self._is_fitted: bool = False
        self._feature_names_out: list[str] = []
        self._n_events_seen: int = 0

    @abstractmethod
    def partial_fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> Self:
        """Incrementally fit on a batch or single event.

        Args:
            X: Input data (one or more rows).
            y: Optional target variable.

        Returns:
            Self for method chaining.
        """

    @abstractmethod
    def partial_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform a batch or single event using current state.

        Args:
            X: Input data (one or more rows).

        Returns:
            DataFrame with generated streaming features.
        """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Batch fit by delegating to partial_fit.

        Args:
            X: Input DataFrame.
            y: Optional target variable.

        Returns:
            Self for method chaining.
        """
        self.reset()
        return self.partial_fit(X, y)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Batch transform by delegating to partial_transform.

        Args:
            X: Input DataFrame.

        Returns:
            DataFrame with generated features.
        """
        self._check_is_fitted()
        return self.partial_transform(X)

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> list[str]:
        """Get output feature names."""
        self._check_is_fitted()
        return self._feature_names_out.copy()

    def reset(self) -> None:
        """Reset all internal state."""
        self._is_fitted = False
        self._feature_names_out = []
        self._n_events_seen = 0

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)

    def get_state(self) -> dict[str, Any]:
        """Serialize internal state for checkpointing."""
        return {
            "n_events_seen": self._n_events_seen,
            "feature_names_out": self._feature_names_out,
            "is_fitted": self._is_fitted,
        }


class IncrementalAggregator(StreamingFeatureGenerator):
    """Compute running aggregations using Welford's online algorithm.

    Maintains running mean, variance, min, max, count, and sum
    for specified numeric columns without storing all data.

    Args:
        columns: Columns to aggregate. None for all numeric.
        stats: Statistics to compute. Default: mean, std, min, max.

    Example:
        >>> agg = IncrementalAggregator(columns=["price", "quantity"])
        >>> for batch in data_stream:
        ...     agg.partial_fit(batch)
        ...     features = agg.partial_transform(batch)
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        stats: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.columns = columns
        self.stats = stats or ["mean", "std", "min", "max"]
        # Welford accumulators per column
        self._n: dict[str, int] = {}
        self._mean: dict[str, float] = {}
        self._m2: dict[str, float] = {}
        self._sum: dict[str, float] = {}
        self._min: dict[str, float] = {}
        self._max: dict[str, float] = {}
        self._resolved_columns: list[str] = []

    def partial_fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> Self:
        """Update running statistics with new data."""
        if not isinstance(X, pd.DataFrame):
            raise ValidationError(f"Expected DataFrame, got {type(X).__name__}")

        if not self._resolved_columns:
            if self.columns is not None:
                self._resolved_columns = [
                    c for c in self.columns if c in X.columns
                ]
            else:
                self._resolved_columns = list(
                    X.select_dtypes(include=[np.number]).columns
                )
            for col in self._resolved_columns:
                self._n[col] = 0
                self._mean[col] = 0.0
                self._m2[col] = 0.0
                self._sum[col] = 0.0
                self._min[col] = float("inf")
                self._max[col] = float("-inf")

        for col in self._resolved_columns:
            if col not in X.columns:
                continue
            values = X[col].dropna().values
            for val in values:
                v = float(val)
                self._n[col] += 1
                delta = v - self._mean[col]
                self._mean[col] += delta / self._n[col]
                delta2 = v - self._mean[col]
                self._m2[col] += delta * delta2
                self._sum[col] += v
                self._min[col] = min(self._min[col], v)
                self._max[col] = max(self._max[col], v)

        self._n_events_seen += len(X)
        self._feature_names_out = [
            f"{col}_running_{stat}"
            for col in self._resolved_columns
            for stat in self.stats
        ]
        self._is_fitted = True
        return self

    def partial_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate running statistic features for the current state."""
        self._check_is_fitted()
        result: dict[str, list[float]] = {}

        for col in self._resolved_columns:
            for stat in self.stats:
                fname = f"{col}_running_{stat}"
                val = self._get_stat(col, stat)
                result[fname] = [val] * len(X)

        return pd.DataFrame(result, index=X.index)

    def _get_stat(self, col: str, stat: str) -> float:
        n = self._n.get(col, 0)
        if n == 0:
            return np.nan
        if stat == "mean":
            return self._mean[col]
        if stat == "std":
            if n < 2:
                return np.nan
            return float(np.sqrt(self._m2[col] / (n - 1)))
        if stat == "var":
            if n < 2:
                return np.nan
            return self._m2[col] / (n - 1)
        if stat == "min":
            return self._min[col]
        if stat == "max":
            return self._max[col]
        if stat == "sum":
            return self._sum[col]
        if stat == "count":
            return float(n)
        return np.nan

    def reset(self) -> None:
        """Reset all accumulators."""
        super().reset()
        self._n.clear()
        self._mean.clear()
        self._m2.clear()
        self._sum.clear()
        self._min.clear()
        self._max.clear()
        self._resolved_columns = []


class StreamingWindowAggregator(StreamingFeatureGenerator):
    """Compute windowed aggregations over a sliding window.

    Uses SlidingWindow internally for efficient O(1) updates.

    Args:
        columns: Columns to aggregate. None for all numeric.
        window_size: Number of events in the sliding window.
        stats: Statistics to compute per window.
        min_periods: Minimum events before producing output.

    Example:
        >>> agg = StreamingWindowAggregator(
        ...     columns=["price"], window_size=100, stats=["mean", "std"]
        ... )
        >>> agg.partial_fit(batch)
        >>> features = agg.partial_transform(batch)
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        window_size: int = 100,
        stats: list[str] | None = None,
        min_periods: int = 1,
    ) -> None:
        super().__init__()
        self.columns = columns
        self.window_size = window_size
        self.stats = stats or ["mean", "std", "min", "max"]
        self.min_periods = min_periods
        self._windows: dict[str, SlidingWindow] = {}
        self._resolved_columns: list[str] = []

    def partial_fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> Self:
        """Update sliding windows with new data."""
        if not isinstance(X, pd.DataFrame):
            raise ValidationError(f"Expected DataFrame, got {type(X).__name__}")

        if not self._resolved_columns:
            if self.columns is not None:
                self._resolved_columns = [
                    c for c in self.columns if c in X.columns
                ]
            else:
                self._resolved_columns = list(
                    X.select_dtypes(include=[np.number]).columns
                )
            for col in self._resolved_columns:
                self._windows[col] = SlidingWindow(
                    size=self.window_size, min_periods=self.min_periods
                )

        for col in self._resolved_columns:
            if col not in X.columns:
                continue
            for val in X[col].dropna().values:
                self._windows[col].add(float(val))

        self._n_events_seen += len(X)
        self._feature_names_out = [
            f"{col}_window{self.window_size}_{stat}"
            for col in self._resolved_columns
            for stat in self.stats
        ]
        self._is_fitted = True
        return self

    def partial_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate windowed features for current window state."""
        self._check_is_fitted()
        result: dict[str, list[float]] = {}

        for col in self._resolved_columns:
            window = self._windows.get(col)
            for stat in self.stats:
                fname = f"{col}_window{self.window_size}_{stat}"
                if window is None or not window.is_ready:
                    val = np.nan
                else:
                    val = getattr(window, stat)()
                result[fname] = [val] * len(X)

        return pd.DataFrame(result, index=X.index)

    def reset(self) -> None:
        """Reset all windows."""
        super().reset()
        for w in self._windows.values():
            w.reset()
        self._windows.clear()
        self._resolved_columns = []


class IncrementalInteractionGenerator(StreamingFeatureGenerator):
    """Generate interaction features incrementally.

    Computes pairwise products and ratios for numeric columns
    in a streaming fashion.

    Args:
        columns: Columns to interact. None for all numeric.
        operations: Operations to apply. Default: multiply, ratio.
        max_interactions: Maximum number of interaction pairs.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        operations: list[str] | None = None,
        max_interactions: int = 50,
    ) -> None:
        super().__init__()
        self.columns = columns
        self.operations = operations or ["multiply", "ratio"]
        self.max_interactions = max_interactions
        self._resolved_columns: list[str] = []
        self._pairs: list[tuple[str, str]] = []

    def partial_fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> Self:
        """Determine interaction pairs from the data."""
        if not isinstance(X, pd.DataFrame):
            raise ValidationError(f"Expected DataFrame, got {type(X).__name__}")

        if not self._resolved_columns:
            if self.columns is not None:
                self._resolved_columns = [
                    c for c in self.columns if c in X.columns
                ]
            else:
                self._resolved_columns = list(
                    X.select_dtypes(include=[np.number]).columns
                )

            # Generate pairs up to max_interactions
            pairs: list[tuple[str, str]] = []
            for i, c1 in enumerate(self._resolved_columns):
                for c2 in self._resolved_columns[i + 1 :]:
                    pairs.append((c1, c2))
                    if len(pairs) >= self.max_interactions:
                        break
                if len(pairs) >= self.max_interactions:
                    break
            self._pairs = pairs

        self._feature_names_out = []
        for c1, c2 in self._pairs:
            for op in self.operations:
                self._feature_names_out.append(f"{c1}_{op}_{c2}")

        self._n_events_seen += len(X)
        self._is_fitted = True
        return self

    def partial_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute interaction features for the given data."""
        self._check_is_fitted()
        result: dict[str, Any] = {}

        for c1, c2 in self._pairs:
            if c1 not in X.columns or c2 not in X.columns:
                continue
            for op in self.operations:
                fname = f"{c1}_{op}_{c2}"
                if op == "multiply":
                    result[fname] = X[c1] * X[c2]
                elif op == "ratio":
                    result[fname] = X[c1] / X[c2].replace(0, np.nan)
                elif op == "add":
                    result[fname] = X[c1] + X[c2]
                elif op == "subtract":
                    result[fname] = X[c1] - X[c2]

        return pd.DataFrame(result, index=X.index)

    def reset(self) -> None:
        """Reset state."""
        super().reset()
        self._resolved_columns = []
        self._pairs = []
