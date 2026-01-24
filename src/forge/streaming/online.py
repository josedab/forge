"""Online statistics engine and stream connectors.

Provides incrementally-updatable statistics (running mean/variance,
t-digest quantiles, exponential moving averages, frequency counts),
stream connector abstractions, and adaptive feature re-computation.

Example:
    >>> from forge.streaming.online import OnlineStatistics, StreamConnector
    >>> stats = OnlineStatistics(columns=["price", "quantity"])
    >>> for row in data_stream:
    ...     stats.update(row)
    >>> print(stats.mean("price"), stats.quantile("price", 0.95))
"""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WelfordState:
    """Welford's online algorithm state for mean/variance."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_val: float = float("inf")
    max_val: float = float("-inf")

    def update(self, value: float) -> None:
        """Update with a new observation."""
        if math.isnan(value):
            return
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)

    @property
    def variance(self) -> float:
        """Population variance."""
        if self.count < 2:
            return 0.0
        return self.m2 / self.count

    @property
    def std(self) -> float:
        """Population standard deviation."""
        return math.sqrt(self.variance)


class TDigest:
    """Simplified t-digest for online quantile estimation.

    Uses a sorted list of centroids to approximate the distribution.
    """

    def __init__(self, compression: float = 100.0) -> None:
        self.compression = compression
        self._centroids: list[tuple[float, int]] = []  # (mean, weight)
        self._total_weight: int = 0
        self._min: float = float("inf")
        self._max: float = float("-inf")

    def update(self, value: float) -> None:
        """Add a single value."""
        if math.isnan(value):
            return
        self._centroids.append((value, 1))
        self._total_weight += 1
        self._min = min(self._min, value)
        self._max = max(self._max, value)
        # Compress periodically
        if len(self._centroids) > int(self.compression) * 3:
            self._compress()

    def quantile(self, q: float) -> float:
        """Estimate quantile.

        Args:
            q: Quantile in [0, 1].

        Returns:
            Estimated value at quantile.
        """
        if not self._centroids:
            return 0.0
        if q <= 0:
            return self._min
        if q >= 1:
            return self._max

        self._compress()
        target = q * self._total_weight
        cumulative = 0.0

        for i, (mean, weight) in enumerate(self._centroids):
            cumulative += weight
            if cumulative >= target:
                return mean
        return self._centroids[-1][0]

    def _compress(self) -> None:
        """Compress centroids by merging nearby ones."""
        if len(self._centroids) <= 1:
            return
        self._centroids.sort(key=lambda c: c[0])
        merged: list[tuple[float, int]] = [self._centroids[0]]

        for mean, weight in self._centroids[1:]:
            last_mean, last_weight = merged[-1]
            new_weight = last_weight + weight
            if new_weight <= max(1, int(self.compression / len(merged) + 1)):
                new_mean = (last_mean * last_weight + mean * weight) / new_weight
                merged[-1] = (new_mean, new_weight)
            else:
                merged.append((mean, weight))

        self._centroids = merged

    @property
    def count(self) -> int:
        """Total observations."""
        return self._total_weight


class ExponentialMovingAverage:
    """Exponential moving average tracker."""

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._value: float | None = None
        self._count: int = 0

    def update(self, value: float) -> None:
        """Update EMA with new value."""
        if math.isnan(value):
            return
        self._count += 1
        if self._value is None:
            self._value = value
        else:
            self._value = self.alpha * value + (1 - self.alpha) * self._value

    @property
    def value(self) -> float:
        """Current EMA value."""
        return self._value if self._value is not None else 0.0

    @property
    def count(self) -> int:
        """Number of observations."""
        return self._count


class FrequencyCounter:
    """Online frequency counter for categorical values."""

    def __init__(self, max_categories: int = 1000) -> None:
        self.max_categories = max_categories
        self._counts: dict[Any, int] = defaultdict(int)
        self._total: int = 0

    def update(self, value: Any) -> None:
        """Count a value occurrence."""
        if len(self._counts) >= self.max_categories and value not in self._counts:
            return  # Skip if at capacity
        self._counts[value] += 1
        self._total += 1

    def frequency(self, value: Any) -> float:
        """Get relative frequency of a value."""
        if self._total == 0:
            return 0.0
        return self._counts.get(value, 0) / self._total

    @property
    def top_k(self) -> list[tuple[Any, int]]:
        """Get top-k by count."""
        return sorted(self._counts.items(), key=lambda x: -x[1])[:10]

    @property
    def n_unique(self) -> int:
        """Number of unique values seen."""
        return len(self._counts)

    @property
    def total(self) -> int:
        """Total observations."""
        return self._total


class OnlineStatistics:
    """Maintains online statistics for multiple columns.

    Tracks running mean, variance, quantiles, EMA, and
    frequency counts incrementally.

    Parameters:
        columns: Column names to track.
        ema_alpha: Alpha for exponential moving average.
        tdigest_compression: T-digest compression factor.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        ema_alpha: float = 0.1,
        tdigest_compression: float = 100.0,
    ) -> None:
        self.columns = columns
        self.ema_alpha = ema_alpha
        self.tdigest_compression = tdigest_compression
        self._welford: dict[str, WelfordState] = {}
        self._tdigest: dict[str, TDigest] = {}
        self._ema: dict[str, ExponentialMovingAverage] = {}
        self._freq: dict[str, FrequencyCounter] = {}
        self._n_rows: int = 0

        if columns:
            for col in columns:
                self._init_column(col)

    def _init_column(self, col: str) -> None:
        """Initialize tracking for a column."""
        if col not in self._welford:
            self._welford[col] = WelfordState()
            self._tdigest[col] = TDigest(self.tdigest_compression)
            self._ema[col] = ExponentialMovingAverage(self.ema_alpha)
            self._freq[col] = FrequencyCounter()

    def update(self, row: dict[str, Any] | pd.Series) -> None:
        """Update statistics with a single row.

        Args:
            row: Single data row as dict or Series.
        """
        if isinstance(row, pd.Series):
            row = row.to_dict()

        self._n_rows += 1

        for col, value in row.items():
            col_str = str(col)
            if self.columns and col_str not in self.columns:
                continue

            self._init_column(col_str)

            if isinstance(value, (int, float)) and not math.isnan(value):
                self._welford[col_str].update(float(value))
                self._tdigest[col_str].update(float(value))
                self._ema[col_str].update(float(value))
            self._freq[col_str].update(value)

    def update_batch(self, df: pd.DataFrame) -> None:
        """Update with a batch of rows.

        Args:
            df: DataFrame with multiple rows.
        """
        for _, row in df.iterrows():
            self.update(row)

    def mean(self, column: str) -> float:
        """Get running mean for a column."""
        state = self._welford.get(column)
        return state.mean if state else 0.0

    def variance(self, column: str) -> float:
        """Get running variance for a column."""
        state = self._welford.get(column)
        return state.variance if state else 0.0

    def std(self, column: str) -> float:
        """Get running standard deviation for a column."""
        state = self._welford.get(column)
        return state.std if state else 0.0

    def min(self, column: str) -> float:
        """Get running min for a column."""
        state = self._welford.get(column)
        return state.min_val if state else float("inf")

    def max(self, column: str) -> float:
        """Get running max for a column."""
        state = self._welford.get(column)
        return state.max_val if state else float("-inf")

    def quantile(self, column: str, q: float) -> float:
        """Get estimated quantile for a column."""
        td = self._tdigest.get(column)
        return td.quantile(q) if td else 0.0

    def ema(self, column: str) -> float:
        """Get exponential moving average for a column."""
        e = self._ema.get(column)
        return e.value if e else 0.0

    def frequency(self, column: str, value: Any) -> float:
        """Get relative frequency of a categorical value."""
        fc = self._freq.get(column)
        return fc.frequency(value) if fc else 0.0

    def summary(self, column: str) -> dict[str, Any]:
        """Get full summary statistics for a column.

        Returns:
            Dictionary with mean, std, min, max, quantiles, ema, count.
        """
        state = self._welford.get(column, WelfordState())
        return {
            "count": state.count,
            "mean": state.mean,
            "std": state.std,
            "min": state.min_val if state.count > 0 else None,
            "max": state.max_val if state.count > 0 else None,
            "q25": self.quantile(column, 0.25),
            "q50": self.quantile(column, 0.50),
            "q75": self.quantile(column, 0.75),
            "q95": self.quantile(column, 0.95),
            "ema": self.ema(column),
        }

    @property
    def tracked_columns(self) -> list[str]:
        """Get all tracked column names."""
        return sorted(self._welford.keys())

    @property
    def n_rows(self) -> int:
        """Total rows processed."""
        return self._n_rows


@dataclass
class StreamEvent:
    """A single event from a data stream."""

    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    key: str = ""
    source: str = ""


class StreamConnector(ABC):
    """Abstract base for stream connectors.

    Provides a unified interface for consuming events from
    different streaming sources (Kafka, Redis, in-memory).
    """

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the stream."""

    @abstractmethod
    def consume(self, timeout: float = 1.0) -> list[StreamEvent]:
        """Consume available events.

        Args:
            timeout: Max time to wait for events.

        Returns:
            List of StreamEvent objects.
        """

    @abstractmethod
    def close(self) -> None:
        """Close the connection."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the connector is connected."""


class InMemoryStreamConnector(StreamConnector):
    """In-memory stream connector for testing and development.

    Events are pushed via `push()` and consumed via `consume()`.
    """

    def __init__(self) -> None:
        self._buffer: list[StreamEvent] = []
        self._connected: bool = False

    def connect(self) -> None:
        """Establish connection."""
        self._connected = True

    def push(self, data: dict[str, Any], key: str = "") -> None:
        """Push an event to the stream.

        Args:
            data: Event data.
            key: Optional event key.
        """
        self._buffer.append(
            StreamEvent(data=data, key=key, source="in-memory")
        )

    def consume(self, timeout: float = 1.0) -> list[StreamEvent]:
        """Consume all buffered events."""
        events = list(self._buffer)
        self._buffer.clear()
        return events

    def close(self) -> None:
        """Close the connection."""
        self._connected = False
        self._buffer.clear()

    @property
    def is_connected(self) -> bool:
        """Whether connected."""
        return self._connected

    @property
    def buffer_size(self) -> int:
        """Number of events in buffer."""
        return len(self._buffer)


class AdaptiveFeatureEngine:
    """Adaptive feature engine with drift-triggered recomputation.

    Monitors online statistics and triggers re-computation when
    distribution drift is detected beyond configured thresholds.

    Parameters:
        stats: OnlineStatistics to monitor.
        drift_threshold: Z-score threshold for drift detection.
        window_size: Number of recent observations for drift window.
        on_drift: Optional callback when drift is detected.
    """

    def __init__(
        self,
        stats: OnlineStatistics | None = None,
        drift_threshold: float = 3.0,
        window_size: int = 100,
        on_drift: Callable[[str, float], None] | None = None,
    ) -> None:
        self.stats = stats or OnlineStatistics()
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self.on_drift = on_drift
        self._recent_values: dict[str, list[float]] = defaultdict(list)
        self._drift_events: list[dict[str, Any]] = []
        self._n_processed: int = 0

    def process(self, row: dict[str, Any] | pd.Series) -> dict[str, bool]:
        """Process a row and check for drift.

        Args:
            row: Data row to process.

        Returns:
            Dictionary of column -> drift_detected.
        """
        if isinstance(row, pd.Series):
            row = row.to_dict()

        drift_flags: dict[str, bool] = {}
        self._n_processed += 1

        for col, value in row.items():
            col_str = str(col)
            if not isinstance(value, (int, float)) or math.isnan(value):
                continue

            self._recent_values[col_str].append(float(value))
            if len(self._recent_values[col_str]) > self.window_size:
                self._recent_values[col_str] = self._recent_values[col_str][-self.window_size:]

            # Check drift against running stats
            overall_mean = self.stats.mean(col_str)
            overall_std = self.stats.std(col_str)

            if overall_std > 0 and len(self._recent_values[col_str]) >= 10:
                window_mean = np.mean(self._recent_values[col_str])
                z_score = abs(window_mean - overall_mean) / overall_std
                drift_detected = z_score > self.drift_threshold
                drift_flags[col_str] = drift_detected

                if drift_detected:
                    self._drift_events.append({
                        "column": col_str,
                        "z_score": float(z_score),
                        "overall_mean": overall_mean,
                        "window_mean": float(window_mean),
                        "n_processed": self._n_processed,
                    })
                    if self.on_drift:
                        self.on_drift(col_str, z_score)

        self.stats.update(row)
        return drift_flags

    @property
    def drift_events(self) -> list[dict[str, Any]]:
        """All recorded drift events."""
        return list(self._drift_events)

    @property
    def n_processed(self) -> int:
        """Total rows processed."""
        return self._n_processed
