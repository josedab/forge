"""Window implementations for streaming aggregations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class WindowConfig:
    """Configuration for a streaming window.

    Args:
        size: Window size (number of events or time duration in seconds).
        slide: Slide interval for sliding windows (defaults to 1).
        min_periods: Minimum events before producing output.
    """

    size: int
    slide: int = 1
    min_periods: int = 1


class TumblingWindow:
    """Fixed-size, non-overlapping window for streaming aggregations.

    Events accumulate until the window is full, then the window
    emits its aggregate and resets.

    Args:
        size: Number of events per window.
        min_periods: Minimum events before producing output.

    Example:
        >>> window = TumblingWindow(size=100)
        >>> window.add(42.0)
        >>> if window.is_ready:
        ...     print(window.mean())
    """

    def __init__(self, size: int, min_periods: int = 1) -> None:
        self.size = size
        self.min_periods = min_periods
        self._buffer: list[float] = []
        self._count = 0

    def add(self, value: float) -> None:
        """Add a value to the window."""
        self._buffer.append(value)
        self._count += 1

    @property
    def is_ready(self) -> bool:
        """Whether the window has enough data to produce output."""
        return len(self._buffer) >= self.min_periods

    @property
    def is_full(self) -> bool:
        """Whether the window is full and should be flushed."""
        return len(self._buffer) >= self.size

    def flush(self) -> list[float]:
        """Flush the window and return accumulated values."""
        values = self._buffer.copy()
        self._buffer.clear()
        return values

    def mean(self) -> float:
        """Compute mean of current window."""
        if not self._buffer:
            return np.nan
        return float(np.mean(self._buffer))

    def std(self) -> float:
        """Compute std of current window."""
        if len(self._buffer) < 2:
            return np.nan
        return float(np.std(self._buffer, ddof=1))

    def min(self) -> float:
        """Compute min of current window."""
        if not self._buffer:
            return np.nan
        return float(np.min(self._buffer))

    def max(self) -> float:
        """Compute max of current window."""
        if not self._buffer:
            return np.nan
        return float(np.max(self._buffer))

    def sum(self) -> float:
        """Compute sum of current window."""
        return float(np.sum(self._buffer))

    def count(self) -> int:
        """Return count of values in window."""
        return len(self._buffer)

    def reset(self) -> None:
        """Reset the window."""
        self._buffer.clear()


class SlidingWindow:
    """Fixed-size sliding window using a deque for O(1) append/pop.

    Maintains the most recent `size` values and provides
    efficient incremental statistics via Welford's online algorithm.

    Args:
        size: Maximum number of values to retain.
        min_periods: Minimum values before producing output.

    Example:
        >>> window = SlidingWindow(size=100)
        >>> for value in stream:
        ...     window.add(value)
        ...     if window.is_ready:
        ...         print(f"mean={window.mean():.2f}, std={window.std():.2f}")
    """

    def __init__(self, size: int, min_periods: int = 1) -> None:
        self.size = size
        self.min_periods = min_periods
        self._buffer: deque[float] = deque(maxlen=size)
        # Welford's online stats
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._sum = 0.0
        self._min = float("inf")
        self._max = float("-inf")

    def add(self, value: float) -> None:
        """Add a value, evicting the oldest if the window is full."""
        if np.isnan(value):
            return

        evicted = None
        if len(self._buffer) == self.size:
            evicted = self._buffer[0]

        self._buffer.append(value)

        if evicted is not None:
            # Update Welford's stats for removal + addition
            self._n = len(self._buffer)
            old_mean = self._mean
            self._sum = self._sum - evicted + value
            self._mean = self._sum / self._n
            # Approximate M2 update for window replacement
            self._m2 += (value - old_mean) * (value - self._mean)
            self._m2 -= (evicted - old_mean) * (evicted - self._mean)
            self._m2 = max(self._m2, 0.0)  # Guard against floating point drift
        else:
            # Welford's online update (addition only)
            self._n += 1
            self._sum += value
            delta = value - self._mean
            self._mean += delta / self._n
            delta2 = value - self._mean
            self._m2 += delta * delta2

        # Track min/max (recompute from buffer when evictions happen)
        if evicted is not None and (evicted == self._min or evicted == self._max):
            self._recompute_minmax()
        else:
            self._min = min(self._min, value)
            self._max = max(self._max, value)

    def _recompute_minmax(self) -> None:
        """Recompute min/max from the buffer."""
        if self._buffer:
            self._min = min(self._buffer)
            self._max = max(self._buffer)
        else:
            self._min = float("inf")
            self._max = float("-inf")

    @property
    def is_ready(self) -> bool:
        """Whether the window has enough data."""
        return len(self._buffer) >= self.min_periods

    def mean(self) -> float:
        """Return the current mean."""
        if not self._buffer:
            return np.nan
        return self._mean

    def std(self) -> float:
        """Return the current standard deviation."""
        if self._n < 2:
            return np.nan
        return float(np.sqrt(self._m2 / (self._n - 1)))

    def var(self) -> float:
        """Return the current variance."""
        if self._n < 2:
            return np.nan
        return self._m2 / (self._n - 1)

    def min(self) -> float:
        """Return the current minimum."""
        if not self._buffer:
            return np.nan
        return self._min

    def max(self) -> float:
        """Return the current maximum."""
        if not self._buffer:
            return np.nan
        return self._max

    def sum(self) -> float:
        """Return the current sum."""
        return self._sum

    def count(self) -> int:
        """Return the number of values in the window."""
        return len(self._buffer)

    def values(self) -> list[float]:
        """Return a copy of the current window values."""
        return list(self._buffer)

    def reset(self) -> None:
        """Reset all state."""
        self._buffer.clear()
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._sum = 0.0
        self._min = float("inf")
        self._max = float("-inf")
