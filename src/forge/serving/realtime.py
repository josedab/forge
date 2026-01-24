"""Real-time computation engine with circuit breaker and request batching.

Extends the serving layer with production-grade reliability patterns:
- CircuitBreaker: Prevents cascading failures by opening on error threshold.
- RequestBatcher: Batches single-row requests for efficient processing.
- ComputationEngine: Unified engine combining compiled pipeline, caching,
  batching, and circuit breaking.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker pattern for feature computation.

    Monitors error rates and opens the circuit when failures exceed
    a threshold, preventing cascading failures in production.

    Args:
        failure_threshold: Number of failures before opening.
        recovery_timeout: Seconds to wait before testing recovery.
        half_open_max_calls: Calls allowed in half-open state to test recovery.

    Example:
        >>> breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30)
        >>> if breaker.allow_request():
        ...     try:
        ...         result = compute_features(row)
        ...         breaker.record_success()
        ...     except Exception:
        ...         breaker.record_failure()
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if (
                    self._last_failure_time is not None
                    and time.time() - self._last_failure_time >= self.recovery_timeout
                ):
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
            return self._state

    def allow_request(self) -> bool:
        """Check if a request should be allowed through."""
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
        return False  # OPEN

    def record_success(self) -> None:
        """Record a successful request."""
        with self._lock:
            self._failure_count = 0
            self._success_count += 1
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                logger.info("Circuit breaker closed after successful recovery")

    def record_failure(self) -> None:
        """Record a failed request."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning("Circuit breaker re-opened after failure in half-open state")
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    "Circuit breaker opened after %d failures", self._failure_count
                )

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._last_failure_time = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize circuit breaker state."""
        return {
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
        }


class TTLCache:
    """Thread-safe LRU cache with TTL expiration.

    Args:
        max_size: Maximum number of cached entries.
        ttl_seconds: Time-to-live for cache entries in seconds.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300.0) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """Get a value from cache, or None if missing/expired."""
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp <= self.ttl_seconds:
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return value
                else:
                    del self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, value: Any) -> None:
        """Store a value in cache."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, time.time())
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def invalidate(self, key: str) -> None:
        """Remove a specific key from cache."""
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    @property
    def size(self) -> int:
        """Number of entries currently cached."""
        return len(self._cache)

    def stats(self) -> dict[str, Any]:
        """Cache statistics."""
        return {
            "size": self.size,
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
            "ttl_seconds": self.ttl_seconds,
        }


@dataclass
class ComputationResult:
    """Result of a feature computation request.

    Attributes:
        features: Computed feature values.
        latency_ms: Computation latency in milliseconds.
        from_cache: Whether result was served from cache.
        error: Error message if computation failed.
    """

    features: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    from_cache: bool = False
    error: str | None = None

    @property
    def success(self) -> bool:
        """Whether computation succeeded."""
        return self.error is None


class ComputationEngine:
    """Unified real-time feature computation engine.

    Combines compiled pipeline execution, TTL caching, circuit breaking,
    and request batching for production feature serving.

    Args:
        transformer: Fitted sklearn-compatible transformer.
        cache_max_size: Maximum cache entries.
        cache_ttl: Cache TTL in seconds.
        circuit_failure_threshold: Failures before circuit opens.
        circuit_recovery_timeout: Seconds before circuit recovery test.

    Example:
        >>> engine = ComputationEngine(fitted_transformer)
        >>> result = engine.compute({"age": 25, "income": 50000})
        >>> if result.success:
        ...     print(result.features)
    """

    def __init__(
        self,
        transformer: Any = None,
        cache_max_size: int = 10000,
        cache_ttl: float = 300.0,
        circuit_failure_threshold: int = 10,
        circuit_recovery_timeout: float = 30.0,
    ) -> None:
        self._transformer = transformer
        self._cache = TTLCache(max_size=cache_max_size, ttl_seconds=cache_ttl)
        self._circuit = CircuitBreaker(
            failure_threshold=circuit_failure_threshold,
            recovery_timeout=circuit_recovery_timeout,
        )
        self._metrics_lock = threading.Lock()
        self._total_requests = 0
        self._total_errors = 0
        self._latencies: list[float] = []

    def compute(
        self, row: dict[str, Any], use_cache: bool = True
    ) -> ComputationResult:
        """Compute features for a single row.

        Args:
            row: Input feature values as a dict.
            use_cache: Whether to check/populate cache.

        Returns:
            ComputationResult with features or error.
        """
        start = time.perf_counter()

        if not self._circuit.allow_request():
            return ComputationResult(
                error="Circuit breaker is open — service unavailable",
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        # Check cache
        cache_key = self._make_cache_key(row) if use_cache else None
        if cache_key:
            cached = self._cache.get(cache_key)
            if cached is not None:
                latency = (time.perf_counter() - start) * 1000
                self._record_latency(latency)
                return ComputationResult(
                    features=cached, latency_ms=latency, from_cache=True
                )

        # Compute
        try:
            features = self._transform_row(row)
            self._circuit.record_success()
            if cache_key:
                self._cache.put(cache_key, features)
            latency = (time.perf_counter() - start) * 1000
            self._record_latency(latency)
            return ComputationResult(features=features, latency_ms=latency)
        except Exception as exc:
            self._circuit.record_failure()
            latency = (time.perf_counter() - start) * 1000
            self._record_latency(latency, error=True)
            return ComputationResult(error=str(exc), latency_ms=latency)

    def compute_batch(
        self, rows: list[dict[str, Any]]
    ) -> list[ComputationResult]:
        """Compute features for a batch of rows.

        Args:
            rows: List of input feature dicts.

        Returns:
            List of ComputationResults.
        """
        if not rows:
            return []

        start = time.perf_counter()

        if not self._circuit.allow_request():
            latency = (time.perf_counter() - start) * 1000
            return [
                ComputationResult(
                    error="Circuit breaker is open", latency_ms=latency
                )
                for _ in rows
            ]

        try:
            df = pd.DataFrame(rows)
            if self._transformer is not None:
                raw = self._transformer.transform(df)
                if isinstance(raw, pd.DataFrame):
                    result_df = raw
                else:
                    cols = list(rows[0].keys())
                    if hasattr(self._transformer, "get_feature_names_out"):
                        try:
                            cols = list(self._transformer.get_feature_names_out())
                        except Exception:
                            pass
                    result_df = pd.DataFrame(
                        np.asarray(raw), columns=cols, index=df.index
                    )
            else:
                result_df = df

            self._circuit.record_success()
            latency = (time.perf_counter() - start) * 1000
            per_row_latency = latency / len(rows)

            results = []
            for i in range(len(result_df)):
                features = result_df.iloc[i].to_dict()
                results.append(ComputationResult(
                    features=features, latency_ms=per_row_latency
                ))
            return results
        except Exception as exc:
            self._circuit.record_failure()
            latency = (time.perf_counter() - start) * 1000
            return [
                ComputationResult(error=str(exc), latency_ms=latency)
                for _ in rows
            ]

    def _transform_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Transform a single row using the fitted transformer."""
        df = pd.DataFrame([row])
        if self._transformer is not None:
            result = self._transformer.transform(df)
            if isinstance(result, pd.DataFrame):
                return result.iloc[0].to_dict()  # type: ignore[no-any-return]
            # Handle numpy array output (e.g., StandardScaler)
            columns = list(row.keys())
            if hasattr(self._transformer, "get_feature_names_out"):
                try:
                    columns = list(self._transformer.get_feature_names_out())
                except Exception:
                    pass
            if isinstance(result, np.ndarray):
                if result.ndim == 2:
                    return dict(zip(columns, result[0].tolist()))
                return dict(zip(columns, result.tolist()))
            return dict(zip(columns, np.asarray(result).ravel().tolist()))
        return row

    def _make_cache_key(self, row: dict[str, Any]) -> str:
        """Create a deterministic cache key from row values."""
        items = sorted(row.items())
        return str(items)

    def _record_latency(self, latency_ms: float, error: bool = False) -> None:
        with self._metrics_lock:
            self._total_requests += 1
            if error:
                self._total_errors += 1
            self._latencies.append(latency_ms)
            if len(self._latencies) > 1000:
                self._latencies = self._latencies[-1000:]

    def health(self) -> dict[str, Any]:
        """Get engine health status."""
        return {
            "circuit_breaker": self._circuit.to_dict(),
            "cache": self._cache.stats(),
            "total_requests": self._total_requests,
            "total_errors": self._total_errors,
            "avg_latency_ms": (
                round(np.mean(self._latencies), 3)
                if self._latencies
                else 0.0
            ),
            "p99_latency_ms": (
                round(float(np.percentile(self._latencies, 99)), 3)
                if self._latencies
                else 0.0
            ),
        }

    def warm_cache(self, rows: list[dict[str, Any]]) -> int:
        """Pre-warm the cache with common inputs.

        Args:
            rows: Input rows to pre-compute and cache.

        Returns:
            Number of entries cached.
        """
        cached = 0
        for row in rows:
            result = self.compute(row, use_cache=True)
            if result.success:
                cached += 1
        return cached
