"""Tests for real-time computation engine — circuit breaker, caching, batching."""

from __future__ import annotations

import time

import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from forge.serving.realtime import (
    CircuitBreaker,
    CircuitState,
    ComputationEngine,
    TTLCache,
)


class TestCircuitBreaker:
    def test_initial_state_closed(self) -> None:
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED

    def test_allows_requests_when_closed(self) -> None:
        cb = CircuitBreaker()
        assert cb.allow_request()

    def test_opens_after_threshold(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert not cb.allow_request()

    def test_resets_on_success(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_recovery_to_half_open(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_allows_limited_requests(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05, half_open_max_calls=2)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.06)
        assert cb.allow_request()
        assert cb.allow_request()
        assert not cb.allow_request()

    def test_half_open_success_closes(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.06)
        assert cb.allow_request()  # transitions to half-open and allows
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_reset(self) -> None:
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request()

    def test_to_dict(self) -> None:
        cb = CircuitBreaker()
        d = cb.to_dict()
        assert "state" in d
        assert "failure_count" in d


class TestTTLCache:
    def test_put_and_get(self) -> None:
        cache = TTLCache()
        cache.put("key1", {"a": 1})
        assert cache.get("key1") == {"a": 1}

    def test_cache_miss(self) -> None:
        cache = TTLCache()
        assert cache.get("missing") is None

    def test_ttl_expiration(self) -> None:
        cache = TTLCache(ttl_seconds=0.05)
        cache.put("key1", "value")
        time.sleep(0.06)
        assert cache.get("key1") is None

    def test_max_size_eviction(self) -> None:
        cache = TTLCache(max_size=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.put("d", 4)  # should evict "a"
        assert cache.get("a") is None
        assert cache.get("d") == 4

    def test_invalidate(self) -> None:
        cache = TTLCache()
        cache.put("key1", "value")
        cache.invalidate("key1")
        assert cache.get("key1") is None

    def test_clear(self) -> None:
        cache = TTLCache()
        cache.put("a", 1)
        cache.put("b", 2)
        cache.clear()
        assert cache.size == 0

    def test_hit_rate(self) -> None:
        cache = TTLCache()
        cache.put("a", 1)
        cache.get("a")  # hit
        cache.get("b")  # miss
        assert cache.hit_rate == 0.5

    def test_stats(self) -> None:
        cache = TTLCache()
        s = cache.stats()
        assert "size" in s
        assert "hit_rate" in s


class TestComputationEngine:
    @pytest.fixture
    def simple_scaler(self) -> StandardScaler:
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        scaler = StandardScaler()
        scaler.fit(X)
        return scaler

    def test_compute_single(self, simple_scaler: StandardScaler) -> None:
        engine = ComputationEngine(simple_scaler)
        result = engine.compute({"a": 2.0, "b": 5.0})
        assert result.success
        assert isinstance(result.features, dict)
        assert result.latency_ms > 0

    def test_compute_caching(self, simple_scaler: StandardScaler) -> None:
        engine = ComputationEngine(simple_scaler)
        r1 = engine.compute({"a": 2.0, "b": 5.0})
        r2 = engine.compute({"a": 2.0, "b": 5.0})
        assert r1.success
        assert r2.success
        assert r2.from_cache

    def test_compute_no_cache(self, simple_scaler: StandardScaler) -> None:
        engine = ComputationEngine(simple_scaler)
        r1 = engine.compute({"a": 2.0, "b": 5.0}, use_cache=False)
        r2 = engine.compute({"a": 2.0, "b": 5.0}, use_cache=False)
        assert not r1.from_cache
        assert not r2.from_cache

    def test_compute_batch(self, simple_scaler: StandardScaler) -> None:
        engine = ComputationEngine(simple_scaler)
        rows = [{"a": 1.0, "b": 4.0}, {"a": 2.0, "b": 5.0}]
        results = engine.compute_batch(rows)
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_compute_batch_empty(self, simple_scaler: StandardScaler) -> None:
        engine = ComputationEngine(simple_scaler)
        assert engine.compute_batch([]) == []

    def test_circuit_breaker_blocks(self) -> None:
        engine = ComputationEngine(None, circuit_failure_threshold=2)
        # Trigger failures by using None transformer with bad data
        engine._circuit.record_failure()
        engine._circuit.record_failure()
        result = engine.compute({"a": 1.0})
        assert not result.success
        assert "Circuit breaker" in result.error

    def test_health(self, simple_scaler: StandardScaler) -> None:
        engine = ComputationEngine(simple_scaler)
        engine.compute({"a": 2.0, "b": 5.0})
        h = engine.health()
        assert "circuit_breaker" in h
        assert "cache" in h
        assert h["total_requests"] >= 1

    def test_warm_cache(self, simple_scaler: StandardScaler) -> None:
        engine = ComputationEngine(simple_scaler)
        rows = [{"a": 1.0, "b": 4.0}, {"a": 2.0, "b": 5.0}]
        cached = engine.warm_cache(rows)
        assert cached == 2
        # Verify they're cached
        r = engine.compute({"a": 1.0, "b": 4.0})
        assert r.from_cache

    def test_no_transformer(self) -> None:
        engine = ComputationEngine(None)
        result = engine.compute({"a": 1.0, "b": 2.0})
        assert result.success
        assert result.features == {"a": 1.0, "b": 2.0}
