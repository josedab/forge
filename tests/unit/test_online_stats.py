"""Tests for online statistics engine and stream connectors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.streaming.online import (
    AdaptiveFeatureEngine,
    ExponentialMovingAverage,
    FrequencyCounter,
    InMemoryStreamConnector,
    OnlineStatistics,
    TDigest,
    WelfordState,
)


class TestWelfordState:
    def test_basic_stats(self):
        state = WelfordState()
        for v in [1, 2, 3, 4, 5]:
            state.update(float(v))
        assert state.count == 5
        assert state.mean == pytest.approx(3.0)
        assert state.min_val == 1.0
        assert state.max_val == 5.0
        assert state.variance > 0

    def test_nan_ignored(self):
        state = WelfordState()
        state.update(1.0)
        state.update(float("nan"))
        assert state.count == 1

    def test_empty_state(self):
        state = WelfordState()
        assert state.count == 0
        assert state.variance == 0.0
        assert state.std == 0.0


class TestTDigest:
    def test_quantile_basic(self):
        td = TDigest()
        for v in range(1, 101):
            td.update(float(v))
        assert td.count == 100
        assert td.quantile(0.5) == pytest.approx(50.0, abs=5)
        assert td.quantile(0.0) <= 2
        assert td.quantile(1.0) >= 99

    def test_empty_quantile(self):
        td = TDigest()
        assert td.quantile(0.5) == 0.0

    def test_nan_ignored(self):
        td = TDigest()
        td.update(float("nan"))
        assert td.count == 0


class TestExponentialMovingAverage:
    def test_basic(self):
        ema = ExponentialMovingAverage(alpha=0.5)
        ema.update(10.0)
        assert ema.value == 10.0
        ema.update(20.0)
        assert ema.value == pytest.approx(15.0)
        assert ema.count == 2

    def test_empty(self):
        ema = ExponentialMovingAverage(alpha=0.1)
        assert ema.value == 0.0
        assert ema.count == 0

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            ExponentialMovingAverage(alpha=0.0)
        with pytest.raises(ValueError):
            ExponentialMovingAverage(alpha=1.5)


class TestFrequencyCounter:
    def test_basic(self):
        fc = FrequencyCounter()
        for v in ["a", "b", "a", "c", "a"]:
            fc.update(v)
        assert fc.frequency("a") == pytest.approx(0.6)
        assert fc.n_unique == 3
        assert fc.total == 5

    def test_max_categories(self):
        fc = FrequencyCounter(max_categories=3)
        for v in ["a", "b", "c", "d", "e"]:
            fc.update(v)
        assert fc.n_unique <= 3

    def test_top_k(self):
        fc = FrequencyCounter()
        for v in ["a", "a", "b"]:
            fc.update(v)
        top = fc.top_k
        assert top[0][0] == "a"


class TestOnlineStatistics:
    def test_update_dict(self):
        stats = OnlineStatistics(columns=["x", "y"])
        stats.update({"x": 10.0, "y": 20.0})
        stats.update({"x": 20.0, "y": 30.0})
        assert stats.mean("x") == pytest.approx(15.0)
        assert stats.n_rows == 2

    def test_update_series(self):
        stats = OnlineStatistics(columns=["a"])
        stats.update(pd.Series({"a": 5.0}))
        assert stats.mean("a") == pytest.approx(5.0)

    def test_update_batch(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        stats = OnlineStatistics(columns=["x"])
        stats.update_batch(df)
        assert stats.mean("x") == pytest.approx(3.0)
        assert stats.n_rows == 5

    def test_variance_std(self):
        stats = OnlineStatistics(columns=["x"])
        for v in [2, 4, 4, 4, 5, 5, 7, 9]:
            stats.update({"x": float(v)})
        assert stats.variance("x") > 0
        assert stats.std("x") > 0

    def test_min_max(self):
        stats = OnlineStatistics(columns=["x"])
        for v in [3, 1, 5, 2]:
            stats.update({"x": float(v)})
        assert stats.min("x") == 1.0
        assert stats.max("x") == 5.0

    def test_quantile(self):
        stats = OnlineStatistics(columns=["x"])
        for v in range(1, 101):
            stats.update({"x": float(v)})
        q50 = stats.quantile("x", 0.5)
        assert 40 < q50 < 60

    def test_ema(self):
        stats = OnlineStatistics(columns=["x"], ema_alpha=0.5)
        stats.update({"x": 10.0})
        stats.update({"x": 20.0})
        assert stats.ema("x") == pytest.approx(15.0)

    def test_frequency(self):
        stats = OnlineStatistics(columns=["cat"])
        stats.update({"cat": "a"})
        stats.update({"cat": "a"})
        stats.update({"cat": "b"})
        assert stats.frequency("cat", "a") == pytest.approx(2 / 3)

    def test_summary(self):
        stats = OnlineStatistics(columns=["x"])
        for v in [1, 2, 3, 4, 5]:
            stats.update({"x": float(v)})
        s = stats.summary("x")
        assert "mean" in s
        assert "std" in s
        assert "q50" in s
        assert s["count"] == 5

    def test_tracked_columns(self):
        stats = OnlineStatistics(columns=["a", "b"])
        stats.update({"a": 1.0, "b": 2.0})
        assert stats.tracked_columns == ["a", "b"]

    def test_auto_detect_columns(self):
        stats = OnlineStatistics()
        stats.update({"x": 1.0, "y": 2.0})
        assert "x" in stats.tracked_columns

    def test_missing_column_defaults(self):
        stats = OnlineStatistics()
        assert stats.mean("nonexistent") == 0.0
        assert stats.variance("nonexistent") == 0.0
        assert stats.ema("nonexistent") == 0.0


class TestInMemoryStreamConnector:
    def test_connect_consume(self):
        conn = InMemoryStreamConnector()
        conn.connect()
        assert conn.is_connected

        conn.push({"x": 1.0})
        conn.push({"x": 2.0})
        assert conn.buffer_size == 2

        events = conn.consume()
        assert len(events) == 2
        assert conn.buffer_size == 0

    def test_close(self):
        conn = InMemoryStreamConnector()
        conn.connect()
        conn.push({"x": 1.0})
        conn.close()
        assert not conn.is_connected
        assert conn.buffer_size == 0


class TestAdaptiveFeatureEngine:
    def test_no_drift_stable_data(self):
        stats = OnlineStatistics(columns=["x"])
        for _ in range(100):
            stats.update({"x": float(np.random.normal(0, 1))})

        engine = AdaptiveFeatureEngine(stats=stats, drift_threshold=5.0)
        flags = engine.process({"x": 0.5})
        assert engine.n_processed == 1

    def test_drift_detected(self):
        stats = OnlineStatistics(columns=["x"])
        for _ in range(200):
            stats.update({"x": float(np.random.normal(0, 1))})

        engine = AdaptiveFeatureEngine(stats=stats, drift_threshold=2.0, window_size=20)

        # Feed many shifted values to fill the window
        for _ in range(25):
            engine.process({"x": 100.0})

        assert len(engine.drift_events) > 0

    def test_drift_callback(self):
        stats = OnlineStatistics(columns=["x"])
        for _ in range(200):
            stats.update({"x": float(np.random.normal(0, 1))})

        callbacks: list[tuple[str, float]] = []
        engine = AdaptiveFeatureEngine(
            stats=stats,
            drift_threshold=2.0,
            window_size=20,
            on_drift=lambda col, z: callbacks.append((col, z)),
        )

        for _ in range(25):
            engine.process({"x": 100.0})

        assert len(callbacks) > 0
        assert callbacks[0][0] == "x"

    def test_handles_nan(self):
        engine = AdaptiveFeatureEngine()
        flags = engine.process({"x": float("nan")})
        assert len(flags) == 0
