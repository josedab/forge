"""Tests for the streaming feature engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.streaming import (
    IncrementalAggregator,
    IncrementalInteractionGenerator,
    StreamingEngine,
    StreamingWindowAggregator,
)
from forge.streaming.window import SlidingWindow, TumblingWindow


class TestSlidingWindow:
    def test_basic_add_and_stats(self):
        window = SlidingWindow(size=5, min_periods=1)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            window.add(v)
        assert window.count() == 5
        assert window.mean() == pytest.approx(3.0)
        assert window.min() == pytest.approx(1.0)
        assert window.max() == pytest.approx(5.0)
        assert window.sum() == pytest.approx(15.0)

    def test_sliding_eviction(self):
        window = SlidingWindow(size=3)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            window.add(v)
        assert window.count() == 3
        assert window.mean() == pytest.approx(4.0)
        assert set(window.values()) == {3.0, 4.0, 5.0}

    def test_std_with_few_values(self):
        window = SlidingWindow(size=10)
        window.add(5.0)
        assert np.isnan(window.std())
        window.add(10.0)
        assert not np.isnan(window.std())

    def test_reset(self):
        window = SlidingWindow(size=5)
        window.add(1.0)
        window.add(2.0)
        window.reset()
        assert window.count() == 0
        assert np.isnan(window.mean())

    def test_nan_values_ignored(self):
        window = SlidingWindow(size=5)
        window.add(1.0)
        window.add(np.nan)
        window.add(3.0)
        assert window.count() == 2


class TestTumblingWindow:
    def test_basic_operations(self):
        window = TumblingWindow(size=3)
        window.add(1.0)
        window.add(2.0)
        assert not window.is_full
        window.add(3.0)
        assert window.is_full
        assert window.mean() == pytest.approx(2.0)

    def test_flush(self):
        window = TumblingWindow(size=2)
        window.add(10.0)
        window.add(20.0)
        values = window.flush()
        assert values == [10.0, 20.0]
        assert window.count() == 0


class TestIncrementalAggregator:
    def test_running_stats(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [5.0, 4.0, 3.0, 2.0, 1.0]})
        agg = IncrementalAggregator(columns=["a", "b"], stats=["mean", "std", "min", "max"])
        agg.partial_fit(df)
        result = agg.partial_transform(df)

        assert "a_running_mean" in result.columns
        assert result["a_running_mean"].iloc[0] == pytest.approx(3.0)
        assert result["a_running_min"].iloc[0] == pytest.approx(1.0)
        assert result["a_running_max"].iloc[0] == pytest.approx(5.0)

    def test_incremental_updates(self):
        agg = IncrementalAggregator(columns=["x"], stats=["mean", "count"])
        batch1 = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        batch2 = pd.DataFrame({"x": [4.0, 5.0, 6.0]})

        agg.partial_fit(batch1)
        result1 = agg.partial_transform(batch1)
        assert result1["x_running_mean"].iloc[0] == pytest.approx(2.0)
        assert result1["x_running_count"].iloc[0] == pytest.approx(3.0)

        agg.partial_fit(batch2)
        result2 = agg.partial_transform(batch2)
        assert result2["x_running_mean"].iloc[0] == pytest.approx(3.5)
        assert result2["x_running_count"].iloc[0] == pytest.approx(6.0)

    def test_auto_detect_numeric_columns(self):
        df = pd.DataFrame({"num": [1.0, 2.0], "cat": ["a", "b"]})
        agg = IncrementalAggregator(stats=["mean"])
        agg.partial_fit(df)
        result = agg.partial_transform(df)
        assert "num_running_mean" in result.columns
        assert "cat_running_mean" not in result.columns

    def test_reset(self):
        agg = IncrementalAggregator(columns=["x"], stats=["mean"])
        agg.partial_fit(pd.DataFrame({"x": [1.0, 2.0]}))
        agg.reset()
        assert not agg._is_fitted


class TestStreamingWindowAggregator:
    def test_windowed_stats(self):
        df = pd.DataFrame({"a": list(range(1, 11))})
        agg = StreamingWindowAggregator(
            columns=["a"], window_size=5, stats=["mean", "min", "max"]
        )
        agg.partial_fit(df)
        result = agg.partial_transform(df)

        assert "a_window5_mean" in result.columns
        assert "a_window5_min" in result.columns
        assert result["a_window5_mean"].iloc[0] == pytest.approx(8.0, abs=1.0)


class TestIncrementalInteractionGenerator:
    def test_interactions(self):
        df = pd.DataFrame({"a": [2.0, 3.0], "b": [4.0, 5.0]})
        gen = IncrementalInteractionGenerator(
            columns=["a", "b"], operations=["multiply", "ratio"]
        )
        gen.partial_fit(df)
        result = gen.partial_transform(df)
        assert "a_multiply_b" in result.columns
        assert result["a_multiply_b"].iloc[0] == pytest.approx(8.0)
        assert result["a_ratio_b"].iloc[0] == pytest.approx(0.5)


class TestStreamingEngine:
    def test_process_single_batch(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        engine = StreamingEngine(
            generators=[IncrementalAggregator(stats=["mean"])],
            passthrough=True,
        )
        result = engine.process(df)
        assert "x" in result.columns
        assert "x_running_mean" in result.columns
        assert engine._n_batches_processed == 1

    def test_process_stream(self):
        batches = [
            pd.DataFrame({"x": [1.0, 2.0]}),
            pd.DataFrame({"x": [3.0, 4.0]}),
        ]
        engine = StreamingEngine(
            generators=[IncrementalAggregator(columns=["x"], stats=["mean"])],
            passthrough=False,
        )
        results = list(engine.process_stream(iter(batches)))
        assert len(results) == 2
        assert engine._n_events_processed == 4

    def test_reset(self):
        engine = StreamingEngine()
        engine.process(pd.DataFrame({"x": [1.0]}))
        engine.reset()
        assert engine._n_batches_processed == 0

    def test_add_generator(self):
        engine = StreamingEngine(generators=[])
        engine.add_generator(IncrementalAggregator(stats=["mean"]))
        assert len(engine.generators) == 1
