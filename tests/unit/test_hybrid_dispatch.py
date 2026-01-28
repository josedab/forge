"""Tests for hybrid GPU/CPU dispatch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from forge.backends.hybrid import ExecutionStats, HybridConfig, HybridDispatcher


@pytest.fixture
def small_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.standard_normal((100, 5)), columns=[f"f{i}" for i in range(5)])


@pytest.fixture
def large_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.standard_normal((50_000, 10)), columns=[f"f{i}" for i in range(10)])


class TestHybridConfig:
    def test_defaults(self) -> None:
        config = HybridConfig()
        assert config.gpu_threshold_rows == 10_000
        assert config.fallback_to_cpu is True

    def test_custom(self) -> None:
        config = HybridConfig(gpu_threshold_rows=500, prefer_gpu=False)
        assert config.gpu_threshold_rows == 500
        assert config.prefer_gpu is False


class TestHybridDispatcher:
    def test_should_use_gpu_small_data(self, small_df: pd.DataFrame) -> None:
        dispatcher = HybridDispatcher()
        use_gpu, reason = dispatcher.should_use_gpu(small_df)
        # Small data should not use GPU (below threshold)
        if not dispatcher.gpu_available:
            assert use_gpu is False
            assert "not available" in reason
        else:
            assert use_gpu is False
            assert "threshold" in reason.lower()

    def test_should_use_gpu_large_data(self, large_df: pd.DataFrame) -> None:
        config = HybridConfig(gpu_threshold_rows=10_000)
        dispatcher = HybridDispatcher(config=config)
        use_gpu, reason = dispatcher.should_use_gpu(large_df)
        if not dispatcher.gpu_available:
            assert use_gpu is False
        else:
            assert use_gpu is True

    def test_should_use_gpu_prefer_disabled(self, large_df: pd.DataFrame) -> None:
        config = HybridConfig(prefer_gpu=False)
        dispatcher = HybridDispatcher(config=config)
        use_gpu, reason = dispatcher.should_use_gpu(large_df)
        assert use_gpu is False
        if dispatcher.gpu_available:
            assert "not preferred" in reason

    def test_transform_cpu_fallback(self, small_df: pd.DataFrame) -> None:
        dispatcher = HybridDispatcher()
        scaler = StandardScaler()
        scaler.fit(small_df)

        result, stats = dispatcher.transform(small_df, scaler)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == small_df.shape
        assert stats.device_used == "cpu"
        assert stats.rows_processed == len(small_df)
        assert stats.latency_ms > 0

    def test_transform_preserves_values(self, small_df: pd.DataFrame) -> None:
        dispatcher = HybridDispatcher()
        scaler = StandardScaler()
        scaler.fit(small_df)

        result, _ = dispatcher.transform(small_df, scaler)
        expected = scaler.transform(small_df)
        np.testing.assert_array_almost_equal(result.values, expected)

    def test_execution_summary_empty(self) -> None:
        dispatcher = HybridDispatcher()
        summary = dispatcher.get_execution_summary()
        assert summary["total_executions"] == 0

    def test_execution_summary_after_transforms(self, small_df: pd.DataFrame) -> None:
        dispatcher = HybridDispatcher()
        scaler = StandardScaler()
        scaler.fit(small_df)

        dispatcher.transform(small_df, scaler)
        dispatcher.transform(small_df, scaler)

        summary = dispatcher.get_execution_summary()
        assert summary["total_executions"] == 2
        assert summary["cpu_executions"] == 2
        assert summary["avg_latency_ms"] > 0

    def test_memory_limit(self, large_df: pd.DataFrame) -> None:
        config = HybridConfig(max_gpu_memory_mb=1)  # 1MB limit
        dispatcher = HybridDispatcher(config=config)
        use_gpu, reason = dispatcher.should_use_gpu(large_df)
        if dispatcher.gpu_available:
            assert use_gpu is False
            assert "memory" in reason.lower()


class TestExecutionStats:
    def test_defaults(self) -> None:
        stats = ExecutionStats()
        assert stats.device_used == "cpu"
        assert stats.latency_ms == 0.0
        assert stats.fallback_occurred is False

    def test_custom(self) -> None:
        stats = ExecutionStats(
            device_used="gpu",
            latency_ms=5.3,
            rows_processed=10000,
            reason="Data size suitable for GPU",
        )
        assert stats.device_used == "gpu"
        assert stats.rows_processed == 10000
