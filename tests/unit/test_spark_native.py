"""Tests for Spark native backend (pandas fallback mode)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.backends.spark_native import SparkNativeBackend


@pytest.fixture
def backend():
    return SparkNativeBackend()


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        "cat": ["x", "y", "x", "y", "x"],
    })


class TestSparkNativeBackend:
    def test_device(self, backend):
        from forge.backends.base import ComputeDevice
        assert backend.device == ComputeDevice.AUTO

    def test_add(self, backend, sample_df):
        result = backend.add(sample_df["a"], sample_df["b"])
        assert result.iloc[0] == pytest.approx(11.0)

    def test_multiply(self, backend, sample_df):
        result = backend.multiply(sample_df["a"], sample_df["b"])
        assert result.iloc[0] == pytest.approx(10.0)

    def test_divide(self, backend, sample_df):
        result = backend.divide(sample_df["a"], sample_df["b"])
        assert result.iloc[0] == pytest.approx(0.1)

    def test_divide_by_zero(self, backend):
        a = pd.Series([1.0, 2.0])
        b = pd.Series([0.0, 1.0])
        result = backend.divide(a, b)
        assert np.isnan(result.iloc[0])

    def test_power(self, backend, sample_df):
        result = backend.power(sample_df["a"], 2.0)
        assert result.iloc[2] == pytest.approx(9.0)

    def test_log(self, backend, sample_df):
        result = backend.log(sample_df["a"])
        assert result.iloc[0] > 0

    def test_sqrt(self, backend, sample_df):
        result = backend.sqrt(sample_df["a"])
        assert result.iloc[3] == pytest.approx(2.0)

    def test_group_agg(self, backend, sample_df):
        result = backend.group_agg(sample_df, "cat", "a", "mean")
        assert len(result) == 2

    def test_rolling_agg(self, backend, sample_df):
        result = backend.rolling_agg(sample_df["a"], window=2, agg_func="mean")
        assert len(result) == 5

    def test_one_hot_encode(self, backend, sample_df):
        result = backend.one_hot_encode(sample_df["cat"])
        assert result.shape[1] == 2

    def test_ordinal_encode(self, backend, sample_df):
        result = backend.ordinal_encode(sample_df["cat"])
        assert set(result.dropna().unique()) == {0, 1}

    def test_polynomial_features(self, backend, sample_df):
        result = backend.polynomial_features(sample_df[["a", "b"]], degree=2)
        assert "a^2" in result.columns

    def test_interaction_features(self, backend, sample_df):
        result = backend.interaction_features(sample_df, columns=["a", "b"])
        assert "a_x_b" in result.columns

    def test_quantile_bin(self, backend, sample_df):
        result = backend.quantile_bin(sample_df["a"], n_bins=3)
        assert len(result) == 5

    def test_to_frame(self, backend):
        result = backend.to_frame({"x": [1, 2], "y": [3, 4]})
        assert isinstance(result, pd.DataFrame)

    def test_to_pandas(self, backend, sample_df):
        result = backend.to_pandas(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_materialize_to_delta_fallback(self, backend, sample_df, tmp_path):
        path = str(tmp_path / "features.parquet")
        result = backend.materialize_to_delta(sample_df[["a", "b"]], path)
        assert result["rows"] == 5
