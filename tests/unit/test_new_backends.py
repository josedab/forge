"""Tests for Polars and DuckDB compute backends."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.backends.base import ComputeDevice
from forge.backends.dispatch import get_backend
from forge.backends.duckdb_backend import _DUCKDB_AVAILABLE, DuckDBBackend
from forge.backends.polars_backend import _POLARS_AVAILABLE, PolarsBackend


@pytest.fixture
def series_a():
    return pd.Series([1.0, 2.0, 3.0, 4.0], name="a")


@pytest.fixture
def series_b():
    return pd.Series([2.0, 0.0, 4.0, 1.0], name="b")


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [10.0, 20.0, 30.0, 40.0],
        "g": ["a", "b", "a", "b"],
    })


class TestPolarsBackend:
    @pytest.fixture(autouse=True)
    def _skip_if_no_polars(self):
        if not _POLARS_AVAILABLE:
            pytest.skip("Polars not installed")

    def test_is_available(self):
        b = PolarsBackend()
        assert b.is_available()

    def test_device(self):
        b = PolarsBackend()
        assert b.device == ComputeDevice.CPU

    def test_add(self, series_a, series_b):
        b = PolarsBackend()
        result = b.add(series_a, series_b)
        assert list(result) == [3.0, 2.0, 7.0, 5.0]

    def test_multiply(self, series_a, series_b):
        b = PolarsBackend()
        result = b.multiply(series_a, series_b)
        assert list(result) == [2.0, 0.0, 12.0, 4.0]

    def test_divide_safe(self, series_a, series_b):
        b = PolarsBackend()
        result = b.divide(series_a, series_b)
        assert result.iloc[0] == pytest.approx(0.5)
        assert pd.isna(result.iloc[1])  # 2/0 = NaN

    def test_power(self, series_a):
        b = PolarsBackend()
        result = b.power(series_a, 2.0)
        assert list(result) == [1.0, 4.0, 9.0, 16.0]

    def test_log(self, series_a):
        b = PolarsBackend()
        result = b.log(series_a)
        assert result.iloc[0] == pytest.approx(0.0)
        assert result.iloc[2] == pytest.approx(np.log(3.0))

    def test_sqrt(self, series_a):
        b = PolarsBackend()
        result = b.sqrt(series_a)
        assert result.iloc[0] == pytest.approx(1.0)
        assert result.iloc[3] == pytest.approx(2.0)

    def test_polynomial_features(self, sample_df):
        b = PolarsBackend()
        result = b.polynomial_features(sample_df, degree=2)
        assert "x^2" in result.columns
        assert "x*y" in result.columns

    def test_interaction_features(self, sample_df):
        b = PolarsBackend()
        result = b.interaction_features(sample_df, ["x", "y"])
        assert "x*y" in result.columns
        assert result["x*y"].iloc[0] == pytest.approx(10.0)

    def test_to_frame(self):
        b = PolarsBackend()
        df = pd.DataFrame({"a": [1, 2]})
        result = b.to_frame(df)
        assert isinstance(result, pd.DataFrame)

    def test_one_hot_encode(self):
        b = PolarsBackend()
        series = pd.Series(["a", "b", "a", "c"], name="cat")
        result = b.one_hot_encode(series)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 3

    def test_rolling_agg(self, series_a):
        b = PolarsBackend()
        result = b.rolling_agg(series_a, window=2, agg_func="mean")
        assert len(result) == 4
        assert result.iloc[1] == pytest.approx(1.5)


class TestDuckDBBackend:
    @pytest.fixture(autouse=True)
    def _skip_if_no_duckdb(self):
        if not _DUCKDB_AVAILABLE:
            pytest.skip("DuckDB not installed")

    def test_is_available(self):
        b = DuckDBBackend()
        assert b.is_available()

    def test_add(self, series_a, series_b):
        b = DuckDBBackend()
        result = b.add(series_a, series_b)
        assert list(result) == [3.0, 2.0, 7.0, 5.0]

    def test_multiply(self, series_a, series_b):
        b = DuckDBBackend()
        result = b.multiply(series_a, series_b)
        assert list(result) == [2.0, 0.0, 12.0, 4.0]

    def test_divide_safe(self, series_a, series_b):
        b = DuckDBBackend()
        result = b.divide(series_a, series_b)
        assert result.iloc[0] == pytest.approx(0.5)
        assert pd.isna(result.iloc[1])

    def test_power(self, series_a):
        b = DuckDBBackend()
        result = b.power(series_a, 2.0)
        np.testing.assert_allclose(result.values, [1.0, 4.0, 9.0, 16.0])

    def test_log(self, series_a):
        b = DuckDBBackend()
        result = b.log(series_a)
        assert result.iloc[0] == pytest.approx(0.0, abs=1e-6)

    def test_sqrt(self, series_a):
        b = DuckDBBackend()
        result = b.sqrt(series_a)
        assert result.iloc[3] == pytest.approx(2.0)

    def test_group_agg(self, sample_df):
        b = DuckDBBackend()
        result = b.group_agg(sample_df, "g", "x", "mean")
        assert len(result) == 4
        # Group "a" has x=[1,3], mean=2
        assert result.iloc[0] == pytest.approx(2.0)

    def test_rolling_agg(self, series_a):
        b = DuckDBBackend()
        result = b.rolling_agg(series_a, window=2, agg_func="mean")
        assert len(result) == 4

    def test_polynomial_features(self, sample_df):
        b = DuckDBBackend()
        result = b.polynomial_features(sample_df, degree=2)
        assert "x^2" in result.columns
        assert "x*y" in result.columns

    def test_interaction_features(self, sample_df):
        b = DuckDBBackend()
        result = b.interaction_features(sample_df, ["x", "y"])
        assert "x*y" in result.columns

    def test_close(self):
        b = DuckDBBackend()
        b._get_conn()
        b.close()
        assert b._conn is None


class TestDispatch:
    def test_get_polars_backend(self):
        if not _POLARS_AVAILABLE:
            pytest.skip("Polars not installed")
        b = get_backend("polars")
        assert isinstance(b, PolarsBackend)

    def test_get_duckdb_backend(self):
        if not _DUCKDB_AVAILABLE:
            pytest.skip("DuckDB not installed")
        b = get_backend("duckdb")
        assert isinstance(b, DuckDBBackend)

    def test_get_cpu_backend(self):
        from forge.backends.cpu import CPUBackend
        b = get_backend("cpu")
        assert isinstance(b, CPUBackend)
