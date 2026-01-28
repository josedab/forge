"""Tests for compute backends."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.backends import ComputeDevice, CPUBackend, GPUBackend, get_backend, set_backend


@pytest.fixture
def cpu():
    return CPUBackend()


@pytest.fixture
def series_a():
    return pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def series_b():
    return pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])


class TestCPUBackend:
    def test_device(self, cpu):
        assert cpu.device == ComputeDevice.CPU

    def test_is_available(self, cpu):
        assert cpu.is_available()

    def test_add(self, cpu, series_a, series_b):
        result = cpu.add(series_a, series_b)
        assert list(result) == [6.0, 6.0, 6.0, 6.0, 6.0]

    def test_multiply(self, cpu, series_a, series_b):
        result = cpu.multiply(series_a, series_b)
        assert result.iloc[0] == pytest.approx(5.0)

    def test_divide(self, cpu, series_a, series_b):
        result = cpu.divide(series_a, series_b)
        assert result.iloc[0] == pytest.approx(0.2)

    def test_divide_by_zero(self, cpu):
        a = pd.Series([1.0, 2.0])
        b = pd.Series([0.0, 2.0])
        result = cpu.divide(a, b)
        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == pytest.approx(1.0)

    def test_power(self, cpu, series_a):
        result = cpu.power(series_a, 2)
        assert result.iloc[2] == pytest.approx(9.0)

    def test_log(self, cpu):
        s = pd.Series([1.0, np.e, np.e**2])
        result = cpu.log(s)
        assert result.iloc[0] == pytest.approx(0.0)
        assert result.iloc[1] == pytest.approx(1.0)

    def test_sqrt(self, cpu):
        s = pd.Series([4.0, 9.0, 16.0])
        result = cpu.sqrt(s)
        assert result.iloc[0] == pytest.approx(2.0)

    def test_group_agg(self, cpu):
        df = pd.DataFrame({"g": ["a", "a", "b", "b"], "v": [1.0, 3.0, 2.0, 4.0]})
        result = cpu.group_agg(df, "g", "v", "mean")
        assert result.iloc[0] == pytest.approx(2.0)
        assert result.iloc[2] == pytest.approx(3.0)

    def test_rolling_agg(self, cpu, series_a):
        result = cpu.rolling_agg(series_a, window=3, agg_func="mean")
        assert result.iloc[2] == pytest.approx(2.0)

    def test_one_hot_encode(self, cpu):
        s = pd.Series(["a", "b", "a", "c"], name="cat")
        result = cpu.one_hot_encode(s)
        assert result.shape[1] == 3
        assert result["cat_a"].iloc[0] == 1

    def test_ordinal_encode(self, cpu):
        s = pd.Series(["low", "mid", "high", "mid"])
        result = cpu.ordinal_encode(s)
        assert result.nunique() == 3

    def test_polynomial_features(self, cpu):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        result = cpu.polynomial_features(df, degree=2)
        assert "x^2" in result.columns
        assert "x*y" in result.columns

    def test_interaction_features(self, cpu):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
        result = cpu.interaction_features(df, ["a", "b", "c"])
        assert "a*b" in result.columns
        assert "a*c" in result.columns
        assert "b*c" in result.columns

    def test_quantile_bin(self, cpu):
        s = pd.Series(range(100), dtype=float)
        result = cpu.quantile_bin(s, n_bins=4)
        assert result.nunique() == 4

    def test_to_frame_passthrough(self, cpu):
        df = pd.DataFrame({"x": [1, 2]})
        assert cpu.to_frame(df) is df

    def test_to_frame_from_dict(self, cpu):
        result = cpu.to_frame({"x": [1, 2]})
        assert isinstance(result, pd.DataFrame)


class TestGPUBackend:
    def test_device(self):
        gpu = GPUBackend()
        assert gpu.device == ComputeDevice.GPU

    def test_fallback_add(self):
        gpu = GPUBackend()
        a = pd.Series([1.0, 2.0])
        b = pd.Series([3.0, 4.0])
        result = gpu.add(a, b)
        assert list(result) == [4.0, 6.0]

    def test_fallback_multiply(self):
        gpu = GPUBackend()
        a = pd.Series([2.0, 3.0])
        b = pd.Series([4.0, 5.0])
        result = gpu.multiply(a, b)
        assert result.iloc[0] == pytest.approx(8.0)

    def test_fallback_polynomial(self):
        gpu = GPUBackend()
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        result = gpu.polynomial_features(df, degree=2)
        assert "x^2" in result.columns


class TestDispatch:
    def test_auto_fallback_to_cpu(self):
        backend = get_backend("auto")
        # On systems without RAPIDS, should fall back to CPU
        assert backend.is_available()

    def test_explicit_cpu(self):
        backend = get_backend("cpu")
        assert isinstance(backend, CPUBackend)

    def test_set_backend(self):
        cpu = CPUBackend()
        set_backend(cpu)
        from forge.backends.dispatch import current_backend
        assert current_backend() is cpu
