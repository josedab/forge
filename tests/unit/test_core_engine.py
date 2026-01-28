"""Tests for high-performance core engine."""

from __future__ import annotations

import numpy as np
import pytest

from forge.core_engine import (
    Engine,
    fast_interaction,
    fast_log_transform,
    fast_null_indicator,
    fast_one_hot,
    fast_polynomial_features,
    fast_quantile_bin,
    fast_rolling_stats,
    fast_target_encode,
)


class TestFastOneHot:
    def test_basic(self):
        values = np.array(["a", "b", "c", "a"])
        result = fast_one_hot(values)
        assert result.shape == (4, 3)
        assert result[0, 0] == 1.0
        assert result[1, 1] == 1.0
        assert result[3, 0] == 1.0  # Same as first

    def test_with_categories(self):
        values = np.array(["a", "b"])
        cats = np.array(["a", "b", "c"])
        result = fast_one_hot(values, cats)
        assert result.shape == (2, 3)
        assert result[0, 2] == 0.0  # "c" not present

    def test_unknown_category(self):
        values = np.array(["a", "x"])
        cats = np.array(["a", "b"])
        result = fast_one_hot(values, cats)
        assert result[1, :].sum() == 0.0  # "x" not in categories


class TestFastTargetEncode:
    def test_basic(self):
        values = np.array(["a", "a", "b", "b"])
        targets = np.array([1.0, 1.0, 0.0, 0.0])
        encoded, mapping = fast_target_encode(values, targets, smoothing=0.0001)
        assert encoded[0] > encoded[2]  # "a" has higher target mean

    def test_smoothing(self):
        values = np.array(["a", "b"])
        targets = np.array([1.0, 0.0])
        _, mapping_low = fast_target_encode(values, targets, smoothing=0.001)
        _, mapping_high = fast_target_encode(values, targets, smoothing=100.0)
        # Higher smoothing → closer to global mean
        global_mean = 0.5
        assert abs(mapping_high["a"] - global_mean) < abs(mapping_low["a"] - global_mean)


class TestFastInteraction:
    def test_multiply(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = fast_interaction(a, b, "multiply")
        np.testing.assert_array_equal(result, [4.0, 10.0, 18.0])

    def test_add(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        np.testing.assert_array_equal(fast_interaction(a, b, "add"), [4.0, 6.0])

    def test_divide_by_zero(self):
        a = np.array([1.0, 2.0])
        b = np.array([0.0, 2.0])
        result = fast_interaction(a, b, "divide")
        assert result[0] == 0.0
        assert result[1] == 1.0

    def test_unknown_op(self):
        with pytest.raises(ValueError, match="Unknown"):
            fast_interaction(np.array([1.0]), np.array([2.0]), "power")


class TestFastQuantileBin:
    def test_basic(self):
        values = np.arange(100, dtype=float)
        bins, edges = fast_quantile_bin(values, n_bins=4)
        assert len(np.unique(bins)) == 4
        assert len(edges) == 5

    def test_uniform(self):
        rng = np.random.RandomState(42)
        values = rng.uniform(0, 1, 1000)
        bins, _ = fast_quantile_bin(values, n_bins=10)
        counts = np.bincount(bins, minlength=10)
        assert all(c > 50 for c in counts)  # Roughly equal


class TestFastRollingStats:
    def test_mean(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = fast_rolling_stats(values, window=3, stats=["mean"])
        assert np.isnan(result["mean"][0])
        assert result["mean"][2] == pytest.approx(2.0)
        assert result["mean"][4] == pytest.approx(4.0)

    def test_std(self):
        values = np.array([1.0, 1.0, 1.0, 1.0])
        result = fast_rolling_stats(values, window=2, stats=["std"])
        assert result["std"][1] == pytest.approx(0.0)

    def test_min_max(self):
        values = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        result = fast_rolling_stats(values, window=3, stats=["min", "max"])
        assert result["min"][2] == 1.0
        assert result["max"][2] == 4.0


class TestFastPolynomialFeatures:
    def test_degree2(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = fast_polynomial_features(X, degree=2)
        # original (2) + squared (2) + interaction (1) = 5
        assert result.shape == (2, 5)

    def test_interaction_only(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = fast_polynomial_features(X, degree=2, interaction_only=True)
        # original (2) + interaction (1) = 3
        assert result.shape == (2, 3)
        assert result[0, 2] == 2.0  # 1*2
        assert result[1, 2] == 12.0  # 3*4


class TestFastLogTransform:
    def test_basic(self):
        values = np.array([0.0, 1.0, 9.0])
        result = fast_log_transform(values, offset=1.0)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(np.log(2.0))

    def test_negative_values(self):
        values = np.array([-10.0])
        result = fast_log_transform(values, offset=1.0)
        assert np.isfinite(result[0])  # Should not error


class TestFastNullIndicator:
    def test_basic(self):
        values = np.array([1.0, np.nan, 3.0, np.nan])
        result = fast_null_indicator(values)
        np.testing.assert_array_equal(result, [0.0, 1.0, 0.0, 1.0])


class TestEngine:
    def test_active_backend(self):
        engine = Engine()
        assert engine.active_backend == "numpy"

    def test_one_hot(self):
        engine = Engine()
        result = engine.one_hot(np.array(["x", "y", "x"]))
        assert result.shape == (3, 2)

    def test_interaction_features(self):
        engine = Engine()
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = engine.interaction_features(X, [(0, 1, "multiply"), (1, 2, "add")])
        assert result.shape == (2, 2)
        assert result[0, 0] == 2.0
        assert result[0, 1] == 5.0

    def test_batch_transform_log(self):
        engine = Engine()
        X = np.array([[0.0, 1.0], [1.0, 2.0]])
        result = engine.batch_transform(X, [{"op": "log", "col": 0}])
        assert result.shape == (2, 3)  # original 2 + 1 log

    def test_batch_transform_interaction(self):
        engine = Engine()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = engine.batch_transform(X, [{"op": "interaction", "cols": (0, 1)}])
        assert result.shape == (2, 3)
        assert result[0, 2] == 2.0

    def test_batch_unknown_op(self):
        engine = Engine()
        X = np.array([[1.0]])
        with pytest.raises(ValueError, match="Unknown"):
            engine.batch_transform(X, [{"op": "foobar"}])

    def test_rust_fallback(self):
        engine = Engine(backend="rust")
        assert engine.active_backend == "numpy"  # Falls back gracefully
