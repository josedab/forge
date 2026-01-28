"""Tests for Polars backend: eager operations, lazy pipeline, and streaming."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.backends.polars_backend import PolarsBackend

# Skip all tests if Polars is not installed
polars = pytest.importorskip("polars")

from forge.backends.polars_lazy import (
    PolarsFeatureGenerator,
    PolarsLazyPipeline,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "price": np.random.uniform(10, 100, 50),
        "quantity": np.random.randint(1, 20, 50),
        "revenue": np.random.uniform(100, 5000, 50),
        "category": np.random.choice(["A", "B", "C"], 50),
    })


class TestPolarsBackendEager:
    """Tests for the existing eager Polars backend."""

    def test_is_available(self) -> None:
        backend = PolarsBackend()
        assert backend.is_available()

    def test_add(self, sample_df: pd.DataFrame) -> None:
        backend = PolarsBackend()
        result = backend.add(sample_df["price"], sample_df["quantity"])
        expected = sample_df["price"] + sample_df["quantity"]
        np.testing.assert_array_almost_equal(result.values, expected.values)

    def test_multiply(self, sample_df: pd.DataFrame) -> None:
        backend = PolarsBackend()
        result = backend.multiply(sample_df["price"], sample_df["quantity"])
        expected = sample_df["price"] * sample_df["quantity"]
        np.testing.assert_array_almost_equal(result.values, expected.values)

    def test_divide(self, sample_df: pd.DataFrame) -> None:
        backend = PolarsBackend()
        # Use float columns to avoid Polars type mismatch
        result = backend.divide(sample_df["price"], sample_df["revenue"])
        assert len(result) == 50

    def test_divide_by_zero(self) -> None:
        backend = PolarsBackend()
        a = pd.Series([10.0, 20.0, 30.0])
        b = pd.Series([2.0, 0.0, 5.0])
        result = backend.divide(a, b)
        assert result.iloc[0] == 5.0
        # Division by zero should produce NaN
        assert pd.isna(result.iloc[1])

    def test_power(self, sample_df: pd.DataFrame) -> None:
        backend = PolarsBackend()
        result = backend.power(sample_df["price"], 2.0)
        expected = sample_df["price"] ** 2
        np.testing.assert_array_almost_equal(result.values, expected.values)

    def test_log(self, sample_df: pd.DataFrame) -> None:
        backend = PolarsBackend()
        result = backend.log(sample_df["price"])
        assert len(result) == 50
        assert not result.isna().any()

    def test_sqrt(self, sample_df: pd.DataFrame) -> None:
        backend = PolarsBackend()
        result = backend.sqrt(sample_df["price"])
        expected = np.sqrt(sample_df["price"])
        np.testing.assert_array_almost_equal(result.values, expected.values, decimal=4)

    def test_polynomial_features(self, sample_df: pd.DataFrame) -> None:
        backend = PolarsBackend()
        numeric_df = sample_df[["price", "quantity"]].astype(float)
        result = backend.polynomial_features(numeric_df, degree=2)
        assert "price^2" in result.columns
        assert "price*quantity" in result.columns

    def test_interaction_features(self, sample_df: pd.DataFrame) -> None:
        backend = PolarsBackend()
        result = backend.interaction_features(sample_df, ["price", "quantity", "revenue"])
        assert "price*quantity" in result.columns
        assert "price*revenue" in result.columns

    def test_rolling_agg(self, sample_df: pd.DataFrame) -> None:
        backend = PolarsBackend()
        result = backend.rolling_agg(sample_df["price"], window=3, agg_func="mean")
        assert len(result) == 50

    def test_one_hot_encode(self, sample_df: pd.DataFrame) -> None:
        backend = PolarsBackend()
        result = backend.one_hot_encode(sample_df["category"])
        assert result.shape[1] == 3  # A, B, C

    def test_quantile_bin(self, sample_df: pd.DataFrame) -> None:
        backend = PolarsBackend()
        result = backend.quantile_bin(sample_df["price"], n_bins=5)
        assert result.nunique() <= 5


class TestPolarsLazyPipeline:
    """Tests for the lazy Polars pipeline."""

    def test_add_interaction(self, sample_df: pd.DataFrame) -> None:
        pipeline = PolarsLazyPipeline()
        pipeline.add_interaction("price", "quantity")
        result = pipeline.execute(sample_df)
        assert "price_x_quantity" in result.columns
        expected = sample_df["price"] * sample_df["quantity"]
        np.testing.assert_array_almost_equal(
            result["price_x_quantity"].values, expected.values
        )

    def test_add_ratio(self, sample_df: pd.DataFrame) -> None:
        pipeline = PolarsLazyPipeline()
        pipeline.add_ratio("revenue", "quantity")
        result = pipeline.execute(sample_df)
        assert "revenue_div_quantity" in result.columns

    def test_add_log(self, sample_df: pd.DataFrame) -> None:
        pipeline = PolarsLazyPipeline()
        pipeline.add_log("price")
        result = pipeline.execute(sample_df)
        assert "price_log1p" in result.columns

    def test_add_sqrt(self, sample_df: pd.DataFrame) -> None:
        pipeline = PolarsLazyPipeline()
        pipeline.add_sqrt("price")
        result = pipeline.execute(sample_df)
        assert "price_sqrt" in result.columns

    def test_add_polynomial(self, sample_df: pd.DataFrame) -> None:
        pipeline = PolarsLazyPipeline()
        pipeline.add_polynomial("price", degree=3)
        result = pipeline.execute(sample_df)
        assert "price_pow2" in result.columns
        assert "price_pow3" in result.columns

    def test_all_interactions(self, sample_df: pd.DataFrame) -> None:
        pipeline = PolarsLazyPipeline()
        pipeline.add_all_interactions(["price", "quantity", "revenue"])
        result = pipeline.execute(sample_df)
        assert "price_x_quantity" in result.columns
        assert "price_x_revenue" in result.columns
        assert "quantity_x_revenue" in result.columns

    def test_chaining(self, sample_df: pd.DataFrame) -> None:
        pipeline = (
            PolarsLazyPipeline()
            .add_interaction("price", "quantity")
            .add_log("revenue")
            .add_sqrt("price")
        )
        result = pipeline.execute(sample_df)
        assert pipeline.operation_count == 3
        assert len(result.columns) == len(sample_df.columns) + 3

    def test_describe(self) -> None:
        pipeline = PolarsLazyPipeline()
        pipeline.add_interaction("a", "b")
        pipeline.add_log("c")
        desc = pipeline.describe()
        assert len(desc) == 2
        assert "interaction" in desc[0]

    def test_include_originals_false(self, sample_df: pd.DataFrame) -> None:
        pipeline = PolarsLazyPipeline(include_originals=False)
        pipeline.add_interaction("price", "quantity")
        pipeline.add_log("revenue")
        result = pipeline.execute(sample_df)
        # Only generated features, no originals
        assert "price_x_quantity" in result.columns
        assert "revenue_log1p" in result.columns
        assert "price" not in result.columns

    def test_lazy_execution(self, sample_df: pd.DataFrame) -> None:
        """Test that execute_lazy returns a LazyFrame."""
        import polars as pl
        pipeline = PolarsLazyPipeline()
        pipeline.add_interaction("price", "quantity")
        lf = pl.from_pandas(sample_df).lazy()
        result_lf = pipeline.execute_lazy(lf)
        # Should still be lazy
        assert isinstance(result_lf, pl.LazyFrame)
        # Collect should work
        result = result_lf.collect().to_pandas()
        assert "price_x_quantity" in result.columns

    def test_preserves_index(self, sample_df: pd.DataFrame) -> None:
        sample_df.index = range(100, 150)
        pipeline = PolarsLazyPipeline()
        pipeline.add_interaction("price", "quantity")
        result = pipeline.execute(sample_df)
        assert list(result.index) == list(range(100, 150))


class TestPolarsFeatureGenerator:
    """Tests for the high-level Polars feature generator."""

    def test_fit_transform(self, sample_df: pd.DataFrame) -> None:
        gen = PolarsFeatureGenerator(interactions=True, polynomials=False)
        result = gen.fit_transform(sample_df)
        assert result.shape[0] == 50
        assert result.shape[1] > sample_df.shape[1]

    def test_with_polynomials(self, sample_df: pd.DataFrame) -> None:
        gen = PolarsFeatureGenerator(interactions=False, polynomials=True, max_degree=2)
        result = gen.fit_transform(sample_df)
        assert any("pow2" in c for c in result.columns)

    def test_with_log_transforms(self) -> None:
        # Create a skewed dataset
        np.random.seed(42)
        df = pd.DataFrame({
            "skewed": np.exp(np.random.randn(100)),
            "normal": np.random.randn(100),
        })
        gen = PolarsFeatureGenerator(
            interactions=False, log_transforms=True, skewness_threshold=1.0
        )
        result = gen.fit_transform(df)
        assert any("log1p" in c for c in result.columns)

    def test_not_fitted_raises(self) -> None:
        gen = PolarsFeatureGenerator()
        with pytest.raises(RuntimeError, match="not fitted"):
            gen.transform(pd.DataFrame({"a": [1]}))

    def test_get_feature_names_out(self, sample_df: pd.DataFrame) -> None:
        gen = PolarsFeatureGenerator(interactions=True, polynomials=False)
        gen.fit(sample_df)
        names = gen.get_feature_names_out()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_separate_fit_transform(self, sample_df: pd.DataFrame) -> None:
        gen = PolarsFeatureGenerator(interactions=True, polynomials=False)
        gen.fit(sample_df)
        result = gen.transform(sample_df)
        assert result.shape[0] == 50
