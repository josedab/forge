"""Tests for numeric feature generators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from forge.generators.numeric import (
    AggregationGenerator,
    BinningTransformer,
    InteractionGenerator,
    LogTransformer,
    NumericTransformer,
    PolynomialGenerator,
    PowerTransformer,
)


class TestLogTransformer:
    """Tests for LogTransformer."""

    def test_log_transform(self, sample_numeric_df: pd.DataFrame):
        """Test log transformation."""
        transformer = LogTransformer(columns=["income"])
        result = transformer.fit_transform(sample_numeric_df)

        assert "income_log1p" in result.columns
        assert result["income_log1p"].notna().all()

    def test_multiple_columns(self, sample_numeric_df: pd.DataFrame):
        """Test log transformation on multiple columns."""
        transformer = LogTransformer(columns=["age", "income"])
        result = transformer.fit_transform(sample_numeric_df)

        assert "age_log1p" in result.columns
        assert "income_log1p" in result.columns


class TestPowerTransformer:
    """Tests for PowerTransformer."""

    def test_sqrt_transform(self, sample_numeric_df: pd.DataFrame):
        """Test square root transformation."""
        transformer = PowerTransformer(
            columns=["score"], transforms=["sqrt"]
        )
        result = transformer.fit_transform(sample_numeric_df)

        assert "score_sqrt" in result.columns

    def test_square_transform(self, sample_numeric_df: pd.DataFrame):
        """Test square transformation."""
        transformer = PowerTransformer(
            columns=["age"], transforms=["square"]
        )
        result = transformer.fit_transform(sample_numeric_df)

        assert "age_square" in result.columns
        assert np.allclose(
            result["age_square"], sample_numeric_df["age"] ** 2
        )

    def test_multiple_transformations(self, sample_numeric_df: pd.DataFrame):
        """Test multiple transformations."""
        transformer = PowerTransformer(
            columns=["income"],
            transforms=["sqrt", "square"],
        )
        result = transformer.fit_transform(sample_numeric_df)

        assert "income_sqrt" in result.columns
        assert "income_square" in result.columns


class TestBinningTransformer:
    """Tests for BinningTransformer."""

    def test_binning(self, sample_numeric_df: pd.DataFrame):
        """Test binning transformation."""
        transformer = BinningTransformer(columns=["age"], n_bins=5)
        result = transformer.fit_transform(sample_numeric_df)

        assert "age_binned" in result.columns
        assert result["age_binned"].nunique() <= 5


class TestNumericTransformer:
    """Tests for NumericTransformer (combined transformer)."""

    def test_log_sqrt(self, sample_numeric_df: pd.DataFrame):
        """Test log and sqrt transformations."""
        transformer = NumericTransformer(
            columns=["income"], log=True, sqrt=True, square=False
        )
        result = transformer.fit_transform(sample_numeric_df)

        feature_names = transformer.get_feature_names_out()
        assert any("log1p" in n for n in feature_names)
        assert any("sqrt" in n for n in feature_names)

    def test_with_binning(self, sample_numeric_df: pd.DataFrame):
        """Test with binning enabled."""
        transformer = NumericTransformer(
            columns=["age"], log=False, sqrt=False, binning=True, n_bins=3
        )
        result = transformer.fit_transform(sample_numeric_df)

        feature_names = transformer.get_feature_names_out()
        assert any("bin" in n for n in feature_names)


class TestInteractionGenerator:
    """Tests for InteractionGenerator."""

    def test_multiply_interaction(self, sample_numeric_df: pd.DataFrame):
        """Test multiplication interaction."""
        generator = InteractionGenerator(
            columns=["age", "income"], operations=["multiply"]
        )
        result = generator.fit_transform(sample_numeric_df)

        assert "age_x_income" in result.columns
        expected = sample_numeric_df["age"] * sample_numeric_df["income"]
        assert np.allclose(result["age_x_income"], expected)

    def test_add_interaction(self, sample_numeric_df: pd.DataFrame):
        """Test addition interaction."""
        generator = InteractionGenerator(
            columns=["age", "score"], operations=["add"]
        )
        result = generator.fit_transform(sample_numeric_df)

        assert "age_plus_score" in result.columns

    def test_divide_interaction(self, sample_numeric_df: pd.DataFrame):
        """Test division interaction."""
        generator = InteractionGenerator(
            columns=["income", "age"], operations=["divide"]
        )
        result = generator.fit_transform(sample_numeric_df)

        assert "income_div_age" in result.columns

    def test_all_pairs(self, sample_numeric_df: pd.DataFrame):
        """Test all pairs generation."""
        generator = InteractionGenerator(
            columns=["age", "income", "score"],
            operations=["multiply"],
        )
        result = generator.fit_transform(sample_numeric_df)

        # Should have pairs: age*income, age*score, income*score
        assert len(result.columns) >= 3

    def test_self_interactions(self, sample_numeric_df: pd.DataFrame):
        """Test self interactions (squared)."""
        generator = InteractionGenerator(
            columns=["age", "income"],
            operations=["multiply"],
            include_self_interactions=True,
        )
        result = generator.fit_transform(sample_numeric_df)

        # Should include age_squared, income_squared
        assert any("squared" in c for c in result.columns)


class TestPolynomialGenerator:
    """Tests for PolynomialGenerator."""

    def test_degree_2(self, sample_numeric_df: pd.DataFrame):
        """Test degree 2 polynomials."""
        generator = PolynomialGenerator(columns=["age"], degree=2)
        result = generator.fit_transform(sample_numeric_df)

        assert "age^2" in result.columns
        assert np.allclose(
            result["age^2"], sample_numeric_df["age"] ** 2
        )

    def test_degree_3(self, sample_numeric_df: pd.DataFrame):
        """Test degree 3 polynomials."""
        generator = PolynomialGenerator(columns=["age"], degree=3)
        result = generator.fit_transform(sample_numeric_df)

        assert "age^2" in result.columns
        assert "age^3" in result.columns

    def test_multiple_columns(self, sample_numeric_df: pd.DataFrame):
        """Test polynomials for multiple columns."""
        generator = PolynomialGenerator(
            columns=["age", "score"], degree=2
        )
        result = generator.fit_transform(sample_numeric_df)

        assert "age^2" in result.columns
        assert "score^2" in result.columns


class TestAggregationGenerator:
    """Tests for AggregationGenerator."""

    def test_basic_aggregation(self, sample_mixed_df: pd.DataFrame):
        """Test basic aggregation."""
        generator = AggregationGenerator(
            group_cols=["category"],
            agg_cols=["income"],
            agg_funcs=["mean"],
        )
        result = generator.fit_transform(sample_mixed_df)

        # Column name is {group_col}_{agg_col}_{func}
        assert "category_income_mean" in result.columns
        assert len(result) == len(sample_mixed_df)

    def test_multiple_agg_functions(self, sample_mixed_df: pd.DataFrame):
        """Test multiple aggregation functions."""
        generator = AggregationGenerator(
            group_cols=["category"],
            agg_cols=["income"],
            agg_funcs=["mean", "std"],
        )
        result = generator.fit_transform(sample_mixed_df)

        assert "category_income_mean" in result.columns
        assert "category_income_std" in result.columns

    def test_transform_preserves_index(self, sample_mixed_df: pd.DataFrame):
        """Test that transform preserves original index."""
        generator = AggregationGenerator(
            group_cols=["category"],
            agg_cols=["income"],
            agg_funcs=["mean"],
        )
        generator.fit(sample_mixed_df)
        result = generator.transform(sample_mixed_df)

        assert len(result) == len(sample_mixed_df)
        assert (result.index == sample_mixed_df.index).all()
