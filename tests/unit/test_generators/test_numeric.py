"""Tests for numeric feature generators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.generators.numeric import (
    AggregationGenerator,
    InteractionGenerator,
    PolynomialGenerator,
    TransformationGenerator,
)


class TestTransformationGenerator:
    """Tests for TransformationGenerator."""

    def test_log_transform(self, sample_numeric_df: pd.DataFrame):
        """Test log transformation."""
        generator = TransformationGenerator(
            columns=["income"], transformations=["log"]
        )
        result = generator.fit_transform(sample_numeric_df)

        assert "income_log" in result.columns
        # Log should reduce variance for positive values
        assert result["income_log"].notna().all()

    def test_sqrt_transform(self, sample_numeric_df: pd.DataFrame):
        """Test square root transformation."""
        generator = TransformationGenerator(
            columns=["score"], transformations=["sqrt"]
        )
        result = generator.fit_transform(sample_numeric_df)

        assert "score_sqrt" in result.columns

    def test_power_transform(self, sample_numeric_df: pd.DataFrame):
        """Test power transformation."""
        generator = TransformationGenerator(
            columns=["age"], transformations=["square"]
        )
        result = generator.fit_transform(sample_numeric_df)

        assert "age_square" in result.columns
        assert np.allclose(
            result["age_square"], sample_numeric_df["age"] ** 2
        )

    def test_multiple_transformations(self, sample_numeric_df: pd.DataFrame):
        """Test multiple transformations."""
        generator = TransformationGenerator(
            columns=["income"],
            transformations=["log", "sqrt", "square"],
        )
        result = generator.fit_transform(sample_numeric_df)

        assert "income_log" in result.columns
        assert "income_sqrt" in result.columns
        assert "income_square" in result.columns

    def test_binning(self, sample_numeric_df: pd.DataFrame):
        """Test binning transformation."""
        generator = TransformationGenerator(
            columns=["age"], transformations=["bin"], n_bins=5
        )
        result = generator.fit_transform(sample_numeric_df)

        assert "age_bin" in result.columns
        assert result["age_bin"].nunique() <= 5

    def test_get_feature_names_out(self, sample_numeric_df: pd.DataFrame):
        """Test feature names output."""
        generator = TransformationGenerator(
            columns=["age", "income"], transformations=["log"]
        )
        generator.fit(sample_numeric_df)

        names = generator.get_feature_names_out()
        assert "age_log" in names
        assert "income_log" in names


class TestInteractionGenerator:
    """Tests for InteractionGenerator."""

    def test_multiply_interaction(self, sample_numeric_df: pd.DataFrame):
        """Test multiplication interaction."""
        generator = InteractionGenerator(
            columns=["age", "income"], interaction_type="multiply"
        )
        result = generator.fit_transform(sample_numeric_df)

        assert "age_x_income" in result.columns
        expected = sample_numeric_df["age"] * sample_numeric_df["income"]
        assert np.allclose(result["age_x_income"], expected)

    def test_add_interaction(self, sample_numeric_df: pd.DataFrame):
        """Test addition interaction."""
        generator = InteractionGenerator(
            columns=["age", "score"], interaction_type="add"
        )
        result = generator.fit_transform(sample_numeric_df)

        assert "age_+_score" in result.columns

    def test_divide_interaction(self, sample_numeric_df: pd.DataFrame):
        """Test division interaction."""
        generator = InteractionGenerator(
            columns=["income", "age"], interaction_type="divide"
        )
        result = generator.fit_transform(sample_numeric_df)

        assert "income_/_age" in result.columns

    def test_all_pairs(self, sample_numeric_df: pd.DataFrame):
        """Test all pairs generation."""
        generator = InteractionGenerator(
            columns=["age", "income", "score"],
            interaction_type="multiply",
            include_all_pairs=True,
        )
        result = generator.fit_transform(sample_numeric_df)

        # Should have pairs: age*income, age*score, income*score
        assert len(result.columns) >= 3


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

    def test_include_interaction(self, sample_numeric_df: pd.DataFrame):
        """Test polynomial with interaction terms."""
        generator = PolynomialGenerator(
            columns=["age", "score"],
            degree=2,
            include_interaction=True,
        )
        result = generator.fit_transform(sample_numeric_df)

        # Should include age*score interaction
        assert any("age" in c and "score" in c for c in result.columns)


class TestAggregationGenerator:
    """Tests for AggregationGenerator."""

    def test_basic_aggregation(self, sample_mixed_df: pd.DataFrame):
        """Test basic aggregation."""
        generator = AggregationGenerator(
            group_columns=["category"],
            agg_columns=["income"],
            agg_functions=["mean"],
        )
        result = generator.fit_transform(sample_mixed_df)

        assert "income_mean_by_category" in result.columns
        assert len(result) == len(sample_mixed_df)

    def test_multiple_agg_functions(self, sample_mixed_df: pd.DataFrame):
        """Test multiple aggregation functions."""
        generator = AggregationGenerator(
            group_columns=["category"],
            agg_columns=["income"],
            agg_functions=["mean", "std", "min", "max"],
        )
        result = generator.fit_transform(sample_mixed_df)

        assert "income_mean_by_category" in result.columns
        assert "income_std_by_category" in result.columns
        assert "income_min_by_category" in result.columns
        assert "income_max_by_category" in result.columns

    def test_multiple_group_columns(self, sample_mixed_df: pd.DataFrame):
        """Test aggregation with multiple group columns."""
        generator = AggregationGenerator(
            group_columns=["category", "color"],
            agg_columns=["income"],
            agg_functions=["mean"],
        )
        result = generator.fit_transform(sample_mixed_df)

        assert "income_mean_by_category_color" in result.columns

    def test_transform_preserves_index(self, sample_mixed_df: pd.DataFrame):
        """Test that transform preserves original index."""
        generator = AggregationGenerator(
            group_columns=["category"],
            agg_columns=["income"],
            agg_functions=["mean"],
        )
        generator.fit(sample_mixed_df)
        result = generator.transform(sample_mixed_df)

        assert len(result) == len(sample_mixed_df)
        assert (result.index == sample_mixed_df.index).all()
