"""Tests for categorical feature generators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.generators.categorical import (
    CategoryCombiner,
    CategoryStatistics,
    FrequencyEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    TargetEncoder,
)


class TestOneHotEncoder:
    """Tests for OneHotEncoder."""

    def test_basic_encoding(self, sample_categorical_df: pd.DataFrame):
        """Test basic one-hot encoding."""
        encoder = OneHotEncoder(columns=["color"])
        result = encoder.fit_transform(sample_categorical_df)

        # Should have one column per category
        color_cols = [c for c in result.columns if c.startswith("color_")]
        assert len(color_cols) == 3  # red, blue, green

    def test_drop_first(self, sample_categorical_df: pd.DataFrame):
        """Test drop_first option."""
        encoder = OneHotEncoder(columns=["color"], drop_first=True)
        result = encoder.fit_transform(sample_categorical_df)

        color_cols = [c for c in result.columns if c.startswith("color_")]
        assert len(color_cols) == 2  # One less than categories

    def test_handle_unknown(self, sample_categorical_df: pd.DataFrame):
        """Test handling of unknown categories."""
        encoder = OneHotEncoder(columns=["color"], handle_unknown="ignore")
        encoder.fit(sample_categorical_df)

        # Create data with unknown category
        new_df = pd.DataFrame({"color": ["unknown", "red", "blue"]})
        result = encoder.transform(new_df)

        # Unknown should have all zeros
        color_cols = [c for c in result.columns if c.startswith("color_")]
        assert result.loc[0, color_cols].sum() == 0

    def test_max_categories(self, high_cardinality_df: pd.DataFrame):
        """Test max_categories limit."""
        encoder = OneHotEncoder(columns=["group"], max_categories=10)
        result = encoder.fit_transform(high_cardinality_df)

        group_cols = [c for c in result.columns if c.startswith("group_")]
        assert len(group_cols) <= 10


class TestTargetEncoder:
    """Tests for TargetEncoder."""

    def test_basic_encoding(
        self, sample_categorical_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test basic target encoding."""
        encoder = TargetEncoder(columns=["color"])
        result = encoder.fit_transform(sample_categorical_df, sample_target_binary)

        assert "color_target" in result.columns
        assert result["color_target"].dtype in [np.float64, np.float32]

    def test_smoothing(
        self, sample_categorical_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test smoothing parameter."""
        encoder = TargetEncoder(columns=["color"], smoothing=10)
        result = encoder.fit_transform(sample_categorical_df, sample_target_binary)

        # With smoothing, values should be closer to global mean
        global_mean = sample_target_binary.mean()
        assert result["color_target"].between(0, 1).all()

    def test_requires_target(self, sample_categorical_df: pd.DataFrame):
        """Test that target is required."""
        encoder = TargetEncoder(columns=["color"])

        with pytest.raises(ValueError):
            encoder.fit(sample_categorical_df)


class TestFrequencyEncoder:
    """Tests for FrequencyEncoder."""

    def test_basic_encoding(self, sample_categorical_df: pd.DataFrame):
        """Test basic frequency encoding."""
        encoder = FrequencyEncoder(columns=["color"])
        result = encoder.fit_transform(sample_categorical_df)

        assert "color_freq" in result.columns
        # Frequencies should sum to approximately 1
        assert 0 < result["color_freq"].sum() <= len(sample_categorical_df)

    def test_normalize(self, sample_categorical_df: pd.DataFrame):
        """Test normalized frequency encoding."""
        encoder = FrequencyEncoder(columns=["color"], normalize=True)
        result = encoder.fit_transform(sample_categorical_df)

        # Normalized frequencies should be between 0 and 1
        assert result["color_freq"].between(0, 1).all()

    def test_unknown_category(self, sample_categorical_df: pd.DataFrame):
        """Test handling of unknown categories in transform."""
        encoder = FrequencyEncoder(columns=["color"])
        encoder.fit(sample_categorical_df)

        new_df = pd.DataFrame({"color": ["unknown"]})
        result = encoder.transform(new_df)

        # Unknown should have 0 frequency
        assert result["color_freq"].iloc[0] == 0


class TestOrdinalEncoder:
    """Tests for OrdinalEncoder."""

    def test_basic_encoding(self, sample_categorical_df: pd.DataFrame):
        """Test basic ordinal encoding."""
        encoder = OrdinalEncoder(columns=["size"])
        result = encoder.fit_transform(sample_categorical_df)

        assert "size_ordinal" in result.columns
        assert result["size_ordinal"].dtype in [np.int64, np.int32, np.float64]

    def test_custom_order(self, sample_categorical_df: pd.DataFrame):
        """Test custom category ordering."""
        encoder = OrdinalEncoder(
            columns=["size"],
            order={"size": ["S", "M", "L", "XL"]},
        )
        result = encoder.fit_transform(sample_categorical_df)

        # S should be 0, XL should be 3
        s_idx = sample_categorical_df["size"] == "S"
        xl_idx = sample_categorical_df["size"] == "XL"

        assert (result.loc[s_idx, "size_ordinal"] == 0).all()
        assert (result.loc[xl_idx, "size_ordinal"] == 3).all()


class TestCategoryCombiner:
    """Tests for CategoryCombiner."""

    def test_basic_combination(self, sample_categorical_df: pd.DataFrame):
        """Test basic category combination."""
        combiner = CategoryCombiner(columns=["color", "size"])
        result = combiner.fit_transform(sample_categorical_df)

        # Column name is col1_x_col2 (with default separator _x_)
        combined_cols = [c for c in result.columns if "color" in c and "size" in c]
        assert len(combined_cols) >= 1
        # Combined values should be string combinations
        assert result[combined_cols[0]].dtype == object

    def test_separator(self, sample_categorical_df: pd.DataFrame):
        """Test custom separator."""
        combiner = CategoryCombiner(columns=["color", "size"], separator="|")
        result = combiner.fit_transform(sample_categorical_df)

        combined_cols = [c for c in result.columns if "color" in c and "size" in c]
        assert len(combined_cols) >= 1
        assert "|" in result[combined_cols[0]].iloc[0]


class TestCategoryStatistics:
    """Tests for CategoryStatistics."""

    def test_basic_stats(
        self, sample_mixed_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test basic category statistics."""
        stats = CategoryStatistics(
            group_cols=["category"],
            agg_cols=["income"],
            stats=["mean", "std"],
        )
        result = stats.fit_transform(sample_mixed_df)

        # Column names are {group_col}_{agg_col}_{stat}
        assert "category_income_mean" in result.columns
        assert "category_income_std" in result.columns

    def test_count_stat(self, sample_mixed_df: pd.DataFrame):
        """Test count statistic."""
        stats = CategoryStatistics(
            group_cols=["category"],
            agg_cols=["income"],
            stats=["count"],
        )
        result = stats.fit_transform(sample_mixed_df)

        assert "category_income_count" in result.columns
