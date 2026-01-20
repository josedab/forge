"""Tests for outlier handling module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.outliers import ArbitraryCapper, IQRCapper, Trimmer, Winsorizer


class TestWinsorizer:
    """Tests for Winsorizer."""

    def test_basic_winsorization(self, sample_df_with_outliers: pd.DataFrame):
        """Test basic winsorization."""
        winsorizer = Winsorizer(columns=["with_outliers"])
        result = winsorizer.fit_transform(sample_df_with_outliers)

        # Outliers should be capped
        assert result["with_outliers"].max() < 100
        assert result["with_outliers"].min() > -100

    def test_custom_percentiles(self, sample_df_with_outliers: pd.DataFrame):
        """Test custom percentile bounds."""
        winsorizer = Winsorizer(
            columns=["with_outliers"],
            lower_percentile=0.05,
            upper_percentile=0.95,
        )
        result = winsorizer.fit_transform(sample_df_with_outliers)

        # Should cap at tighter bounds
        original = sample_df_with_outliers["with_outliers"]
        assert result["with_outliers"].max() <= np.percentile(original, 95)
        assert result["with_outliers"].min() >= np.percentile(original, 5)

    def test_auto_column_detection(self, sample_numeric_df: pd.DataFrame):
        """Test automatic numeric column detection."""
        winsorizer = Winsorizer()
        result = winsorizer.fit_transform(sample_numeric_df)

        # All numeric columns should be processed
        assert winsorizer.feature_names_in_ == list(sample_numeric_df.columns)

    def test_preserves_non_numeric(self, sample_mixed_df: pd.DataFrame):
        """Test that non-numeric columns are preserved."""
        winsorizer = Winsorizer()
        result = winsorizer.fit_transform(sample_mixed_df)

        # Categorical columns should be unchanged
        assert "category" in result.columns
        pd.testing.assert_series_equal(
            result["category"], sample_mixed_df["category"], check_names=True
        )

    def test_get_feature_names_out(self, sample_numeric_df: pd.DataFrame):
        """Test feature names output."""
        winsorizer = Winsorizer()
        winsorizer.fit(sample_numeric_df)

        names = winsorizer.get_feature_names_out()
        assert set(names) == set(sample_numeric_df.columns)

    def test_fit_transform_equivalence(self, sample_df_with_outliers: pd.DataFrame):
        """Test that fit_transform equals fit then transform."""
        winsorizer1 = Winsorizer(columns=["with_outliers"])
        winsorizer2 = Winsorizer(columns=["with_outliers"])

        result1 = winsorizer1.fit_transform(sample_df_with_outliers)
        result2 = winsorizer2.fit(sample_df_with_outliers).transform(
            sample_df_with_outliers
        )

        pd.testing.assert_frame_equal(result1, result2)


class TestIQRCapper:
    """Tests for IQRCapper."""

    def test_basic_capping(self, sample_df_with_outliers: pd.DataFrame):
        """Test basic IQR capping."""
        capper = IQRCapper(columns=["with_outliers"])
        result = capper.fit_transform(sample_df_with_outliers)

        # Extreme outliers should be capped
        assert result["with_outliers"].max() < 100
        assert result["with_outliers"].min() > -100

    def test_custom_factor(self, sample_df_with_outliers: pd.DataFrame):
        """Test custom IQR factor."""
        capper = IQRCapper(columns=["with_outliers"], factor=3.0)
        result = capper.fit_transform(sample_df_with_outliers)

        # Higher factor = wider bounds
        assert "with_outliers" in result.columns

    def test_bounds_calculation(self, sample_numeric_df: pd.DataFrame):
        """Test that bounds are calculated correctly."""
        capper = IQRCapper(columns=["income"], factor=1.5)
        capper.fit(sample_numeric_df)

        # Check bounds exist
        assert "income" in capper.bounds_
        lower, upper = capper.bounds_["income"]
        assert lower < upper

        # Calculate expected bounds
        q1 = sample_numeric_df["income"].quantile(0.25)
        q3 = sample_numeric_df["income"].quantile(0.75)
        iqr = q3 - q1
        expected_lower = q1 - 1.5 * iqr
        expected_upper = q3 + 1.5 * iqr

        assert np.isclose(lower, expected_lower)
        assert np.isclose(upper, expected_upper)

    def test_get_feature_names_out(self, sample_numeric_df: pd.DataFrame):
        """Test feature names output."""
        capper = IQRCapper()
        capper.fit(sample_numeric_df)

        names = capper.get_feature_names_out()
        assert set(names) == set(sample_numeric_df.columns)


class TestArbitraryCapper:
    """Tests for ArbitraryCapper."""

    def test_basic_capping(self, sample_df_with_outliers: pd.DataFrame):
        """Test basic arbitrary capping."""
        capper = ArbitraryCapper(
            capping_dict={"with_outliers": {"lower": -10, "upper": 10}}
        )
        result = capper.fit_transform(sample_df_with_outliers)

        assert result["with_outliers"].max() <= 10
        assert result["with_outliers"].min() >= -10

    def test_lower_only(self, sample_df_with_outliers: pd.DataFrame):
        """Test capping with only lower bound."""
        capper = ArbitraryCapper(capping_dict={"with_outliers": {"lower": -5}})
        result = capper.fit_transform(sample_df_with_outliers)

        assert result["with_outliers"].min() >= -5
        # Upper should be unchanged (still 100)
        assert result["with_outliers"].max() == sample_df_with_outliers["with_outliers"].max()

    def test_upper_only(self, sample_df_with_outliers: pd.DataFrame):
        """Test capping with only upper bound."""
        capper = ArbitraryCapper(capping_dict={"with_outliers": {"upper": 5}})
        result = capper.fit_transform(sample_df_with_outliers)

        assert result["with_outliers"].max() <= 5
        # Lower should be unchanged (still -100)
        assert result["with_outliers"].min() == sample_df_with_outliers["with_outliers"].min()

    def test_multiple_columns(self, sample_numeric_df: pd.DataFrame):
        """Test capping multiple columns."""
        capper = ArbitraryCapper(
            capping_dict={
                "income": {"lower": 30000, "upper": 80000},
                "age": {"lower": 20, "upper": 70},
            }
        )
        result = capper.fit_transform(sample_numeric_df)

        assert result["income"].min() >= 30000
        assert result["income"].max() <= 80000
        assert result["age"].min() >= 20
        assert result["age"].max() <= 70

    def test_preserves_other_columns(self, sample_df_with_outliers: pd.DataFrame):
        """Test that columns not in capping_dict are preserved."""
        capper = ArbitraryCapper(
            capping_dict={"with_outliers": {"lower": -10, "upper": 10}}
        )
        result = capper.fit_transform(sample_df_with_outliers)

        # 'normal' column should be unchanged
        pd.testing.assert_series_equal(
            result["normal"], sample_df_with_outliers["normal"], check_names=True
        )


class TestTrimmer:
    """Tests for Trimmer."""

    def test_basic_trimming_iqr(self, sample_df_with_outliers: pd.DataFrame):
        """Test basic IQR-based trimming."""
        trimmer = Trimmer(columns=["with_outliers"], method="iqr")
        result = trimmer.fit_transform(sample_df_with_outliers)

        # Rows with outliers should be removed
        assert len(result) < len(sample_df_with_outliers)
        # Should have removed at least the two extreme outliers
        assert len(result) <= len(sample_df_with_outliers) - 2

    def test_percentile_trimming(self, sample_df_with_outliers: pd.DataFrame):
        """Test percentile-based trimming."""
        trimmer = Trimmer(
            columns=["with_outliers"],
            method="percentile",
            lower_percentile=0.05,
            upper_percentile=0.95,
        )
        result = trimmer.fit_transform(sample_df_with_outliers)

        # Should trim roughly 10% of data
        assert len(result) < len(sample_df_with_outliers)

    def test_custom_iqr_factor(self, sample_df_with_outliers: pd.DataFrame):
        """Test custom IQR factor."""
        trimmer_strict = Trimmer(columns=["with_outliers"], method="iqr", factor=1.0)
        trimmer_loose = Trimmer(columns=["with_outliers"], method="iqr", factor=3.0)

        result_strict = trimmer_strict.fit_transform(sample_df_with_outliers)
        result_loose = trimmer_loose.fit_transform(sample_df_with_outliers)

        # Stricter factor should remove more rows
        assert len(result_strict) <= len(result_loose)

    def test_no_outliers_unchanged(self, sample_numeric_df: pd.DataFrame):
        """Test that data without extreme outliers is mostly preserved."""
        trimmer = Trimmer(method="iqr", factor=3.0)
        result = trimmer.fit_transform(sample_numeric_df)

        # Most data should be preserved with loose factor
        assert len(result) >= len(sample_numeric_df) * 0.9

    def test_reset_index(self, sample_df_with_outliers: pd.DataFrame):
        """Test that index is reset after trimming."""
        trimmer = Trimmer(columns=["with_outliers"], method="iqr")
        result = trimmer.fit_transform(sample_df_with_outliers)

        # Index should be sequential
        assert list(result.index) == list(range(len(result)))

    def test_get_feature_names_out(self, sample_numeric_df: pd.DataFrame):
        """Test feature names output."""
        trimmer = Trimmer()
        trimmer.fit(sample_numeric_df)

        names = trimmer.get_feature_names_out()
        assert set(names) == set(sample_numeric_df.columns)


class TestSklearnCompatibility:
    """Test sklearn compatibility for all outlier handlers."""

    def test_winsorizer_clone(self, sample_df_with_outliers: pd.DataFrame):
        """Test that Winsorizer can be cloned."""
        from sklearn.base import clone

        winsorizer = Winsorizer(lower_percentile=0.05, upper_percentile=0.95)
        cloned = clone(winsorizer)

        assert cloned.lower_percentile == 0.05
        assert cloned.upper_percentile == 0.95

    def test_iqr_capper_clone(self, sample_df_with_outliers: pd.DataFrame):
        """Test that IQRCapper can be cloned."""
        from sklearn.base import clone

        capper = IQRCapper(factor=2.0)
        cloned = clone(capper)

        assert cloned.factor == 2.0

    def test_pipeline_integration(self, sample_df_with_outliers: pd.DataFrame):
        """Test integration with sklearn Pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # Create pipeline with Winsorizer
        pipeline = Pipeline(
            [
                ("winsorize", Winsorizer(columns=["with_outliers", "normal"])),
            ]
        )

        result = pipeline.fit_transform(sample_df_with_outliers)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_df_with_outliers)
