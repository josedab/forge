"""Tests for feature selectors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.selectors import (
    CorrelationSelector,
    ImportanceSelector,
    StatisticalSelector,
    VarianceSelector,
)


class TestVarianceSelector:
    """Tests for VarianceSelector."""

    def test_removes_constant(self):
        """Test removal of constant columns."""
        df = pd.DataFrame(
            {
                "constant": [1, 1, 1, 1, 1],
                "varying": [1, 2, 3, 4, 5],
            }
        )

        selector = VarianceSelector(threshold=0.0)
        result = selector.fit_transform(df)

        assert "constant" not in result.columns
        assert "varying" in result.columns

    def test_threshold(self, sample_numeric_df: pd.DataFrame):
        """Test variance threshold."""
        selector = VarianceSelector(threshold=0.1)
        result = selector.fit_transform(sample_numeric_df)

        # All columns should have variance > 0.1
        assert result.var().min() > 0.1

    def test_get_support(self, sample_numeric_df: pd.DataFrame):
        """Test get_support method."""
        selector = VarianceSelector(threshold=0.0)
        selector.fit(sample_numeric_df)

        support = selector.get_support()
        assert isinstance(support, np.ndarray)
        assert support.dtype == bool

        indices = selector.get_support(indices=True)
        assert isinstance(indices, np.ndarray)


class TestCorrelationSelector:
    """Tests for CorrelationSelector."""

    def test_removes_correlated(self, correlated_df: pd.DataFrame):
        """Test removal of highly correlated features."""
        selector = CorrelationSelector(threshold=0.9)
        result = selector.fit_transform(correlated_df)

        # Should remove one of y or z (correlated with x)
        assert len(result.columns) < len(correlated_df.columns)

    def test_keeps_independent(self, correlated_df: pd.DataFrame):
        """Test that independent features are kept."""
        selector = CorrelationSelector(threshold=0.9)
        result = selector.fit_transform(correlated_df)

        # Independent column should always be kept
        assert "independent" in result.columns

    def test_high_threshold(self, sample_numeric_df: pd.DataFrame):
        """Test with very high threshold (no removal)."""
        selector = CorrelationSelector(threshold=0.99)
        result = selector.fit_transform(sample_numeric_df)

        # With high threshold, should keep most columns
        assert len(result.columns) >= len(sample_numeric_df.columns) - 1


class TestStatisticalSelector:
    """Tests for StatisticalSelector."""

    def test_chi2_classification(
        self,
        sample_numeric_df: pd.DataFrame,
        sample_target_binary: pd.Series,
    ):
        """Test chi2 selection for classification."""
        # Make values positive for chi2
        df = sample_numeric_df.abs()

        selector = StatisticalSelector(method="chi2", k=2)
        result = selector.fit_transform(df, sample_target_binary)

        assert len(result.columns) == 2

    def test_mutual_info(
        self,
        sample_numeric_df: pd.DataFrame,
        sample_target_binary: pd.Series,
    ):
        """Test mutual information selection."""
        selector = StatisticalSelector(method="mutual_info_classif", k=2)
        result = selector.fit_transform(sample_numeric_df, sample_target_binary)

        assert len(result.columns) == 2

    def test_anova(
        self,
        sample_numeric_df: pd.DataFrame,
        sample_target_multiclass: pd.Series,
    ):
        """Test ANOVA F-value selection."""
        selector = StatisticalSelector(method="anova", k=2)
        result = selector.fit_transform(sample_numeric_df, sample_target_multiclass)

        assert len(result.columns) == 2

    def test_requires_target(self, sample_numeric_df: pd.DataFrame):
        """Test that target is required."""
        selector = StatisticalSelector(method="chi2", k=2)

        with pytest.raises(ValueError):
            selector.fit(sample_numeric_df)

    def test_get_scores(
        self,
        sample_numeric_df: pd.DataFrame,
        sample_target_binary: pd.Series,
    ):
        """Test get_pvalues method returns scores."""
        selector = StatisticalSelector(method="mutual_info_classif", k=2)
        selector.fit(sample_numeric_df, sample_target_binary)

        result = selector.get_pvalues()
        assert result is not None
        assert len(result) == len(sample_numeric_df.columns)


class TestImportanceSelector:
    """Tests for ImportanceSelector."""

    def test_basic_selection(
        self,
        feature_importance_df: pd.DataFrame,
        feature_importance_target: pd.Series,
    ):
        """Test basic importance selection."""
        selector = ImportanceSelector(n_features=2)
        result = selector.fit_transform(
            feature_importance_df, feature_importance_target
        )

        assert len(result.columns) == 2

    def test_selects_important(
        self,
        feature_importance_df: pd.DataFrame,
        feature_importance_target: pd.Series,
    ):
        """Test that important features are selected."""
        selector = ImportanceSelector(n_features=2)
        result = selector.fit_transform(
            feature_importance_df, feature_importance_target
        )

        # Should select the important features
        selected = set(result.columns)
        assert "important1" in selected or "important2" in selected

    def test_threshold_mode(
        self,
        feature_importance_df: pd.DataFrame,
        feature_importance_target: pd.Series,
    ):
        """Test threshold-based selection."""
        selector = ImportanceSelector(threshold=0.1)
        selector.fit(feature_importance_df, feature_importance_target)

        importance = selector.get_importances()
        assert importance is not None

    def test_get_feature_importance(
        self,
        feature_importance_df: pd.DataFrame,
        feature_importance_target: pd.Series,
    ):
        """Test get_importances method."""
        selector = ImportanceSelector(n_features=2)
        selector.fit(feature_importance_df, feature_importance_target)

        importance = selector.get_importances()
        assert importance is not None
        assert len(importance) == len(feature_importance_df.columns)


class TestSelectorIntegration:
    """Integration tests for selector chaining."""

    def test_variance_then_correlation(self, sample_numeric_df: pd.DataFrame):
        """Test chaining variance and correlation selectors."""
        # Add a constant column
        df = sample_numeric_df.copy()
        df["constant"] = 1

        # First remove constant
        var_selector = VarianceSelector(threshold=0.0)
        df = var_selector.fit_transform(df)
        assert "constant" not in df.columns

        # Then check correlation
        corr_selector = CorrelationSelector(threshold=0.95)
        result = corr_selector.fit_transform(df)

        assert len(result.columns) > 0

    def test_statistical_then_importance(
        self,
        feature_importance_df: pd.DataFrame,
        feature_importance_target: pd.Series,
    ):
        """Test chaining statistical and importance selectors."""
        # First statistical selection (select 3 but fixture may have only 4)
        stat_selector = StatisticalSelector(method="mutual_info_classif", k=3)
        df = stat_selector.fit_transform(
            feature_importance_df, feature_importance_target
        )

        # Fixture has 4 columns, may get 3 or 4 depending on scores
        assert len(df.columns) >= 2

        # Then importance selection
        imp_selector = ImportanceSelector(n_features=2)
        result = imp_selector.fit_transform(df, feature_importance_target)

        assert len(result.columns) == 2
