"""Tests for AutoFeatureTransformer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge import AutoFeatureTransformer
from forge.exceptions import NotFittedError


class TestAutoFeatureTransformer:
    """Tests for AutoFeatureTransformer."""

    def test_basic_fit_transform(
        self, sample_mixed_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test basic fit_transform."""
        transformer = AutoFeatureTransformer(verbose=0)
        result = transformer.fit_transform(sample_mixed_df, sample_target_binary)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_mixed_df)

    def test_transform_after_fit(
        self, sample_mixed_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test separate fit and transform."""
        transformer = AutoFeatureTransformer(verbose=0)
        transformer.fit(sample_mixed_df, sample_target_binary)

        result = transformer.transform(sample_mixed_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_mixed_df)

    def test_transform_without_fit(self, sample_mixed_df: pd.DataFrame):
        """Test transform without fit raises error."""
        transformer = AutoFeatureTransformer()

        with pytest.raises(NotFittedError):
            transformer.transform(sample_mixed_df)

    def test_get_feature_names_out(
        self, sample_mixed_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test get_feature_names_out."""
        transformer = AutoFeatureTransformer(verbose=0)
        transformer.fit(sample_mixed_df, sample_target_binary)

        names = transformer.get_feature_names_out()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_max_features_int(
        self, sample_mixed_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test max_features as integer."""
        transformer = AutoFeatureTransformer(max_features=5, verbose=0)
        result = transformer.fit_transform(sample_mixed_df, sample_target_binary)

        assert len(result.columns) <= 5

    def test_max_features_float(
        self, sample_mixed_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test max_features as fraction."""
        transformer = AutoFeatureTransformer(max_features=0.5, verbose=0)
        transformer.fit(sample_mixed_df, sample_target_binary)

        # Just verify it completes without error
        result = transformer.transform(sample_mixed_df)
        assert len(result.columns) > 0

    def test_get_feature_importance(
        self, sample_mixed_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test get_feature_importance."""
        transformer = AutoFeatureTransformer(verbose=0)
        transformer.fit(sample_mixed_df, sample_target_binary)

        importance = transformer.get_feature_importance()
        assert isinstance(importance, pd.DataFrame)
        assert "feature" in importance.columns
        assert "importance" in importance.columns

    def test_numeric_only(
        self, sample_numeric_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test with numeric only data."""
        transformer = AutoFeatureTransformer(verbose=0)
        result = transformer.fit_transform(sample_numeric_df, sample_target_binary)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_numeric_df)

    def test_categorical_only(
        self, sample_categorical_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test with categorical only data."""
        transformer = AutoFeatureTransformer(
            categorical_encoding="onehot", verbose=0
        )
        result = transformer.fit_transform(sample_categorical_df, sample_target_binary)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_categorical_df)

    def test_with_missing_values(
        self, sample_df_with_missing: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test handling of missing values."""
        transformer = AutoFeatureTransformer(missing_strategy="auto", verbose=0)
        result = transformer.fit_transform(
            sample_df_with_missing, sample_target_binary
        )

        # Should handle missing values
        assert isinstance(result, pd.DataFrame)

    def test_random_state(
        self, sample_mixed_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test reproducibility with random_state."""
        transformer1 = AutoFeatureTransformer(random_state=42, verbose=0)
        result1 = transformer1.fit_transform(sample_mixed_df, sample_target_binary)

        transformer2 = AutoFeatureTransformer(random_state=42, verbose=0)
        result2 = transformer2.fit_transform(sample_mixed_df, sample_target_binary)

        pd.testing.assert_frame_equal(result1, result2)

    def test_preserves_index(
        self, sample_mixed_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test that transform preserves index."""
        df = sample_mixed_df.copy()
        df.index = pd.Index([f"row_{i}" for i in range(len(df))])

        transformer = AutoFeatureTransformer(verbose=0)
        result = transformer.fit_transform(df, sample_target_binary)

        assert (result.index == df.index).all()


class TestAutoFeatureTransformerConfigs:
    """Test different configuration options."""

    def test_numeric_transformations(
        self, sample_numeric_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test specific numeric transformations."""
        transformer = AutoFeatureTransformer(
            numeric_transformations=["log", "sqrt"],
            selection_method=None,  # Disable selection to see raw generated features
            verbose=0,
        )
        result = transformer.fit_transform(sample_numeric_df, sample_target_binary)

        # Should have generated log and sqrt features (before any selection)
        # Check the generators were created with proper names
        generator_names = [name for name, gen in transformer._generators]
        assert "log" in generator_names
        assert "sqrt" in generator_names

    def test_selection_method(
        self, sample_mixed_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test different selection methods."""
        for method in ["importance", "statistical", "correlation"]:
            transformer = AutoFeatureTransformer(
                selection_method=method,
                max_features=10,
                verbose=0,
            )
            result = transformer.fit_transform(sample_mixed_df, sample_target_binary)

            assert len(result.columns) <= 10

    def test_no_feature_generation(
        self, sample_numeric_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test with feature generation disabled."""
        transformer = AutoFeatureTransformer(
            numeric_transformations=[],
            verbose=0,
        )
        result = transformer.fit_transform(sample_numeric_df, sample_target_binary)

        # Should still work
        assert isinstance(result, pd.DataFrame)
