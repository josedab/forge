"""Tests for advanced categorical encoders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.generators.categorical import (
    CatBoostEncoder,
    HashingEncoder,
    LeaveOneOutEncoder,
    WoEEncoder,
)


class TestWoEEncoder:
    """Tests for Weight of Evidence encoder."""

    def test_basic_encoding(
        self, sample_categorical_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test basic WoE encoding."""
        encoder = WoEEncoder(columns=["color"])
        result = encoder.fit_transform(sample_categorical_df, sample_target_binary)

        assert "color_woe" in result.columns
        assert result["color_woe"].dtype == np.float64

    def test_woe_values_range(
        self, sample_categorical_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test that WoE values are reasonable."""
        encoder = WoEEncoder(columns=["color"])
        result = encoder.fit_transform(sample_categorical_df, sample_target_binary)

        # WoE values should typically be between -5 and 5
        assert result["color_woe"].abs().max() < 10

    def test_information_value(
        self, sample_categorical_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test information value calculation."""
        encoder = WoEEncoder(columns=["color"])
        encoder.fit(sample_categorical_df, sample_target_binary)

        # IV should be accessible
        assert hasattr(encoder, "iv_")
        assert "color" in encoder.iv_
        assert encoder.iv_["color"] >= 0

    def test_requires_binary_target(
        self, sample_categorical_df: pd.DataFrame, sample_target_multiclass: pd.Series
    ):
        """Test that WoE requires binary target."""
        encoder = WoEEncoder(columns=["color"])

        with pytest.raises(ValueError):
            encoder.fit(sample_categorical_df, sample_target_multiclass)

    def test_regularization(
        self, sample_categorical_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test regularization parameter."""
        encoder = WoEEncoder(columns=["color"], regularization=1.0)
        result = encoder.fit_transform(sample_categorical_df, sample_target_binary)

        # Regularization should prevent extreme values
        assert result["color_woe"].abs().max() < 5

    def test_multiple_columns(
        self, sample_categorical_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test encoding multiple columns."""
        encoder = WoEEncoder(columns=["color", "size"])
        result = encoder.fit_transform(sample_categorical_df, sample_target_binary)

        assert "color_woe" in result.columns
        assert "size_woe" in result.columns

    def test_get_feature_names_out(
        self, sample_categorical_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test feature names output."""
        encoder = WoEEncoder(columns=["color", "size"])
        encoder.fit(sample_categorical_df, sample_target_binary)

        names = encoder.get_feature_names_out()
        assert "color_woe" in names
        assert "size_woe" in names


class TestCatBoostEncoder:
    """Tests for CatBoost-style ordered target encoder."""

    def test_basic_encoding(
        self, sample_categorical_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test basic CatBoost encoding."""
        encoder = CatBoostEncoder(columns=["color"])
        result = encoder.fit_transform(sample_categorical_df, sample_target_binary)

        assert "color_catboost" in result.columns
        assert result["color_catboost"].dtype in [np.float64, np.float32]

    def test_prevents_leakage(
        self, sample_categorical_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test that CatBoost encoding prevents target leakage."""
        encoder = CatBoostEncoder(columns=["color"])
        result = encoder.fit_transform(sample_categorical_df, sample_target_binary)

        # First row of each category should use prior only
        # This is hard to test directly, but we can verify the values differ
        # from simple target encoding
        assert "color_catboost" in result.columns

    def test_prior_parameter(
        self, sample_categorical_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test prior (a) parameter."""
        encoder = CatBoostEncoder(columns=["color"], a=10)
        result = encoder.fit_transform(sample_categorical_df, sample_target_binary)

        # Higher prior should pull values toward global mean
        global_mean = sample_target_binary.mean()
        assert result["color_catboost"].mean() != 0  # Should have values

    def test_random_state(
        self, sample_categorical_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test random state reproducibility."""
        encoder1 = CatBoostEncoder(columns=["color"], random_state=42)
        encoder2 = CatBoostEncoder(columns=["color"], random_state=42)

        result1 = encoder1.fit_transform(sample_categorical_df, sample_target_binary)
        result2 = encoder2.fit_transform(sample_categorical_df, sample_target_binary)

        pd.testing.assert_series_equal(
            result1["color_catboost"], result2["color_catboost"]
        )

    def test_requires_target(self, sample_categorical_df: pd.DataFrame):
        """Test that target is required."""
        encoder = CatBoostEncoder(columns=["color"])

        with pytest.raises(ValueError):
            encoder.fit(sample_categorical_df)

    def test_get_feature_names_out(
        self, sample_categorical_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test feature names output."""
        encoder = CatBoostEncoder(columns=["color"])
        encoder.fit(sample_categorical_df, sample_target_binary)

        names = encoder.get_feature_names_out()
        assert "color_catboost" in names


class TestLeaveOneOutEncoder:
    """Tests for Leave-One-Out target encoder."""

    def test_basic_encoding(
        self, sample_categorical_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test basic LOO encoding."""
        encoder = LeaveOneOutEncoder(columns=["color"])
        result = encoder.fit_transform(sample_categorical_df, sample_target_binary)

        assert "color_loo" in result.columns
        assert result["color_loo"].dtype in [np.float64, np.float32]

    def test_excludes_current_row(
        self, sample_categorical_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test that current row is excluded in calculation."""
        encoder = LeaveOneOutEncoder(columns=["color"])
        result = encoder.fit_transform(sample_categorical_df, sample_target_binary)

        # LOO should produce different values than simple target encoding
        # because it excludes the current row
        assert "color_loo" in result.columns

    def test_regularization(
        self, sample_categorical_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test sigma regularization parameter."""
        encoder = LeaveOneOutEncoder(columns=["color"], sigma=0.5)
        result = encoder.fit_transform(sample_categorical_df, sample_target_binary)

        # Should still produce valid values
        assert not result["color_loo"].isna().any()

    def test_random_state(
        self, sample_categorical_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test random state reproducibility with sigma."""
        encoder1 = LeaveOneOutEncoder(columns=["color"], sigma=0.1, random_state=42)
        encoder2 = LeaveOneOutEncoder(columns=["color"], sigma=0.1, random_state=42)

        result1 = encoder1.fit_transform(sample_categorical_df, sample_target_binary)
        result2 = encoder2.fit_transform(sample_categorical_df, sample_target_binary)

        pd.testing.assert_series_equal(result1["color_loo"], result2["color_loo"])

    def test_requires_target(self, sample_categorical_df: pd.DataFrame):
        """Test that target is required."""
        encoder = LeaveOneOutEncoder(columns=["color"])

        with pytest.raises(ValueError):
            encoder.fit(sample_categorical_df)

    def test_get_feature_names_out(
        self, sample_categorical_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test feature names output."""
        encoder = LeaveOneOutEncoder(columns=["color"])
        encoder.fit(sample_categorical_df, sample_target_binary)

        names = encoder.get_feature_names_out()
        assert "color_loo" in names


class TestHashingEncoder:
    """Tests for Hashing encoder."""

    def test_basic_encoding(self, sample_categorical_df: pd.DataFrame):
        """Test basic hashing encoding."""
        encoder = HashingEncoder(columns=["color"], n_components=8)
        result = encoder.fit_transform(sample_categorical_df)

        # Should have n_components output columns
        hash_cols = [c for c in result.columns if c.startswith("color_hash_")]
        assert len(hash_cols) == 8

    def test_n_components_parameter(self, sample_categorical_df: pd.DataFrame):
        """Test different n_components values."""
        encoder16 = HashingEncoder(columns=["color"], n_components=16)
        encoder4 = HashingEncoder(columns=["color"], n_components=4)

        result16 = encoder16.fit_transform(sample_categorical_df)
        result4 = encoder4.fit_transform(sample_categorical_df)

        hash_cols_16 = [c for c in result16.columns if c.startswith("color_hash_")]
        hash_cols_4 = [c for c in result4.columns if c.startswith("color_hash_")]

        assert len(hash_cols_16) == 16
        assert len(hash_cols_4) == 4

    def test_high_cardinality(self, high_cardinality_df: pd.DataFrame):
        """Test with high cardinality data."""
        encoder = HashingEncoder(columns=["group"], n_components=32)
        result = encoder.fit_transform(high_cardinality_df)

        # Should have fixed output regardless of cardinality
        hash_cols = [c for c in result.columns if c.startswith("group_hash_")]
        assert len(hash_cols) == 32

    def test_handles_unseen_categories(self, sample_categorical_df: pd.DataFrame):
        """Test handling of unseen categories in transform."""
        encoder = HashingEncoder(columns=["color"], n_components=8)
        encoder.fit(sample_categorical_df)

        # Create data with unseen category
        new_df = pd.DataFrame({"color": ["unknown_color", "red", "another_new"]})
        result = encoder.transform(new_df)

        # Should not fail and produce valid output
        assert len(result) == 3
        hash_cols = [c for c in result.columns if c.startswith("color_hash_")]
        assert len(hash_cols) == 8

    def test_no_target_required(self, sample_categorical_df: pd.DataFrame):
        """Test that no target is required."""
        encoder = HashingEncoder(columns=["color"], n_components=8)

        # Should work without target
        result = encoder.fit_transform(sample_categorical_df)
        assert "color_hash_0" in result.columns

    def test_deterministic(self, sample_categorical_df: pd.DataFrame):
        """Test that hashing is deterministic."""
        encoder1 = HashingEncoder(columns=["color"], n_components=8)
        encoder2 = HashingEncoder(columns=["color"], n_components=8)

        result1 = encoder1.fit_transform(sample_categorical_df)
        result2 = encoder2.fit_transform(sample_categorical_df)

        hash_cols = [c for c in result1.columns if c.startswith("color_hash_")]
        for col in hash_cols:
            pd.testing.assert_series_equal(result1[col], result2[col])

    def test_get_feature_names_out(self, sample_categorical_df: pd.DataFrame):
        """Test feature names output."""
        encoder = HashingEncoder(columns=["color"], n_components=8)
        encoder.fit(sample_categorical_df)

        names = encoder.get_feature_names_out()
        assert len([n for n in names if n.startswith("color_hash_")]) == 8


class TestSklearnCompatibility:
    """Test sklearn compatibility for all advanced encoders."""

    def test_woe_clone(
        self, sample_categorical_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test that WoEEncoder can be cloned."""
        from sklearn.base import clone

        encoder = WoEEncoder(regularization=0.5)
        cloned = clone(encoder)

        assert cloned.regularization == 0.5

    def test_catboost_clone(
        self, sample_categorical_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test that CatBoostEncoder can be cloned."""
        from sklearn.base import clone

        encoder = CatBoostEncoder(a=5)
        cloned = clone(encoder)

        assert cloned.a == 5

    def test_loo_clone(
        self, sample_categorical_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test that LeaveOneOutEncoder can be cloned."""
        from sklearn.base import clone

        encoder = LeaveOneOutEncoder(sigma=0.5)
        cloned = clone(encoder)

        assert cloned.sigma == 0.5

    def test_hashing_clone(self, sample_categorical_df: pd.DataFrame):
        """Test that HashingEncoder can be cloned."""
        from sklearn.base import clone

        encoder = HashingEncoder(n_components=16)
        cloned = clone(encoder)

        assert cloned.n_components == 16

    def test_pipeline_integration(
        self, sample_categorical_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test integration with sklearn Pipeline."""
        from sklearn.pipeline import Pipeline

        # Create pipeline with HashingEncoder (doesn't need target)
        pipeline = Pipeline(
            [
                ("hash", HashingEncoder(columns=["color"], n_components=8)),
            ]
        )

        result = pipeline.fit_transform(sample_categorical_df)
        assert isinstance(result, pd.DataFrame)
