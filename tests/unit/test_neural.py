"""Tests for neural feature generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.neural import (
    NeuralEmbeddingGenerator,
    NeuralFeatureCross,
    NeuralFeatureGenerator,
)


@pytest.fixture
def numeric_df():
    np.random.seed(42)
    return pd.DataFrame({
        "a": np.random.randn(100),
        "b": np.random.randn(100),
        "c": np.random.randn(100),
    })


@pytest.fixture
def mixed_df():
    np.random.seed(42)
    return pd.DataFrame({
        "num1": np.random.randn(100),
        "num2": np.random.randn(100),
        "cat1": np.random.choice(["red", "green", "blue"], 100),
        "cat2": np.random.choice(["small", "medium", "large"], 100),
    })


class TestNeuralFeatureCross:
    """Tests for NeuralFeatureCross."""

    def test_fit_transform_basic(self, numeric_df):
        cross = NeuralFeatureCross(n_crosses=2, random_state=42)
        result = cross.fit_transform(numeric_df)
        assert result.shape[0] == 100
        assert result.shape[1] > numeric_df.shape[1]

    def test_output_columns_named(self, numeric_df):
        cross = NeuralFeatureCross(n_crosses=1, random_state=42)
        result = cross.fit_transform(numeric_df)
        cross_cols = [c for c in result.columns if c.startswith("cross_")]
        assert len(cross_cols) > 0

    def test_preserves_original_columns(self, numeric_df):
        cross = NeuralFeatureCross(random_state=42)
        result = cross.fit_transform(numeric_df)
        for col in numeric_df.columns:
            assert col in result.columns

    def test_column_selection(self, numeric_df):
        cross = NeuralFeatureCross(columns=["a", "b"], n_crosses=1, random_state=42)
        cross.fit(numeric_df)
        assert cross.feature_names_in_ == ["a", "b"]

    def test_projection_dim(self, numeric_df):
        cross = NeuralFeatureCross(
            n_crosses=1, projection_dim=5, random_state=42
        )
        result = cross.fit_transform(numeric_df)
        cross_cols = [c for c in result.columns if c.startswith("cross_")]
        assert len(cross_cols) == 5

    def test_multiple_crosses(self, numeric_df):
        cross1 = NeuralFeatureCross(n_crosses=1, random_state=42)
        cross3 = NeuralFeatureCross(n_crosses=3, random_state=42)
        r1 = cross1.fit_transform(numeric_df)
        r3 = cross3.fit_transform(numeric_df)
        cross_cols_1 = [c for c in r1.columns if c.startswith("cross_")]
        cross_cols_3 = [c for c in r3.columns if c.startswith("cross_")]
        assert len(cross_cols_3) > len(cross_cols_1)

    def test_get_feature_names_out(self, numeric_df):
        cross = NeuralFeatureCross(n_crosses=2, random_state=42)
        cross.fit(numeric_df)
        names = cross.get_feature_names_out()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_transform_before_fit_raises(self, numeric_df):
        cross = NeuralFeatureCross()
        with pytest.raises(RuntimeError, match="must be fitted"):
            cross.transform(numeric_df)

    def test_reproducibility(self, numeric_df):
        cross1 = NeuralFeatureCross(random_state=42)
        cross2 = NeuralFeatureCross(random_state=42)
        r1 = cross1.fit_transform(numeric_df)
        r2 = cross2.fit_transform(numeric_df)
        pd.testing.assert_frame_equal(r1, r2)

    def test_empty_columns(self):
        df = pd.DataFrame({"cat": ["a", "b", "c"]})
        cross = NeuralFeatureCross(random_state=42)
        result = cross.fit_transform(df)
        assert result.shape == df.shape


class TestNeuralEmbeddingGenerator:
    """Tests for NeuralEmbeddingGenerator."""

    def test_fit_transform_basic(self, mixed_df):
        emb = NeuralEmbeddingGenerator(
            columns=["cat1", "cat2"], random_state=42
        )
        result = emb.fit_transform(mixed_df)
        assert result.shape[0] == 100
        assert result.shape[1] > mixed_df.shape[1]

    def test_auto_detect_categorical(self, mixed_df):
        emb = NeuralEmbeddingGenerator(random_state=42)
        emb.fit(mixed_df)
        assert "cat1" in emb._embedded_columns
        assert "cat2" in emb._embedded_columns

    def test_embedding_dim_auto(self, mixed_df):
        emb = NeuralEmbeddingGenerator(random_state=42)
        emb.fit(mixed_df)
        for col in emb._embedded_columns:
            cardinality = mixed_df[col].nunique()
            expected_dim = min(50, 1 + cardinality // 2)
            assert emb._embedding_dims[col] == expected_dim

    def test_embedding_dim_fixed(self, mixed_df):
        emb = NeuralEmbeddingGenerator(
            columns=["cat1"], embedding_dim=10, random_state=42
        )
        emb.fit(mixed_df)
        assert emb._embedding_dims["cat1"] == 10

    def test_unknown_category_handling(self, mixed_df):
        emb = NeuralEmbeddingGenerator(
            columns=["cat1"], random_state=42
        )
        emb.fit(mixed_df)

        new_df = mixed_df.copy()
        new_df.loc[0, "cat1"] = "unknown_value"
        result = emb.transform(new_df)
        emb_cols = [c for c in result.columns if c.startswith("cat1_emb_")]
        assert not np.any(np.isnan(result[emb_cols].values[0]))

    def test_max_cardinality_filtering(self):
        df = pd.DataFrame({
            "high_card": [str(i) for i in range(200)],
        })
        emb = NeuralEmbeddingGenerator(max_cardinality=50, random_state=42)
        emb.fit(df)
        assert "high_card" not in emb._embedded_columns

    def test_preserves_original_columns(self, mixed_df):
        emb = NeuralEmbeddingGenerator(random_state=42)
        result = emb.fit_transform(mixed_df)
        for col in mixed_df.columns:
            assert col in result.columns

    def test_get_feature_names_out(self, mixed_df):
        emb = NeuralEmbeddingGenerator(random_state=42)
        emb.fit(mixed_df)
        names = emb.get_feature_names_out()
        assert all("emb" in n for n in names)

    def test_transform_before_fit_raises(self, mixed_df):
        emb = NeuralEmbeddingGenerator()
        with pytest.raises(RuntimeError, match="must be fitted"):
            emb.transform(mixed_df)


class TestNeuralFeatureGenerator:
    """Tests for combined NeuralFeatureGenerator."""

    def test_fit_transform_basic(self, mixed_df):
        gen = NeuralFeatureGenerator(random_state=42)
        result = gen.fit_transform(mixed_df)
        assert result.shape[0] == 100
        assert result.shape[1] > mixed_df.shape[1]

    def test_has_both_cross_and_embedding(self, mixed_df):
        gen = NeuralFeatureGenerator(random_state=42)
        result = gen.fit_transform(mixed_df)
        cross_cols = [c for c in result.columns if c.startswith("cross_")]
        emb_cols = [c for c in result.columns if "_emb_" in c]
        assert len(cross_cols) > 0
        assert len(emb_cols) > 0

    def test_column_selection(self, mixed_df):
        gen = NeuralFeatureGenerator(
            numeric_columns=["num1"],
            categorical_columns=["cat1"],
            random_state=42,
        )
        gen.fit(mixed_df)
        names = gen.get_feature_names_out()
        assert any("cross" in n for n in names)
        assert any("cat1_emb" in n for n in names)
        assert not any("cat2_emb" in n for n in names)

    def test_get_feature_names_out(self, mixed_df):
        gen = NeuralFeatureGenerator(random_state=42)
        gen.fit(mixed_df)
        names = gen.get_feature_names_out()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_transform_before_fit_raises(self, mixed_df):
        gen = NeuralFeatureGenerator()
        with pytest.raises(RuntimeError):
            gen.transform(mixed_df)
