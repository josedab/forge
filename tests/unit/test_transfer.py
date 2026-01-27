"""Tests for cross-dataset feature transfer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.transfer import (
    ColumnFingerprint,
    FeatureTransferEngine,
    TransferResult,
)


@pytest.fixture
def source_df():
    np.random.seed(42)
    return pd.DataFrame({
        "price": np.random.lognormal(3, 1, 200),
        "quantity": np.random.poisson(10, 200).astype(float),
        "rating": np.random.uniform(1, 5, 200),
    })


@pytest.fixture
def similar_target_df():
    np.random.seed(99)
    return pd.DataFrame({
        "cost": np.random.lognormal(3.1, 1.1, 150),
        "count": np.random.poisson(11, 150).astype(float),
        "score": np.random.uniform(1, 5, 150),
    })


@pytest.fixture
def different_target_df():
    np.random.seed(99)
    return pd.DataFrame({
        "category": np.random.choice(["A", "B", "C"], 150),
        "flag": np.random.choice([0, 1], 150).astype(float),
    })


class TestColumnFingerprint:
    """Tests for ColumnFingerprint."""

    def test_same_fingerprint_high_similarity(self):
        fp = ColumnFingerprint(
            name="col", dtype="numeric",
            stats={"mean": 5.0, "std": 2.0, "min": 0.0, "max": 10.0},
            distribution=np.array([1, 2, 3, 4, 5, 5, 4, 3, 2, 1], dtype=float),
        )
        assert fp.similarity(fp) > 0.95

    def test_different_dtype_zero_similarity(self):
        fp1 = ColumnFingerprint(name="num", dtype="numeric", stats={"mean": 5.0})
        fp2 = ColumnFingerprint(name="cat", dtype="categorical", stats={"mean": 5.0})
        assert fp1.similarity(fp2) == 0.0

    def test_similar_stats_high_similarity(self):
        fp1 = ColumnFingerprint(
            name="a", dtype="numeric",
            stats={"mean": 5.0, "std": 2.0},
        )
        fp2 = ColumnFingerprint(
            name="b", dtype="numeric",
            stats={"mean": 5.1, "std": 2.1},
        )
        assert fp1.similarity(fp2) > 0.8

    def test_different_stats_low_similarity(self):
        fp1 = ColumnFingerprint(
            name="a", dtype="numeric",
            stats={"mean": 0.0, "std": 1.0},
        )
        fp2 = ColumnFingerprint(
            name="b", dtype="numeric",
            stats={"mean": 100.0, "std": 50.0},
        )
        assert fp1.similarity(fp2) < 0.5


class TestFeatureTransferEngine:
    """Tests for FeatureTransferEngine."""

    def test_fit_basic(self, source_df):
        engine = FeatureTransferEngine()
        engine.fit(source_df)
        assert engine._is_fitted
        assert len(engine.source_fingerprint_.columns) == 3

    def test_fit_learns_transforms(self, source_df):
        engine = FeatureTransferEngine()
        engine.fit(source_df)
        assert len(engine.learned_transforms_) > 0

    def test_transform_similar_data(self, source_df, similar_target_df):
        engine = FeatureTransferEngine(similarity_threshold=0.5)
        engine.fit(source_df)
        result = engine.transform(similar_target_df)
        assert result.shape[1] >= similar_target_df.shape[1]

    def test_transfer_creates_new_columns(self, source_df, similar_target_df):
        engine = FeatureTransferEngine(similarity_threshold=0.5)
        engine.fit(source_df)
        result = engine.transform(similar_target_df)
        transfer_cols = [c for c in result.columns if "_transfer" in c]
        assert len(transfer_cols) > 0

    def test_preserves_original_columns(self, source_df, similar_target_df):
        engine = FeatureTransferEngine(similarity_threshold=0.5)
        engine.fit(source_df)
        result = engine.transform(similar_target_df)
        for col in similar_target_df.columns:
            assert col in result.columns

    def test_no_transfer_for_different_data(self, source_df, different_target_df):
        engine = FeatureTransferEngine(similarity_threshold=0.8)
        engine.fit(source_df)
        result = engine.transform(different_target_df)
        transfer_cols = [c for c in result.columns if "_transfer" in c]
        assert len(transfer_cols) == 0

    def test_match_columns(self, source_df, similar_target_df):
        engine = FeatureTransferEngine(similarity_threshold=0.3)
        engine.fit(source_df)
        matches = engine.match_columns(similar_target_df)
        assert isinstance(matches, dict)

    def test_get_feature_names_out(self, source_df, similar_target_df):
        engine = FeatureTransferEngine(similarity_threshold=0.5)
        engine.fit(source_df)
        engine.transform(similar_target_df)
        names = engine.get_feature_names_out()
        assert all("_transfer" in n for n in names)

    def test_threshold_filtering(self, source_df, similar_target_df):
        engine_low = FeatureTransferEngine(similarity_threshold=0.3)
        engine_high = FeatureTransferEngine(similarity_threshold=0.95)

        engine_low.fit(source_df)
        result_low = engine_low.transform(similar_target_df)

        engine_high.fit(source_df)
        result_high = engine_high.transform(similar_target_df)

        low_transfers = [c for c in result_low.columns if "_transfer" in c]
        high_transfers = [c for c in result_high.columns if "_transfer" in c]
        assert len(low_transfers) >= len(high_transfers)

    def test_transform_before_fit_raises(self, similar_target_df):
        engine = FeatureTransferEngine()
        with pytest.raises(RuntimeError, match="must be fitted"):
            engine.transform(similar_target_df)

    def test_custom_transformations(self, source_df, similar_target_df):
        engine = FeatureTransferEngine(
            transformations=["log", "sqrt"],
            similarity_threshold=0.5,
        )
        engine.fit(source_df)
        result = engine.transform(similar_target_df)
        transfer_cols = [c for c in result.columns if "_transfer" in c]
        for col in transfer_cols:
            assert "log" in col or "sqrt" in col


class TestTransferResult:
    """Tests for TransferResult dataclass."""

    def test_creation(self):
        result = TransferResult(
            source_column="price",
            target_column="cost",
            similarity=0.85,
            transform_name="log",
            success=True,
        )
        assert result.source_column == "price"
        assert result.success is True
