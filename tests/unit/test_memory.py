"""Tests for memory utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.utils.memory import (
    MemoryEstimate,
    estimate_dataframe_size,
    estimate_feature_engineering_memory,
    get_memory_usage_summary,
    suggest_dtype_optimizations,
)


class TestEstimateDataframeSize:
    """Tests for estimate_dataframe_size function."""

    def test_basic_estimation(self, sample_numeric_df: pd.DataFrame):
        """Test basic size estimation."""
        size = estimate_dataframe_size(sample_numeric_df)
        assert size > 0
        assert isinstance(size, float)

    def test_larger_df_larger_size(self):
        """Test that larger DataFrame has larger size."""
        small = pd.DataFrame({"a": range(100)})
        large = pd.DataFrame({"a": range(10000)})

        small_size = estimate_dataframe_size(small)
        large_size = estimate_dataframe_size(large)

        assert large_size > small_size

    def test_more_columns_larger_size(self):
        """Test that more columns increases size."""
        narrow = pd.DataFrame({"a": range(1000)})
        wide = pd.DataFrame({f"col_{i}": range(1000) for i in range(10)})

        narrow_size = estimate_dataframe_size(narrow)
        wide_size = estimate_dataframe_size(wide)

        assert wide_size > narrow_size


class TestEstimateFeatureEngineeringMemory:
    """Tests for estimate_feature_engineering_memory function."""

    def test_returns_memory_estimate(self, sample_numeric_df: pd.DataFrame):
        """Test that function returns MemoryEstimate."""
        estimate = estimate_feature_engineering_memory(sample_numeric_df)

        assert isinstance(estimate, MemoryEstimate)
        assert estimate.input_size_mb > 0
        assert estimate.output_size_mb > 0
        assert estimate.peak_size_mb > 0
        assert estimate.available_mb > 0

    def test_with_max_features(self, sample_numeric_df: pd.DataFrame):
        """Test estimation with max_features limit."""
        estimate_unlimited = estimate_feature_engineering_memory(
            sample_numeric_df, max_features=None
        )
        estimate_limited = estimate_feature_engineering_memory(
            sample_numeric_df, max_features=10
        )

        # Limited should have smaller output
        assert estimate_limited.output_size_mb <= estimate_unlimited.output_size_mb

    def test_recommendation_on_safe(self, sample_numeric_df: pd.DataFrame):
        """Test that safe estimates have appropriate recommendation."""
        estimate = estimate_feature_engineering_memory(sample_numeric_df)

        # Small test data should always be safe
        assert estimate.is_safe is True
        assert "Sufficient" in estimate.recommendation or "safe" in estimate.recommendation.lower()

    def test_str_representation(self, sample_numeric_df: pd.DataFrame):
        """Test string representation."""
        estimate = estimate_feature_engineering_memory(sample_numeric_df)
        string = str(estimate)

        assert "Memory Estimate" in string
        assert "Input" in string
        assert "Output" in string
        assert "Peak" in string


class TestGetMemoryUsageSummary:
    """Tests for get_memory_usage_summary function."""

    def test_returns_all_keys(self, sample_mixed_df: pd.DataFrame):
        """Test that all expected keys are returned."""
        summary = get_memory_usage_summary(sample_mixed_df)

        assert "total_mb" in summary
        assert "numeric_mb" in summary
        assert "object_mb" in summary
        assert "rows" in summary
        assert "columns" in summary

    def test_row_column_counts(self, sample_numeric_df: pd.DataFrame):
        """Test row and column counts are accurate."""
        summary = get_memory_usage_summary(sample_numeric_df)

        assert summary["rows"] == len(sample_numeric_df)
        assert summary["columns"] == len(sample_numeric_df.columns)

    def test_total_matches_sum(self, sample_numeric_df: pd.DataFrame):
        """Test that individual sizes sum approximately to total."""
        summary = get_memory_usage_summary(sample_numeric_df)

        # For numeric-only df, numeric_mb should be close to total
        assert summary["numeric_mb"] <= summary["total_mb"]


class TestSuggestDtypeOptimizations:
    """Tests for suggest_dtype_optimizations function."""

    def test_object_to_category_suggestion(self):
        """Test suggestion for object columns with low cardinality."""
        df = pd.DataFrame({
            "category": ["A", "B", "C"] * 1000,  # Low cardinality
        })

        suggestions = suggest_dtype_optimizations(df)
        assert any("category" in s and "Convert to category" in s for s in suggestions)

    def test_int64_to_smaller_suggestion(self):
        """Test suggestion for int64 that could be smaller."""
        df = pd.DataFrame({
            "small_int": np.array([1, 2, 3, 4, 5] * 100, dtype=np.int64),
        })

        suggestions = suggest_dtype_optimizations(df)
        assert any("small_int" in s for s in suggestions)

    def test_no_suggestion_for_optimal(self):
        """Test no suggestions for already optimal dtypes."""
        df = pd.DataFrame({
            "optimal": np.array([1, 2, 3], dtype=np.int8),
        })

        suggestions = suggest_dtype_optimizations(df)
        # int8 should not have suggestions
        assert not any("optimal" in s for s in suggestions)

    def test_float64_to_float32_suggestion(self):
        """Test suggestion for float64 that could be float32."""
        df = pd.DataFrame({
            "float_col": np.random.randn(100).astype(np.float64),
        })

        suggestions = suggest_dtype_optimizations(df)
        assert any("float32" in s for s in suggestions)
