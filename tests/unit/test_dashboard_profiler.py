"""Tests for the dashboard profiler module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.dashboard.profiler import (
    DatasetProfile,
    DatasetProfiler,
    _generate_pipeline_code,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame for profiling."""
    np.random.seed(42)
    return pd.DataFrame({
        "numeric_a": np.random.randn(100),
        "numeric_b": np.random.uniform(0, 100, 100),
        "category": np.random.choice(["A", "B", "C", "D"], 100),
        "boolean_col": np.random.choice([True, False], 100),
        "constant": [1] * 100,
        "with_nulls": [np.nan if i % 3 == 0 else float(i) for i in range(100)],
        "text_col": ["short"] * 50 + ["a" * 60] * 50,
    })


@pytest.fixture
def profiler() -> DatasetProfiler:
    """Create a profiler instance."""
    return DatasetProfiler(max_categories=50, high_null_threshold=0.3)


class TestDatasetProfiler:
    """Tests for DatasetProfiler."""

    def test_profile_returns_dataset_profile(
        self, profiler: DatasetProfiler, sample_df: pd.DataFrame
    ) -> None:
        profile = profiler.profile(sample_df)
        assert isinstance(profile, DatasetProfile)
        assert profile.n_rows == 100
        assert profile.n_columns == 7

    def test_profile_column_count(
        self, profiler: DatasetProfiler, sample_df: pd.DataFrame
    ) -> None:
        profile = profiler.profile(sample_df)
        assert len(profile.column_profiles) == 7

    def test_type_inference_numeric(
        self, profiler: DatasetProfiler, sample_df: pd.DataFrame
    ) -> None:
        profile = profiler.profile(sample_df)
        num_a = next(p for p in profile.column_profiles if p.name == "numeric_a")
        assert num_a.inferred_type == "numeric"
        assert num_a.mean is not None
        assert num_a.std is not None

    def test_type_inference_categorical(
        self, profiler: DatasetProfiler, sample_df: pd.DataFrame
    ) -> None:
        profile = profiler.profile(sample_df)
        cat = next(p for p in profile.column_profiles if p.name == "category")
        assert cat.inferred_type == "categorical"
        assert len(cat.top_values) > 0

    def test_type_inference_boolean(
        self, profiler: DatasetProfiler, sample_df: pd.DataFrame
    ) -> None:
        profile = profiler.profile(sample_df)
        bool_col = next(p for p in profile.column_profiles if p.name == "boolean_col")
        assert bool_col.inferred_type == "boolean"

    def test_constant_column_detection(
        self, profiler: DatasetProfiler, sample_df: pd.DataFrame
    ) -> None:
        profile = profiler.profile(sample_df)
        const = next(p for p in profile.column_profiles if p.name == "constant")
        assert const.unique_count == 1
        assert any("Constant" in issue or "constant" in issue.lower()
                    for issue in const.quality_issues)
        assert const.quality_score < 1.0

    def test_null_detection(
        self, profiler: DatasetProfiler, sample_df: pd.DataFrame
    ) -> None:
        profile = profiler.profile(sample_df)
        null_col = next(p for p in profile.column_profiles if p.name == "with_nulls")
        assert null_col.null_count > 0
        assert null_col.null_pct > 0

    def test_type_summary(
        self, profiler: DatasetProfiler, sample_df: pd.DataFrame
    ) -> None:
        profile = profiler.profile(sample_df)
        assert isinstance(profile.type_summary, dict)
        assert sum(profile.type_summary.values()) == 7

    def test_memory_calculation(
        self, profiler: DatasetProfiler, sample_df: pd.DataFrame
    ) -> None:
        profile = profiler.profile(sample_df)
        assert profile.memory_mb > 0

    def test_overall_quality(
        self, profiler: DatasetProfiler, sample_df: pd.DataFrame
    ) -> None:
        profile = profiler.profile(sample_df)
        assert 0 <= profile.overall_quality <= 1.0

    def test_duplicate_detection(self, profiler: DatasetProfiler) -> None:
        df = pd.DataFrame({"a": [1, 1, 2, 2], "b": ["x", "x", "y", "y"]})
        profile = profiler.profile(df)
        assert profile.duplicate_rows == 2
        assert profile.duplicate_pct == 50.0

    def test_recommendations_generated(
        self, profiler: DatasetProfiler, sample_df: pd.DataFrame
    ) -> None:
        profile = profiler.profile(sample_df)
        assert isinstance(profile.recommendations, list)
        assert len(profile.recommendations) > 0

    def test_generator_recommendations(
        self, profiler: DatasetProfiler, sample_df: pd.DataFrame
    ) -> None:
        profile = profiler.profile(sample_df)
        num_a = next(p for p in profile.column_profiles if p.name == "numeric_a")
        assert len(num_a.recommended_generators) > 0
        assert "PolynomialGenerator" in num_a.recommended_generators

    def test_to_dataframe(
        self, profiler: DatasetProfiler, sample_df: pd.DataFrame
    ) -> None:
        profile = profiler.profile(sample_df)
        df_result = profile.to_dataframe()
        assert isinstance(df_result, pd.DataFrame)
        assert len(df_result) == 7
        assert "Column" in df_result.columns
        assert "Quality" in df_result.columns


class TestGeneratePipelineCode:
    """Tests for pipeline code generation."""

    def test_generates_valid_code(
        self, profiler: DatasetProfiler, sample_df: pd.DataFrame
    ) -> None:
        profile = profiler.profile(sample_df)
        code = _generate_pipeline_code(profile)
        assert isinstance(code, str)
        assert "from forge" in code
        assert "pipeline" in code.lower()

    def test_includes_numeric_generators(
        self, profiler: DatasetProfiler, sample_df: pd.DataFrame
    ) -> None:
        profile = profiler.profile(sample_df)
        code = _generate_pipeline_code(profile)
        assert "InteractionGenerator" in code

    def test_includes_categorical_generators(
        self, profiler: DatasetProfiler, sample_df: pd.DataFrame
    ) -> None:
        profile = profiler.profile(sample_df)
        code = _generate_pipeline_code(profile)
        assert "TargetEncoder" in code


class TestEmptyDataFrame:
    """Edge case tests for empty or minimal DataFrames."""

    def test_empty_df(self, profiler: DatasetProfiler) -> None:
        df = pd.DataFrame()
        profile = profiler.profile(df)
        assert profile.n_rows == 0
        assert profile.n_columns == 0
        assert profile.overall_quality == 1.0

    def test_single_row(self, profiler: DatasetProfiler) -> None:
        df = pd.DataFrame({"a": [1], "b": ["x"]})
        profile = profiler.profile(df)
        assert profile.n_rows == 1
        assert len(profile.column_profiles) == 2

    def test_all_null_column(self, profiler: DatasetProfiler) -> None:
        df = pd.DataFrame({"a": [None, None, None]})
        profile = profiler.profile(df)
        col = profile.column_profiles[0]
        assert col.null_pct == 100.0


class TestTemporalDetection:
    """Tests for temporal type inference."""

    def test_datetime_column(self, profiler: DatasetProfiler) -> None:
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=10, freq="D")
        })
        profile = profiler.profile(df)
        assert profile.column_profiles[0].inferred_type == "temporal"
        assert "TemporalComponents" in profile.column_profiles[0].recommended_generators
