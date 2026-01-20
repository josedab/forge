"""Tests for the analyzer module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.analyzer import DataAnalyzer
from forge.analyzer.quality import QualityAssessor
from forge.analyzer.statistics import StatisticsProfiler
from forge.analyzer.type_inference import TypeInferrer
from forge.types import ColumnType


class TestTypeInferrer:
    """Tests for TypeInferrer."""

    def test_infer_numeric(self, sample_numeric_df: pd.DataFrame):
        """Test numeric type inference."""
        inferrer = TypeInferrer()
        types = inferrer.infer_types(sample_numeric_df)

        assert types["age"] == ColumnType.NUMERIC
        assert types["income"] == ColumnType.NUMERIC
        assert types["score"] == ColumnType.NUMERIC

    def test_infer_categorical(self, sample_categorical_df: pd.DataFrame):
        """Test categorical type inference."""
        inferrer = TypeInferrer()
        types = inferrer.infer_types(sample_categorical_df)

        assert types["color"] == ColumnType.CATEGORICAL
        assert types["size"] == ColumnType.CATEGORICAL
        assert types["category"] == ColumnType.CATEGORICAL

    def test_infer_boolean(self, sample_categorical_df: pd.DataFrame):
        """Test boolean type inference."""
        inferrer = TypeInferrer()
        types = inferrer.infer_types(sample_categorical_df)

        assert types["flag"] == ColumnType.BOOLEAN

    def test_infer_datetime(self, sample_temporal_df: pd.DataFrame):
        """Test datetime type inference."""
        inferrer = TypeInferrer()
        types = inferrer.infer_types(sample_temporal_df)

        assert types["date"] == ColumnType.DATETIME
        assert types["timestamp"] == ColumnType.DATETIME

    def test_infer_text(self, sample_text_df: pd.DataFrame):
        """Test text type inference."""
        inferrer = TypeInferrer()
        types = inferrer.infer_types(sample_text_df)

        assert types["text"] == ColumnType.TEXT

    def test_infer_mixed(self, sample_mixed_df: pd.DataFrame):
        """Test mixed type inference."""
        inferrer = TypeInferrer()
        types = inferrer.infer_types(sample_mixed_df)

        assert types["age"] == ColumnType.NUMERIC
        assert types["category"] == ColumnType.CATEGORICAL
        assert types["date"] == ColumnType.DATETIME


class TestStatisticsProfiler:
    """Tests for StatisticsProfiler."""

    def test_profile_numeric(self, sample_numeric_df: pd.DataFrame):
        """Test numeric profiling."""
        profiler = StatisticsProfiler()
        stats = profiler.profile(sample_numeric_df)

        assert "age" in stats
        assert "mean" in stats["age"]
        assert "std" in stats["age"]
        assert "min" in stats["age"]
        assert "max" in stats["age"]

    def test_profile_categorical(self, sample_categorical_df: pd.DataFrame):
        """Test categorical profiling."""
        profiler = StatisticsProfiler()
        stats = profiler.profile(sample_categorical_df)

        assert "color" in stats
        assert "unique_count" in stats["color"]
        assert "mode" in stats["color"]

    def test_profile_with_missing(self, sample_df_with_missing: pd.DataFrame):
        """Test profiling with missing values."""
        profiler = StatisticsProfiler()
        stats = profiler.profile(sample_df_with_missing)

        assert stats["few_missing"]["missing_count"] > 0
        assert stats["many_missing"]["missing_count"] > 0


class TestQualityAssessor:
    """Tests for QualityAssessor."""

    def test_assess_missing(self, sample_df_with_missing: pd.DataFrame):
        """Test missing value detection."""
        assessor = QualityAssessor()
        report = assessor.assess(sample_df_with_missing)

        assert "missing" in report
        assert report["missing"]["few_missing"]["count"] > 0
        assert report["missing"]["many_missing"]["count"] > 0

    def test_assess_outliers(self, sample_df_with_outliers: pd.DataFrame):
        """Test outlier detection."""
        assessor = QualityAssessor()
        report = assessor.assess(sample_df_with_outliers)

        assert "outliers" in report
        assert "with_outliers" in report["outliers"]
        assert report["outliers"]["with_outliers"]["count"] >= 2

    def test_assess_high_cardinality(self, high_cardinality_df: pd.DataFrame):
        """Test high cardinality detection."""
        assessor = QualityAssessor()
        report = assessor.assess(high_cardinality_df)

        assert "high_cardinality" in report
        assert "id" in report["high_cardinality"]


class TestDataAnalyzer:
    """Tests for DataAnalyzer."""

    def test_analyze_basic(self, sample_mixed_df: pd.DataFrame):
        """Test basic analysis."""
        analyzer = DataAnalyzer()
        report = analyzer.analyze(sample_mixed_df)

        assert report is not None
        assert len(report.columns) == len(sample_mixed_df.columns)

    def test_analyze_with_target(
        self, sample_mixed_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Test analysis with target variable."""
        analyzer = DataAnalyzer()
        report = analyzer.analyze(sample_mixed_df, sample_target_binary)

        assert report is not None
        assert report.target_info is not None

    def test_infer_types(self, sample_mixed_df: pd.DataFrame):
        """Test type inference through analyzer."""
        analyzer = DataAnalyzer()
        types = analyzer.infer_types(sample_mixed_df)

        assert "age" in types
        assert types["age"] == ColumnType.NUMERIC

    def test_profile_statistics(self, sample_numeric_df: pd.DataFrame):
        """Test statistics profiling through analyzer."""
        analyzer = DataAnalyzer()
        stats = analyzer.profile_statistics(sample_numeric_df)

        assert "age" in stats
        assert "mean" in stats["age"]

    def test_assess_quality(self, sample_df_with_missing: pd.DataFrame):
        """Test quality assessment through analyzer."""
        analyzer = DataAnalyzer()
        quality = analyzer.assess_quality(sample_df_with_missing)

        assert "missing" in quality

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        analyzer = DataAnalyzer()

        with pytest.raises(ValueError):
            analyzer.analyze(pd.DataFrame())

    def test_single_column(self):
        """Test with single column DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        analyzer = DataAnalyzer()
        report = analyzer.analyze(df)

        assert len(report.columns) == 1
