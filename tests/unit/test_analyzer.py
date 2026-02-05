"""Tests for the analyzer module."""

from __future__ import annotations

import pandas as pd

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
        # Use lower cardinality threshold since fixture has limited unique values
        inferrer = TypeInferrer(categorical_threshold=5)
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
        inferrer = TypeInferrer()
        column_types = inferrer.infer_types(sample_numeric_df)

        profiler = StatisticsProfiler()
        stats = profiler.profile(sample_numeric_df, column_types)

        assert "age" in stats
        assert "mean" in stats["age"]
        assert "std" in stats["age"]
        assert "min" in stats["age"]
        assert "max" in stats["age"]

    def test_profile_categorical(self, sample_categorical_df: pd.DataFrame):
        """Test categorical profiling."""
        inferrer = TypeInferrer()
        column_types = inferrer.infer_types(sample_categorical_df)

        profiler = StatisticsProfiler()
        stats = profiler.profile(sample_categorical_df, column_types)

        assert "color" in stats
        assert "unique_count" in stats["color"]
        assert "mode" in stats["color"]

    def test_profile_with_missing(self, sample_df_with_missing: pd.DataFrame):
        """Test profiling with missing values."""
        inferrer = TypeInferrer()
        column_types = inferrer.infer_types(sample_df_with_missing)

        profiler = StatisticsProfiler()
        stats = profiler.profile(sample_df_with_missing, column_types)

        assert stats["few_missing"]["null_count"] > 0
        assert stats["many_missing"]["null_count"] > 0


class TestQualityAssessor:
    """Tests for QualityAssessor."""

    def test_assess_missing(self, sample_df_with_missing: pd.DataFrame):
        """Test missing value detection."""
        inferrer = TypeInferrer()
        column_types = inferrer.infer_types(sample_df_with_missing)

        # Lower threshold to detect the 30% missing in fixture
        assessor = QualityAssessor(missing_threshold=0.25)
        issues = assessor.assess(sample_df_with_missing, column_types)

        # Issues is a list of QualityIssue objects
        missing_issues = [i for i in issues if i.issue_type == "high_missing"]
        assert len(missing_issues) > 0

    def test_assess_outliers(self, sample_df_with_outliers: pd.DataFrame):
        """Test outlier detection."""
        inferrer = TypeInferrer()
        column_types = inferrer.infer_types(sample_df_with_outliers)

        assessor = QualityAssessor()
        issues = assessor.assess(sample_df_with_outliers, column_types)

        # Issues is a list of QualityIssue objects
        outlier_issues = [i for i in issues if i.issue_type == "outliers"]
        assert len(outlier_issues) >= 1

    def test_assess_high_cardinality(self, high_cardinality_df: pd.DataFrame):
        """Test high cardinality detection."""
        inferrer = TypeInferrer()
        column_types = inferrer.infer_types(high_cardinality_df)

        assessor = QualityAssessor()
        issues = assessor.assess(high_cardinality_df, column_types)

        # Issues is a list of QualityIssue objects
        cardinality_issues = [i for i in issues if i.issue_type == "high_cardinality"]
        assert len(cardinality_issues) >= 1


class TestDataAnalyzer:
    """Tests for DataAnalyzer."""

    def test_analyze_basic(self, sample_mixed_df: pd.DataFrame):
        """Test basic analysis."""
        analyzer = DataAnalyzer()
        report = analyzer.analyze(sample_mixed_df)

        assert report is not None
        assert report.n_columns == len(sample_mixed_df.columns)

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

        # Quality is a dict with overall_score, completeness, etc.
        assert "overall_score" in quality or "completeness" in quality

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        analyzer = DataAnalyzer()
        # Empty dataframe should either raise an error or return empty report
        report = analyzer.analyze(pd.DataFrame())
        assert report.n_rows == 0 or report.n_columns == 0

    def test_single_column(self):
        """Test with single column DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        analyzer = DataAnalyzer()
        report = analyzer.analyze(df)

        assert report.n_columns == 1
