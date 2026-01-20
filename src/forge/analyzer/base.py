"""Main DataAnalyzer class for Forge."""

from __future__ import annotations

from typing import Any

import pandas as pd

from forge.analyzer.quality import QualityAssessor
from forge.analyzer.report import ReportBuilder
from forge.analyzer.statistics import StatisticsProfiler
from forge.analyzer.type_inference import TypeInferrer
from forge.types import AnalysisReport, ColumnType


class DataAnalyzer:
    """Analyzes DataFrames to understand data characteristics.

    Combines type inference, statistical profiling, and quality assessment
    to provide comprehensive data analysis.

    Example:
        >>> analyzer = DataAnalyzer()
        >>> report = analyzer.analyze(X, y)
        >>> print(report.summary())
        >>> print(report.column_types)
    """

    def __init__(
        self,
        categorical_threshold: int = 50,
        categorical_ratio: float = 0.05,
        missing_threshold: float = 0.3,
        constant_threshold: float = 0.99
    ) -> None:
        """Initialize the DataAnalyzer.

        Args:
            categorical_threshold: Max unique values for categorical detection.,
            categorical_ratio: Max unique ratio for categorical detection.,
            missing_threshold: Ratio above which missing values are flagged.,
            constant_threshold: Ratio above which columns are flagged as constant.
        """
        self.type_inferrer = TypeInferrer(
            categorical_threshold = categorical_threshold,
            categorical_ratio=categorical_ratio
        )
        self.stats_profiler = StatisticsProfiler()
        self.quality_assessor = QualityAssessor(
            missing_threshold = missing_threshold,
            constant_threshold=constant_threshold
        )
        self.report_builder = ReportBuilder()

    def analyze(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None
    ) -> AnalysisReport:
        """Perform comprehensive data analysis.

        Args:
            X: Input DataFrame.,
            y: Optional target variable.,

        Returns:
            Complete analysis report.
        """
        # Infer column types
        column_types = self.infer_types(X)

        # Compute statistics
        statistics = self.profile_statistics(X, column_types)

        # Assess quality
        quality_issues = self.quality_assessor.assess(X, column_types, y)

        # Build report
        return self.report_builder.build_report(
            df = X,
            column_types = column_types,
            statistics = statistics,
            quality_issues = quality_issues,
            target=y
        )

    def infer_types(self, X: pd.DataFrame) -> dict[str, ColumnType]:
        """Infer semantic types for all columns.

        Args:
            X: Input DataFrame.,

        Returns:
            Dictionary mapping column names to types.
        """
        return self.type_inferrer.infer_types(X)

    def profile_statistics(
        self,
        X: pd.DataFrame,
        column_types: dict[str, ColumnType] | None = None
    ) -> dict[str, dict[str, Any]]:
        """Compute descriptive statistics for all columns.

        Args:
            X: Input DataFrame.,
            column_types: Optional pre-computed column types.,

        Returns:
            Dictionary mapping column names to statistics.
        """
        if column_types is None:
            column_types = self.infer_types(X)

        return self.stats_profiler.profile(X, column_types)

    def assess_quality(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None
    ) -> dict[str, Any]:
        """Assess data quality and return quality metrics.

        Args:
            X: Input DataFrame.,
            y: Optional target variable.,

        Returns:
            Dictionary with quality metrics and scores.
        """
        column_types = self.infer_types(X)
        return self.quality_assessor.get_quality_score(X, column_types)

    def get_column_info(self, X: pd.DataFrame, column: str) -> dict[str, Any]:
        """Get detailed information about a specific column.

        Args:
            X: Input DataFrame.,
            column: Column name.,

        Returns:
            Dictionary with column information.
        """
        if column not in X.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        series = X[column]
        col_type = self.type_inferrer.infer_column_type(series)
        stats = self.stats_profiler.profile_column(series, col_type)

        return {
            "name": column,
            "type": col_type.value,
            "dtype": str(series.dtype),
            **stats,
        }

    def suggest_transformations(
        self,
        X: pd.DataFrame,
        column_types: dict[str, ColumnType] | None = None
    ) -> dict[str, list[str]]:
        """Suggest transformations for each column.

        Args:
            X: Input DataFrame.,
            column_types: Optional pre-computed column types.,

        Returns:
            Dictionary mapping column names to suggested transformations.
        """
        if column_types is None:
            column_types = self.infer_types(X)

        suggestions: dict[str, list[str]] = {}

        for col, col_type in column_types.items():
            series = X[col]
            col_suggestions: list[str] = []

            if col_type == ColumnType.NUMERIC:
                non_null = pd.to_numeric(series.dropna(), errors="coerce").dropna()
                if len(non_null) > 0:
                    # Check for skewness
                    skew = non_null.skew()
                    if abs(skew) > 1:
                        if (non_null > 0).all():
                            col_suggestions.append("log_transform")
                        col_suggestions.append("power_transform")

                    # Check for outliers
                    q1, q3 = non_null.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    if ((non_null < q1 - 1.5 * iqr) | (non_null > q3 + 1.5 * iqr)).any():
                        col_suggestions.append("winsorize")

                    # Suggest binning for high cardinality
                    if series.nunique() > 100:
                        col_suggestions.append("bin")

                    col_suggestions.extend(["standardize", "normalize"])

            elif col_type == ColumnType.CATEGORICAL:
                n_unique = series.nunique()
                if n_unique <= 10:
                    col_suggestions.append("onehot_encode")
                else:
                    col_suggestions.append("target_encode")
                    col_suggestions.append("frequency_encode")

            elif col_type == ColumnType.DATETIME:
                col_suggestions.extend([
                    "extract_year",
                    "extract_month",
                    "extract_day",
                    "extract_weekday",
                    "extract_hour",
                ])

            elif col_type == ColumnType.TEXT:
                col_suggestions.extend([
                    "text_length",
                    "word_count",
                    "tfidf",
                ])

            if series.isna().any():
                col_suggestions.append("impute")
                col_suggestions.append("missing_indicator")

            suggestions[col] = col_suggestions

        return suggestions
