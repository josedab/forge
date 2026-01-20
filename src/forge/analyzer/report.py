"""Analysis report builder for Forge."""

from __future__ import annotations

from typing import Any

import pandas as pd

from forge.types import AnalysisReport, ColumnInfo, ColumnType, QualityIssue


class ReportBuilder:
    """Builds comprehensive analysis reports from profiling results.

    Combines type inference, statistics, and quality assessment into
    a structured report.
    """

    def build_report(
        self,
        df: pd.DataFrame,
        column_types: dict[str, ColumnType],
        statistics: dict[str, dict[str, Any]],
        quality_issues: list[QualityIssue],
        target: pd.Series | None = None
    ) -> AnalysisReport:
        """Build a complete analysis report.

        Args:
            df: Input DataFrame.,
            column_types: Mapping of column names to types.,
            statistics: Statistics for each column.,
            quality_issues: List of quality issues.,
            target: Optional target variable.,

        Returns:
            Complete analysis report.
        """
        column_info = self._build_column_info(df, column_types, statistics)
        target_info = self._analyze_target(target) if target is not None else None

        return AnalysisReport(
            n_rows = len(df),
            n_columns = len(df.columns),
            column_info = column_info,
            column_types = column_types,
            quality_issues = quality_issues,
            statistics = statistics,
            target_info=target_info
        )

    def _build_column_info(
        self,
        df: pd.DataFrame,
        column_types: dict[str, ColumnType],
        statistics: dict[str, dict[str, Any]],
    ) -> dict[str, ColumnInfo]:
        """Build ColumnInfo for each column."""
        column_info: dict[str, ColumnInfo] = {}

        for col in df.columns:
            series = df[col]
            col_type = column_types.get(col, ColumnType.UNKNOWN)
            col_stats = statistics.get(col, {})

            # Get sample values
            non_null = series.dropna()
            sample_values = list(non_null.head(5).values) if len(non_null) > 0 else []

            column_info[col] = ColumnInfo(
                name = col,
                dtype = str(series.dtype),
                inferred_type = col_type,
                null_count = col_stats.get("null_count", 0),
                null_ratio = col_stats.get("null_ratio", 0.0),
                unique_count = col_stats.get("unique_count", 0),
                unique_ratio = col_stats.get("unique_ratio", 0.0),
                sample_values=sample_values
            )

        return column_info

    def _analyze_target(self, target: pd.Series) -> dict[str, Any]:
        """Analyze the target variable.

        Args:
            target: Target variable.,

        Returns:
            Dictionary with target analysis.
        """
        info: dict[str, Any] = {
            "name": target.name if target.name else "target",
            "dtype": str(target.dtype),
            "null_count": int(target.isna().sum()),
            "null_ratio": float(target.isna().mean()),
            "unique_count": int(target.nunique()),
        }

        non_null = target.dropna()

        # Determine task type
        if pd.api.types.is_numeric_dtype(target):
            unique_ratio = target.nunique() / len(target) if len(target) > 0 else 0
            if target.nunique() <= 10 or unique_ratio < 0.05:
                info["task_type"] = "classification"
                info["n_classes"] = int(target.nunique())
                value_counts = non_null.value_counts()
                info["class_distribution"] = {
                    str(k): int(v) for k, v in value_counts.items()
                }
                info["is_balanced"] = (
                    value_counts.min() / value_counts.max() > 0.5
                    if len(value_counts) > 0 else True
                )
            else:
                info["task_type"] = "regression"
                info["mean"] = float(non_null.mean())
                info["std"] = float(non_null.std())
                info["min"] = float(non_null.min())
                info["max"] = float(non_null.max())
        else:
            info["task_type"] = "classification"
            info["n_classes"] = int(target.nunique())
            value_counts = non_null.value_counts()
            info["class_distribution"] = {
                str(k): int(v) for k, v in value_counts.head(10).items()
            }

        return info

    def format_report(self, report: AnalysisReport) -> str:
        """Format report as a readable string.

        Args:
            report: Analysis report.,

        Returns:
            Formatted string representation.
        """
        lines = [
            "=" * 60,
            "DATA ANALYSIS REPORT",
            "=" * 60,
            "",
            f"Dataset: {report.n_rows:,} rows x {report.n_columns} columns",
            "",
        ]

        # Column types summary
        lines.append("COLUMN TYPES:")
        type_counts: dict[ColumnType, int] = {}
        for col_type in report.column_types.values():
            type_counts[col_type] = type_counts.get(col_type, 0) + 1

        for col_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {col_type.value:15} {count}")

        lines.append("")

        # Quality summary
        if report.quality_issues:
            lines.append("QUALITY ISSUES:")
            severity_counts = {"high": 0, "medium": 0, "low": 0}
            for issue in report.quality_issues:
                severity_counts[issue.severity] += 1

            for severity in ["high", "medium", "low"]:
                if severity_counts[severity] > 0:
                    lines.append(f"  {severity.upper():8} {severity_counts[severity]}")

            lines.append("")
            lines.append("Top Issues:")
            for issue in report.quality_issues[:5]:
                lines.append(f"  [{issue.severity}] {issue.column}: {issue.description}")
        else:
            lines.append("QUALITY: No issues found")

        lines.append("")

        # Target summary
        if report.target_info:
            lines.append("TARGET VARIABLE:")
            info = report.target_info
            lines.append(f"  Name: {info['name']}")
            lines.append(f"  Task: {info['task_type']}")
            if info["task_type"] == "classification":
                lines.append(f"  Classes: {info['n_classes']}")
                if not info.get("is_balanced", True):
                    lines.append("  Warning: Imbalanced classes detected")
            else:
                lines.append(f"  Range: [{info['min']:.2f}, {info['max']:.2f}]")
                lines.append(f"  Mean: {info['mean']:.2f} (std: {info['std']:.2f})")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)
