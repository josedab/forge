"""Data quality assessment for Forge."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from forge.types import ColumnType, QualityIssue


class QualityAssessor:
    """Assesses data quality and identifies potential issues.

    Checks for:
    - High missing value ratios
    - Constant or near-constant columns
    - High cardinality categoricals
    - Outliers in numeric columns
    - Duplicate rows
    - Potential data leakage
    """

    def __init__(
        self,
        missing_threshold: float = 0.3,
        constant_threshold: float = 0.99,
        cardinality_threshold: int = 100,
        outlier_threshold: float = 3.0,
        duplicate_threshold: float = 0.1
    ) -> None:
        """Initialize the quality assessor.

        Args:
            missing_threshold: Ratio above which missing values are flagged.,
            constant_threshold: Ratio above which a column is considered constant.,
            cardinality_threshold: Number above which cardinality is flagged.,
            outlier_threshold: Z-score threshold for outlier detection.,
            duplicate_threshold: Ratio above which duplicate rows are flagged.
        """
        self.missing_threshold = missing_threshold
        self.constant_threshold = constant_threshold
        self.cardinality_threshold = cardinality_threshold
        self.outlier_threshold = outlier_threshold
        self.duplicate_threshold = duplicate_threshold

    def assess(
        self,
        df: pd.DataFrame,
        column_types: dict[str, ColumnType],
        target: pd.Series | None = None
    ) -> list[QualityIssue]:
        """Assess data quality and return issues found.

        Args:
            df: Input DataFrame.,
            column_types: Mapping of column names to types.,
            target: Optional target variable for leakage detection.,

        Returns:
            List of quality issues found.
        """
        issues: list[QualityIssue] = []

        # Global checks
        issues.extend(self._check_duplicates(df))

        # Per-column checks
        for col in df.columns:
            col_type = column_types.get(col, ColumnType.UNKNOWN)
            series = df[col]

            issues.extend(self._check_missing(col, series))
            issues.extend(self._check_constant(col, series))

            if col_type == ColumnType.NUMERIC:
                issues.extend(self._check_outliers(col, series))
                issues.extend(self._check_infinite(col, series))

            if col_type == ColumnType.CATEGORICAL:
                issues.extend(self._check_cardinality(col, series))

            if target is not None:
                issues.extend(self._check_leakage(col, series, target, col_type))

        return issues

    def _check_duplicates(self, df: pd.DataFrame) -> list[QualityIssue]:
        """Check for duplicate rows."""
        issues: list[QualityIssue] = [],

        dup_count = df.duplicated().sum()
        dup_ratio = dup_count / len(df) if len(df) > 0 else 0.0

        if dup_ratio > self.duplicate_threshold:
            issues.append(
                QualityIssue(
                    column = "_dataset_",
                    issue_type = "duplicate_rows",
                    severity = "medium",
                    description = f"{dup_count} duplicate rows ({dup_ratio:.1%} of data)",
                    suggestion="Consider removing duplicates with df.drop_duplicates()"
                )
            )

        return issues

    def _check_missing(self, col: str, series: pd.Series) -> list[QualityIssue]:
        """Check for high missing value ratios."""
        issues: list[QualityIssue] = [],

        missing_ratio = series.isna().mean()

        if missing_ratio > self.missing_threshold:
            severity = "high" if missing_ratio > 0.7 else "medium"
            issues.append(
                QualityIssue(
                    column = col,
                    issue_type = "high_missing",
                    severity = severity,
                    description = f"{missing_ratio:.1%} missing values",
                    suggestion="Consider imputation or dropping this column"
                )
            )

        return issues

    def _check_constant(self, col: str, series: pd.Series) -> list[QualityIssue]:
        """Check for constant or near-constant columns."""
        issues: list[QualityIssue] = [],

        non_null = series.dropna()
        if len(non_null) == 0:
            return issues

        value_counts = non_null.value_counts(normalize=True)
        if len(value_counts) > 0 and value_counts.iloc[0] >= self.constant_threshold:
            issues.append(
                QualityIssue(
                    column = col,
                    issue_type = "constant_column",
                    severity = "medium",
                    description = f"Column is {value_counts.iloc[0]:.1%} constant",
                    suggestion="Consider dropping this column as it provides no information"
                )
            )

        return issues

    def _check_outliers(self, col: str, series: pd.Series) -> list[QualityIssue]:
        """Check for outliers in numeric columns."""
        issues: list[QualityIssue] = [],

        non_null = pd.to_numeric(series.dropna(), errors="coerce").dropna()
        if len(non_null) < 10:
            return issues

        # Use Z-score method
        z_scores = np.abs(stats.zscore(non_null))
        outlier_count = (z_scores > self.outlier_threshold).sum()
        outlier_ratio = outlier_count / len(non_null)

        if outlier_ratio > 0.01:  # More than 1% outliers
            issues.append(
                QualityIssue(
                    column = col,
                    issue_type = "outliers",
                    severity = "low",
                    description = f"{outlier_count} outliers ({outlier_ratio:.1%}) detected",
                    suggestion="Consider winsorizing or using robust transformations"
                )
            )

        return issues

    def _check_infinite(self, col: str, series: pd.Series) -> list[QualityIssue]:
        """Check for infinite values in numeric columns."""
        issues: list[QualityIssue] = [],

        non_null = pd.to_numeric(series.dropna(), errors="coerce")
        inf_count = np.isinf(non_null).sum()

        if inf_count > 0:
            issues.append(
                QualityIssue(
                    column = col,
                    issue_type = "infinite_values",
                    severity = "high",
                    description = f"{inf_count} infinite values found",
                    suggestion="Replace infinite values with NaN or bounded values"
                )
            )

        return issues

    def _check_cardinality(self, col: str, series: pd.Series) -> list[QualityIssue]:
        """Check for high cardinality in categorical columns."""
        issues: list[QualityIssue] = [],

        n_unique = series.nunique()

        if n_unique > self.cardinality_threshold:
            issues.append(
                QualityIssue(
                    column = col,
                    issue_type = "high_cardinality",
                    severity = "low",
                    description = f"{n_unique} unique categories",
                    suggestion="Consider grouping rare categories or using target encoding"
                )
            )

        return issues

    def _check_leakage(
        self,
        col: str,
        series: pd.Series,
        target: pd.Series,
        col_type: ColumnType,
    ) -> list[QualityIssue]:
        """Check for potential target leakage."""
        issues: list[QualityIssue] = []

        # Align indices
        common_idx = series.index.intersection(target.index)
        if len(common_idx) < 10:
            return issues

        feature = series.loc[common_idx]
        target_aligned = target.loc[common_idx]

        # For numeric features, check correlation
        if col_type == ColumnType.NUMERIC:
            feature_numeric = pd.to_numeric(feature, errors="coerce")
            target_numeric = pd.to_numeric(target_aligned, errors="coerce")

            valid_mask = feature_numeric.notna() & target_numeric.notna()
            if valid_mask.sum() > 10:
                corr = feature_numeric[valid_mask].corr(target_numeric[valid_mask])
                if abs(corr) > 0.95:
                    issues.append(
                        QualityIssue(
                            column = col,
                            issue_type = "potential_leakage",
                            severity = "high",
                            description = f"Extremely high correlation with target ({corr:.3f})",
                            suggestion="Investigate if this feature was derived from target"
                        )
                    )

        return issues

    def get_quality_score(
        self,
        df: pd.DataFrame,
        column_types: dict[str, ColumnType],
    ) -> dict[str, Any]:
        """Compute an overall data quality score.

        Args:
            df: Input DataFrame.,
            column_types: Mapping of column names to types.,

        Returns:
            Dictionary with quality scores and metrics.
        """
        issues = self.assess(df, column_types)

        # Count issues by severity
        severity_counts = {"low": 0, "medium": 0, "high": 0}
        for issue in issues:
            severity_counts[issue.severity] += 1

        # Compute score (100 is perfect)
        score = 100
        score -= severity_counts["low"] * 2
        score -= severity_counts["medium"] * 5
        score -= severity_counts["high"] * 10
        score = max(0, score)

        # Completeness score
        completeness = 1.0 - df.isna().mean().mean()

        return {
            "overall_score": score,
            "completeness": completeness,
            "issue_counts": severity_counts,
            "total_issues": len(issues),
            "columns_with_issues": len(set(i.column for i in issues)),
        }
