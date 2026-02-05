"""Statistical profiling for Forge."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from forge.types import ColumnType


class StatisticsProfiler:
    """Computes descriptive statistics for DataFrame columns.

    Provides statistics appropriate for each column type:
    - Numeric: mean, std, min, max, quartiles, skewness, kurtosis
    - Categorical: frequencies, mode, entropy
    - Datetime: range, frequency analysis
    - Text: length statistics, word counts
    """

    def __init__(self, percentiles: list[float] | None = None) -> None:
        """Initialize the statistics profiler.

        Args:
            percentiles: Percentiles to compute for numeric columns.
        """
        self.percentiles = percentiles or [0.25, 0.5, 0.75]

    def profile(
        self,
        df: pd.DataFrame,
        column_types: dict[str, ColumnType],
    ) -> dict[str, dict[str, Any]]:
        """Compute statistics for all columns.

        Args:
            df: Input DataFrame.,
            column_types: Mapping of column names to types.,

        Returns:
            Dictionary mapping column names to their statistics.
        """
        stats_dict: dict[str, dict[str, Any]] = {}
        for col in df.columns:
            col_type = column_types.get(col, ColumnType.UNKNOWN)
            stats_dict[col] = self.profile_column(df[col], col_type)
        return stats_dict

    def profile_column(self, series: pd.Series, col_type: ColumnType) -> dict[str, Any]:
        """Compute statistics for a single column.

        Args:
            series: Input Series.,
            col_type: Inferred column type.,

        Returns:
            Dictionary of statistics.
        """
        base_stats = self._base_stats(series)

        if col_type == ColumnType.NUMERIC:
            base_stats.update(self._numeric_stats(series))
        elif col_type == ColumnType.CATEGORICAL:
            base_stats.update(self._categorical_stats(series))
        elif col_type == ColumnType.DATETIME:
            base_stats.update(self._datetime_stats(series))
        elif col_type == ColumnType.TEXT:
            base_stats.update(self._text_stats(series))
        elif col_type == ColumnType.BOOLEAN:
            base_stats.update(self._boolean_stats(series))

        return base_stats

    def _base_stats(self, series: pd.Series) -> dict[str, Any]:
        """Compute base statistics common to all types."""
        return {
            "count": len(series),
            "null_count": int(series.isna().sum()),
            "null_ratio": float(series.isna().mean()),
            "unique_count": int(series.nunique()),
            "unique_ratio": float(series.nunique() / len(series)) if len(series) > 0 else 0.0,
            "memory_usage": int(series.memory_usage(deep=True)),
        }

    def _numeric_stats(self, series: pd.Series) -> dict[str, Any]:
        """Compute statistics for numeric columns."""
        non_null = series.dropna()
        if len(non_null) == 0:
            return {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "range": None,
                "percentiles": {},
                "skewness": None,
                "kurtosis": None,
                "zeros_count": 0,
                "zeros_ratio": 0.0,
                "negatives_count": 0,
                "negatives_ratio": 0.0,
            }

        numeric_vals = pd.to_numeric(non_null, errors="coerce").dropna()
        if len(numeric_vals) == 0:
            return {}

        percentile_values = np.percentile(numeric_vals, [p * 100 for p in self.percentiles])
        percentile_dict = {
            f"p{int(p * 100)}": float(v) for p, v in zip(self.percentiles, percentile_values)
        }

        return {
            "mean": float(numeric_vals.mean()),
            "std": float(numeric_vals.std()),
            "min": float(numeric_vals.min()),
            "max": float(numeric_vals.max()),
            "range": float(numeric_vals.max() - numeric_vals.min()),
            "percentiles": percentile_dict,
            "skewness": float(stats.skew(numeric_vals)),
            "kurtosis": float(stats.kurtosis(numeric_vals)),
            "zeros_count": int((numeric_vals == 0).sum()),
            "zeros_ratio": float((numeric_vals == 0).mean()),
            "negatives_count": int((numeric_vals < 0).sum()),
            "negatives_ratio": float((numeric_vals < 0).mean()),
        }

    def _categorical_stats(self, series: pd.Series) -> dict[str, Any]:
        """Compute statistics for categorical columns."""
        non_null = series.dropna()
        if len(non_null) == 0:
            return {
                "mode": None,
                "mode_count": 0,
                "mode_ratio": 0.0,
                "top_categories": [],
                "entropy": 0.0,
            }

        value_counts = non_null.value_counts()
        mode = value_counts.index[0] if len(value_counts) > 0 else None
        mode_count = int(value_counts.iloc[0]) if len(value_counts) > 0 else 0

        # Compute entropy
        probs = value_counts / value_counts.sum()
        entropy = float(stats.entropy(probs))

        # Top categories
        top_n = min(10, len(value_counts))
        top_categories = [
            {"value": str(val), "count": int(cnt), "ratio": float(cnt / len(non_null))}
            for val, cnt in value_counts.head(top_n).items()
        ]

        return {
            "mode": str(mode) if mode is not None else None,
            "mode_count": mode_count,
            "mode_ratio": float(mode_count / len(non_null)) if len(non_null) > 0 else 0.0,
            "top_categories": top_categories,
            "entropy": entropy,
        }

    def _datetime_stats(self, series: pd.Series) -> dict[str, Any]:
        """Compute statistics for datetime columns."""
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(series):
            dt_series = pd.to_datetime(series, errors="coerce")
        else:
            dt_series = series

        non_null = dt_series.dropna()
        if len(non_null) == 0:
            return {
                "min_date": None,
                "max_date": None,
                "date_range_days": None,
                "most_common_weekday": None,
                "most_common_month": None,
            }

        # Compute date range
        min_date = non_null.min()
        max_date = non_null.max()
        date_range = (max_date - min_date).days

        # Weekday distribution
        weekdays = non_null.dt.dayofweek
        weekday_mode = int(weekdays.mode().iloc[0]) if len(weekdays) > 0 else None
        weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        most_common_weekday = weekday_names[weekday_mode] if weekday_mode is not None else None

        # Month distribution
        months = non_null.dt.month
        month_mode = int(months.mode().iloc[0]) if len(months) > 0 else None

        return {
            "min_date": str(min_date),
            "max_date": str(max_date),
            "date_range_days": int(date_range),
            "most_common_weekday": most_common_weekday,
            "most_common_month": month_mode,
        }

    def _text_stats(self, series: pd.Series) -> dict[str, Any]:
        """Compute statistics for text columns."""
        non_null = series.dropna().astype(str)
        if len(non_null) == 0:
            return {
                "avg_length": 0.0,
                "min_length": 0,
                "max_length": 0,
                "avg_words": 0.0,
                "min_words": 0,
                "max_words": 0,
            }

        lengths = non_null.str.len()
        word_counts = non_null.str.split().str.len()

        return {
            "avg_length": float(lengths.mean()),
            "min_length": int(lengths.min()),
            "max_length": int(lengths.max()),
            "avg_words": float(word_counts.mean()),
            "min_words": int(word_counts.min()),
            "max_words": int(word_counts.max()),
        }

    def _boolean_stats(self, series: pd.Series) -> dict[str, Any]:
        """Compute statistics for boolean columns."""
        non_null = series.dropna()
        if len(non_null) == 0:
            return {
                "true_count": 0,
                "true_ratio": 0.0,
                "false_count": 0,
                "false_ratio": 0.0,
            }

        # Convert to boolean if needed
        if not pd.api.types.is_bool_dtype(non_null):
            bool_map = {
                True: True, False: False,
                1: True, 0: False,
                1.0: True, 0.0: False,
                "true": True, "false": False,
                "True": True, "False": False,
                "TRUE": True, "FALSE": False,
                "yes": True, "no": False,
                "Yes": True, "No": False,
                "YES": True, "NO": False,
                "y": True, "n": False,
                "Y": True, "N": False,
            }
            bool_vals = non_null.map(lambda x: bool_map.get(x)).dropna()
        else:
            bool_vals = non_null

        true_count = int(bool_vals.sum())
        false_count = int((~bool_vals).sum()) if len(bool_vals) > 0 else 0

        return {
            "true_count": true_count,
            "true_ratio": float(true_count / len(bool_vals)) if len(bool_vals) > 0 else 0.0,
            "false_count": false_count,
            "false_ratio": float(false_count / len(bool_vals)) if len(bool_vals) > 0 else 0.0,
        }
