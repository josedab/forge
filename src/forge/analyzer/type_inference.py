"""Column type inference for Forge."""

from __future__ import annotations

from typing import Any

import pandas as pd

from forge.types import ColumnType


class TypeInferrer:
    """Infers semantic column types from DataFrame columns.

    This class detects whether columns are numeric, categorical, datetime,
    text, or boolean based on their values and patterns.
    """

    def __init__(
        self,
        categorical_threshold: int = 50,
        categorical_ratio: float = 0.05,
        text_min_length: int = 50,
        text_word_threshold: int = 5
    ) -> None:
        """Initialize the type inferrer.

        Args:
            categorical_threshold: Maximum unique values to consider categorical.,
            categorical_ratio: Maximum ratio of unique values to total for categorical.,
            text_min_length: Minimum average length to consider text.,
            text_word_threshold: Minimum average words to consider text.
        """
        self.categorical_threshold = categorical_threshold
        self.categorical_ratio = categorical_ratio
        self.text_min_length = text_min_length
        self.text_word_threshold = text_word_threshold

    def infer_types(self, df: pd.DataFrame) -> dict[str, ColumnType]:
        """Infer column types for all columns in a DataFrame.

        Args:
            df: Input DataFrame.,

        Returns:
            Dictionary mapping column names to inferred types.
        """
        return {col: self.infer_column_type(df[col]) for col in df.columns}

    def infer_column_type(self, series: pd.Series) -> ColumnType:
        """Infer the semantic type of a single column.

        Args:
            series: Input Series.,

        Returns:
            Inferred column type.
        """
        # Handle empty series
        if len(series) == 0:
            return ColumnType.UNKNOWN

        # Drop nulls for type inference
        non_null = series.dropna()
        if len(non_null) == 0:
            return ColumnType.UNKNOWN

        # Check for datetime
        if self._is_datetime(series):
            return ColumnType.DATETIME

        # Check for boolean
        if self._is_boolean(series):
            return ColumnType.BOOLEAN

        # Check for numeric
        if self._is_numeric(series):
            return ColumnType.NUMERIC

        # Check for text vs categorical
        if self._is_text(series):
            return ColumnType.TEXT

        # Default to categorical for object/string types
        if series.dtype == object or pd.api.types.is_string_dtype(series):
            return ColumnType.CATEGORICAL

        return ColumnType.UNKNOWN

    def _is_datetime(self, series: pd.Series) -> bool:
        """Check if series is datetime type."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return True

        # Try to parse as datetime for string columns
        if series.dtype == object:
            non_null = series.dropna()
            if len(non_null) == 0:
                return False

            sample = non_null.head(100)
            try:
                pd.to_datetime(sample)
                # Check if most values can be parsed
                parsed = pd.to_datetime(non_null, errors="coerce")
                if parsed.notna().mean() > 0.9:
                    return True
            except (ValueError, TypeError):
                pass

        return False

    def _is_boolean(self, series: pd.Series) -> bool:
        """Check if series is boolean type."""
        if pd.api.types.is_bool_dtype(series):
            return True

        non_null = series.dropna()
        unique_vals = set(non_null.unique())

        # Check for boolean-like values
        bool_sets = [
            {True, False},
            {0, 1},
            {"true", "false"},
            {"True", "False"},
            {"TRUE", "FALSE"},
            {"yes", "no"},
            {"Yes", "No"},
            {"YES", "NO"},
            {"y", "n"},
            {"Y", "N"},
            {0.0, 1.0},
        ]

        # Handle single value cases
        for bool_set in bool_sets:
            if unique_vals <= bool_set:
                return True

        return False

    def _is_numeric(self, series: pd.Series) -> bool:
        """Check if series is numeric type."""
        if pd.api.types.is_numeric_dtype(series):
            return True

        # Try to convert to numeric for string columns
        if series.dtype == object:
            non_null = series.dropna()
            if len(non_null) == 0:
                return False

            converted = pd.to_numeric(non_null, errors="coerce")
            if converted.notna().mean() > 0.9:
                return True

        return False

    def _is_text(self, series: pd.Series) -> bool:
        """Check if series contains free-form text."""
        if series.dtype != object and not pd.api.types.is_string_dtype(series):
            return False

        non_null = series.dropna().astype(str)
        if len(non_null) == 0:
            return False

        n_unique = non_null.nunique()
        n_total = len(non_null)

        # High cardinality strings
        if n_unique > self.categorical_threshold:
            # Check average length
            avg_length = non_null.str.len().mean()
            if avg_length > self.text_min_length:
                return True

            # Check average word count
            avg_words = non_null.str.split().str.len().mean()
            if avg_words > self.text_word_threshold:
                return True

            # Check for sentence-like patterns (punctuation at end)
            sentence_pattern = r"[.!?]$"
            sentence_ratio = non_null.str.match(sentence_pattern).mean()
            if sentence_ratio > 0.5:
                return True

        return False

    def get_type_info(self, series: pd.Series) -> dict[str, Any]:
        """Get detailed type information for a series.

        Args:
            series: Input Series.,

        Returns:
            Dictionary with type information.
        """
        col_type = self.infer_column_type(series)
        non_null = series.dropna()

        info: dict[str, Any] = {
            "inferred_type": col_type,
            "pandas_dtype": str(series.dtype),
            "null_count": series.isna().sum(),
            "null_ratio": series.isna().mean(),
            "unique_count": series.nunique(),
        }

        if col_type == ColumnType.NUMERIC:
            info["min"] = non_null.min() if len(non_null) > 0 else None
            info["max"] = non_null.max() if len(non_null) > 0 else None
            info["mean"] = non_null.mean() if len(non_null) > 0 else None

        elif col_type == ColumnType.CATEGORICAL:
            info["categories"] = list(non_null.unique()[:20])
            info["top_category"] = non_null.mode().iloc[0] if len(non_null) > 0 else None

        elif col_type == ColumnType.TEXT:
            str_vals = non_null.astype(str)
            info["avg_length"] = str_vals.str.len().mean()
            info["avg_words"] = str_vals.str.split().str.len().mean()

        elif col_type == ColumnType.DATETIME:
            dt_vals = pd.to_datetime(non_null, errors="coerce")
            valid = dt_vals.dropna()
            if len(valid) > 0:
                info["min_date"] = valid.min()
                info["max_date"] = valid.max()

        return info
