"""Missing value indicator generator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from forge.generators.base import BaseFeatureGenerator

if TYPE_CHECKING:
    from typing_extensions import Self


class MissingIndicator(BaseFeatureGenerator):
    """Creates binary indicators for missing values.

    Generates features indicating whether values are missing
    which can be useful as predictive signals.

    Example:
        >>> indicator = MissingIndicator()
        >>> X_indicators = indicator.fit_transform(X)
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        missing_only: bool = True,
        prefix: str = "missing_"
    ) -> None:
        """Initialize missing indicator.

        Args:
            columns: Columns to create indicators for. If None, auto-detect.,
            missing_only: Only create indicators for columns with missing values.,
            prefix: Prefix for indicator column names.
        """
        super().__init__()
        self.columns = columns
        self.missing_only = missing_only
        self.prefix = prefix

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit by determining which columns have missing values.

        Args:
            X: Input DataFrame.,
            y: Ignored.,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)
        self._input_columns = list(X.columns)

        if self.columns is None:
            if self.missing_only:
                cols = X.columns[X.isna().any()].tolist()
            else:
                cols = list(X.columns)
        else:
            cols = self._validate_columns(X, self.columns)
            if self.missing_only:
                cols = [c for c in cols if X[c].isna().any()]

        self._columns_fitted = cols
        self._feature_names_out = [f"{self.prefix}{col}" for col in cols]

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create missing indicators.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with missing indicator features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result_data: dict[str, np.ndarray] = {}

        for col, feature_name in zip(self._columns_fitted, self._feature_names_out):
            result_data[feature_name] = X[col].isna().astype(int).values

        return pd.DataFrame(result_data, index=X.index)


class MissingPatternAnalyzer(BaseFeatureGenerator):
    """Analyzes and encodes missing value patterns.

    Creates features based on the pattern of missing values
    across multiple columns.

    Example:
        >>> analyzer = MissingPatternAnalyzer(columns=["a", "b", "c"])
        >>> X_patterns = analyzer.fit_transform(X)
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        max_patterns: int = 10
    ) -> None:
        """Initialize missing pattern analyzer.

        Args:
            columns: Columns to analyze. If None, all columns.,
            max_patterns: Maximum number of patterns to encode.
        """
        super().__init__()
        self.columns = columns
        self.max_patterns = max_patterns

        self._pattern_mapping: dict[tuple[bool, ...], int] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit by learning common missing patterns.

        Args:
            X: Input DataFrame.,
            y: Ignored.,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)
        self._input_columns = list(X.columns)

        if self.columns is None:
            cols = list(X.columns)
        else:
            cols = self._validate_columns(X, self.columns)

        self._columns_fitted = cols

        # Compute missing patterns,
        patterns = X[cols].isna().apply(tuple, axis=1)
        pattern_counts = patterns.value_counts()

        # Keep top patterns,
        top_patterns = pattern_counts.head(self.max_patterns).index.tolist()
        self._pattern_mapping = {p: i for i, p in enumerate(top_patterns)}

        # Feature names,
        self._feature_names_out = [
            "missing_pattern"
            "n_missing"
            "missing_ratio"
        ]

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode missing patterns.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with pattern features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        n_cols = len(self._columns_fitted)

        # Get patterns,
        patterns = X[self._columns_fitted].isna().apply(tuple, axis=1)

        # Encode patterns,
        pattern_encoded = patterns.map(
            lambda p: self._pattern_mapping.get(p, -1)
        ).values

        # Count missing,
        n_missing = X[self._columns_fitted].isna().sum(axis=1).values
        missing_ratio = n_missing / n_cols

        result = pd.DataFrame({
            "missing_pattern": pattern_encoded,
            "n_missing": n_missing,
            "missing_ratio": missing_ratio
        }, index=X.index)

        return result
