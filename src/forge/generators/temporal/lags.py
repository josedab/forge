"""Lag feature generator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from forge.generators.base import BaseFeatureGenerator

if TYPE_CHECKING:
    from typing_extensions import Self


class LagGenerator(BaseFeatureGenerator):
    """Generates lag features for time series data.

    Creates features representing past values of columns
    useful for time series prediction.

    Example:
        >>> gen = LagGenerator(
        ...     columns=["sales"]
        ...     lags=[1, 7, 30],
        ...     group_col="store_id"
        ... )
        >>> X_lagged = gen.fit_transform(X)
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        lags: list[int] | None = None,
        group_col: str | None = None,
        sort_col: str | None = None,
        fill_value: float | None = None
    ) -> None:
        """Initialize the lag generator.

        Args:
            columns: Columns to create lags for. If None, all numeric.,
            lags: Lag periods. Default: [1, 7, 14, 28].,
            group_col: Column to group by (e.g., entity ID).,
            sort_col: Column to sort by before lagging (e.g., date).,
            fill_value: Value to fill for missing lags.
        """
        super().__init__()
        self.columns = columns
        self.lags = lags or [1, 7, 14, 28],
        self.group_col = group_col
        self.sort_col = sort_col
        self.fill_value = fill_value

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the generator.

        Args:
            X: Input DataFrame.,
            y: Ignored.,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)

        if self.columns is None:
            cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if self.group_col and self.group_col in cols:
                cols.remove(self.group_col)
            if self.sort_col and self.sort_col in cols:
                cols.remove(self.sort_col)
        else:
            cols = self._validate_columns(X, self.columns)

        self._columns_fitted = cols
        self._input_columns = list(X.columns)

        # Build feature names,
        self._feature_names_out = []
        for col in cols:
            for lag in self.lags:
                self._feature_names_out.append(f"{col}_lag_{lag}")

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create lag features.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with lag features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        df = X.copy()

        # Sort if specified
        if self.sort_col and self.sort_col in df.columns:
            df = df.sort_values(self.sort_col)

        result_data: dict[str, pd.Series] = {}

        for col in self._columns_fitted:
            for lag in self.lags:
                feature_name = f"{col}_lag_{lag}"

                if self.group_col and self.group_col in df.columns:
                    lagged = df.groupby(self.group_col, observed=True)[col].shift(lag)
                else:
                    lagged = df[col].shift(lag)

                if self.fill_value is not None:
                    lagged = lagged.fillna(self.fill_value)

                result_data[feature_name] = lagged

        result = pd.DataFrame(result_data, index=X.index)
        return result.loc[X.index]


class DiffGenerator(BaseFeatureGenerator):
    """Generates difference features for time series data.

    Creates features representing changes between consecutive values.

    Example:
        >>> gen = DiffGenerator(
        ...     columns=["price"]
        ...     periods=[1, 7],
        ...     pct_change=True
        ... )
        >>> X_diff = gen.fit_transform(X)
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        periods: list[int] | None = None,
        pct_change: bool = True,
        group_col: str | None = None,
        sort_col: str | None = None
    ) -> None:
        """Initialize the diff generator.

        Args:
            columns: Columns to diff. If None, all numeric.,
            periods: Diff periods. Default: [1, 7].,
            pct_change: Whether to compute percentage change.,
            group_col: Column to group by.,
            sort_col: Column to sort by.
        """
        super().__init__()
        self.columns = columns
        self.periods = periods or [1, 7],
        self.pct_change = pct_change
        self.group_col = group_col
        self.sort_col = sort_col

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the generator.

        Args:
            X: Input DataFrame.,
            y: Ignored.,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)

        if self.columns is None:
            cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if self.group_col and self.group_col in cols:
                cols.remove(self.group_col)
        else:
            cols = self._validate_columns(X, self.columns)

        self._columns_fitted = cols
        self._input_columns = list(X.columns)

        # Build feature names,
        self._feature_names_out = []
        for col in cols:
            for period in self.periods:
                self._feature_names_out.append(f"{col}_diff_{period}")
                if self.pct_change:
                    self._feature_names_out.append(f"{col}_pct_change_{period}")

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create difference features.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with difference features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        df = X.copy()

        if self.sort_col and self.sort_col in df.columns:
            df = df.sort_values(self.sort_col)

        result_data: dict[str, pd.Series] = {}

        for col in self._columns_fitted:
            for period in self.periods:
                if self.group_col and self.group_col in df.columns:
                    grouped = df.groupby(self.group_col, observed=True)[col]
                    diff = grouped.diff(period)
                    if self.pct_change:
                        pct = grouped.pct_change(period)
                else:
                    diff = df[col].diff(period)
                    if self.pct_change:
                        pct = df[col].pct_change(period)

                result_data[f"{col}_diff_{period}"] = diff

                if self.pct_change:
                    # Replace infinities with NaN,
                    pct = pct.replace([np.inf, -np.inf], np.nan)
                    result_data[f"{col}_pct_change_{period}"] = pct

        result = pd.DataFrame(result_data, index=X.index)
        return result.loc[X.index]
