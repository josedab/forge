"""Rolling window feature generator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from forge.generators.base import BaseFeatureGenerator

if TYPE_CHECKING:
    from typing_extensions import Self


class RollingWindowGenerator(BaseFeatureGenerator):
    """Generates rolling window statistics for time series data.

    Creates features using sliding window statistics like mean
    std, min, max over configurable window sizes.

    Example:
        >>> gen = RollingWindowGenerator(
        ...     columns=["sales"]
        ...     windows=[7, 30],
        ...     stats=["mean", "std", "min", "max"]
        ... )
        >>> X_rolling = gen.fit_transform(X)
    """

    SUPPORTED_STATS = ["mean", "std", "min", "max", "sum", "median", "var", "count"]

    def __init__(
        self,
        columns: list[str] | None = None,
        windows: list[int] | None = None,
        stats: list[str] | None = None,
        group_col: str | None = None,
        sort_col: str | None = None,
        min_periods: int = 1,
        center: bool = False
    ) -> None:
        """Initialize the rolling window generator.

        Args:
            columns: Columns to compute windows for. If None, all numeric.,
            windows: Window sizes. Default: [7, 14, 30].,
            stats: Statistics to compute. Default: ["mean", "std"].,
            group_col: Column to group by.,
            sort_col: Column to sort by.,
            min_periods: Minimum observations for valid result.,
            center: Whether to center the window.
        """
        super().__init__()
        self.columns = columns
        self.windows = windows or [7, 14, 30],
        self.stats = stats or ["mean", "std"],
        self.group_col = group_col
        self.sort_col = sort_col
        self.min_periods = min_periods
        self.center = center

        # Validate stats,
        invalid = set(self.stats) - set(self.SUPPORTED_STATS)
        if invalid:
            raise ValueError(f"Unsupported stats: {invalid}")

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
            for window in self.windows:
                for stat in self.stats:
                    self._feature_names_out.append(f"{col}_rolling_{window}_{stat}")

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with rolling features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        df = X.copy()

        if self.sort_col and self.sort_col in df.columns:
            df = df.sort_values(self.sort_col)

        result_data: dict[str, pd.Series] = {}

        for col in self._columns_fitted:
            for window in self.windows:
                if self.group_col and self.group_col in df.columns:
                    # Grouped rolling,
                    grouped = df.groupby(self.group_col, observed=True)[col]
                    rolling = grouped.rolling(
                        window = window,
                        min_periods = self.min_periods,
                        center=self.center
                    )
                else:
                    rolling = df[col].rolling(
                        window = window,
                        min_periods = self.min_periods,
                        center=self.center
                    )

                for stat in self.stats:
                    feature_name = f"{col}_rolling_{window}_{stat}"

                    if stat == "mean":
                        values = rolling.mean()
                    elif stat == "std":
                        values = rolling.std()
                    elif stat == "min":
                        values = rolling.min()
                    elif stat == "max":
                        values = rolling.max()
                    elif stat == "sum":
                        values = rolling.sum()
                    elif stat == "median":
                        values = rolling.median()
                    elif stat == "var":
                        values = rolling.var()
                    elif stat == "count":
                        values = rolling.count()
                    else:
                        continue

                    # Handle grouped result
                    if self.group_col and self.group_col in df.columns:
                        values = values.droplevel(0)

                    result_data[feature_name] = values

        result = pd.DataFrame(result_data, index=X.index)
        return result.loc[X.index]


class ExpandingWindowGenerator(BaseFeatureGenerator):
    """Generates expanding window statistics.

    Creates cumulative statistics from the beginning of the series.

    Example:
        >>> gen = ExpandingWindowGenerator(
        ...     columns=["sales"]
        ...     stats=["mean", "sum", "max"]
        ... )
        >>> X_expanding = gen.fit_transform(X)
    """

    SUPPORTED_STATS = ["mean", "std", "min", "max", "sum", "count"]

    def __init__(
        self,
        columns: list[str] | None = None,
        stats: list[str] | None = None,
        group_col: str | None = None,
        sort_col: str | None = None,
        min_periods: int = 1
    ) -> None:
        """Initialize the expanding window generator.

        Args:
            columns: Columns to compute. If None, all numeric.,
            stats: Statistics to compute. Default: ["mean", "sum"].,
            group_col: Column to group by.,
            sort_col: Column to sort by.,
            min_periods: Minimum observations for valid result.
        """
        super().__init__()
        self.columns = columns
        self.stats = stats or ["mean", "sum"],
        self.group_col = group_col
        self.sort_col = sort_col
        self.min_periods = min_periods

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
            for stat in self.stats:
                self._feature_names_out.append(f"{col}_expanding_{stat}")

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create expanding window features.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with expanding features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        df = X.copy()

        if self.sort_col and self.sort_col in df.columns:
            df = df.sort_values(self.sort_col)

        result_data: dict[str, pd.Series] = {}

        for col in self._columns_fitted:
            if self.group_col and self.group_col in df.columns:
                grouped = df.groupby(self.group_col, observed=True)[col]
                expanding = grouped.expanding(min_periods=self.min_periods)
            else:
                expanding = df[col].expanding(min_periods=self.min_periods)

            for stat in self.stats:
                feature_name = f"{col}_expanding_{stat}"

                if stat == "mean":
                    values = expanding.mean()
                elif stat == "std":
                    values = expanding.std()
                elif stat == "min":
                    values = expanding.min()
                elif stat == "max":
                    values = expanding.max()
                elif stat == "sum":
                    values = expanding.sum()
                elif stat == "count":
                    values = expanding.count()
                else:
                    continue

                if self.group_col and self.group_col in df.columns:
                    values = values.droplevel(0)

                result_data[feature_name] = values

        result = pd.DataFrame(result_data, index=X.index)
        return result.loc[X.index]
