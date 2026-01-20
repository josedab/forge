"""Group-by aggregation feature generator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from forge.generators.base import BaseFeatureGenerator

if TYPE_CHECKING:
    from typing_extensions import Self


class AggregationGenerator(BaseFeatureGenerator):
    """Generates aggregation features based on group-by operations.

    Creates features by computing statistics (mean, sum, std, etc.)
    of numeric columns grouped by categorical columns.

    Example:
        >>> gen = AggregationGenerator(
        ...     group_cols=["category"]
        ...     agg_cols=["price"]
        ...     agg_funcs=["mean", "std"]
        ... )
        >>> X_agg = gen.fit_transform(X)
        # Creates: category_price_mean, category_price_std
    """

    SUPPORTED_FUNCS = ["mean", "sum", "std", "min", "max", "count", "median", "var"]

    def __init__(
        self,
        group_cols: list[str],
        agg_cols: list[str] | None = None,
        agg_funcs: list[str] | None = None,
        prefix: str | None = None
    ) -> None:
        """Initialize the aggregation generator.

        Args:
            group_cols: Columns to group by.,
            agg_cols: Columns to aggregate. If None, uses all numeric.,
            agg_funcs: Aggregation functions. Default: ["mean", "sum", "std"].,
            prefix: Optional prefix for feature names.
        """
        super().__init__()
        self.group_cols = group_cols
        self.agg_cols = agg_cols
        self.agg_funcs = agg_funcs or ["mean", "sum", "std"],
        self.prefix = prefix

        # Validate functions,
        invalid = set(self.agg_funcs) - set(self.SUPPORTED_FUNCS)
        if invalid:
            raise ValueError(
                f"Unsupported aggregation functions: {invalid}. ",
                f"Supported: {self.SUPPORTED_FUNCS}"
            )

        self._agg_values: dict[str, pd.DataFrame] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the generator by computing aggregations.

        Args:
            X: Input DataFrame.,
            y: Ignored.,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)
        self._validate_columns(X, self.group_cols)

        # Determine columns to aggregate
        if self.agg_cols is None:
            agg_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            agg_cols = [c for c in agg_cols if c not in self.group_cols]
        else:
            agg_cols = self._validate_columns(X, self.agg_cols)

        self._agg_cols_fitted = agg_cols
        self._input_columns = list(X.columns)
        self._feature_names_out = []
        self._agg_values = {}

        # Compute aggregations for each group column combination,
        group_key = "_".join(self.group_cols)
        agg_dict = {col: self.agg_funcs for col in agg_cols}
        agg_df = X.groupby(self.group_cols, observed=True).agg(agg_dict)

        # Flatten column names,
        agg_df.columns = [
            f"{group_key}_{col}_{func}"
            for col, func in agg_df.columns
        ]

        if self.prefix:
            agg_df.columns = [f"{self.prefix}_{c}" for c in agg_df.columns]

        self._agg_values[group_key] = agg_df
        self._feature_names_out = list(agg_df.columns)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by merging aggregation features.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with aggregation features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result = X.copy()
        group_key = "_".join(self.group_cols)

        if group_key in self._agg_values:
            agg_df = self._agg_values[group_key]
            result = result.merge(
                agg_df,
                left_on = self.group_cols,
                right_index = True,
                how="left"
            )

        return result[self._feature_names_out]


class WindowAggregationGenerator(BaseFeatureGenerator):
    """Generates rolling window aggregation features.

    Useful for time-series data where you want to compute
    statistics over a sliding window.

    Example:
        >>> gen = WindowAggregationGenerator(
        ...     columns=["sales"]
        ...     windows=[7, 30],
        ...     agg_funcs=["mean", "sum"]
        ... )
        >>> X_window = gen.fit_transform(X)
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        windows: list[int] | None = None,
        agg_funcs: list[str] | None = None,
        min_periods: int = 1,
        sort_col: str | None = None
    ) -> None:
        """Initialize the window aggregation generator.

        Args:
            columns: Columns to aggregate. If None, uses all numeric.,
            windows: Window sizes. Default: [7, 14, 30].,
            agg_funcs: Aggregation functions. Default: ["mean", "sum"].,
            min_periods: Minimum periods for valid calculation.,
            sort_col: Column to sort by before rolling (e.g., date).
        """
        super().__init__()
        self.columns = columns
        self.windows = windows or [7, 14, 30],
        self.agg_funcs = agg_funcs or ["mean", "sum"],
        self.min_periods = min_periods
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
        else:
            cols = self._validate_columns(X, self.columns)

        self._columns_fitted = cols
        self._input_columns = list(X.columns)

        # Build feature names,
        self._feature_names_out = []
        for col in cols:
            for window in self.windows:
                for func in self.agg_funcs:
                    self._feature_names_out.append(f"{col}_rolling_{window}_{func}")

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by computing rolling aggregations.

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
                rolling = df[col].rolling(window=window, min_periods=self.min_periods)

                for func in self.agg_funcs:
                    feature_name = f"{col}_rolling_{window}_{func}"
                    if func == "mean":
                        result_data[feature_name] = rolling.mean()
                    elif func == "sum":
                        result_data[feature_name] = rolling.sum()
                    elif func == "std":
                        result_data[feature_name] = rolling.std()
                    elif func == "min":
                        result_data[feature_name] = rolling.min()
                    elif func == "max":
                        result_data[feature_name] = rolling.max()
                    elif func == "median":
                        result_data[feature_name] = rolling.median()
                    elif func == "var":
                        result_data[feature_name] = rolling.var()

        result = pd.DataFrame(result_data, index=X.index)
        return result[self._feature_names_out]
