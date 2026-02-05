"""Categorical statistics generator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from forge.generators.base import BaseFeatureGenerator

if TYPE_CHECKING:
    from typing_extensions import Self


class CategoryStatistics(BaseFeatureGenerator):
    """Generates statistical features based on categorical groupings.

    Computes statistics of numeric columns grouped by categorical columns
    creating features that capture category-level aggregations.

    Example:
        >>> gen = CategoryStatistics(
        ...     group_cols=["category"]
        ...     agg_cols=["price"]
        ...     stats=["mean", "std", "count"]
        ... )
        >>> X_stats = gen.fit_transform(X)
    """

    SUPPORTED_STATS = ["mean", "std", "min", "max", "median", "count", "sum", "nunique"]

    def __init__(
        self,
        group_cols: list[str],
        agg_cols: list[str] | None = None,
        stats: list[str] | None = None,
        suffix: str = ""
    ) -> None:
        """Initialize the category statistics generator.

        Args:
            group_cols: Categorical columns to group by.,
            agg_cols: Numeric columns to aggregate. If None, all numeric.,
            stats: Statistics to compute. Default: ["mean", "std", "count"].,
            suffix: Optional suffix for feature names.
        """
        super().__init__()
        self.group_cols = group_cols
        self.agg_cols = agg_cols
        self.stats = stats or ["mean", "std", "count"]
        self.suffix = suffix

        # Validate stats,
        invalid = set(self.stats) - set(self.SUPPORTED_STATS)
        if invalid:
            raise ValueError(f"Unsupported stats: {invalid}")

        self._group_stats: dict[str, pd.DataFrame] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit by computing category statistics.

        Args:
            X: Input DataFrame.,
            y: Ignored.,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)
        self._validate_columns(X, self.group_cols)

        if self.agg_cols is None:
            agg_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            agg_cols = [c for c in agg_cols if c not in self.group_cols]
        else:
            agg_cols = self._validate_columns(X, self.agg_cols)

        self._agg_cols_fitted = agg_cols
        self._input_columns = list(X.columns)
        self._feature_names_out = []
        self._group_stats = {}

        # Compute statistics
        for group_col in self.group_cols:
            agg_dict = dict.fromkeys(agg_cols, self.stats)
            grouped = X.groupby(group_col, observed=True).agg(agg_dict)

            # Flatten column names,
            new_cols = []
            for col, stat in grouped.columns:
                suffix_str = f"_{self.suffix}" if self.suffix else ""
                new_name = f"{group_col}_{col}_{stat}{suffix_str}"
                new_cols.append(new_name)

            grouped.columns = new_cols
            self._group_stats[group_col] = grouped
            self._feature_names_out.extend(new_cols)

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by merging category statistics.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with category statistics features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result = X.copy()

        for group_col in self.group_cols:
            stats_df = self._group_stats[group_col]
            result = result.merge(
                stats_df,
                left_on = group_col,
                right_index = True,
                how="left"
            )

        return result[self._feature_names_out]


class CategoryTargetStatistics(BaseFeatureGenerator):
    """Generates target-based statistics for categorical columns.

    Computes target variable statistics per category, with options
    for smoothing and leave-one-out encoding.

    Example:
        >>> gen = CategoryTargetStatistics(
        ...     columns=["category"]
        ...     stats=["mean", "std"]
        ... )
        >>> X_stats = gen.fit_transform(X, y)
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        stats: list[str] | None = None,
        smoothing: float = 10.0,
        leave_one_out: bool = False
    ) -> None:
        """Initialize the target statistics generator.

        Args:
            columns: Categorical columns. If None, all categorical.,
            stats: Statistics to compute. Default: ["mean", "std"].,
            smoothing: Smoothing parameter for regularization.,
            leave_one_out: Whether to use leave-one-out encoding.
        """
        super().__init__()
        self.columns = columns
        self.stats = stats or ["mean", "std"]
        self.smoothing = smoothing
        self.leave_one_out = leave_one_out

        self._target_stats: dict[str, dict[str, dict[Any, float]]] = {}
        self._global_stats: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit by computing target statistics per category.

        Args:
            X: Input DataFrame.,
            y: Target variable (required).,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)

        if y is None:
            raise ValueError("Target variable y is required")

        if self.columns is None:
            cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        else:
            cols = self._validate_columns(X, self.columns)

        self._columns_fitted = cols
        self._input_columns = list(X.columns)
        self._target_stats = {}
        self._feature_names_out = []

        # Compute global stats,
        self._global_stats = {
            "mean": float(y.mean()),
            "std": float(y.std()),
            "median": float(y.median()),
        }

        df_temp = pd.DataFrame({"target": y})
        for col in cols:
            df_temp[col] = X[col].values

        for col in cols:
            self._target_stats[col] = {}

            for stat in self.stats:
                grouped = df_temp.groupby(col, observed=True)["target"]

                if stat == "mean":
                    agg = grouped.mean()
                    counts = grouped.count()
                    # Apply smoothing,
                    global_mean = self._global_stats["mean"]
                    agg = (counts * agg + self.smoothing * global_mean) / (
                        counts + self.smoothing
                    )
                elif stat == "std":
                    agg = grouped.std().fillna(0)
                elif stat == "median":
                    agg = grouped.median()
                elif stat == "count":
                    agg = grouped.count()
                elif stat == "min":
                    agg = grouped.min()
                elif stat == "max":
                    agg = grouped.max()
                else:
                    continue

                self._target_stats[col][stat] = agg.to_dict()
                self._feature_names_out.append(f"{col}_target_{stat}")

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by applying target statistics.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with target statistics features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result_data: dict[str, np.ndarray] = {}

        for col in self._columns_fitted:
            for stat in self.stats:
                feature_name = f"{col}_target_{stat}"
                stat_dict = self._target_stats[col].get(stat, {})
                global_val = self._global_stats.get(stat, 0.0)

                values = X[col].map(stat_dict).fillna(global_val).values
                result_data[feature_name] = values

        return pd.DataFrame(result_data, index=X.index)
