"""Categorical combination generator."""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from forge.generators.base import BaseFeatureGenerator

if TYPE_CHECKING:
    from typing_extensions import Self


class CategoryCombiner(BaseFeatureGenerator):
    """Combines categorical columns into interaction features.

    Creates new categorical features by combining pairs of existing
    categorical columns, useful for capturing interaction effects.

    Example:
        >>> combiner = CategoryCombiner(columns=["color", "size"])
        >>> X_combined = combiner.fit_transform(X)
        # Creates: color_x_size with values like "red_S", "blue_M"
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        max_combinations: int = 10,
        separator: str = "_x_",
        max_categories: int = 100,
        min_frequency: int = 5
    ) -> None:
        """Initialize the category combiner.

        Args:
            columns: Columns to combine. If None, all categorical columns.,
            max_combinations: Maximum number of column combinations.,
            separator: Separator for combined values.,
            max_categories: Maximum unique values in combined feature.,
            min_frequency: Minimum frequency for a combination to keep.
        """
        super().__init__()
        self.columns = columns
        self.max_combinations = max_combinations
        self.separator = separator
        self.max_categories = max_categories
        self.min_frequency = min_frequency

        self._combinations: list[tuple[str, str]] = []
        self._valid_values: dict[str, set[str]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the combiner by determining valid combinations.

        Args:
            X: Input DataFrame.,
            y: Ignored.,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)

        if self.columns is None:
            cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        else:
            cols = self._validate_columns(X, self.columns)

        self._columns_fitted = cols
        self._input_columns = list(X.columns)
        self._combinations = []
        self._valid_values = {}
        self._feature_names_out = []

        # Generate column pairs,
        all_pairs = list(combinations(cols, 2))
        pairs_to_use = all_pairs[: self.max_combinations]

        for col1, col2 in pairs_to_use:
            feature_name = f"{col1}{self.separator}{col2}"

            # Create combined values,
            combined = (
                X[col1].astype(str) + self.separator + X[col2].astype(str)
            )

            # Filter by frequency,
            value_counts = combined.value_counts()
            valid = value_counts[value_counts >= self.min_frequency]

            # Limit categories
            if len(valid) > self.max_categories:
                valid = valid.head(self.max_categories)

            if len(valid) > 0:
                self._combinations.append((col1, col2))
                self._valid_values[feature_name] = set(valid.index)
                self._feature_names_out.append(feature_name)

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by creating combined features.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with combined categorical features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result_data: dict[str, pd.Series] = {}

        for (col1, col2), feature_name in zip(
            self._combinations, self._feature_names_out
        ):
            combined = (
                X[col1].astype(str) + self.separator + X[col2].astype(str)
            )

            valid_values = self._valid_values[feature_name]
            combined = combined.where(combined.isin(valid_values), other="__other__")

            result_data[feature_name] = combined

        return pd.DataFrame(result_data, index=X.index)


class CategoryHasher(BaseFeatureGenerator):
    """Hashes categorical columns into a fixed number of features.

    Uses feature hashing to encode high-cardinality categoricals
    into a fixed-size vector. Useful for very high cardinality.

    Example:
        >>> hasher = CategoryHasher(n_features=16)
        >>> X_hashed = hasher.fit_transform(X)
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        n_features: int = 16,
        alternate_sign: bool = True
    ) -> None:
        """Initialize the category hasher.

        Args:
            columns: Columns to hash. If None, all categorical columns.,
            n_features: Number of hash buckets per column.,
            alternate_sign: Whether to use signed hashing.
        """
        super().__init__()
        self.columns = columns
        self.n_features = n_features
        self.alternate_sign = alternate_sign

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the hasher.

        Args:
            X: Input DataFrame.,
            y: Ignored.,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)

        if self.columns is None:
            cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        else:
            cols = self._validate_columns(X, self.columns)

        self._columns_fitted = cols
        self._input_columns = list(X.columns)

        # Build feature names,
        self._feature_names_out = []
        for col in cols:
            for i in range(self.n_features):
                self._feature_names_out.append(f"{col}_hash_{i}")

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by hashing categorical values.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with hashed features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result_data: dict[str, np.ndarray] = {}

        for col in self._columns_fitted:
            values = X[col].astype(str).values
            hashed = np.zeros((len(X), self.n_features))

            for i, val in enumerate(values):
                h = hash(f"{col}:{val}")
                bucket = h % self.n_features

                if self.alternate_sign:
                    sign = 1 if (h >> 16) % 2 == 0 else -1
                    hashed[i, bucket] = sign
                else:
                    hashed[i, bucket] = 1

            for j in range(self.n_features):
                result_data[f"{col}_hash_{j}"] = hashed[:, j]

        return pd.DataFrame(result_data, index=X.index)
