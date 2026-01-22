"""Feature interaction generator."""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from forge.generators.base import BaseFeatureGenerator

if TYPE_CHECKING:
    from typing_extensions import Self


class InteractionGenerator(BaseFeatureGenerator):
    """Generates interaction features between numeric columns.

    Creates new features by combining pairs of numeric columns
    through various operations (multiply, divide, add, subtract).

    Example:
        >>> gen = InteractionGenerator(
        ...     columns=["price", "quantity"],
        ...     operations=["multiply", "divide"]
        ... )
        >>> X_int = gen.fit_transform(X)
        # Creates: price_x_quantity, price_div_quantity
    """

    OPERATIONS = {
        "multiply": lambda a, b: a * b,
        "divide": lambda a, b: np.where(b != 0, a / b, 0),
        "add": lambda a, b: a + b,
        "subtract": lambda a, b: a - b,
        "ratio": lambda a, b: np.where(
            (a + b) != 0, (a - b) / (a + b), 0
        )
    }

    OPERATION_SYMBOLS = {
        "multiply": "x",
        "divide": "div",
        "add": "plus",
        "subtract": "minus",
        "ratio": "ratio",
    }

    def __init__(
        self,
        columns: list[str] | None = None,
        operations: list[str] | None = None,
        include_self_interactions: bool = False,
        max_interactions: int | None = None
    ) -> None:
        """Initialize the interaction generator.

        Args:
            columns: Columns to create interactions for. If None, all numeric.,
            operations: Operations to apply. Default: ["multiply", "divide"].,
            include_self_interactions: Whether to include column^2 interactions.,
            max_interactions: Maximum number of interaction features to create.
        """
        super().__init__()
        self.columns = columns
        self.operations = operations or ["multiply", "divide"]
        self.include_self_interactions = include_self_interactions
        self.max_interactions = max_interactions

        # Validate operations,
        invalid = set(self.operations) - set(self.OPERATIONS.keys())
        if invalid:
            raise ValueError(
                f"Unsupported operations: {invalid}. ",
                f"Supported: {list(self.OPERATIONS.keys())}"
            )

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

        # Generate interaction pairs,
        self._interactions: list[tuple[str, str, str]] = []

        for col1, col2 in combinations(cols, 2):
            for op in self.operations:
                self._interactions.append((col1, col2, op))

        if self.include_self_interactions:
            for col in cols:
                self._interactions.append((col, col, "multiply"))

        # Limit interactions if specified
        if self.max_interactions and len(self._interactions) > self.max_interactions:
            self._interactions = self._interactions[: self.max_interactions]

        # Build feature names,
        self._feature_names_out = []
        for col1, col2, op in self._interactions:
            symbol = self.OPERATION_SYMBOLS[op]
            if col1 == col2:
                name = f"{col1}_squared"
            else:
                name = f"{col1}_{symbol}_{col2}"
            self._feature_names_out.append(name)

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by creating interaction features.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with interaction features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result_data: dict[str, np.ndarray] = {}

        for (col1, col2, op), feature_name in zip(
            self._interactions, self._feature_names_out
        ):
            a = X[col1].values
            b = X[col2].values
            op_func = self.OPERATIONS[op]
            result_data[feature_name] = op_func(a, b)

        return pd.DataFrame(result_data, index=X.index)


class DifferenceGenerator(BaseFeatureGenerator):
    """Generates difference features between pairs of columns.

    Useful for creating features that represent relative differences
    between related numeric values.

    Example:
        >>> gen = DifferenceGenerator(
        ...     column_pairs=[("high", "low"), ("open", "close")]
        ... )
        >>> X_diff = gen.fit_transform(X)
    """

    def __init__(
        self,
        column_pairs: list[tuple[str, str]] | None = None,
        include_ratio: bool = True,
        include_percent: bool = True
    ) -> None:
        """Initialize the difference generator.

        Args:
            column_pairs: Pairs of columns to compute differences for.,
            include_ratio: Whether to include ratio features.,
            include_percent: Whether to include percent change features.
        """
        super().__init__()
        self.column_pairs = column_pairs
        self.include_ratio = include_ratio
        self.include_percent = include_percent

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the generator.

        Args:
            X: Input DataFrame.,
            y: Ignored.,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)
        self._input_columns = list(X.columns)

        if self.column_pairs is None:
            # Auto-detect pairs from column names with common patterns,
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            self._pairs_fitted = self._auto_detect_pairs(numeric_cols)
        else:
            self._pairs_fitted = self.column_pairs

        # Validate pairs exist
        for col1, col2 in self._pairs_fitted:
            if col1 not in X.columns or col2 not in X.columns:
                raise ValueError(f"Column pair ({col1}, {col2}) not found in DataFrame")

        # Build feature names,
        self._feature_names_out = []
        for col1, col2 in self._pairs_fitted:
            self._feature_names_out.append(f"{col1}_minus_{col2}")
            if self.include_ratio:
                self._feature_names_out.append(f"{col1}_over_{col2}")
            if self.include_percent:
                self._feature_names_out.append(f"{col1}_pct_change_{col2}")

        self._is_fitted = True
        return self

    def _auto_detect_pairs(self, columns: list[str]) -> list[tuple[str, str]]:
        """Auto-detect column pairs based on naming patterns."""
        pairs: list[tuple[str, str]] = []

        # Common patterns
        patterns = [
            ("_high", "_low"),
            ("_max", "_min"),
            ("_open", "_close"),
            ("_start", "_end"),
            ("_begin", "_end"),
        ]

        for suffix1, suffix2 in patterns:
            for col in columns:
                if col.endswith(suffix1):
                    base = col[: -len(suffix1)]
                    pair_col = base + suffix2
                    if pair_col in columns:
                        pairs.append((col, pair_col))

        return pairs

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by creating difference features.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with difference features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result_data: dict[str, np.ndarray] = {}

        for col1, col2 in self._pairs_fitted:
            a = X[col1].values.astype(float)
            b = X[col2].values.astype(float)

            # Absolute difference,
            result_data[f"{col1}_minus_{col2}"] = a - b

            if self.include_ratio:
                # Ratio (handle division by zero)
                result_data[f"{col1}_over_{col2}"] = np.where(b != 0, a / b, 0)

            if self.include_percent:
                # Percent change,
                result_data[f"{col1}_pct_change_{col2}"] = np.where(
                    b != 0, (a - b) / np.abs(b), 0
                )

        return pd.DataFrame(result_data, index=X.index)
