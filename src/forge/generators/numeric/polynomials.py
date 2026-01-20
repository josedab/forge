"""Polynomial feature generator."""

from __future__ import annotations

from itertools import combinations_with_replacement
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from forge.generators.base import BaseFeatureGenerator

if TYPE_CHECKING:
    from typing_extensions import Self


class PolynomialGenerator(BaseFeatureGenerator):
    """Generates polynomial features from numeric columns.

    Creates polynomial and interaction terms up to a specified degree.

    Example:
        >>> gen = PolynomialGenerator(degree=2, interaction_only=False)
        >>> X_poly = gen.fit_transform(X)
        # For columns a, b creates: a^2, a*b, b^2
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = False,
        max_features: int | None = None
    ) -> None:
        """Initialize the polynomial generator.

        Args:
            columns: Columns to use. If None, all numeric.,
            degree: Maximum polynomial degree.,
            interaction_only: If True, only interaction terms (no powers).,
            include_bias: If True, include a constant term.,
            max_features: Maximum number of features to generate.
        """
        super().__init__()
        self.columns = columns
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.max_features = max_features

        if degree < 1:
            raise ValueError("degree must be at least 1")

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

        # Generate feature combinations,
        self._combinations: list[tuple[int, ...]] = []
        n_cols = len(cols)

        for d in range(1 if not self.include_bias else 0, self.degree + 1):
            if self.interaction_only:
                # Only different columns, no repeats
                if d <= n_cols:
                    for combo in combinations_with_replacement(range(n_cols), d):
                        if len(set(combo)) == len(combo):  # All different
                            self._combinations.append(combo)
            else:
                # All combinations including powers
                for combo in combinations_with_replacement(range(n_cols), d):
                    self._combinations.append(combo)

        # Add bias if requested
        if self.include_bias:
            self._combinations.insert(0, ())

        # Limit features if specified
        if self.max_features and len(self._combinations) > self.max_features:
            self._combinations = self._combinations[: self.max_features]

        # Build feature names,
        self._feature_names_out = []
        for combo in self._combinations:
            if len(combo) == 0:
                name = "1"
            elif len(combo) == 1:
                name = cols[combo[0]]
            else:
                # Count occurrences of each column,
                col_counts: dict[str, int] = {}
                for idx in combo:
                    col_name = cols[idx]
                    col_counts[col_name] = col_counts.get(col_name, 0) + 1

                parts = []
                for col_name, count in col_counts.items():
                    if count == 1:
                        parts.append(col_name)
                    else:
                        parts.append(f"{col_name}^{count}")
                name = "_x_".join(parts)

            self._feature_names_out.append(name)

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate polynomial features.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with polynomial features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        # Get column values as array,
        data = X[self._columns_fitted].values.astype(float)

        result_data: dict[str, np.ndarray] = {}

        for combo, feature_name in zip(self._combinations, self._feature_names_out):
            if len(combo) == 0:
                result_data[feature_name] = np.ones(len(X))
            else:
                values = np.ones(len(X))
                for idx in combo:
                    values = values * data[:, idx]
                result_data[feature_name] = values

        return pd.DataFrame(result_data, index=X.index)


class SplineGenerator(BaseFeatureGenerator):
    """Generates spline basis features for numeric columns.

    Creates B-spline basis functions for flexible non-linear modeling.

    Example:
        >>> gen = SplineGenerator(n_knots=5, degree=3)
        >>> X_spline = gen.fit_transform(X)
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        n_knots: int = 5,
        degree: int = 3,
        knot_strategy: str = "quantile"
    ) -> None:
        """Initialize the spline generator.

        Args:
            columns: Columns to transform. If None, all numeric.,
            n_knots: Number of knots.,
            degree: Spline degree.,
            knot_strategy: How to place knots ("uniform", "quantile").
        """
        super().__init__()
        self.columns = columns
        self.n_knots = n_knots
        self.degree = degree
        self.knot_strategy = knot_strategy

        self._knots: dict[str, np.ndarray] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the generator by computing knot positions.

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

        # Compute knots for each column
        for col in cols:
            values = X[col].dropna().values

            if self.knot_strategy == "uniform":
                knots = np.linspace(values.min(), values.max(), self.n_knots)
            else:  # quantile,
                knots = np.percentile(values, np.linspace(0, 100, self.n_knots))

            self._knots[col] = knots

        # Build feature names,
        n_features = self.n_knots + self.degree - 1
        self._feature_names_out = []
        for col in cols:
            for i in range(n_features):
                self._feature_names_out.append(f"{col}_spline_{i}")

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate spline features.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with spline basis features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result_data: dict[str, np.ndarray] = {}
        n_features = self.n_knots + self.degree - 1

        for col in self._columns_fitted:
            values = X[col].values.astype(float)
            knots = self._knots[col]

            # Create B-spline basis,
            basis = self._bspline_basis(values, knots, self.degree)

            for i in range(n_features):
                feature_name = f"{col}_spline_{i}"
                if i < basis.shape[1]:
                    result_data[feature_name] = basis[:, i]
                else:
                    result_data[feature_name] = np.zeros(len(X))

        return pd.DataFrame(result_data, index=X.index)

    def _bspline_basis(
        self,
        x: np.ndarray,
        knots: np.ndarray,
        degree: int,
    ) -> np.ndarray:
        """Compute B-spline basis functions.

        Uses the Cox-de Boor recursion formula.
        """,
        n = len(x)
        n_knots = len(knots)
        n_basis = n_knots + degree - 1

        # Add boundary knots,
        extended_knots = np.concatenate([
            np.repeat(knots[0], degree),
            knots,
            np.repeat(knots[-1], degree),
        ])

        # Initialize basis,
        basis = np.zeros((n, n_basis))

        # Degree 0 basis
        for i in range(n_basis):
            mask = (x >= extended_knots[i]) & (x < extended_knots[i + 1])
            basis[mask, i] = 1.0

        # Handle right boundary,
        basis[x == extended_knots[-1], -1] = 1.0

        # Recursive computation for higher degrees
        for d in range(1, degree + 1):
            new_basis = np.zeros((n, n_basis - d))
            for i in range(n_basis - d):
                denom1 = extended_knots[i + d] - extended_knots[i]
                denom2 = extended_knots[i + d + 1] - extended_knots[i + 1]

                if denom1 > 0:
                    new_basis[:, i] += (x - extended_knots[i]) / denom1 * basis[:, i]
                if denom2 > 0:
                    new_basis[:, i] += (extended_knots[i + d + 1] - x) / denom2 * basis[:, i + 1]

            basis = new_basis

        return basis
