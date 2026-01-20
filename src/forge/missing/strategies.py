"""Missing value imputation strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer as SKLearnKNNImputer
from sklearn.impute import SimpleImputer

from forge.generators.base import BaseFeatureGenerator

if TYPE_CHECKING:
    from typing_extensions import Self


class MeanMedianImputer(BaseFeatureGenerator):
    """Imputes missing values with mean or median.

    For numeric columns, replaces NaN values with the column's
    mean or median computed from the training data.

    Example:
        >>> imputer = MeanMedianImputer(strategy="median")
        >>> X_imputed = imputer.fit_transform(X)
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        strategy: str = "mean"
    ) -> None:
        """Initialize the imputer.

        Args:
            columns: Columns to impute. If None, all numeric.,
            strategy: Imputation strategy ("mean" or "median").
        """
        super().__init__()
        self.columns = columns
        self.strategy = strategy

        if strategy not in ("mean", "median"):
            raise ValueError(f"strategy must be 'mean' or 'median', got '{strategy}'")

        self._fill_values: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the imputer by computing fill values.

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
        self._feature_names_out = cols

        # Compute fill values
        for col in cols:
            if self.strategy == "mean":
                self._fill_values[col] = float(X[col].mean())
            else:
                self._fill_values[col] = float(X[col].median())

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with imputed values.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result = X[self._columns_fitted].copy()

        for col in self._columns_fitted:
            result[col] = result[col].fillna(self._fill_values[col])

        return result


class ModeImputer(BaseFeatureGenerator):
    """Imputes missing values with mode (most frequent value).

    Suitable for categorical columns.

    Example:
        >>> imputer = ModeImputer(columns=["category"])
        >>> X_imputed = imputer.fit_transform(X)
    """

    def __init__(
        self,
        columns: list[str] | None = None
    ) -> None:
        """Initialize the mode imputer.

        Args:
            columns: Columns to impute. If None, all object/category columns.
        """
        super().__init__()
        self.columns = columns

        self._fill_values: dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit by computing modes.

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
        self._feature_names_out = cols

        for col in cols:
            mode = X[col].mode()
            self._fill_values[col] = mode.iloc[0] if len(mode) > 0 else None

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values with mode.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with imputed values.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result = X[self._columns_fitted].copy()

        for col in self._columns_fitted:
            if self._fill_values[col] is not None:
                result[col] = result[col].fillna(self._fill_values[col])

        return result


class ConstantImputer(BaseFeatureGenerator):
    """Imputes missing values with a constant.

    Example:
        >>> imputer = ConstantImputer(fill_value=0)
        >>> X_imputed = imputer.fit_transform(X)
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        fill_value: Any = 0,
        fill_values: dict[str, Any] | None = None
    ) -> None:
        """Initialize constant imputer.

        Args:
            columns: Columns to impute. If None, all columns.,
            fill_value: Default fill value for all columns.,
            fill_values: Column-specific fill values.
        """
        super().__init__()
        self.columns = columns
        self.fill_value = fill_value
        self.fill_values = fill_values or {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the imputer.

        Args:
            X: Input DataFrame.,
            y: Ignored.,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)

        if self.columns is None:
            cols = list(X.columns)
        else:
            cols = self._validate_columns(X, self.columns)

        self._columns_fitted = cols
        self._input_columns = list(X.columns)
        self._feature_names_out = cols
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute with constant values.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with imputed values.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result = X[self._columns_fitted].copy()

        for col in self._columns_fitted:
            fill_val = self.fill_values.get(col, self.fill_value)
            result[col] = result[col].fillna(fill_val)

        return result


class KNNImputer(BaseFeatureGenerator):
    """Imputes missing values using K-Nearest Neighbors.

    Uses the K nearest neighbors to estimate missing values
    based on similar samples.

    Example:
        >>> imputer = KNNImputer(n_neighbors=5)
        >>> X_imputed = imputer.fit_transform(X)
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        n_neighbors: int = 5,
        weights: str = "uniform"
    ) -> None:
        """Initialize KNN imputer.

        Args:
            columns: Columns to impute. If None, all numeric.,
            n_neighbors: Number of neighbors to use.,
            weights: Weight function ("uniform" or "distance").
        """
        super().__init__()
        self.columns = columns
        self.n_neighbors = n_neighbors
        self.weights = weights

        self._imputer: SKLearnKNNImputer | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the KNN imputer.

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
        self._feature_names_out = cols

        self._imputer = SKLearnKNNImputer(
            n_neighbors = self.n_neighbors,
            weights=self.weights
        )
        self._imputer.fit(X[cols])

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute using KNN.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with imputed values.
        """
        self._check_is_fitted()
        self._validate_input(X)

        imputed = self._imputer.transform(X[self._columns_fitted])
        return pd.DataFrame(imputed, index=X.index, columns=self._columns_fitted)


class AutoImputer(BaseFeatureGenerator):
    """Automatically selects imputation strategy per column.

    Chooses the best imputation strategy based on column type
    and data characteristics.

    Example:
        >>> imputer = AutoImputer()
        >>> X_imputed = imputer.fit_transform(X)
    """

    def __init__(
        self,
        numeric_strategy: str = "median",
        categorical_strategy: str = "mode",
        add_indicator: bool = False
    ) -> None:
        """Initialize auto imputer.

        Args:
            numeric_strategy: Strategy for numeric columns.,
            categorical_strategy: Strategy for categorical columns.,
            add_indicator: Whether to add missing indicator columns.
        """
        super().__init__()
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.add_indicator = add_indicator

        self._numeric_imputer: MeanMedianImputer | None = None
        self._categorical_imputer: ModeImputer | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit imputers for each column type.

        Args:
            X: Input DataFrame.,
            y: Ignored.,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)
        self._input_columns = list(X.columns)

        # Fit numeric imputer,
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            self._numeric_imputer = MeanMedianImputer(
                columns = numeric_cols,
                strategy=self.numeric_strategy
            )
            self._numeric_imputer.fit(X)

        # Fit categorical imputer,
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            self._categorical_imputer = ModeImputer(columns=cat_cols)
            self._categorical_imputer.fit(X)

        # Build feature names,
        self._feature_names_out = list(X.columns)

        if self.add_indicator:
            cols_with_missing = X.columns[X.isna().any()].tolist()
            for col in cols_with_missing:
                self._feature_names_out.append(f"{col}_missing")

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply auto imputation.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with imputed values.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result = X.copy()

        # Apply numeric imputation
        if self._numeric_imputer is not None:
            numeric_imputed = self._numeric_imputer.transform(X)
            for col in numeric_imputed.columns:
                result[col] = numeric_imputed[col]

        # Apply categorical imputation
        if self._categorical_imputer is not None:
            cat_imputed = self._categorical_imputer.transform(X)
            for col in cat_imputed.columns:
                result[col] = cat_imputed[col]

        # Add indicators if requested
        if self.add_indicator:
            for col in X.columns:
                if X[col].isna().any():
                    result[f"{col}_missing"] = X[col].isna().astype(int)

        return result[self._feature_names_out]
