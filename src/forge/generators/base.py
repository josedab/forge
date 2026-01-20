"""Base classes for feature generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from forge.exceptions import NotFittedError, ValidationError

if TYPE_CHECKING:
    from typing_extensions import Self


class BaseFeatureGenerator(BaseEstimator, TransformerMixin, ABC):
    """Abstract base class for all feature generators.

    All feature generators inherit from this class and implement
    the fit and transform methods. This class provides sklearn
    compatibility through BaseEstimator and TransformerMixin.

    Example:
        >>> class MyGenerator(BaseFeatureGenerator):
        ...     def fit(self, X, y=None):
        ...         self._is_fitted = True
        ...         return self
        ...     def transform(self, X):
        ...         self._check_is_fitted()
        ...         return X_transformed
    """

    def __init__(self) -> None:
        """Initialize the generator."""
        self._is_fitted: bool = False
        self._feature_names_out: list[str] = []
        self._input_columns: list[str] = []

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the generator to the data.

        Args:
            X: Input DataFrame.,
            y: Optional target variable.,

        Returns:
            Self for method chaining.
        """
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data to generate new features.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with generated features.
        """
        pass

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            X: Input DataFrame.,
            y: Optional target variable.,

        Returns:
            DataFrame with generated features.
        """
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Get names of output features.

        Args:
            input_features: Ignored, for sklearn compatibility.

        Returns:
            List of feature names.
        """
        self._check_is_fitted()
        return self._feature_names_out.copy()

    def _check_is_fitted(self) -> None:
        """Check if the generator has been fitted."""
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)

    def _validate_input(self, X: pd.DataFrame) -> None:
        """Validate input DataFrame.

        Args:
            X: Input DataFrame.,

        Raises:
            ValidationError: If input is invalid.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValidationError(
                f"Expected pandas DataFrame, got {type(X).__name__}"
            )

        if len(X) == 0:
            raise ValidationError("Input DataFrame is empty")

    def _validate_columns(
        self,
        X: pd.DataFrame,
        columns: list[str] | None,
        required: bool = True
    ) -> list[str]:
        """Validate and get columns to process.

        Args:
            X: Input DataFrame.,
            columns: Columns to validate, or None for all.
            required: Whether columns are required to exist.,

        Returns:
            List of valid columns.
        """
        if columns is None:
            return list(X.columns)

        missing = [c for c in columns if c not in X.columns]
        if missing and required:
            raise ValidationError(
                f"Columns not found in DataFrame: {missing}. "
                f"Available columns: {list(X.columns)}"
            )

        return [c for c in columns if c in X.columns]

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this generator.

        Args:
            deep: If True, return parameters of sub-estimators.

        Returns:
            Parameter dictionary.
        """
        return super().get_params(deep=deep)

    def set_params(self, **params: Any) -> Self:
        """Set parameters for this generator.

        Args:
            **params: Parameter key-value pairs.

        Returns:
            Self for method chaining.
        """
        return super().set_params(**params)


class ColumnSelector(BaseFeatureGenerator):
    """Selects specific columns from a DataFrame.

    Useful as the first step in a pipeline to select columns
    for downstream processing.

    Example:
        >>> selector = ColumnSelector(columns=["a", "b", "c"])
        >>> X_selected = selector.fit_transform(X)
    """

    def __init__(self, columns: list[str] | None = None) -> None:
        """Initialize the column selector.

        Args:
            columns: Columns to select. If None, selects all.
        """
        super().__init__()
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the selector.

        Args:
            X: Input DataFrame.,
            y: Ignored.,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)
        self._input_columns = list(X.columns)

        if self.columns is None:
            self._feature_names_out = list(X.columns)
        else:
            self._feature_names_out = self._validate_columns(X, self.columns)

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select columns from the DataFrame.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with selected columns.
        """
        self._check_is_fitted()
        return X[self._feature_names_out].copy()


class PassthroughGenerator(BaseFeatureGenerator):
    """Passes input through unchanged.

    Useful as a placeholder or for combining with other generators.
    """

    def __init__(self) -> None:
        """Initialize the passthrough generator."""
        super().__init__()

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
        self._feature_names_out = list(X.columns)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return input unchanged.

        Args:
            X: Input DataFrame.,

        Returns:
            Copy of input DataFrame.
        """
        self._check_is_fitted()
        return X.copy()
