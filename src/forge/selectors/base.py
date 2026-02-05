"""Base class for feature selectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from forge.exceptions import NotFittedError, ValidationError

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from typing_extensions import Self


class BaseFeatureSelector(BaseEstimator, TransformerMixin, ABC):
    """Abstract base class for all feature selectors.

    Feature selectors reduce the dimensionality of feature sets by
    selecting the most important or relevant features.

    Example:
        >>> class MySelector(BaseFeatureSelector):
        ...     def fit(self, X, y=None):
        ...         self._support_mask = np.array([True, False, True])
        ...         self._is_fitted = True
        ...         return self
    """

    def __init__(self) -> None:
        """Initialize the selector."""
        self._is_fitted: bool = False
        self._support_mask: NDArray[np.bool_] = np.array([], dtype=bool)
        self._feature_names_in: list[str] = []
        self._feature_names_out: list[str] = []
        self._scores: NDArray[np.floating[Any]] | None = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the selector to the data.

        Args:
            X: Input features.
            y: Target variable (may be required by some selectors).

        Returns:
            Self for method chaining.
        """
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by selecting features.

        Args:
            X: Input DataFrame.

        Returns:
            DataFrame with selected features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        selected_cols = self.get_support(indices=False)
        if isinstance(selected_cols, np.ndarray):
            cols_to_keep = [
                col for col, keep in zip(self._feature_names_in, selected_cols)
                if keep
            ]
        else:
            cols_to_keep = selected_cols

        return X[cols_to_keep].copy()

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            X: Input DataFrame.
            y: Optional target variable.

        Returns:
            DataFrame with selected features.
        """
        return self.fit(X, y).transform(X)

    def get_support(self, indices: bool = False) -> NDArray[Any] | list[str]:
        """Get the mask or indices of selected features.

        Args:
            indices: If True, return indices. If False, return mask.

        Returns:
            Boolean mask or feature names of selected features.
        """
        self._check_is_fitted()

        if indices:
            return np.where(self._support_mask)[0]
        return self._support_mask

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Get names of selected features.

        Args:
            input_features: Ignored, for sklearn compatibility.

        Returns:
            List of selected feature names.
        """
        self._check_is_fitted()
        return self._feature_names_out.copy()

    def get_scores(self) -> pd.DataFrame | None:
        """Get feature scores from the selection process.

        Returns:
            DataFrame with feature names and scores, or None.
        """
        self._check_is_fitted()

        if self._scores is None:
            return None

        return pd.DataFrame({
            "feature": self._feature_names_in,
            "score": self._scores,
            "selected": self._support_mask
        }).sort_values("score", ascending=False)

    def _check_is_fitted(self) -> None:
        """Check if the selector has been fitted."""
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)

    def _validate_input(self, X: pd.DataFrame) -> None:
        """Validate input DataFrame.

        Args:
            X: Input DataFrame.

        Raises:
            ValidationError: If input is invalid.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValidationError(
                f"Expected pandas DataFrame, got {type(X).__name__}"
            )

        if len(X) == 0:
            raise ValidationError("Input DataFrame is empty")

    def _finalize_fit(self, X: pd.DataFrame) -> None:
        """Finalize fit by setting feature names.

        Args:
            X: Input DataFrame.
        """
        self._feature_names_in = list(X.columns)
        self._feature_names_out = [
            col for col, keep in zip(self._feature_names_in, self._support_mask)
            if keep
        ]
        self._is_fitted = True

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform is not supported for selection.

        Args:
            X: Input DataFrame.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "inverse_transform is not supported for feature selection"
        )
