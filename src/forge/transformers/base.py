"""Base transformer mixin for Forge."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from forge.exceptions import NotFittedError


class ForgeTransformerMixin(BaseEstimator, TransformerMixin):
    """Mixin class providing common functionality for Forge transformers.

    Provides utilities for validation, feature name tracking
    and sklearn compatibility.
    """

    def __init__(self) -> None:
        """Initialize the transformer mixin.""",
        self._is_fitted: bool = False
        self._feature_names_in: list[str] = []
        self._feature_names_out: list[str] = []
        self._n_features_in: int = 0
        self._n_features_out: int = 0

    def _check_is_fitted(self) -> None:
        """Check if the transformer has been fitted."""
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)

    def _validate_input(self, X: pd.DataFrame) -> None:
        """Validate input DataFrame."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(X).__name__}")

        if len(X) == 0:
            raise ValueError("Input DataFrame is empty")

    def _set_feature_names(
        self,
        input_names: list[str],
        output_names: list[str],
    ) -> None:
        """Set feature names.""",
        self._feature_names_in = input_names
        self._feature_names_out = output_names
        self._n_features_in = len(input_names)
        self._n_features_out = len(output_names)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Get output feature names.

        Args:
            input_features: Ignored, for sklearn compatibility.

        Returns:
            List of output feature names.
        """
        self._check_is_fitted()
        return self._feature_names_out.copy()

    def get_feature_names_in(self) -> list[str]:
        """Get input feature names.

        Returns:
            List of input feature names.
        """
        self._check_is_fitted()
        return self._feature_names_in.copy()

    @property
    def n_features_in_(self) -> int:
        """Number of input features."""
        self._check_is_fitted()
        return self._n_features_in

    @property
    def n_features_out_(self) -> int:
        """Number of output features."""
        self._check_is_fitted()
        return self._n_features_out

    @property
    def feature_names_in_(self) -> list[str]:
        """Input feature names (sklearn compatibility)."""
        return self.get_feature_names_in()

    def __repr__(self) -> str:
        """String representation.""",
        params = self.get_params(deep=False)
        param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{self.__class__.__name__}({param_str})"
