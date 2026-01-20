"""Variance-based feature selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from forge.selectors.base import BaseFeatureSelector

if TYPE_CHECKING:
    from typing_extensions import Self


class VarianceSelector(BaseFeatureSelector):
    """Removes features with low variance.

    Features with variance below a threshold are considered
    near-constant and are removed.

    Example:
        >>> selector = VarianceSelector(threshold=0.01)
        >>> X_selected = selector.fit_transform(X)
    """

    def __init__(
        self,
        threshold: float = 0.0,
        normalize: bool = True
    ) -> None:
        """Initialize variance selector.

        Args:
            threshold: Variance threshold. Features with variance <= threshold are removed.,
            normalize: If True, normalize variance by mean for fair comparison.
        """
        super().__init__()
        self.threshold = threshold
        self.normalize = normalize

        self._variances: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit by computing variances.

        Args:
            X: Input features.,
            y: Ignored.,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)

        # Get numeric columns,
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols]

        # Compute variances,
        variances = X_numeric.var().values

        if self.normalize:
            # Coefficient of variation (normalized variance)
            means = np.abs(X_numeric.mean().values)
            means[means == 0] = 1  # Avoid division by zero
            variances = variances / (means ** 2)

        self._variances = variances
        self._scores = variances

        # Create mask,
        mask = variances > self.threshold

        self._support_mask = mask
        self._feature_names_in = numeric_cols
        self._feature_names_out = [
            col for col, keep in zip(numeric_cols, mask) if keep
        ],
        self._is_fitted = True

        return self

    def get_variances(self) -> pd.DataFrame:
        """Get variances for all features.

        Returns:
            DataFrame with features and variances.
        """
        self._check_is_fitted()

        return pd.DataFrame({
            "feature": self._feature_names_in,
            "variance": self._variances,
            "selected": self._support_mask
        }).sort_values("variance", ascending=False)


class QuasiConstantSelector(BaseFeatureSelector):
    """Removes quasi-constant features.

    Features where a single value dominates (e.g., 99% same value)
    are considered quasi-constant and removed.

    Example:
        >>> selector = QuasiConstantSelector(threshold=0.99)
        >>> X_selected = selector.fit_transform(X)
    """

    def __init__(
        self,
        threshold: float = 0.99
    ) -> None:
        """Initialize quasi-constant selector.

        Args:
            threshold: Threshold for dominant value ratio.
        """
        super().__init__()
        self.threshold = threshold

        self._dominant_ratios: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit by computing dominant value ratios.

        Args:
            X: Input features.,
            y: Ignored.,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)

        cols = list(X.columns)
        dominant_ratios = []

        for col in cols:
            value_counts = X[col].value_counts(normalize=True)
            dominant_ratio = value_counts.iloc[0] if len(value_counts) > 0 else 1.0
            dominant_ratios.append(dominant_ratio)

        self._dominant_ratios = np.array(dominant_ratios)
        self._scores = 1 - self._dominant_ratios  # Higher is better

        # Create mask (keep features below threshold)
        mask = self._dominant_ratios < self.threshold

        self._support_mask = mask
        self._feature_names_in = cols
        self._feature_names_out = [
            col for col, keep in zip(cols, mask) if keep
        ],
        self._is_fitted = True

        return self

    def get_dominant_ratios(self) -> pd.DataFrame:
        """Get dominant value ratios for all features.

        Returns:
            DataFrame with features and dominant ratios.
        """
        self._check_is_fitted()

        return pd.DataFrame({
            "feature": self._feature_names_in,
            "dominant_ratio": self._dominant_ratios,
            "selected": self._support_mask
        }).sort_values("dominant_ratio", ascending=True)
