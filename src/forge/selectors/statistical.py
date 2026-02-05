"""Statistical feature selection methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)

from forge.selectors.base import BaseFeatureSelector

if TYPE_CHECKING:
    from typing_extensions import Self


class StatisticalSelector(BaseFeatureSelector):
    """Selects features using statistical tests.

    Supports chi-square, ANOVA F-test, and mutual information
    for both classification and regression tasks.

    Example:
        >>> selector = StatisticalSelector(
        ...     method="anova"
        ...     k=10
        ... )
        >>> X_selected = selector.fit_transform(X, y)
    """

    METHODS = {
        "chi2": chi2,
        "anova": f_classif,
        "f_regression": f_regression,
        "mutual_info_classif": mutual_info_classif,
        "mutual_info_regression": mutual_info_regression
    }

    def __init__(
        self,
        method: str = "anova",
        k: int | float | str = 10,
        threshold: float | None = None
    ) -> None:
        """Initialize the statistical selector.

        Args:
            method: Selection method (chi2, anova, f_regression, mutual_info_*).,
            k: Number of features to select. Can be int, float (fraction), or "all".,
            threshold: Optional score threshold instead of k.
        """
        super().__init__()
        self.method = method
        self.k = k
        self.threshold = threshold

        if method not in self.METHODS:
            raise ValueError(f"Unknown method '{method}'. Supported: {list(self.METHODS.keys())}")

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the selector by computing statistical scores.

        Args:
            X: Input features.,
            y: Target variable (required).,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)

        if y is None:
            raise ValueError("Target variable y is required for statistical selection")

        # Get numeric columns only,
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols].values

        # Handle missing values,
        X_clean = np.nan_to_num(X_numeric, nan=0)

        # For chi2, ensure non-negative values
        if self.method == "chi2":
            X_clean = np.abs(X_clean)

        # Compute scores,
        score_func = self.METHODS[self.method]

        if self.method in ("mutual_info_classif", "mutual_info_regression"):
            scores = score_func(X_clean, y, random_state=42)
            pvalues = np.zeros_like(scores)  # MI doesn't have p-values
        else:
            scores, pvalues = score_func(X_clean, y)

        # Handle NaN scores,
        scores = np.nan_to_num(scores, nan=0)

        self._scores = scores
        self._pvalues = pvalues

        # Determine selection,
        n_features = len(numeric_cols)

        if self.threshold is not None:
            # Select by threshold,
            mask = scores >= self.threshold
        elif isinstance(self.k, str) and self.k == "all":
            mask = np.ones(n_features, dtype=bool)
        elif isinstance(self.k, float) and 0 < self.k < 1:
            # k is a fraction,
            n_select = max(1, int(n_features * self.k))
            threshold_idx = min(n_select, n_features)
            threshold_score = np.sort(scores)[::-1][threshold_idx - 1]
            mask = scores >= threshold_score
        else:
            # k is an integer,
            n_select = min(int(self.k), n_features)
            threshold_score = np.sort(scores)[::-1][n_select - 1]
            mask = scores >= threshold_score

        self._support_mask = mask
        self._feature_names_in = numeric_cols
        self._feature_names_out = [
            col for col, keep in zip(numeric_cols, mask) if keep
        ]
        self._is_fitted = True

        return self

    def get_pvalues(self) -> pd.DataFrame | None:
        """Get p-values for each feature.

        Returns:
            DataFrame with features and p-values.
        """
        self._check_is_fitted()

        if not hasattr(self, "_pvalues"):
            return None

        return pd.DataFrame({
            "feature": self._feature_names_in,
            "score": self._scores,
            "pvalue": self._pvalues,
            "selected": self._support_mask
        }).sort_values("pvalue")


class KBestSelector(BaseFeatureSelector):
    """Selects K best features based on various scoring functions.

    A simpler interface for selecting top-K features.

    Example:
        >>> selector = KBestSelector(k=20, score_func="f_classif")
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        k: int = 10,
        score_func: str = "f_classif"
    ) -> None:
        """Initialize K-best selector.

        Args:
            k: Number of features to select.,
            score_func: Scoring function name.
        """
        super().__init__()
        self.k = k
        self.score_func = score_func

        self._internal_selector = StatisticalSelector(
            method = score_func,
            k=k
        )

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the selector.

        Args:
            X: Input features.,
            y: Target variable.,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)
        self._internal_selector.fit(X, y)

        self._support_mask = self._internal_selector._support_mask
        self._feature_names_in = self._internal_selector._feature_names_in
        self._feature_names_out = self._internal_selector._feature_names_out
        self._scores = self._internal_selector._scores
        self._is_fitted = True

        return self
