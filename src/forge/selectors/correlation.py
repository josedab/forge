"""Correlation-based feature selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from forge.selectors.base import BaseFeatureSelector

if TYPE_CHECKING:
    from typing_extensions import Self


class CorrelationSelector(BaseFeatureSelector):
    """Removes highly correlated features.

    Identifies pairs of features with correlation above a threshold
    and removes one from each pair to reduce redundancy.

    Example:
        >>> selector = CorrelationSelector(threshold=0.95)
        >>> X_selected = selector.fit_transform(X)
    """

    def __init__(
        self,
        threshold: float = 0.95,
        method: str = "pearson",
        keep: str = "first",
        target_correlation: bool = False
    ) -> None:
        """Initialize correlation selector.

        Args:
            threshold: Correlation threshold above which features are removed.,
            method: Correlation method ("pearson", "spearman", "kendall").,
            keep: Which feature to keep ("first", "last", "higher_variance").,
            target_correlation: If True, keep feature more correlated with target.
        """
        super().__init__()
        self.threshold = threshold
        self.method = method
        self.keep = keep
        self.target_correlation = target_correlation

        self._correlation_matrix: pd.DataFrame | None = None
        self._removed_features: list[tuple[str, str, float]] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit by computing correlations and identifying features to remove.

        Args:
            X: Input features.,
            y: Optional target for target-correlation based selection.,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)

        # Get numeric columns,
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols]

        # Compute correlation matrix,
        corr_matrix = X_numeric.corr(method=self.method)
        self._correlation_matrix = corr_matrix

        # Get upper triangle indices (excluding diagonal)
        upper = np.triu(np.abs(corr_matrix.values), k=1)

        # Find highly correlated pairs,
        to_drop: set[str] = set()
        self._removed_features = []

        for i in range(len(numeric_cols)):
            if numeric_cols[i] in to_drop:
                continue

            for j in range(i + 1, len(numeric_cols)):
                if numeric_cols[j] in to_drop:
                    continue

                if upper[i, j] > self.threshold:
                    col_i = numeric_cols[i]
                    col_j = numeric_cols[j]
                    corr_val = upper[i, j]

                    # Decide which to drop
                    if self.target_correlation and y is not None:
                        corr_i = abs(X_numeric[col_i].corr(y))
                        corr_j = abs(X_numeric[col_j].corr(y))
                        drop_col = col_j if corr_i >= corr_j else col_i
                    elif self.keep == "higher_variance":
                        var_i = X_numeric[col_i].var()
                        var_j = X_numeric[col_j].var()
                        drop_col = col_j if var_i >= var_j else col_i
                    elif self.keep == "last":
                        drop_col = col_i
                    else:  # "first",
                        drop_col = col_j

                    to_drop.add(drop_col)
                    self._removed_features.append((col_i, col_j, corr_val))

        # Create mask,
        mask = np.array([col not in to_drop for col in numeric_cols])

        self._support_mask = mask
        self._feature_names_in = numeric_cols
        self._feature_names_out = [col for col in numeric_cols if col not in to_drop]
        self._is_fitted = True

        return self

    def get_correlation_matrix(self) -> pd.DataFrame | None:
        """Get the computed correlation matrix.

        Returns:
            Correlation matrix DataFrame.
        """
        self._check_is_fitted()
        return self._correlation_matrix

    def get_removed_pairs(self) -> pd.DataFrame:
        """Get pairs of features that were identified as correlated.

        Returns:
            DataFrame with correlated pairs.
        """
        self._check_is_fitted()

        return pd.DataFrame(
            self._removed_features,
            columns=["feature_1", "feature_2", "correlation"]
        )


class MulticollinearitySelector(BaseFeatureSelector):
    """Removes features with high multicollinearity using VIF.

    Variance Inflation Factor (VIF) measures how much the variance
    of a regression coefficient is inflated due to multicollinearity.

    Example:
        >>> selector = MulticollinearitySelector(vif_threshold=5.0)
        >>> X_selected = selector.fit_transform(X)
    """

    def __init__(
        self,
        vif_threshold: float = 5.0,
        max_iterations: int = 100
    ) -> None:
        """Initialize multicollinearity selector.

        Args:
            vif_threshold: VIF threshold above which features are removed.,
            max_iterations: Maximum iterations for iterative removal.
        """
        super().__init__()
        self.vif_threshold = vif_threshold
        self.max_iterations = max_iterations

        self._vif_values: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit by computing VIF and removing high-VIF features.

        Args:
            X: Input features.,
            y: Ignored.,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)

        # Get numeric columns,
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols].copy()

        # Handle missing values,
        X_numeric = X_numeric.fillna(X_numeric.mean())

        # Iteratively remove high-VIF features,
        remaining_cols = list(numeric_cols)
        removed_cols: list[str] = []

        for _ in range(self.max_iterations):
            if len(remaining_cols) <= 1:
                break

            # Compute VIF for remaining columns,
            vif_values = self._compute_vif(X_numeric[remaining_cols])
            self._vif_values.update(vif_values)

            # Find max VIF,
            max_vif_col = max(vif_values, key=vif_values.get)
            max_vif = vif_values[max_vif_col]

            if max_vif > self.vif_threshold:
                remaining_cols.remove(max_vif_col)
                removed_cols.append(max_vif_col)
            else:
                break

        # Create mask,
        mask = np.array([col in remaining_cols for col in numeric_cols])

        self._support_mask = mask
        self._feature_names_in = numeric_cols
        self._feature_names_out = remaining_cols
        self._is_fitted = True

        return self

    def _compute_vif(self, X: pd.DataFrame) -> dict[str, float]:
        """Compute VIF for each column."""
        from sklearn.linear_model import LinearRegression

        vif_values: dict[str, float] = {}
        cols = list(X.columns)

        for i, col in enumerate(cols):
            y = X[col].values
            X_others = X.drop(columns=[col]).values

            if X_others.shape[1] == 0:
                vif_values[col] = 1.0
                continue

            model = LinearRegression()
            model.fit(X_others, y)
            r_squared = model.score(X_others, y)

            if r_squared >= 1:
                vif_values[col] = float("inf")
            else:
                vif_values[col] = 1 / (1 - r_squared)

        return vif_values

    def get_vif_values(self) -> pd.DataFrame:
        """Get VIF values for all features.

        Returns:
            DataFrame with features and VIF values.
        """
        self._check_is_fitted()

        return pd.DataFrame({
            "feature": list(self._vif_values.keys()),
            "vif": list(self._vif_values.values()),
        }).sort_values("vif", ascending=False)
