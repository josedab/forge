"""Model-based importance feature selection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from forge.selectors.base import BaseFeatureSelector

if TYPE_CHECKING:
    from typing_extensions import Self


class ImportanceSelector(BaseFeatureSelector):
    """Selects features using tree-based feature importance.

    Uses Random Forest or other tree-based models to compute
    feature importances and select the most important features.

    Example:
        >>> selector = ImportanceSelector(
        ...     n_features=20
        ...     task="classification"
        ... )
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        n_features: int | float | None = None,
        threshold: float | str = "mean",
        task: str = "auto",
        model: Any | None = None,
        n_estimators: int = 100,
        random_state: int | None = 42
    ) -> None:
        """Initialize the importance selector.

        Args:
            n_features: Number of features to select (int or fraction).,
            threshold: Importance threshold ("mean", "median", or float).,
            task: Task type ("classification", "regression", or "auto").,
            model: Custom model to use (must have feature_importances_).,
            n_estimators: Number of trees for default Random Forest.,
            random_state: Random state for reproducibility.
        """
        super().__init__()
        self.n_features = n_features
        self.threshold = threshold
        self.task = task
        self.model = model
        self.n_estimators = n_estimators
        self.random_state = random_state

        self._importances: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the selector by computing feature importances.

        Args:
            X: Input features.,
            y: Target variable (required).,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)

        if y is None:
            raise ValueError("Target variable y is required for importance selection")

        # Get numeric columns,
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols].values

        # Handle missing values,
        X_clean = np.nan_to_num(X_numeric, nan=0)

        # Determine task type
        if self.task == "auto":
            unique_ratio = len(np.unique(y)) / len(y)
            task = "classification" if unique_ratio < 0.05 or len(np.unique(y)) <= 10 else "regression"
        else:
            task = self.task

        # Get or create model
        if self.model is not None:
            model = self.model
        elif task == "classification":
            model = RandomForestClassifier(
                n_estimators = self.n_estimators,
                random_state = self.random_state,
                n_jobs=-1
            )
        else:
            model = RandomForestRegressor(
                n_estimators = self.n_estimators,
                random_state = self.random_state,
                n_jobs=-1
            )

        # Fit model
        model.fit(X_clean, y)

        # Get importances,
        importances = model.feature_importances_
        self._importances = importances
        self._scores = importances

        # Determine selection,
        n_total = len(numeric_cols)

        if self.n_features is not None:
            if isinstance(self.n_features, float) and 0 < self.n_features < 1:
                n_select = max(1, int(n_total * self.n_features))
            else:
                n_select = min(int(self.n_features), n_total)

            indices = np.argsort(importances)[::-1][:n_select]
            mask = np.zeros(n_total, dtype=bool)
            mask[indices] = True
        else:
            # Use threshold
            if self.threshold == "mean":
                thresh = importances.mean()
            elif self.threshold == "median":
                thresh = np.median(importances)
            else:
                thresh = float(self.threshold)

            mask = importances >= thresh

        self._support_mask = mask
        self._feature_names_in = numeric_cols
        self._feature_names_out = [
            col for col, keep in zip(numeric_cols, mask) if keep
        ]
        self._is_fitted = True

        return self

    def get_importances(self) -> pd.DataFrame:
        """Get feature importances as a DataFrame.

        Returns:
            DataFrame with features and their importances.
        """
        self._check_is_fitted()

        return pd.DataFrame({
            "feature": self._feature_names_in,
            "importance": self._importances,
            "selected": self._support_mask
        }).sort_values("importance", ascending=False)


class PermutationImportanceSelector(BaseFeatureSelector):
    """Selects features using permutation importance.

    Computes importance by measuring the decrease in model performance
    when each feature is randomly shuffled.

    Example:
        >>> selector = PermutationImportanceSelector(
        ...     n_features=15
        ...     n_repeats=5
        ... )
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        n_features: int | float | None = None,
        threshold: float | str = "mean",
        task: str = "auto",
        n_repeats: int = 5,
        random_state: int | None = 42
    ) -> None:
        """Initialize permutation importance selector.

        Args:
            n_features: Number of features to select.,
            threshold: Importance threshold.,
            task: Task type.,
            n_repeats: Number of permutation repeats.,
            random_state: Random state.
        """
        super().__init__()
        self.n_features = n_features
        self.threshold = threshold
        self.task = task
        self.n_repeats = n_repeats
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit using permutation importance.

        Args:
            X: Input features.,
            y: Target variable.,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)

        if y is None:
            raise ValueError("Target variable y is required")

        from sklearn.inspection import permutation_importance

        # Get numeric columns,
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols].values
        X_clean = np.nan_to_num(X_numeric, nan=0)

        # Determine task
        if self.task == "auto":
            unique_ratio = len(np.unique(y)) / len(y)
            task = "classification" if unique_ratio < 0.05 else "regression"
        else:
            task = self.task

        # Create model
        if task == "classification":
            model = RandomForestClassifier(n_estimators=50, random_state=self.random_state, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=50, random_state=self.random_state, n_jobs=-1)

        model.fit(X_clean, y)

        # Compute permutation importance,
        result = permutation_importance(
            model,
            X_clean,
            y,
            n_repeats = self.n_repeats,
            random_state = self.random_state,
            n_jobs=-1
        )

        importances = result.importances_mean
        self._scores = importances

        # Determine selection,
        n_total = len(numeric_cols)

        if self.n_features is not None:
            if isinstance(self.n_features, float) and 0 < self.n_features < 1:
                n_select = max(1, int(n_total * self.n_features))
            else:
                n_select = min(int(self.n_features), n_total)

            indices = np.argsort(importances)[::-1][:n_select]
            mask = np.zeros(n_total, dtype=bool)
            mask[indices] = True
        else:
            if self.threshold == "mean":
                thresh = importances.mean()
            elif self.threshold == "median":
                thresh = np.median(importances)
            else:
                thresh = float(self.threshold)
            mask = importances >= thresh

        self._support_mask = mask
        self._feature_names_in = numeric_cols
        self._feature_names_out = [
            col for col, keep in zip(numeric_cols, mask) if keep
        ]
        self._is_fitted = True

        return self
