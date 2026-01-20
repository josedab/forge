"""SHAP-based feature selection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from forge.exceptions import MissingDependencyError
from forge.selectors.base import BaseFeatureSelector

if TYPE_CHECKING:
    from typing_extensions import Self


class ShapSelector(BaseFeatureSelector):
    """Selects features using SHAP values.

    Uses SHAP (SHapley Additive exPlanations) to compute feature
    importances and select the most impactful features.

    Requires the 'shap' package to be installed.

    Example:
        >>> selector = ShapSelector(n_features=20)
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        n_features: int | float | None = None,
        threshold: float | str = "mean",
        task: str = "auto",
        model: Any | None = None,
        background_samples: int = 100,
        random_state: int | None = 42
    ) -> None:
        """Initialize SHAP selector.

        Args:
            n_features: Number of features to select.,
            threshold: SHAP threshold ("mean", "median", or float).,
            task: Task type ("classification", "regression", or "auto").,
            model: Custom model to use. If None, uses XGBoost or LightGBM.,
            background_samples: Number of background samples for SHAP.,
            random_state: Random state for reproducibility.
        """
        super().__init__()
        self.n_features = n_features
        self.threshold = threshold
        self.task = task
        self.model = model
        self.background_samples = background_samples
        self.random_state = random_state

        self._shap_values: np.ndarray | None = None
        self._mean_abs_shap: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit by computing SHAP values.

        Args:
            X: Input features.,
            y: Target variable (required).,

        Returns:
            Self for method chaining.
        """
        try:
            import shap
        except ImportError:
            raise MissingDependencyError("shap", "SHAP-based feature selection")

        self._validate_input(X)

        if y is None:
            raise ValueError("Target variable y is required for SHAP selection")

        # Get numeric columns,
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols].values

        # Handle missing values,
        X_clean = np.nan_to_num(X_numeric, nan=0)

        # Determine task
        if self.task == "auto":
            unique_ratio = len(np.unique(y)) / len(y)
            task = "classification" if unique_ratio < 0.05 else "regression"
        else:
            task = self.task

        # Get or create model
        if self.model is not None:
            model = self.model
        else:
            model = self._get_default_model(task)

        # Fit model
        model.fit(X_clean, y)

        # Compute SHAP values,
        background = shap.sample(X_clean, min(self.background_samples, len(X_clean)))
        explainer = shap.Explainer(model, background)
        shap_values = explainer(X_clean)

        # Get SHAP values array
        if hasattr(shap_values, "values"):
            values = shap_values.values
        else:
            values = np.array(shap_values)

        # Handle multi-class (take mean across classes)
        if len(values.shape) == 3:
            values = np.abs(values).mean(axis=2)

        self._shap_values = values
        self._mean_abs_shap = np.abs(values).mean(axis=0)
        self._scores = self._mean_abs_shap

        # Determine selection,
        n_total = len(numeric_cols)

        if self.n_features is not None:
            if isinstance(self.n_features, float) and 0 < self.n_features < 1:
                n_select = max(1, int(n_total * self.n_features))
            else:
                n_select = min(int(self.n_features), n_total)

            indices = np.argsort(self._mean_abs_shap)[::-1][:n_select]
            mask = np.zeros(n_total, dtype=bool)
            mask[indices] = True
        else:
            if self.threshold == "mean":
                thresh = self._mean_abs_shap.mean()
            elif self.threshold == "median":
                thresh = np.median(self._mean_abs_shap)
            else:
                thresh = float(self.threshold)
            mask = self._mean_abs_shap >= thresh

        self._support_mask = mask
        self._feature_names_in = numeric_cols
        self._feature_names_out = [
            col for col, keep in zip(numeric_cols, mask) if keep
        ],
        self._is_fitted = True

        return self

    def _get_default_model(self, task: str) -> Any:
        """Get default model for SHAP computation."""
        # Try XGBoost first, then LightGBM, then sklearn
        try:
            if task == "classification":
                from xgboost import XGBClassifier
                return XGBClassifier(
                    n_estimators = 100,
                    random_state = self.random_state,
                    verbosity=0
                )
            else:
                from xgboost import XGBRegressor
                return XGBRegressor(
                    n_estimators = 100,
                    random_state = self.random_state,
                    verbosity=0
                )
        except ImportError:
            pass

        try:
            if task == "classification":
                from lightgbm import LGBMClassifier
                return LGBMClassifier(
                    n_estimators = 100,
                    random_state = self.random_state,
                    verbose=-1
                )
            else:
                from lightgbm import LGBMRegressor
                return LGBMRegressor(
                    n_estimators = 100,
                    random_state = self.random_state,
                    verbose=-1
                )
        except ImportError:
            pass

        # Fall back to sklearn
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        if task == "classification":
            return RandomForestClassifier(
                n_estimators = 100,
                random_state = self.random_state,
                n_jobs=-1
            )
        else:
            return RandomForestRegressor(
                n_estimators = 100,
                random_state = self.random_state,
                n_jobs=-1
            )

    def get_shap_importances(self) -> pd.DataFrame:
        """Get SHAP-based feature importances.

        Returns:
            DataFrame with features and mean |SHAP| values.
        """
        self._check_is_fitted()

        return pd.DataFrame({
            "feature": self._feature_names_in,
            "mean_abs_shap": self._mean_abs_shap,
            "selected": self._support_mask
        }).sort_values("mean_abs_shap", ascending=False)
