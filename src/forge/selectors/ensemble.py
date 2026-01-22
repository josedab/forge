"""Ensemble feature importance and selection."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import (
    LassoCV,
    RidgeCV,
    LogisticRegressionCV,
)
from sklearn.preprocessing import StandardScaler

from forge.exceptions import NotFittedError, ValidationError
from forge.selectors.base import BaseFeatureSelector

if TYPE_CHECKING:
    from typing_extensions import Self
    from numpy.typing import NDArray


@dataclass
class ImportanceMethod:
    """Configuration for an importance method."""

    name: str
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)


class EnsembleImportanceSelector(BaseFeatureSelector):
    """Feature selection using ensemble of importance methods.

    Combines multiple importance methods (tree-based, permutation,
    linear, SHAP) with configurable weights to produce robust
    importance scores.

    Example:
        >>> selector = EnsembleImportanceSelector(
        ...     methods=["tree", "permutation", "linear"],
        ...     n_features=20,
        ... )
        >>> X_selected = selector.fit_transform(X, y)
    """

    AVAILABLE_METHODS = [
        "tree", "permutation", "linear", "shap", "mutual_info", "gradient_boost"
    ]

    def __init__(
        self,
        methods: list[str | ImportanceMethod] | None = None,
        n_features: int | float | None = None,
        threshold: float | str = "mean",
        aggregation: Literal["mean", "median", "rank", "vote"] = "rank",
        task: str = "auto",
        normalize: bool = True,
        n_jobs: int = -1,
        random_state: int | None = 42,
        verbose: int = 0,
    ) -> None:
        """Initialize ensemble importance selector.

        Args:
            methods: List of importance methods or ImportanceMethod objects.
                Available: "tree", "permutation", "linear", "shap",
                "mutual_info", "gradient_boost".
            n_features: Number of features to select.
            threshold: Selection threshold ("mean", "median", or float).
            aggregation: How to combine importance scores.
            task: Task type ("classification", "regression", "auto").
            normalize: Whether to normalize scores before combining.
            n_jobs: Number of parallel jobs.
            random_state: Random seed.
            verbose: Verbosity level.
        """
        super().__init__()
        self.methods = methods or ["tree", "permutation", "linear"]
        self.n_features = n_features
        self.threshold = threshold
        self.aggregation = aggregation
        self.task = task
        self.normalize = normalize
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        self._method_importances: dict[str, np.ndarray] = {}
        self._method_weights: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the ensemble selector.

        Args:
            X: Input features.
            y: Target variable (required).

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)
        if y is None:
            raise ValidationError("Target variable y is required")

        # Get numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValidationError("No numeric columns found")

        X_numeric = X[numeric_cols].copy()
        self._feature_names_in = numeric_cols

        # Handle missing values
        X_clean = X_numeric.fillna(0)

        # Determine task
        if self.task == "auto":
            unique_vals = len(np.unique(y))
            self._task = "classification" if unique_vals <= 20 else "regression"
        else:
            self._task = self.task

        # Parse methods
        method_configs = self._parse_methods()

        # Compute importance for each method
        for config in method_configs:
            if self.verbose > 0:
                print(f"Computing {config.name} importance...")

            try:
                importances = self._compute_importance(X_clean, y, config)
                self._method_importances[config.name] = importances
                self._method_weights[config.name] = config.weight
            except Exception as e:
                if self.verbose > 0:
                    print(f"Warning: {config.name} failed: {e}")

        if not self._method_importances:
            raise ValidationError("All importance methods failed")

        # Aggregate importances
        combined = self._aggregate_importances()
        self._scores = combined

        # Select features
        n_total = len(numeric_cols)
        if self.n_features is not None:
            if isinstance(self.n_features, float) and 0 < self.n_features < 1:
                n_select = max(1, int(n_total * self.n_features))
            else:
                n_select = min(int(self.n_features), n_total)

            indices = np.argsort(combined)[::-1][:n_select]
            mask = np.zeros(n_total, dtype=bool)
            mask[indices] = True
        else:
            if self.threshold == "mean":
                thresh = combined.mean()
            elif self.threshold == "median":
                thresh = np.median(combined)
            else:
                thresh = float(self.threshold)
            mask = combined >= thresh

        self._support_mask = mask
        self._finalize_fit(X_numeric)
        return self

    def _parse_methods(self) -> list[ImportanceMethod]:
        """Parse method specifications."""
        configs = []
        for method in self.methods:
            if isinstance(method, str):
                if method not in self.AVAILABLE_METHODS:
                    raise ValueError(f"Unknown method: {method}")
                configs.append(ImportanceMethod(name=method))
            elif isinstance(method, ImportanceMethod):
                configs.append(method)
            else:
                raise ValueError(f"Invalid method specification: {method}")
        return configs

    def _compute_importance(
        self, X: pd.DataFrame, y: pd.Series, config: ImportanceMethod
    ) -> np.ndarray:
        """Compute importance using specified method."""
        X_arr = X.values

        if config.name == "tree":
            return self._tree_importance(X_arr, y, config.params)
        elif config.name == "gradient_boost":
            return self._gradient_boost_importance(X_arr, y, config.params)
        elif config.name == "permutation":
            return self._permutation_importance(X_arr, y, config.params)
        elif config.name == "linear":
            return self._linear_importance(X_arr, y, config.params)
        elif config.name == "mutual_info":
            return self._mutual_info_importance(X_arr, y, config.params)
        elif config.name == "shap":
            return self._shap_importance(X_arr, y, config.params)
        else:
            raise ValueError(f"Unknown method: {config.name}")

    def _tree_importance(
        self, X: np.ndarray, y: pd.Series, params: dict[str, Any]
    ) -> np.ndarray:
        """Compute tree-based importance."""
        n_estimators = params.get("n_estimators", 100)

        if self._task == "classification":
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)

        return model.feature_importances_

    def _gradient_boost_importance(
        self, X: np.ndarray, y: pd.Series, params: dict[str, Any]
    ) -> np.ndarray:
        """Compute gradient boosting importance."""
        n_estimators = params.get("n_estimators", 100)

        if self._task == "classification":
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                random_state=self.random_state,
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                random_state=self.random_state,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)

        return model.feature_importances_

    def _permutation_importance(
        self, X: np.ndarray, y: pd.Series, params: dict[str, Any]
    ) -> np.ndarray:
        """Compute permutation importance."""
        from sklearn.inspection import permutation_importance

        n_repeats = params.get("n_repeats", 10)

        if self._task == "classification":
            model = RandomForestClassifier(
                n_estimators=50,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=50,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
            result = permutation_importance(
                model, X, y,
                n_repeats=n_repeats,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )

        return result.importances_mean

    def _linear_importance(
        self, X: np.ndarray, y: pd.Series, params: dict[str, Any]
    ) -> np.ndarray:
        """Compute linear model importance (coefficients)."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self._task == "classification":
                model = LogisticRegressionCV(
                    cv=5,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    max_iter=1000,
                )
                model.fit(X_scaled, y)
                # For multi-class, take mean absolute coefficient
                coef = np.abs(model.coef_)
                if coef.ndim > 1:
                    coef = coef.mean(axis=0)
            else:
                model = LassoCV(cv=5, random_state=self.random_state, n_jobs=self.n_jobs)
                model.fit(X_scaled, y)
                coef = np.abs(model.coef_)

        return coef

    def _mutual_info_importance(
        self, X: np.ndarray, y: pd.Series, params: dict[str, Any]
    ) -> np.ndarray:
        """Compute mutual information importance."""
        if self._task == "classification":
            from sklearn.feature_selection import mutual_info_classif
            mi = mutual_info_classif(X, y, random_state=self.random_state)
        else:
            from sklearn.feature_selection import mutual_info_regression
            mi = mutual_info_regression(X, y, random_state=self.random_state)

        return mi

    def _shap_importance(
        self, X: np.ndarray, y: pd.Series, params: dict[str, Any]
    ) -> np.ndarray:
        """Compute SHAP importance."""
        try:
            import shap
        except ImportError:
            raise ImportError("shap package required for SHAP importance")

        if self._task == "classification":
            model = RandomForestClassifier(
                n_estimators=50,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=50,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)

            # Use a sample for SHAP
            n_samples = min(100, X.shape[0])
            X_sample = X[:n_samples]

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            # Handle multi-class
            if isinstance(shap_values, list):
                shap_values = np.abs(np.array(shap_values)).mean(axis=0)

            return np.abs(shap_values).mean(axis=0)

    def _aggregate_importances(self) -> np.ndarray:
        """Aggregate importance scores from all methods."""
        n_features = len(self._feature_names_in)

        if self.aggregation == "rank":
            # Convert to ranks and average
            rank_scores = np.zeros(n_features)
            total_weight = 0

            for method, importances in self._method_importances.items():
                weight = self._method_weights[method]
                # Rank (higher importance = lower rank number)
                ranks = n_features - np.argsort(np.argsort(importances))
                if self.normalize:
                    ranks = ranks / n_features
                rank_scores += weight * ranks
                total_weight += weight

            return rank_scores / total_weight

        elif self.aggregation == "vote":
            # Each method votes for top features
            votes = np.zeros(n_features)
            n_top = max(1, n_features // 4)

            for method, importances in self._method_importances.items():
                weight = self._method_weights[method]
                top_indices = np.argsort(importances)[-n_top:]
                votes[top_indices] += weight

            return votes

        else:
            # Mean or median aggregation
            all_scores = []
            weights = []

            for method, importances in self._method_importances.items():
                weight = self._method_weights[method]
                if self.normalize:
                    # Min-max normalize
                    imp_min, imp_max = importances.min(), importances.max()
                    if imp_max > imp_min:
                        importances = (importances - imp_min) / (imp_max - imp_min)
                    else:
                        importances = np.ones_like(importances) * 0.5
                all_scores.append(importances)
                weights.append(weight)

            all_scores = np.array(all_scores)
            weights = np.array(weights)

            if self.aggregation == "mean":
                return np.average(all_scores, axis=0, weights=weights)
            else:  # median
                return np.median(all_scores, axis=0)

    def get_method_importances(self) -> pd.DataFrame:
        """Get importances from each method.

        Returns:
            DataFrame with importances from each method.
        """
        self._check_is_fitted()

        data = {"feature": self._feature_names_in}
        for method, importances in self._method_importances.items():
            data[method] = importances
        data["combined"] = self._scores
        data["selected"] = self._support_mask

        return pd.DataFrame(data)

    def get_method_agreement(self) -> pd.DataFrame:
        """Get agreement statistics between methods.

        Returns:
            DataFrame with pairwise correlations.
        """
        self._check_is_fitted()

        methods = list(self._method_importances.keys())
        n_methods = len(methods)

        correlations = np.zeros((n_methods, n_methods))
        for i, m1 in enumerate(methods):
            for j, m2 in enumerate(methods):
                correlations[i, j] = np.corrcoef(
                    self._method_importances[m1],
                    self._method_importances[m2]
                )[0, 1]

        return pd.DataFrame(correlations, index=methods, columns=methods)


class StabilitySelector(BaseFeatureSelector):
    """Feature selection with stability estimation.

    Uses subsampling to estimate feature selection stability,
    selecting features that are consistently important across
    random subsamples.

    Example:
        >>> selector = StabilitySelector(
        ...     n_iterations=100,
        ...     sample_fraction=0.8,
        ...     threshold=0.7,
        ... )
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        base_selector: BaseFeatureSelector | None = None,
        n_iterations: int = 100,
        sample_fraction: float = 0.8,
        threshold: float = 0.6,
        n_features: int | None = None,
        random_state: int | None = 42,
        n_jobs: int = 1,
        verbose: int = 0,
    ) -> None:
        """Initialize stability selector.

        Args:
            base_selector: Base feature selector to use.
            n_iterations: Number of subsampling iterations.
            sample_fraction: Fraction of samples per iteration.
            threshold: Minimum selection frequency threshold.
            n_features: Optional fixed number of features.
            random_state: Random seed.
            n_jobs: Parallel jobs.
            verbose: Verbosity level.
        """
        super().__init__()
        self.base_selector = base_selector
        self.n_iterations = n_iterations
        self.sample_fraction = sample_fraction
        self.threshold = threshold
        self.n_features = n_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        self._selection_frequencies: np.ndarray | None = None
        self._stability_scores: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit with stability estimation.

        Args:
            X: Input features.
            y: Target variable.

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)
        if y is None:
            raise ValidationError("Target variable y is required")

        # Setup
        rng = np.random.RandomState(self.random_state)
        n_samples = len(X)
        sample_size = int(n_samples * self.sample_fraction)

        self._feature_names_in = list(X.columns)
        n_features = len(self._feature_names_in)

        # Create base selector if not provided
        if self.base_selector is None:
            from forge.selectors.importance import ImportanceSelector
            base = ImportanceSelector(n_features=max(1, n_features // 2))
        else:
            base = self.base_selector

        # Track selection frequency
        selection_counts = np.zeros(n_features)

        for iteration in range(self.n_iterations):
            # Subsample
            indices = rng.choice(n_samples, size=sample_size, replace=False)
            X_sub = X.iloc[indices]
            y_sub = y.iloc[indices]

            # Fit selector
            try:
                selector = clone(base)
                selector.fit(X_sub, y_sub)
                selected_mask = selector.get_support(indices=False)

                # Map back to original features
                if hasattr(selector, "_feature_names_in"):
                    for i, col in enumerate(selector._feature_names_in):
                        if col in self._feature_names_in:
                            orig_idx = self._feature_names_in.index(col)
                            if i < len(selected_mask) and selected_mask[i]:
                                selection_counts[orig_idx] += 1
                else:
                    selection_counts[:len(selected_mask)] += selected_mask

            except Exception as e:
                if self.verbose > 0:
                    print(f"Iteration {iteration} failed: {e}")

            if self.verbose > 0 and (iteration + 1) % 10 == 0:
                print(f"Completed {iteration + 1}/{self.n_iterations} iterations")

        # Compute frequencies
        self._selection_frequencies = selection_counts / self.n_iterations
        self._scores = self._selection_frequencies

        # Select features
        if self.n_features is not None:
            n_select = min(self.n_features, n_features)
            indices = np.argsort(self._selection_frequencies)[::-1][:n_select]
            mask = np.zeros(n_features, dtype=bool)
            mask[indices] = True
        else:
            mask = self._selection_frequencies >= self.threshold

        self._support_mask = mask
        self._finalize_fit(X)
        return self

    def get_stability_scores(self) -> pd.DataFrame:
        """Get feature stability scores.

        Returns:
            DataFrame with selection frequencies.
        """
        self._check_is_fitted()

        return pd.DataFrame({
            "feature": self._feature_names_in,
            "selection_frequency": self._selection_frequencies,
            "selected": self._support_mask,
        }).sort_values("selection_frequency", ascending=False)


def ensemble_importance(
    X: pd.DataFrame,
    y: pd.Series,
    methods: list[str] | None = None,
    n_features: int | float | None = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, EnsembleImportanceSelector]:
    """Convenience function for ensemble importance selection.

    Args:
        X: Input features.
        y: Target variable.
        methods: Importance methods to use.
        n_features: Number of features to select.
        **kwargs: Additional arguments for selector.

    Returns:
        Tuple of (selected features, fitted selector).
    """
    selector = EnsembleImportanceSelector(
        methods=methods,
        n_features=n_features,
        **kwargs,
    )
    X_selected = selector.fit_transform(X, y)
    return X_selected, selector
