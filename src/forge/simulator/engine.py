"""Feature impact simulation engine.

Estimates the impact of adding or removing features on model performance
using ablation studies, permutation importance, and surrogate models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from sklearn.base import BaseEstimator

if TYPE_CHECKING:
    import pandas as pd
    from typing_extensions import Self

logger = logging.getLogger(__name__)


@dataclass
class ImpactReport:
    """Report of feature impact simulation.

    Attributes:
    ----------
    feature : str
        Feature name evaluated.
    action : str
        'add' or 'remove'.
    baseline_score : float
        Model score with all features.
    simulated_score : float
        Estimated score after action.
    delta : float
        Score change (simulated - baseline).
    relative_delta : float
        Relative change as fraction of baseline.
    confidence : float
        Confidence in the estimate (0-1).
    method : str
        Method used for simulation.
    """

    feature: str
    action: str
    baseline_score: float
    simulated_score: float
    delta: float
    relative_delta: float
    confidence: float
    method: str

    def __str__(self) -> str:
        """Human-readable impact summary."""
        direction = "+" if self.delta >= 0 else ""
        return (
            f"{self.action.upper()} '{self.feature}': "
            f"{direction}{self.delta:.4f} ({direction}{self.relative_delta:.2%}) "
            f"[confidence: {self.confidence:.2f}]"
        )


class FeatureImpactSimulator(BaseEstimator):
    """Simulate the impact of feature changes on model performance.

    Uses ablation studies, permutation importance, and lightweight
    surrogate models to estimate performance changes without
    full retraining.

    Parameters
    ----------
    model : Any
        A fitted sklearn-compatible estimator.
    scoring : str
        Scoring metric name compatible with sklearn (e.g., 'accuracy',
        'r2', 'neg_mean_squared_error').
    method : str
        Simulation method: 'permutation', 'ablation', or 'surrogate'.
    n_repeats : int
        Number of repetitions for permutation-based methods.
    cv_folds : int
        Cross-validation folds for surrogate method.
    random_state : int | None
        Random seed.

    Attributes:
    ----------
    baseline_score_ : float
        Baseline model performance computed during fit.
    feature_impacts_ : dict[str, ImpactReport]
        Impact reports for each analyzed feature.
    feature_names_in_ : list[str]
        Feature names from the fitted dataset.

    Examples:
    --------
    >>> from forge.simulator import FeatureImpactSimulator
    >>> from sklearn.ensemble import RandomForestClassifier
    >>>
    >>> model = RandomForestClassifier().fit(X_train, y_train)
    >>> sim = FeatureImpactSimulator(model=model, scoring='accuracy')
    >>> sim.fit(X_val, y_val)
    >>>
    >>> # What if we remove a feature?
    >>> report = sim.simulate_remove("feature_a")
    >>> print(report)
    >>>
    >>> # What if we add a new feature?
    >>> report = sim.simulate_add("new_feature", X_val_with_new)
    """

    def __init__(
        self,
        model: Any = None,
        scoring: str = "accuracy",
        method: Literal["permutation", "ablation", "surrogate"] = "permutation",
        n_repeats: int = 5,
        cv_folds: int = 3,
        random_state: int | None = None,
    ) -> None:
        self.model = model
        self.scoring = scoring
        self.method = method
        self.n_repeats = n_repeats
        self.cv_folds = cv_folds
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        """Compute baseline score and prepare for simulation.

        Parameters
        ----------
        X : pd.DataFrame
            Validation data (features).
        y : Any
            Target variable.

        Returns:
        -------
        Self
            Fitted simulator.
        """
        import pandas as pd

        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"Expected pd.DataFrame, got {type(X).__name__}")
        if y is None:
            raise ValueError("Target y is required for impact simulation.")
        if self.model is None:
            raise ValueError("A fitted model is required.")

        self.feature_names_in_ = list(X.columns)
        self._X = X.copy()
        self._y = np.asarray(y)

        self.baseline_score_ = self._score_model(X, self._y)
        self.feature_impacts_: dict[str, ImpactReport] = {}

        self._is_fitted = True
        return self

    def simulate_remove(self, feature: str) -> ImpactReport:
        """Simulate removing a feature.

        Parameters
        ----------
        feature : str
            Feature to remove.

        Returns:
        -------
        ImpactReport
            Estimated impact of removal.

        Raises:
        ------
        ValueError
            If feature not found or simulator not fitted.
        """
        self._check_fitted()

        if feature not in self.feature_names_in_:
            raise ValueError(
                f"Feature '{feature}' not found. "
                f"Available: {self.feature_names_in_}"
            )

        if self.method == "permutation":
            report = self._permutation_remove(feature)
        elif self.method == "ablation":
            report = self._ablation_remove(feature)
        else:
            report = self._surrogate_remove(feature)

        self.feature_impacts_[f"remove_{feature}"] = report
        return report

    def simulate_add(
        self, feature_name: str, X_with_feature: pd.DataFrame
    ) -> ImpactReport:
        """Simulate adding a new feature.

        Uses a surrogate model to estimate performance gain from adding
        a feature. Trains a lightweight model with and without the feature
        and compares performance.

        Parameters
        ----------
        feature_name : str
            Name of the new feature column.
        X_with_feature : pd.DataFrame
            Data including both original features and the new feature.

        Returns:
        -------
        ImpactReport
            Estimated impact of adding the feature.
        """
        self._check_fitted()

        if feature_name not in X_with_feature.columns:
            raise ValueError(
                f"Feature '{feature_name}' not found in provided DataFrame."
            )

        from sklearn.model_selection import cross_val_score
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        is_classifier = hasattr(self.model, "predict_proba") or hasattr(
            self.model, "classes_"
        )
        surrogate = (
            DecisionTreeClassifier(max_depth=5, random_state=self.random_state)
            if is_classifier
            else DecisionTreeRegressor(max_depth=5, random_state=self.random_state)
        )

        # Score without new feature
        X_base = X_with_feature[self.feature_names_in_]
        scores_without = cross_val_score(
            surrogate, X_base, self._y,
            scoring=self.scoring, cv=self.cv_folds,
        )

        # Score with new feature
        all_cols = self.feature_names_in_ + [feature_name]
        X_extended = X_with_feature[all_cols]
        scores_with = cross_val_score(
            surrogate, X_extended, self._y,
            scoring=self.scoring, cv=self.cv_folds,
        )

        mean_without = float(np.mean(scores_without))
        mean_with = float(np.mean(scores_with))
        delta = mean_with - mean_without

        # Confidence based on score variability
        std_combined = float(np.std(np.concatenate([scores_without, scores_with])))
        confidence = max(0.0, min(1.0, 1.0 - std_combined))

        baseline = self.baseline_score_
        relative = delta / abs(baseline) if abs(baseline) > 1e-10 else 0.0

        report = ImpactReport(
            feature=feature_name,
            action="add",
            baseline_score=baseline,
            simulated_score=baseline + delta,
            delta=delta,
            relative_delta=relative,
            confidence=confidence,
            method="surrogate",
        )

        self.feature_impacts_[f"add_{feature_name}"] = report
        return report

    def what_if(
        self,
        add: list[str] | None = None,
        remove: list[str] | None = None,
        X_extended: pd.DataFrame | None = None,
    ) -> list[ImpactReport]:
        """Batch what-if analysis for multiple features.

        Parameters
        ----------
        add : list[str] | None
            Features to add (requires X_extended).
        remove : list[str] | None
            Features to remove.
        X_extended : pd.DataFrame | None
            Data with additional feature columns (required if add is set).

        Returns:
        -------
        list[ImpactReport]
            Impact reports for all simulated changes.
        """
        self._check_fitted()

        reports: list[ImpactReport] = []

        if remove:
            for feat in remove:
                if feat in self.feature_names_in_:
                    reports.append(self.simulate_remove(feat))

        if add and X_extended is not None:
            for feat in add:
                if feat in X_extended.columns:
                    reports.append(self.simulate_add(feat, X_extended))

        return reports

    def rank_features(self) -> list[ImpactReport]:
        """Rank all features by their removal impact.

        Returns:
        -------
        list[ImpactReport]
            Features ranked by impact (most impactful first).
        """
        self._check_fitted()

        reports: list[ImpactReport] = []
        for feat in self.feature_names_in_:
            reports.append(self.simulate_remove(feat))

        reports.sort(key=lambda r: abs(r.delta), reverse=True)
        return reports

    def _permutation_remove(self, feature: str) -> ImpactReport:
        """Estimate removal impact via permutation."""
        rng = np.random.RandomState(self.random_state)
        scores: list[float] = []

        for _ in range(self.n_repeats):
            X_perm = self._X.copy()
            X_perm[feature] = rng.permutation(X_perm[feature].values)
            scores.append(self._score_model(X_perm, self._y))

        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        delta = mean_score - self.baseline_score_
        confidence = max(0.0, min(1.0, 1.0 - std_score / (abs(delta) + 1e-10)))
        relative = delta / abs(self.baseline_score_) if abs(self.baseline_score_) > 1e-10 else 0.0

        return ImpactReport(
            feature=feature,
            action="remove",
            baseline_score=self.baseline_score_,
            simulated_score=mean_score,
            delta=delta,
            relative_delta=relative,
            confidence=confidence,
            method="permutation",
        )

    def _ablation_remove(self, feature: str) -> ImpactReport:
        """Estimate removal impact via zeroing the feature."""
        X_ablated = self._X.copy()
        X_ablated[feature] = 0.0

        score = self._score_model(X_ablated, self._y)
        delta = score - self.baseline_score_
        relative = delta / abs(self.baseline_score_) if abs(self.baseline_score_) > 1e-10 else 0.0

        return ImpactReport(
            feature=feature,
            action="remove",
            baseline_score=self.baseline_score_,
            simulated_score=score,
            delta=delta,
            relative_delta=relative,
            confidence=0.7,  # Lower confidence for ablation
            method="ablation",
        )

    def _surrogate_remove(self, feature: str) -> ImpactReport:
        """Estimate removal impact via surrogate model."""
        from sklearn.model_selection import cross_val_score
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        is_classifier = hasattr(self.model, "predict_proba") or hasattr(
            self.model, "classes_"
        )
        surrogate = (
            DecisionTreeClassifier(max_depth=5, random_state=self.random_state)
            if is_classifier
            else DecisionTreeRegressor(max_depth=5, random_state=self.random_state)
        )

        remaining = [c for c in self.feature_names_in_ if c != feature]

        scores_all = cross_val_score(
            surrogate, self._X[self.feature_names_in_], self._y,
            scoring=self.scoring, cv=self.cv_folds,
        )
        scores_without = cross_val_score(
            surrogate, self._X[remaining], self._y,
            scoring=self.scoring, cv=self.cv_folds,
        )

        mean_all = float(np.mean(scores_all))
        mean_without = float(np.mean(scores_without))
        delta = mean_without - mean_all

        std_combined = float(np.std(np.concatenate([scores_all, scores_without])))
        confidence = max(0.0, min(1.0, 1.0 - std_combined))

        baseline = self.baseline_score_
        relative = delta / abs(baseline) if abs(baseline) > 1e-10 else 0.0

        return ImpactReport(
            feature=feature,
            action="remove",
            baseline_score=baseline,
            simulated_score=baseline + delta,
            delta=delta,
            relative_delta=relative,
            confidence=confidence,
            method="surrogate",
        )

    def _score_model(self, X: pd.DataFrame, y: np.ndarray) -> float:
        """Score the model using the configured scoring metric."""
        from sklearn.metrics import get_scorer

        scorer = get_scorer(self.scoring)
        return float(scorer(self.model, X, y))

    def _check_fitted(self) -> None:
        """Check that simulator is fitted."""
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise RuntimeError("FeatureImpactSimulator must be fitted first.")

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all simulations.

        Returns:
        -------
        dict
            Summary of baseline and all impact reports.
        """
        return {
            "baseline_score": self.baseline_score_ if hasattr(self, "baseline_score_") else None,
            "scoring": self.scoring,
            "method": self.method,
            "simulations": {
                key: {
                    "feature": r.feature,
                    "action": r.action,
                    "delta": r.delta,
                    "relative_delta": r.relative_delta,
                    "confidence": r.confidence,
                }
                for key, r in self.feature_impacts_.items()
            },
        }
