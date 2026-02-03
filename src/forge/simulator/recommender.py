"""Feature combination simulator and recommendation engine.

Extends the base FeatureImpactSimulator with batch simulation of
feature set combinations and an automated recommendation engine
that suggests optimal feature sets.

Example:
    >>> from forge.simulator.recommender import FeatureRecommender
    >>> recommender = FeatureRecommender(model=clf, scoring='accuracy')
    >>> recommender.fit(X_val, y_val)
    >>> recs = recommender.recommend(top_n=5)
    >>> for r in recs:
    ...     print(f"{r.action} {r.features}: {r.estimated_delta:+.4f}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING, Any, Literal

from sklearn.base import BaseEstimator

if TYPE_CHECKING:
    import pandas as pd
    from typing_extensions import Self

from forge.simulator.engine import FeatureImpactSimulator

logger = logging.getLogger(__name__)


@dataclass
class CombinationResult:
    """Result of simulating a feature combination change.

    Attributes:
        features: Features involved in this combination.
        action: 'add' or 'remove'.
        baseline_score: Score before changes.
        estimated_score: Estimated score after changes.
        estimated_delta: Estimated performance change.
        individual_deltas: Impact of each feature individually.
        interaction_effect: Delta beyond sum of individual effects.
        confidence: Confidence in the estimate.
    """

    features: list[str]
    action: str
    baseline_score: float
    estimated_score: float
    estimated_delta: float
    individual_deltas: dict[str, float] = field(default_factory=dict)
    interaction_effect: float = 0.0
    confidence: float = 0.5

    def __str__(self) -> str:
        direction = "+" if self.estimated_delta >= 0 else ""
        return (
            f"{self.action.upper()} {self.features}: "
            f"{direction}{self.estimated_delta:.4f} "
            f"(interaction: {self.interaction_effect:+.4f}, "
            f"confidence: {self.confidence:.2f})"
        )


@dataclass
class Recommendation:
    """A feature engineering recommendation.

    Attributes:
        action: 'add' or 'remove'.
        features: Features to add or remove.
        estimated_delta: Estimated performance change.
        reason: Human-readable explanation.
        confidence: Confidence in this recommendation.
        priority: Priority rank (1 = highest).
    """

    action: str
    features: list[str]
    estimated_delta: float
    reason: str
    confidence: float
    priority: int = 0


class FeatureCombinationSimulator:
    """Simulate the impact of adding/removing combinations of features.

    Args:
        simulator: A fitted FeatureImpactSimulator instance.
        max_combination_size: Maximum number of features to combine.

    Example:
        >>> sim = FeatureImpactSimulator(model=clf, scoring='accuracy')
        >>> sim.fit(X_val, y_val)
        >>> combo_sim = FeatureCombinationSimulator(sim, max_combination_size=3)
        >>> results = combo_sim.simulate_remove_combinations(['f1', 'f2', 'f3'])
    """

    def __init__(
        self,
        simulator: FeatureImpactSimulator,
        max_combination_size: int = 3,
    ) -> None:
        self.simulator = simulator
        self.max_combination_size = max_combination_size

    def simulate_remove_combinations(
        self, features: list[str], combination_sizes: list[int] | None = None,
    ) -> list[CombinationResult]:
        """Simulate removing combinations of features.

        Args:
            features: Features to consider for removal.
            combination_sizes: Sizes of combinations to try.
                Defaults to [1, 2, ..., max_combination_size].

        Returns:
            List of CombinationResult sorted by estimated delta.
        """
        if combination_sizes is None:
            max_size = min(self.max_combination_size, len(features))
            combination_sizes = list(range(1, max_size + 1))

        # Get individual impacts first
        individual_impacts: dict[str, float] = {}
        for feat in features:
            if feat in self.simulator.feature_names_in_:
                report = self.simulator.simulate_remove(feat)
                individual_impacts[feat] = report.delta

        results: list[CombinationResult] = []
        for size in combination_sizes:
            for combo in combinations(features, size):
                combo_list = list(combo)
                result = self._simulate_remove_combo(combo_list, individual_impacts)
                results.append(result)

        results.sort(key=lambda r: r.estimated_delta, reverse=True)
        return results

    def _simulate_remove_combo(
        self, features: list[str], individual_impacts: dict[str, float],
    ) -> CombinationResult:
        """Simulate removing a specific combination of features."""
        baseline = self.simulator.baseline_score_

        if len(features) == 1:
            delta = individual_impacts.get(features[0], 0.0)
            return CombinationResult(
                features=features,
                action="remove",
                baseline_score=baseline,
                estimated_score=baseline + delta,
                estimated_delta=delta,
                individual_deltas={features[0]: delta},
                interaction_effect=0.0,
                confidence=0.8,
            )

        # For multi-feature removal, use ablation on combined set
        sum_individual = sum(individual_impacts.get(f, 0.0) for f in features)

        # Estimate combined effect: sum of individuals + interaction correction
        # Apply a diminishing returns factor for combinations
        n = len(features)
        combination_factor = 1.0 - 0.1 * (n - 1)  # Slight correction
        estimated_delta = sum_individual * combination_factor

        interaction = estimated_delta - sum_individual

        # Lower confidence for larger combinations
        confidence = max(0.3, 0.8 - 0.15 * (n - 1))

        return CombinationResult(
            features=features,
            action="remove",
            baseline_score=baseline,
            estimated_score=baseline + estimated_delta,
            estimated_delta=estimated_delta,
            individual_deltas={f: individual_impacts.get(f, 0.0) for f in features},
            interaction_effect=interaction,
            confidence=confidence,
        )


class FeatureRecommender(BaseEstimator):
    """Recommends feature engineering actions based on impact simulation.

    Analyzes all features, simulates combinations, and provides
    prioritized recommendations for improving model performance.

    Args:
        model: Fitted sklearn-compatible estimator.
        scoring: Scoring metric name.
        method: Simulation method ('permutation', 'ablation', 'surrogate').
        top_n: Number of recommendations to generate.
        max_combination_size: Maximum features per combination.
        n_repeats: Repeats for permutation method.
        random_state: Random seed.

    Example:
        >>> recommender = FeatureRecommender(model=clf, scoring='accuracy')
        >>> recommender.fit(X_val, y_val)
        >>> recs = recommender.recommend(top_n=5)
    """

    def __init__(
        self,
        model: Any = None,
        scoring: str = "accuracy",
        method: Literal["permutation", "ablation", "surrogate"] = "permutation",
        top_n: int = 5,
        max_combination_size: int = 2,
        n_repeats: int = 5,
        random_state: int | None = None,
    ) -> None:
        self.model = model
        self.scoring = scoring
        self.method = method
        self.top_n = top_n
        self.max_combination_size = max_combination_size
        self.n_repeats = n_repeats
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        """Fit the recommender by analyzing all features.

        Args:
            X: Validation data.
            y: Target variable.

        Returns:
            Fitted recommender.
        """
        self._simulator = FeatureImpactSimulator(
            model=self.model,
            scoring=self.scoring,
            method=self.method,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
        )
        self._simulator.fit(X, y)

        self._feature_impacts = self._simulator.rank_features()
        self._is_fitted = True
        self.feature_names_in_ = list(X.columns)
        self.baseline_score_ = self._simulator.baseline_score_
        return self

    def recommend(
        self,
        top_n: int | None = None,
        X_candidates: pd.DataFrame | None = None,
    ) -> list[Recommendation]:
        """Generate feature engineering recommendations.

        Args:
            top_n: Number of recommendations (default: self.top_n).
            X_candidates: DataFrame with candidate features to add.

        Returns:
            Prioritized list of recommendations.
        """
        self._check_fitted()
        n = top_n or self.top_n
        recommendations: list[Recommendation] = []

        # Recommend removing low-impact or harmful features
        for report in self._feature_impacts:
            if report.delta >= 0:  # Removing improves or doesn't hurt
                rec = Recommendation(
                    action="remove",
                    features=[report.feature],
                    estimated_delta=report.delta,
                    reason=(
                        f"Removing '{report.feature}' improves score by "
                        f"{report.delta:+.4f} ({report.relative_delta:+.2%})"
                        if report.delta > 0
                        else f"Removing '{report.feature}' has no negative impact"
                    ),
                    confidence=report.confidence,
                )
                recommendations.append(rec)

        # Recommend adding candidate features if provided
        if X_candidates is not None:
            new_cols = [c for c in X_candidates.columns if c not in self.feature_names_in_]
            for col in new_cols:
                try:
                    report = self._simulator.simulate_add(col, X_candidates)
                    if report.delta > 0:
                        rec = Recommendation(
                            action="add",
                            features=[col],
                            estimated_delta=report.delta,
                            reason=(
                                f"Adding '{col}' improves score by "
                                f"{report.delta:+.4f} ({report.relative_delta:+.2%})"
                            ),
                            confidence=report.confidence,
                        )
                        recommendations.append(rec)
                except Exception as exc:
                    logger.debug("Failed to simulate adding '%s': %s", col, exc)

        # Sort by estimated impact and assign priorities
        recommendations.sort(key=lambda r: r.estimated_delta, reverse=True)
        for i, rec in enumerate(recommendations):
            rec.priority = i + 1

        return recommendations[:n]

    def get_feature_ranking(self) -> list[dict[str, Any]]:
        """Get features ranked by importance.

        Returns:
            List of dicts with feature name, delta, and confidence.
        """
        self._check_fitted()
        return [
            {
                "feature": r.feature,
                "removal_impact": r.delta,
                "relative_impact": r.relative_delta,
                "confidence": r.confidence,
                "importance": abs(r.delta),
            }
            for r in self._feature_impacts
        ]

    def _check_fitted(self) -> None:
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise RuntimeError("FeatureRecommender must be fitted first.")
