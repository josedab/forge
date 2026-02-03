"""Tests for feature combination simulator and recommender."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from forge.simulator.engine import FeatureImpactSimulator
from forge.simulator.recommender import (
    CombinationResult,
    FeatureCombinationSimulator,
    FeatureRecommender,
    Recommendation,
)


@pytest.fixture
def classification_data() -> tuple[pd.DataFrame, pd.Series, RandomForestClassifier]:
    X_arr, y_arr = make_classification(
        n_samples=200, n_features=6, n_informative=3,
        n_redundant=1, n_classes=2, random_state=42,
    )
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(6)])
    y = pd.Series(y_arr, name="target")
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X, y)
    return X, y, model


@pytest.fixture
def fitted_simulator(
    classification_data: tuple[pd.DataFrame, pd.Series, RandomForestClassifier],
) -> FeatureImpactSimulator:
    X, y, model = classification_data
    sim = FeatureImpactSimulator(model=model, scoring="accuracy", method="permutation", n_repeats=2)
    sim.fit(X, y)
    return sim


class TestCombinationResult:
    def test_str(self) -> None:
        r = CombinationResult(
            features=["f1", "f2"],
            action="remove",
            baseline_score=0.9,
            estimated_score=0.88,
            estimated_delta=-0.02,
            interaction_effect=-0.005,
            confidence=0.7,
        )
        s = str(r)
        assert "remove" in s.lower()
        assert "f1" in s
        assert "f2" in s

    def test_defaults(self) -> None:
        r = CombinationResult(
            features=["a"],
            action="add",
            baseline_score=0.5,
            estimated_score=0.6,
            estimated_delta=0.1,
        )
        assert r.confidence == 0.5
        assert r.interaction_effect == 0.0


class TestRecommendation:
    def test_fields(self) -> None:
        r = Recommendation(
            action="remove",
            features=["f1"],
            estimated_delta=0.02,
            reason="Improves score",
            confidence=0.8,
            priority=1,
        )
        assert r.priority == 1
        assert r.action == "remove"


class TestFeatureCombinationSimulator:
    def test_single_feature_removal(self, fitted_simulator: FeatureImpactSimulator) -> None:
        combo_sim = FeatureCombinationSimulator(fitted_simulator, max_combination_size=2)
        results = combo_sim.simulate_remove_combinations(["f0", "f1"], combination_sizes=[1])
        assert len(results) == 2
        assert all(isinstance(r, CombinationResult) for r in results)
        assert all(r.action == "remove" for r in results)

    def test_multi_feature_removal(self, fitted_simulator: FeatureImpactSimulator) -> None:
        combo_sim = FeatureCombinationSimulator(fitted_simulator, max_combination_size=2)
        results = combo_sim.simulate_remove_combinations(["f0", "f1", "f2"])
        # Sizes 1 and 2: C(3,1) + C(3,2) = 3 + 3 = 6
        assert len(results) == 6
        # Results are sorted
        for i in range(len(results) - 1):
            assert results[i].estimated_delta >= results[i + 1].estimated_delta

    def test_combination_sizes(self, fitted_simulator: FeatureImpactSimulator) -> None:
        combo_sim = FeatureCombinationSimulator(fitted_simulator, max_combination_size=3)
        results = combo_sim.simulate_remove_combinations(["f0", "f1"], combination_sizes=[2])
        assert len(results) == 1  # Only C(2,2)=1 combination
        assert len(results[0].features) == 2

    def test_interaction_effect(self, fitted_simulator: FeatureImpactSimulator) -> None:
        combo_sim = FeatureCombinationSimulator(fitted_simulator)
        results = combo_sim.simulate_remove_combinations(["f0", "f1"], combination_sizes=[2])
        # Multi-feature combinations have non-zero interaction
        for r in results:
            if len(r.features) > 1:
                assert r.interaction_effect != 0.0 or r.estimated_delta == 0.0

    def test_confidence_decreases_with_size(self, fitted_simulator: FeatureImpactSimulator) -> None:
        combo_sim = FeatureCombinationSimulator(fitted_simulator, max_combination_size=3)
        results = combo_sim.simulate_remove_combinations(["f0", "f1", "f2"])
        singles = [r for r in results if len(r.features) == 1]
        pairs = [r for r in results if len(r.features) == 2]
        if singles and pairs:
            avg_single_conf = np.mean([r.confidence for r in singles])
            avg_pair_conf = np.mean([r.confidence for r in pairs])
            assert avg_single_conf >= avg_pair_conf


class TestFeatureRecommender:
    def test_fit(
        self,
        classification_data: tuple[pd.DataFrame, pd.Series, RandomForestClassifier],
    ) -> None:
        X, y, model = classification_data
        rec = FeatureRecommender(model=model, scoring="accuracy", method="permutation", n_repeats=2)
        rec.fit(X, y)
        assert hasattr(rec, "baseline_score_")
        assert hasattr(rec, "feature_names_in_")
        assert len(rec.feature_names_in_) == 6

    def test_recommend(
        self,
        classification_data: tuple[pd.DataFrame, pd.Series, RandomForestClassifier],
    ) -> None:
        X, y, model = classification_data
        rec = FeatureRecommender(model=model, scoring="accuracy", method="permutation", n_repeats=2, top_n=3)
        rec.fit(X, y)
        recs = rec.recommend()
        assert len(recs) <= 3
        for r in recs:
            assert isinstance(r, Recommendation)
            assert r.priority > 0

    def test_recommend_with_candidates(
        self,
        classification_data: tuple[pd.DataFrame, pd.Series, RandomForestClassifier],
    ) -> None:
        X, y, model = classification_data
        rec = FeatureRecommender(model=model, scoring="accuracy", method="permutation", n_repeats=2, top_n=10)
        rec.fit(X, y)

        rng = np.random.default_rng(42)
        X_candidates = X.copy()
        X_candidates["new_feat"] = rng.standard_normal(len(X))

        recs = rec.recommend(X_candidates=X_candidates)
        assert isinstance(recs, list)

    def test_get_feature_ranking(
        self,
        classification_data: tuple[pd.DataFrame, pd.Series, RandomForestClassifier],
    ) -> None:
        X, y, model = classification_data
        rec = FeatureRecommender(model=model, scoring="accuracy", method="permutation", n_repeats=2)
        rec.fit(X, y)
        ranking = rec.get_feature_ranking()
        assert len(ranking) == 6
        assert all("feature" in r for r in ranking)
        assert all("importance" in r for r in ranking)

    def test_not_fitted_error(self) -> None:
        rec = FeatureRecommender()
        with pytest.raises(RuntimeError, match="must be fitted"):
            rec.recommend()

    def test_priorities_ordered(
        self,
        classification_data: tuple[pd.DataFrame, pd.Series, RandomForestClassifier],
    ) -> None:
        X, y, model = classification_data
        rec = FeatureRecommender(model=model, scoring="accuracy", method="permutation", n_repeats=2)
        rec.fit(X, y)
        recs = rec.recommend(top_n=10)
        for i, r in enumerate(recs):
            assert r.priority == i + 1
