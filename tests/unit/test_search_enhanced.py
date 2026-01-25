"""Tests for Bayesian and multi-fidelity search strategies."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from forge.search.bayesian import BayesianFeatureSearch, MultiFidelitySearch
from forge.search.grammar import TransformNode, TransformOp
from forge.search.pareto import (
    EarlyStoppingMonitor,
    ParetoPoint,
    pareto_optimal_sets,
    select_knee_point,
)
from forge.search.search import AutoFeatureSearch, SearchResult


@pytest.fixture
def classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create a small classification dataset."""
    X_arr, y_arr = make_classification(
        n_samples=100, n_features=5, n_informative=3,
        n_redundant=1, random_state=42,
    )
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(5)])
    y = pd.Series(y_arr, name="target")
    return X, y


class TestBayesianFeatureSearch:
    """Tests for BayesianFeatureSearch."""

    def test_fit_returns_self(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = classification_data
        search = BayesianFeatureSearch(
            n_initial=10, n_iterations=3, batch_size=2, random_state=42
        )
        result = search.fit(X, y)
        assert result is search

    def test_produces_results(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = classification_data
        search = BayesianFeatureSearch(
            n_initial=10, n_iterations=3, batch_size=2, random_state=42
        )
        search.fit(X, y)
        assert len(search.results_) > 0
        assert len(search.best_features_) > 0

    def test_results_are_sorted(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = classification_data
        search = BayesianFeatureSearch(
            n_initial=15, n_iterations=3, batch_size=3, random_state=42
        )
        search.fit(X, y)
        scores = [r.score for r in search.results_]
        assert scores == sorted(scores, reverse=True)

    def test_history_tracked(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = classification_data
        search = BayesianFeatureSearch(
            n_initial=10, n_iterations=5, batch_size=2, random_state=42
        )
        search.fit(X, y)
        assert len(search.history_) == 5
        assert "best_score" in search.history_[0]

    def test_search_result_has_name(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = classification_data
        search = BayesianFeatureSearch(
            n_initial=10, n_iterations=2, batch_size=2, random_state=42
        )
        search.fit(X, y)
        for r in search.best_features_:
            assert isinstance(r.name, str)
            assert len(r.name) > 0


class TestMultiFidelitySearch:
    """Tests for MultiFidelitySearch."""

    def test_fit_returns_self(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = classification_data
        search = MultiFidelitySearch(
            n_candidates=20, halving_rounds=2, random_state=42
        )
        result = search.fit(X, y)
        assert result is search

    def test_produces_results(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = classification_data
        search = MultiFidelitySearch(
            n_candidates=20, halving_rounds=2, random_state=42
        )
        search.fit(X, y)
        assert len(search.results_) > 0

    def test_successive_halving(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = classification_data
        search = MultiFidelitySearch(
            n_candidates=30, halving_rounds=3, min_fidelity=0.3, random_state=42
        )
        search.fit(X, y)
        # After 3 rounds of halving 30 candidates, we should have a small set
        assert len(search.best_features_) <= 30
        assert search._is_fitted

    def test_results_sorted(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = classification_data
        search = MultiFidelitySearch(
            n_candidates=20, halving_rounds=2, random_state=42
        )
        search.fit(X, y)
        scores = [r.score for r in search.results_]
        assert scores == sorted(scores, reverse=True)


class TestAutoFeatureSearchSklearn:
    """Tests for sklearn compatibility of AutoFeatureSearch."""

    def test_random_search_sklearn(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = classification_data
        search = AutoFeatureSearch(
            method="random", n_features=5, n_candidates=20, random_state=42
        )
        X_out = search.fit_transform(X, y)
        assert isinstance(X_out, pd.DataFrame)
        assert X_out.shape[0] == len(X)

    def test_get_feature_names_out(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = classification_data
        search = AutoFeatureSearch(
            method="random", n_features=5, n_candidates=20, random_state=42
        )
        search.fit(X, y)
        names = search.get_feature_names_out()
        assert isinstance(names, list)

    def test_requires_y(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, _ = classification_data
        search = AutoFeatureSearch()
        with pytest.raises(Exception):
            search.fit(X)


class TestParetoOptimalSets:
    """Tests for Pareto-optimal feature set selection."""

    @pytest.fixture
    def sample_results(self) -> list[SearchResult]:
        return [
            SearchResult(
                program=[TransformNode(op=TransformOp.LOG, columns=["a"])],
                name=f"feat_{i}", score=0.9 - i * 0.03, generation=0,
            )
            for i in range(10)
        ]

    def test_empty_results(self) -> None:
        assert pareto_optimal_sets([]) == []

    def test_returns_pareto_points(self, sample_results: list[SearchResult]) -> None:
        frontier = pareto_optimal_sets(sample_results)
        assert len(frontier) > 0
        assert all(isinstance(p, ParetoPoint) for p in frontier)

    def test_sorted_by_n_features(self, sample_results: list[SearchResult]) -> None:
        frontier = pareto_optimal_sets(sample_results)
        ns = [p.n_features for p in frontier]
        assert ns == sorted(ns)

    def test_max_features_cap(self, sample_results: list[SearchResult]) -> None:
        frontier = pareto_optimal_sets(sample_results, max_features=3)
        for p in frontier:
            assert p.n_features <= 3


class TestSelectKneePoint:
    """Tests for knee-point detection."""

    def test_empty(self) -> None:
        assert select_knee_point([]) is None

    def test_single(self) -> None:
        p = ParetoPoint(n_features=5, score=0.9)
        assert select_knee_point([p]) is p

    def test_detects_knee(self) -> None:
        points = [
            ParetoPoint(n_features=1, score=0.60),
            ParetoPoint(n_features=3, score=0.88),
            ParetoPoint(n_features=5, score=0.90),
            ParetoPoint(n_features=10, score=0.91),
        ]
        knee = select_knee_point(points)
        assert knee is not None
        assert knee.n_features <= 5


class TestEarlyStoppingMonitor:
    """Tests for early stopping."""

    def test_no_stop_improving(self) -> None:
        m = EarlyStoppingMonitor(patience=3)
        assert not m.should_stop(0.80)
        assert not m.should_stop(0.85)
        assert not m.should_stop(0.90)

    def test_stop_after_patience(self) -> None:
        m = EarlyStoppingMonitor(patience=2, min_delta=0.01)
        m.should_stop(0.90)
        m.should_stop(0.90)
        assert m.should_stop(0.90)

    def test_tracks_best_score(self) -> None:
        m = EarlyStoppingMonitor(patience=5)
        m.should_stop(0.80)
        m.should_stop(0.90)
        m.should_stop(0.85)
        assert m.best_score == 0.90


class TestAutoFeatureSearchAllMethods:
    """Tests for extended AutoFeatureSearch methods."""

    def test_bayesian_method(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = classification_data
        search = AutoFeatureSearch(
            method="bayesian", n_features=5, n_iterations=3, random_state=42
        )
        result = search.fit_transform(X, y)
        assert isinstance(result, pd.DataFrame)

    def test_multi_fidelity_method(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = classification_data
        search = AutoFeatureSearch(
            method="multi_fidelity", n_features=5, n_candidates=20, random_state=42
        )
        result = search.fit_transform(X, y)
        assert isinstance(result, pd.DataFrame)

    def test_pareto_select(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = classification_data
        search = AutoFeatureSearch(
            method="random", n_features=10, n_candidates=30,
            pareto_select=True, random_state=42,
        )
        result = search.fit_transform(X, y)
        assert isinstance(result, pd.DataFrame)
        assert hasattr(search, "pareto_frontier_")

    def test_early_stopping(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = classification_data
        search = AutoFeatureSearch(
            method="evolutionary", n_features=5, population_size=10,
            n_generations=50, early_stopping_patience=3, random_state=42,
        )
        result = search.fit_transform(X, y)
        assert isinstance(result, pd.DataFrame)

    def test_invalid_method(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = classification_data
        search = AutoFeatureSearch(method="invalid")
        with pytest.raises(Exception):
            search.fit(X, y)
