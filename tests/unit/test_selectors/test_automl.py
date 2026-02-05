"""Tests for AutoML feature selectors."""

import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from forge.exceptions import NotFittedError, ValidationError
from forge.selectors.automl import (
    BayesianFeatureSelector,
    GeneticFeatureSelector,
    SequentialFeatureSelector,
    auto_select_features,
)


@pytest.fixture
def classification_data():
    """Create classification dataset."""
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
    y = pd.Series(y)
    return X, y


@pytest.fixture
def regression_data():
    """Create regression dataset."""
    X, y = make_regression(
        n_samples=200,
        n_features=15,
        n_informative=8,
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(15)])
    y = pd.Series(y)
    return X, y


class TestBayesianFeatureSelector:
    """Tests for BayesianFeatureSelector."""

    def test_fit_transform(self, classification_data):
        """Test basic fit and transform."""
        X, y = classification_data
        selector = BayesianFeatureSelector(
            n_iterations=10,
            n_initial_random=5,
            random_state=42,
        )
        X_selected = selector.fit_transform(X, y)

        assert isinstance(X_selected, pd.DataFrame)
        assert len(X_selected) == len(X)
        assert len(X_selected.columns) <= len(X.columns)

    def test_requires_target(self, classification_data):
        """Test that target is required."""
        X, _ = classification_data
        selector = BayesianFeatureSelector()

        with pytest.raises(ValidationError):
            selector.fit(X)

    def test_get_support(self, classification_data):
        """Test get_support method."""
        X, y = classification_data
        selector = BayesianFeatureSelector(n_iterations=5, random_state=42)
        selector.fit(X, y)

        mask = selector.get_support(indices=False)
        assert len(mask) == len(X.columns)
        assert mask.dtype == bool

        indices = selector.get_support(indices=True)
        assert len(indices) == mask.sum()

    def test_optimization_history(self, classification_data):
        """Test optimization history tracking."""
        X, y = classification_data
        selector = BayesianFeatureSelector(n_iterations=10, random_state=42)
        selector.fit(X, y)

        history = selector.get_optimization_history()
        assert len(history) == 10
        assert "score" in history.columns
        assert "n_features" in history.columns

    def test_best_score(self, classification_data):
        """Test best score retrieval."""
        X, y = classification_data
        selector = BayesianFeatureSelector(n_iterations=10, random_state=42)
        selector.fit(X, y)

        best_score = selector.get_best_score()
        assert isinstance(best_score, float)
        assert best_score > 0

    def test_not_fitted_error(self, classification_data):
        """Test NotFittedError before fitting."""
        X, _ = classification_data
        selector = BayesianFeatureSelector()

        with pytest.raises(NotFittedError):
            selector.transform(X)


class TestSequentialFeatureSelector:
    """Tests for SequentialFeatureSelector."""

    def test_forward_selection(self, classification_data):
        """Test forward selection."""
        X, y = classification_data
        selector = SequentialFeatureSelector(
            direction="forward",
            n_features_to_select=5,
        )
        X_selected = selector.fit_transform(X, y)

        assert len(X_selected.columns) == 5

    def test_backward_selection(self, classification_data):
        """Test backward selection."""
        X, y = classification_data
        selector = SequentialFeatureSelector(
            direction="backward",
            n_features_to_select=10,
        )
        X_selected = selector.fit_transform(X, y)

        assert len(X_selected.columns) == 10

    def test_floating_selection(self, classification_data):
        """Test floating variant."""
        X, y = classification_data
        selector = SequentialFeatureSelector(
            direction="forward",
            n_features_to_select=5,
            floating=True,
        )
        X_selected = selector.fit_transform(X, y)

        assert isinstance(X_selected, pd.DataFrame)

    def test_selection_history(self, classification_data):
        """Test selection history."""
        X, y = classification_data
        selector = SequentialFeatureSelector(n_features_to_select=5)
        selector.fit(X, y)

        history = selector.get_selection_history()
        assert len(history) > 0


class TestGeneticFeatureSelector:
    """Tests for GeneticFeatureSelector."""

    def test_fit_transform(self, classification_data):
        """Test genetic algorithm selection."""
        X, y = classification_data
        selector = GeneticFeatureSelector(
            population_size=20,
            n_generations=5,
            random_state=42,
        )
        X_selected = selector.fit_transform(X, y)

        assert isinstance(X_selected, pd.DataFrame)
        assert len(X_selected) == len(X)

    def test_generation_history(self, classification_data):
        """Test generation history tracking."""
        X, y = classification_data
        selector = GeneticFeatureSelector(
            population_size=10,
            n_generations=5,
            random_state=42,
        )
        selector.fit(X, y)

        history = selector.get_generation_history()
        assert len(history) == 5
        assert "best_fitness" in history.columns
        assert "mean_fitness" in history.columns

    def test_min_max_features(self, classification_data):
        """Test min/max features constraints."""
        X, y = classification_data
        selector = GeneticFeatureSelector(
            population_size=10,
            n_generations=5,
            min_features=3,
            max_features=10,
            random_state=42,
        )
        selector.fit(X, y)

        n_selected = selector.get_support(indices=False).sum()
        assert 3 <= n_selected <= 10


class TestAutoSelectFeatures:
    """Tests for auto_select_features function."""

    def test_bayesian_method(self, classification_data):
        """Test Bayesian method."""
        X, y = classification_data
        X_selected, selector = auto_select_features(
            X, y, method="bayesian", n_iterations=5
        )

        assert isinstance(X_selected, pd.DataFrame)
        assert isinstance(selector, BayesianFeatureSelector)

    def test_sequential_method(self, classification_data):
        """Test sequential method."""
        X, y = classification_data
        X_selected, selector = auto_select_features(
            X, y, method="sequential", n_features=5
        )

        assert isinstance(X_selected, pd.DataFrame)
        assert isinstance(selector, SequentialFeatureSelector)

    def test_genetic_method(self, classification_data):
        """Test genetic method."""
        X, y = classification_data
        X_selected, selector = auto_select_features(
            X, y, method="genetic", population_size=10, n_generations=3
        )

        assert isinstance(X_selected, pd.DataFrame)
        assert isinstance(selector, GeneticFeatureSelector)

    def test_invalid_method(self, classification_data):
        """Test invalid method raises error."""
        X, y = classification_data

        with pytest.raises(ValueError):
            auto_select_features(X, y, method="invalid")
