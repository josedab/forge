"""Tests for ensemble feature selectors."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from forge.selectors.ensemble import (
    EnsembleImportanceSelector,
    StabilitySelector,
    ImportanceMethod,
    ensemble_importance,
)
from forge.exceptions import NotFittedError, ValidationError


@pytest.fixture
def classification_data():
    """Create classification dataset."""
    X, y = make_classification(
        n_samples=200,
        n_features=15,
        n_informative=8,
        n_redundant=3,
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(15)])
    y = pd.Series(y)
    return X, y


class TestEnsembleImportanceSelector:
    """Tests for EnsembleImportanceSelector."""

    def test_fit_transform_basic(self, classification_data):
        """Test basic fit and transform."""
        X, y = classification_data
        selector = EnsembleImportanceSelector(
            methods=["tree", "linear"],
            n_features=10,
            random_state=42,
        )
        X_selected = selector.fit_transform(X, y)

        assert isinstance(X_selected, pd.DataFrame)
        assert len(X_selected.columns) == 10

    def test_multiple_methods(self, classification_data):
        """Test with multiple importance methods."""
        X, y = classification_data
        selector = EnsembleImportanceSelector(
            methods=["tree", "permutation", "mutual_info"],
            n_features=8,
            random_state=42,
        )
        selector.fit(X, y)

        method_importances = selector.get_method_importances()
        assert "tree" in method_importances.columns
        assert "permutation" in method_importances.columns
        assert "mutual_info" in method_importances.columns

    def test_custom_method_weights(self, classification_data):
        """Test with custom method weights."""
        X, y = classification_data
        methods = [
            ImportanceMethod(name="tree", weight=2.0),
            ImportanceMethod(name="linear", weight=1.0),
        ]
        selector = EnsembleImportanceSelector(
            methods=methods,
            n_features=5,
            random_state=42,
        )
        X_selected = selector.fit_transform(X, y)

        assert len(X_selected.columns) == 5

    def test_aggregation_methods(self, classification_data):
        """Test different aggregation methods."""
        X, y = classification_data

        for aggregation in ["mean", "median", "rank", "vote"]:
            selector = EnsembleImportanceSelector(
                methods=["tree", "linear"],
                n_features=5,
                aggregation=aggregation,
                random_state=42,
            )
            X_selected = selector.fit_transform(X, y)
            assert len(X_selected.columns) == 5

    def test_method_agreement(self, classification_data):
        """Test method agreement calculation."""
        X, y = classification_data
        selector = EnsembleImportanceSelector(
            methods=["tree", "linear", "mutual_info"],
            random_state=42,
        )
        selector.fit(X, y)

        agreement = selector.get_method_agreement()
        assert agreement.shape == (3, 3)
        # Diagonal should be 1.0 (perfect correlation with self)
        np.testing.assert_array_almost_equal(np.diag(agreement), [1.0, 1.0, 1.0])

    def test_requires_target(self, classification_data):
        """Test that target is required."""
        X, _ = classification_data
        selector = EnsembleImportanceSelector()

        with pytest.raises(ValidationError):
            selector.fit(X)


class TestStabilitySelector:
    """Tests for StabilitySelector."""

    def test_fit_transform(self, classification_data):
        """Test stability selection."""
        X, y = classification_data
        selector = StabilitySelector(
            n_iterations=20,
            sample_fraction=0.7,
            threshold=0.5,
            random_state=42,
        )
        X_selected = selector.fit_transform(X, y)

        assert isinstance(X_selected, pd.DataFrame)
        assert len(X_selected.columns) > 0

    def test_stability_scores(self, classification_data):
        """Test stability score retrieval."""
        X, y = classification_data
        selector = StabilitySelector(
            n_iterations=20,
            random_state=42,
        )
        selector.fit(X, y)

        stability = selector.get_stability_scores()
        assert "selection_frequency" in stability.columns
        assert all(0 <= f <= 1 for f in stability["selection_frequency"])

    def test_n_features_option(self, classification_data):
        """Test with fixed number of features."""
        X, y = classification_data
        selector = StabilitySelector(
            n_iterations=20,
            n_features=5,
            random_state=42,
        )
        X_selected = selector.fit_transform(X, y)

        assert len(X_selected.columns) == 5


class TestEnsembleImportanceFunction:
    """Tests for ensemble_importance convenience function."""

    def test_basic_usage(self, classification_data):
        """Test basic function usage."""
        X, y = classification_data
        X_selected, selector = ensemble_importance(
            X, y,
            methods=["tree", "linear"],
            n_features=8,
        )

        assert isinstance(X_selected, pd.DataFrame)
        assert isinstance(selector, EnsembleImportanceSelector)
        assert len(X_selected.columns) == 8
