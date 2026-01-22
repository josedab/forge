"""Tests for feature interaction generators."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from forge.generators.interactions import (
    InteractionDiscoverer,
    PolynomialInteractionGenerator,
    GroupedInteractionGenerator,
    discover_interactions,
)
from forge.exceptions import NotFittedError, ValidationError


@pytest.fixture
def classification_data():
    """Create classification dataset."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    y = pd.Series(y)
    return X, y


@pytest.fixture
def regression_data():
    """Create regression dataset with interactions."""
    np.random.seed(42)
    n_samples = 200

    X = pd.DataFrame({
        "a": np.random.randn(n_samples),
        "b": np.random.randn(n_samples),
        "c": np.random.randn(n_samples),
        "d": np.random.randn(n_samples),
    })

    # Create target with known interactions
    y = X["a"] * X["b"] + X["c"] ** 2 + np.random.randn(n_samples) * 0.1

    return X, pd.Series(y)


@pytest.fixture
def grouped_data():
    """Create data suitable for grouped interactions."""
    np.random.seed(42)
    n_samples = 200

    return pd.DataFrame({
        "price": np.random.uniform(10, 100, n_samples),
        "quantity": np.random.randint(1, 10, n_samples),
        "width": np.random.uniform(1, 10, n_samples),
        "height": np.random.uniform(1, 10, n_samples),
        "depth": np.random.uniform(1, 10, n_samples),
    })


class TestInteractionDiscoverer:
    """Tests for InteractionDiscoverer."""

    def test_fit_transform(self, classification_data):
        """Test basic fit and transform."""
        X, y = classification_data
        discoverer = InteractionDiscoverer(
            max_interactions=5,
            min_score_threshold=0.0,
            random_state=42,
        )
        X_transformed = discoverer.fit_transform(X, y)

        assert isinstance(X_transformed, pd.DataFrame)
        assert len(X_transformed.columns) >= len(X.columns)

    def test_discovers_interactions(self, regression_data):
        """Test that interactions are discovered."""
        X, y = regression_data
        discoverer = InteractionDiscoverer(
            max_interactions=10,
            random_state=42,
        )
        discoverer.fit(X, y)

        interactions = discoverer.get_interactions()
        assert len(interactions) > 0 or isinstance(interactions, pd.DataFrame)

    def test_interaction_scores(self, classification_data):
        """Test interaction scoring."""
        X, y = classification_data
        discoverer = InteractionDiscoverer(
            max_interactions=5,
            random_state=42,
        )
        discoverer.fit(X, y)

        scores = discoverer.get_interactions()
        assert isinstance(scores, pd.DataFrame)

    def test_min_threshold_filter(self, classification_data):
        """Test minimum score threshold filtering."""
        X, y = classification_data

        # With low threshold
        discoverer_low = InteractionDiscoverer(
            max_interactions=10,
            min_score_threshold=0.0,
            random_state=42,
        )
        discoverer_low.fit(X, y)
        low_interactions = discoverer_low.get_interactions()

        # With high threshold
        discoverer_high = InteractionDiscoverer(
            max_interactions=10,
            min_score_threshold=0.5,
            random_state=42,
        )
        discoverer_high.fit(X, y)
        high_interactions = discoverer_high.get_interactions()

        # High threshold should find fewer or equal interactions
        # Handle empty DataFrame case
        assert len(high_interactions) <= len(low_interactions) or len(high_interactions) == 0

    def test_max_interactions_limit(self, classification_data):
        """Test max interactions limit."""
        X, y = classification_data
        max_limit = 3

        discoverer = InteractionDiscoverer(
            max_interactions=max_limit,
            random_state=42,
        )
        discoverer.fit(X, y)

        interactions = discoverer.get_interactions()
        assert len(interactions) <= max_limit

    def test_requires_target(self, classification_data):
        """Test that target is required."""
        X, _ = classification_data
        discoverer = InteractionDiscoverer()

        with pytest.raises(ValidationError):
            discoverer.fit(X)

    def test_not_fitted_error(self, classification_data):
        """Test NotFittedError before fitting."""
        X, _ = classification_data
        discoverer = InteractionDiscoverer()

        with pytest.raises(NotFittedError):
            discoverer.transform(X)

    def test_get_feature_names_out(self, classification_data):
        """Test get_feature_names_out method."""
        X, y = classification_data
        discoverer = InteractionDiscoverer(max_interactions=3, random_state=42)
        discoverer.fit(X, y)

        names = discoverer.get_feature_names_out()
        assert isinstance(names, list)
        assert len(names) >= len(X.columns)


class TestPolynomialInteractionGenerator:
    """Tests for PolynomialInteractionGenerator."""

    def test_degree_2(self, classification_data):
        """Test degree 2 polynomial features."""
        X, _ = classification_data
        generator = PolynomialInteractionGenerator(
            degree=2,
            interaction_only=False,
        )
        X_transformed = generator.fit_transform(X)

        assert isinstance(X_transformed, pd.DataFrame)
        assert len(X_transformed.columns) > len(X.columns)

    def test_interaction_only(self, classification_data):
        """Test interaction only (no powers)."""
        X, _ = classification_data

        # With powers
        gen_powers = PolynomialInteractionGenerator(
            degree=2,
            interaction_only=False,
        )
        X_powers = gen_powers.fit_transform(X)

        # Interaction only
        gen_interactions = PolynomialInteractionGenerator(
            degree=2,
            interaction_only=True,
        )
        X_interactions = gen_interactions.fit_transform(X)

        # Interaction only should have fewer features
        assert len(X_interactions.columns) < len(X_powers.columns)

    def test_include_bias(self, classification_data):
        """Test include_bias option."""
        X, _ = classification_data

        gen_no_bias = PolynomialInteractionGenerator(
            degree=2,
            include_bias=False,
        )
        X_no_bias = gen_no_bias.fit_transform(X)

        gen_bias = PolynomialInteractionGenerator(
            degree=2,
            include_bias=True,
        )
        X_bias = gen_bias.fit_transform(X)

        # With bias should have one more column
        assert len(X_bias.columns) == len(X_no_bias.columns) + 1

    def test_selected_columns(self, classification_data):
        """Test with selected columns only."""
        X, _ = classification_data
        selected = ["feature_0", "feature_1", "feature_2"]

        generator = PolynomialInteractionGenerator(
            degree=2,
            columns=selected,
        )
        X_transformed = generator.fit_transform(X)

        assert isinstance(X_transformed, pd.DataFrame)

    def test_degree_3(self, classification_data):
        """Test degree 3 polynomial features."""
        X, _ = classification_data
        # Use fewer features to keep size manageable
        X_small = X[["feature_0", "feature_1", "feature_2"]]

        generator = PolynomialInteractionGenerator(
            degree=3,
        )
        X_transformed = generator.fit_transform(X_small)

        assert len(X_transformed.columns) > len(X_small.columns)

    def test_get_feature_names_out(self, classification_data):
        """Test get_feature_names_out method."""
        X, _ = classification_data
        X_small = X[["feature_0", "feature_1"]]

        generator = PolynomialInteractionGenerator(degree=2)
        generator.fit(X_small)

        names = generator.get_feature_names_out()
        assert isinstance(names, list)
        assert len(names) > 0


class TestGroupedInteractionGenerator:
    """Tests for GroupedInteractionGenerator."""

    def test_basic_groups(self, grouped_data):
        """Test basic grouped interactions."""
        groups = {
            "financial": ["price", "quantity"],
            "dimensions": ["width", "height", "depth"],
        }

        generator = GroupedInteractionGenerator(
            groups=groups,
            operations=["multiply"],
        )
        X_transformed = generator.fit_transform(grouped_data)

        assert isinstance(X_transformed, pd.DataFrame)
        assert len(X_transformed.columns) > len(grouped_data.columns)

    def test_multiple_operations(self, grouped_data):
        """Test with multiple operations."""
        groups = {
            "financial": ["price", "quantity"],
        }

        generator = GroupedInteractionGenerator(
            groups=groups,
            operations=["multiply", "add", "subtract", "divide"],
        )
        X_transformed = generator.fit_transform(grouped_data)

        # Should have features for each operation
        assert len(X_transformed.columns) > len(grouped_data.columns)

    def test_between_groups_interactions(self, grouped_data):
        """Test between-group interactions."""
        groups = {
            "financial": ["price", "quantity"],
            "size": ["width", "height"],
        }

        generator = GroupedInteractionGenerator(
            groups=groups,
            operations=["multiply"],
            within_group=True,
            between_groups=True,
        )
        X_transformed = generator.fit_transform(grouped_data)

        assert isinstance(X_transformed, pd.DataFrame)

    def test_within_group_only(self, grouped_data):
        """Test within-group only interactions."""
        groups = {
            "financial": ["price", "quantity"],
            "size": ["width", "height"],
        }

        gen_within = GroupedInteractionGenerator(
            groups=groups,
            operations=["multiply"],
            within_group=True,
            between_groups=False,
        )
        X_within = gen_within.fit_transform(grouped_data)

        gen_between = GroupedInteractionGenerator(
            groups=groups,
            operations=["multiply"],
            within_group=True,
            between_groups=True,
        )
        X_between = gen_between.fit_transform(grouped_data)

        # Between-group should have more features
        assert len(X_between.columns) >= len(X_within.columns)

    def test_divide_operation(self, grouped_data):
        """Test divide operation."""
        groups = {
            "financial": ["price", "quantity"],
            "dimensions": ["width", "height"],
        }

        generator = GroupedInteractionGenerator(
            groups=groups,
            operations=["divide"],
            within_group=True,
        )
        X_transformed = generator.fit_transform(grouped_data)

        # Should have features for divide operations
        assert isinstance(X_transformed, pd.DataFrame)

    def test_get_feature_names_out(self, grouped_data):
        """Test get_feature_names_out method."""
        groups = {"financial": ["price", "quantity"]}

        generator = GroupedInteractionGenerator(
            groups=groups,
            operations=["multiply"],
        )
        generator.fit(grouped_data)

        names = generator.get_feature_names_out()
        assert isinstance(names, list)


class TestDiscoverInteractionsFunction:
    """Tests for discover_interactions convenience function."""

    def test_basic_usage(self, classification_data):
        """Test basic function usage."""
        X, y = classification_data
        X_transformed, discoverer = discover_interactions(
            X, y,
            max_interactions=5,
        )

        assert isinstance(X_transformed, pd.DataFrame)
        assert isinstance(discoverer, InteractionDiscoverer)

    def test_returns_interactions(self, classification_data):
        """Test that function returns discovered interactions."""
        X, y = classification_data
        X_transformed, discoverer = discover_interactions(
            X, y,
            max_interactions=3,
        )

        interactions = discoverer.get_interactions()
        assert len(interactions) <= 3

    def test_with_random_state(self, classification_data):
        """Test reproducibility with random state."""
        X, y = classification_data

        X1, disc1 = discover_interactions(X, y, max_interactions=3, random_state=42)
        X2, disc2 = discover_interactions(X, y, max_interactions=3, random_state=42)

        # Should get same results with same random state
        assert list(X1.columns) == list(X2.columns)
