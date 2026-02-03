"""Tests for feature impact simulator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from forge.simulator import FeatureImpactSimulator, ImpactReport


@pytest.fixture
def classification_data():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        "important": np.random.randn(n) * 2,
        "useful": np.random.randn(n),
        "noise": np.random.randn(n),
    })
    y = (X["important"] + 0.5 * X["useful"] > 0).astype(int)
    return X, y


@pytest.fixture
def fitted_classifier(classification_data):
    X, y = classification_data
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def regression_data():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        "x1": np.random.randn(n),
        "x2": np.random.randn(n),
        "x3": np.random.randn(n),
    })
    y = 3 * X["x1"] + 2 * X["x2"] + np.random.randn(n) * 0.1
    return X, y


@pytest.fixture
def fitted_regressor(regression_data):
    X, y = regression_data
    model = LinearRegression()
    model.fit(X, y)
    return model


class TestFeatureImpactSimulator:
    """Tests for FeatureImpactSimulator."""

    def test_fit_basic(self, classification_data, fitted_classifier):
        X, y = classification_data
        sim = FeatureImpactSimulator(model=fitted_classifier, scoring="accuracy")
        sim.fit(X, y)
        assert sim._is_fitted
        assert sim.baseline_score_ > 0
        assert len(sim.feature_names_in_) == 3

    def test_fit_requires_y(self, classification_data, fitted_classifier):
        X, _ = classification_data
        sim = FeatureImpactSimulator(model=fitted_classifier)
        with pytest.raises(ValueError, match="Target y is required"):
            sim.fit(X, y=None)

    def test_fit_requires_model(self, classification_data):
        X, y = classification_data
        sim = FeatureImpactSimulator(model=None)
        with pytest.raises(ValueError, match="fitted model is required"):
            sim.fit(X, y)

    def test_simulate_remove_permutation(self, classification_data, fitted_classifier):
        X, y = classification_data
        sim = FeatureImpactSimulator(
            model=fitted_classifier, scoring="accuracy",
            method="permutation", random_state=42,
        )
        sim.fit(X, y)
        report = sim.simulate_remove("important")
        assert report.action == "remove"
        assert report.feature == "important"
        assert report.delta <= 0  # Removing important feature should hurt
        assert report.method == "permutation"

    def test_simulate_remove_ablation(self, classification_data, fitted_classifier):
        X, y = classification_data
        sim = FeatureImpactSimulator(
            model=fitted_classifier, scoring="accuracy",
            method="ablation", random_state=42,
        )
        sim.fit(X, y)
        report = sim.simulate_remove("important")
        assert report.method == "ablation"
        assert report.delta <= 0

    def test_simulate_remove_surrogate(self, classification_data, fitted_classifier):
        X, y = classification_data
        sim = FeatureImpactSimulator(
            model=fitted_classifier, scoring="accuracy",
            method="surrogate", random_state=42,
        )
        sim.fit(X, y)
        report = sim.simulate_remove("noise")
        assert report.method == "surrogate"

    def test_simulate_remove_invalid_feature(self, classification_data, fitted_classifier):
        X, y = classification_data
        sim = FeatureImpactSimulator(model=fitted_classifier)
        sim.fit(X, y)
        with pytest.raises(ValueError, match="not found"):
            sim.simulate_remove("nonexistent")

    def test_simulate_add(self, classification_data, fitted_classifier):
        X, y = classification_data
        X_ext = X.copy()
        X_ext["new_useful"] = X["important"] * 2 + np.random.randn(len(X)) * 0.1
        sim = FeatureImpactSimulator(
            model=fitted_classifier, scoring="accuracy", random_state=42,
        )
        sim.fit(X, y)
        report = sim.simulate_add("new_useful", X_ext)
        assert report.action == "add"
        assert report.feature == "new_useful"

    def test_what_if_batch(self, classification_data, fitted_classifier):
        X, y = classification_data
        sim = FeatureImpactSimulator(
            model=fitted_classifier, scoring="accuracy",
            method="permutation", random_state=42,
        )
        sim.fit(X, y)
        reports = sim.what_if(remove=["noise", "useful"])
        assert len(reports) == 2

    def test_rank_features(self, classification_data, fitted_classifier):
        X, y = classification_data
        sim = FeatureImpactSimulator(
            model=fitted_classifier, scoring="accuracy",
            method="permutation", random_state=42, n_repeats=3,
        )
        sim.fit(X, y)
        ranked = sim.rank_features()
        assert len(ranked) == 3
        assert abs(ranked[0].delta) >= abs(ranked[-1].delta)

    def test_get_summary(self, classification_data, fitted_classifier):
        X, y = classification_data
        sim = FeatureImpactSimulator(
            model=fitted_classifier, scoring="accuracy", random_state=42,
        )
        sim.fit(X, y)
        sim.simulate_remove("noise")
        summary = sim.get_summary()
        assert "baseline_score" in summary
        assert "simulations" in summary
        assert len(summary["simulations"]) == 1

    def test_regression_model(self, regression_data, fitted_regressor):
        X, y = regression_data
        sim = FeatureImpactSimulator(
            model=fitted_regressor, scoring="r2",
            method="permutation", random_state=42,
        )
        sim.fit(X, y)
        report = sim.simulate_remove("x1")
        assert report.delta < 0  # x1 is most important

    def test_not_fitted_raises(self, classification_data):
        X, y = classification_data
        sim = FeatureImpactSimulator(model=RandomForestClassifier())
        with pytest.raises(RuntimeError, match="must be fitted"):
            sim.simulate_remove("important")


class TestImpactReport:
    """Tests for ImpactReport dataclass."""

    def test_str_representation(self):
        report = ImpactReport(
            feature="col_a",
            action="remove",
            baseline_score=0.9,
            simulated_score=0.85,
            delta=-0.05,
            relative_delta=-0.0556,
            confidence=0.8,
            method="permutation",
        )
        s = str(report)
        assert "REMOVE" in s
        assert "col_a" in s
        assert "-0.05" in s

    def test_str_positive_delta(self):
        report = ImpactReport(
            feature="new_feat",
            action="add",
            baseline_score=0.9,
            simulated_score=0.95,
            delta=0.05,
            relative_delta=0.0556,
            confidence=0.8,
            method="surrogate",
        )
        s = str(report)
        assert "ADD" in s
        assert "+0.05" in s
