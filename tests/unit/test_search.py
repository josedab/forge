"""Tests for the AutoFeature search module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.search import (
    AutoFeatureSearch,
    EvolutionarySearch,
    RandomSearch,
    TransformGrammar,
    TransformNode,
    TransformOp,
)


@pytest.fixture
def sample_data():
    rng = np.random.RandomState(42)
    n = 200
    X = pd.DataFrame({
        "a": rng.randn(n),
        "b": rng.randn(n) * 2 + 1,
        "c": rng.rand(n) * 10,
    })
    y = pd.Series((X["a"] * X["b"] > 0).astype(int), name="target")
    return X, y


class TestTransformGrammar:
    def test_sample_programs(self, sample_data):
        X, _ = sample_data
        grammar = TransformGrammar(max_depth=2)
        programs = grammar.sample_programs(X, n=10, rng=np.random.RandomState(42))
        assert len(programs) == 10
        assert all(len(p) >= 1 for p in programs)

    def test_evaluate_program(self, sample_data):
        X, _ = sample_data
        grammar = TransformGrammar()
        prog = [TransformNode(op=TransformOp.LOG1P, columns=["c"])]
        result = grammar.evaluate_program(prog, X)
        assert len(result) == len(X)
        assert not result.isna().all()

    def test_program_name(self):
        grammar = TransformGrammar()
        prog = [
            TransformNode(op=TransformOp.MULTIPLY, columns=["a", "b"]),
            TransformNode(op=TransformOp.LOG1P, columns=["_prev"]),
        ]
        name = grammar.program_name(prog)
        assert "multiply" in name
        assert "log1p" in name


class TestTransformNode:
    def test_unary_apply(self, sample_data):
        X, _ = sample_data
        node = TransformNode(op=TransformOp.SQUARE, columns=["a"])
        result = node.apply(X)
        assert result.iloc[0] == pytest.approx(X["a"].iloc[0] ** 2)

    def test_binary_apply(self, sample_data):
        X, _ = sample_data
        node = TransformNode(op=TransformOp.ADD, columns=["a", "b"])
        result = node.apply(X)
        assert result.iloc[0] == pytest.approx(X["a"].iloc[0] + X["b"].iloc[0])

    def test_identity(self, sample_data):
        X, _ = sample_data
        node = TransformNode(op=TransformOp.IDENTITY, columns=["a"])
        result = node.apply(X)
        pd.testing.assert_series_equal(result, X["a"])

    def test_name_generation(self):
        node = TransformNode(op=TransformOp.SQRT, columns=["price"])
        assert node.name == "sqrt_price"


class TestRandomSearch:
    def test_basic_search(self, sample_data):
        X, y = sample_data
        search = RandomSearch(n_candidates=20, cv=2, random_state=42)
        search.fit(X, y)
        assert search._is_fitted
        assert len(search.results_) > 0
        assert all(r.score is not None for r in search.results_)

    def test_results_sorted(self, sample_data):
        X, y = sample_data
        search = RandomSearch(n_candidates=30, cv=2, random_state=42)
        search.fit(X, y)
        scores = [r.score for r in search.results_]
        assert scores == sorted(scores, reverse=True)


class TestEvolutionarySearch:
    def test_basic_evolution(self, sample_data):
        X, y = sample_data
        search = EvolutionarySearch(
            population_size=15, n_generations=3, cv=2, random_state=42
        )
        search.fit(X, y)
        assert search._is_fitted
        assert len(search.results_) > 0
        assert len(search.history_) == 3

    def test_history_recorded(self, sample_data):
        X, y = sample_data
        search = EvolutionarySearch(
            population_size=10, n_generations=2, cv=2, random_state=42
        )
        search.fit(X, y)
        assert all("best_score" in h for h in search.history_)


class TestAutoFeatureSearch:
    def test_random_method(self, sample_data):
        X, y = sample_data
        search = AutoFeatureSearch(
            method="random", n_features=5, n_candidates=20,
            cv=2, random_state=42
        )
        result = search.fit_transform(X, y)
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) <= 5
        assert len(result) == len(X)

    def test_evolutionary_method(self, sample_data):
        X, y = sample_data
        search = AutoFeatureSearch(
            method="evolutionary", n_features=5,
            population_size=10, n_generations=2,
            cv=2, random_state=42
        )
        result = search.fit_transform(X, y)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(X)

    def test_get_feature_names_out(self, sample_data):
        X, y = sample_data
        search = AutoFeatureSearch(
            method="random", n_features=5, n_candidates=20,
            cv=2, random_state=42
        )
        search.fit(X, y)
        names = search.get_feature_names_out()
        assert isinstance(names, list)
        assert len(names) <= 5

    def test_requires_target(self, sample_data):
        X, _ = sample_data
        search = AutoFeatureSearch()
        with pytest.raises(Exception):
            search.fit(X, None)

    def test_invalid_method(self, sample_data):
        X, y = sample_data
        search = AutoFeatureSearch(method="invalid")
        with pytest.raises(Exception):
            search.fit(X, y)
