"""Tests for RL feature search."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.search.rl_search import RLFeatureSearch, SimplePolicy, _compute_state


@pytest.fixture
def clf_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(42)
    X = pd.DataFrame({
        "a": rng.randn(80),
        "b": rng.randn(80),
        "c": rng.randn(80),
    })
    y = pd.Series((X["a"] + X["b"] > 0).astype(int), name="target")
    return X, y


class TestSimplePolicy:
    def test_select_action(self) -> None:
        p = SimplePolicy(n_actions=3, state_dim=4)
        state = np.ones(4)
        action = p.select_action(state)
        assert 0 <= action < 3

    def test_update_does_not_crash(self) -> None:
        p = SimplePolicy(n_actions=2, state_dim=4)
        states = [np.ones(4), np.ones(4)]
        actions = [0, 1]
        p.update(states, actions, reward=1.0)


class TestComputeState:
    def test_returns_correct_shape(self) -> None:
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        state = _compute_state(X, ["feat1"])
        assert state.shape == (8,)


class TestRLFeatureSearch:
    def test_fit_transform(self, clf_data: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = clf_data
        rl = RLFeatureSearch(n_episodes=3, max_steps=3, random_state=0)
        result = rl.fit_transform(X, y)
        assert isinstance(result, pd.DataFrame)

    def test_get_feature_names_out(self, clf_data: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = clf_data
        rl = RLFeatureSearch(n_episodes=3, max_steps=3, random_state=0)
        rl.fit(X, y)
        names = rl.get_feature_names_out()
        assert isinstance(names, list)

    def test_episode_history(self, clf_data: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = clf_data
        rl = RLFeatureSearch(n_episodes=5, max_steps=2, random_state=0)
        rl.fit(X, y)
        assert len(rl.episode_history) == 5

    def test_best_reward(self, clf_data: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = clf_data
        rl = RLFeatureSearch(n_episodes=3, max_steps=3, random_state=0)
        rl.fit(X, y)
        assert rl.best_reward > float("-inf")

    def test_not_fitted_raises(self) -> None:
        rl = RLFeatureSearch()
        with pytest.raises(Exception):
            rl.transform(pd.DataFrame({"a": [1]}))

    def test_no_target_raises(self) -> None:
        rl = RLFeatureSearch()
        with pytest.raises(Exception):
            rl.fit(pd.DataFrame({"a": [1, 2, 3]}))
