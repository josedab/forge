"""Reinforcement-learning based feature search.

Uses a simple policy-gradient (REINFORCE) approach to learn which
feature transformations to apply.  The agent observes dataset
statistics and selects actions (add feature, remove feature, stop).

Example:
    >>> from forge.search.rl_search import RLFeatureSearch
    >>> search = RLFeatureSearch(n_episodes=50, scoring="accuracy")
    >>> search.fit(X, y)
    >>> X_new = search.transform(X)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score

from forge.exceptions import NotFittedError, ValidationError
from forge.search.grammar import TransformGrammar, TransformNode

if TYPE_CHECKING:
    from typing_extensions import Self

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """Record of a single RL search episode."""

    actions: list[int]
    programs: list[list[TransformNode]]
    reward: float
    n_features: int


class SimplePolicy:
    """Lightweight softmax policy over discrete actions.

    Parameters
    ----------
    n_actions : int
        Number of possible actions.
    state_dim : int
        Dimensionality of state vector.
    learning_rate : float
        Step size for policy gradient updates.
    """

    def __init__(
        self,
        n_actions: int,
        state_dim: int = 8,
        learning_rate: float = 0.01,
    ) -> None:
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self._rng = np.random.RandomState(0)
        # Weight matrix: state_dim x n_actions
        self.weights = np.zeros((state_dim, n_actions))

    def select_action(self, state: np.ndarray) -> int:
        """Sample an action from the policy distribution."""
        logits = state @ self.weights
        probs = self._softmax(logits)
        return int(self._rng.choice(self.n_actions, p=probs))

    def update(
        self,
        states: list[np.ndarray],
        actions: list[int],
        reward: float,
    ) -> None:
        """REINFORCE update: increase probability of actions that led to high reward."""
        for state, action in zip(states, actions):
            logits = state @ self.weights
            probs = self._softmax(logits)
            grad = -probs.copy()
            grad[action] += 1.0
            self.weights += self.learning_rate * reward * np.outer(state, grad)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


def _compute_state(
    X: pd.DataFrame,
    current_features: list[str],
) -> np.ndarray:
    """Compute a fixed-size state vector from the dataset."""
    n_rows, n_cols = X.shape
    numeric = X.select_dtypes(include=[np.number])
    n_numeric = numeric.shape[1]
    mean_corr = 0.0
    if n_numeric >= 2:
        corr = numeric.corr().values
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        vals = corr[mask]
        mean_corr = float(np.nanmean(np.abs(vals))) if len(vals) > 0 else 0.0

    return np.array([
        np.log1p(n_rows),
        np.log1p(n_cols),
        n_numeric / max(n_cols, 1),
        mean_corr,
        len(current_features) / max(n_cols, 1),
        float(numeric.isna().mean().mean()) if n_numeric > 0 else 0.0,
        np.log1p(len(current_features)),
        1.0,  # bias
    ], dtype=float)


class RLFeatureSearch(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Reinforcement-learning feature search.

    Uses a lightweight REINFORCE agent to learn an optimal sequence
    of feature generation actions.

    Parameters
    ----------
    n_episodes : int
        Number of RL episodes to run.
    max_steps : int
        Max actions per episode.
    n_candidates_per_step : int
        Random feature candidates sampled at each step.
    scoring : str | None
        sklearn scoring metric.
    cv : int
        Cross-validation folds.
    learning_rate : float
        Policy learning rate.
    random_state : int | None
        Random seed.

    Example:
    -------
    >>> rl = RLFeatureSearch(n_episodes=30, scoring="accuracy")
    >>> X_new = rl.fit_transform(X, y)
    """

    def __init__(
        self,
        n_episodes: int = 20,
        max_steps: int = 10,
        n_candidates_per_step: int = 5,
        scoring: str | None = None,
        cv: int = 3,
        learning_rate: float = 0.01,
        random_state: int | None = None,
    ) -> None:
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.n_candidates_per_step = n_candidates_per_step
        self.scoring = scoring
        self.cv = cv
        self.learning_rate = learning_rate
        self.random_state = random_state

        self._is_fitted = False
        self._grammar = TransformGrammar()
        self._selected_programs: list[tuple[str, list[TransformNode]]] = []
        self._feature_names_out: list[str] = []
        self._episodes: list[Episode] = []
        self._best_reward: float = float("-inf")

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Run RL search to discover useful feature programs.

        Args:
            X: Input features.
            y: Target variable (required).

        Returns:
            Self.
        """
        if y is None:
            raise ValidationError("Target y is required for RLFeatureSearch.")
        if not isinstance(X, pd.DataFrame):
            raise ValidationError(f"Expected DataFrame, got {type(X).__name__}")

        rng = np.random.RandomState(self.random_state)
        scoring = self.scoring or self._infer_scoring(y)
        estimator = self._get_estimator(y)

        # Actions: 0 = add random feature, 1 = stop
        n_actions = 2
        policy = SimplePolicy(
            n_actions=n_actions,
            state_dim=8,
            learning_rate=self.learning_rate,
        )
        policy._rng = rng

        best_programs: list[tuple[str, list[TransformNode]]] = []
        best_reward = float("-inf")

        for _ep in range(self.n_episodes):
            current_programs: list[tuple[str, list[TransformNode]]] = []
            states: list[np.ndarray] = []
            actions: list[int] = []
            X_aug = X.copy()

            for _step in range(self.max_steps):
                state = _compute_state(X_aug, [n for n, _ in current_programs])
                action = policy.select_action(state)
                states.append(state)
                actions.append(action)

                if action == 1:  # stop
                    break

                # Add a random feature
                candidates = self._grammar.sample_programs(X, n=self.n_candidates_per_step, rng=rng)
                added = False
                for prog in candidates:
                    try:
                        feature = self._grammar.evaluate_program(prog, X)
                        if feature.isna().all() or feature.std() == 0:
                            continue
                        name = self._grammar.program_name(prog)
                        if name not in X_aug.columns:
                            X_aug[name] = feature
                            current_programs.append((name, prog))
                            added = True
                            break
                    except Exception:  # noqa: S112
                        continue
                if not added:
                    break

            # Evaluate reward
            reward = self._evaluate_features(X_aug, y, estimator, scoring)
            policy.update(states, actions, reward)

            self._episodes.append(Episode(
                actions=actions,
                programs=[p for _, p in current_programs],
                reward=reward,
                n_features=len(current_programs),
            ))

            if reward > best_reward:
                best_reward = reward
                best_programs = list(current_programs)

        self._selected_programs = best_programs
        self._best_reward = best_reward
        self._feature_names_out = [n for n, _ in best_programs]
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate discovered features.

        Args:
            X: Input DataFrame.

        Returns:
            DataFrame with discovered features.
        """
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)

        result: dict[str, pd.Series] = {}
        for name, prog in self._selected_programs:
            try:
                feature = self._grammar.evaluate_program(prog, X)
                result[name] = feature
            except Exception:
                result[name] = pd.Series(np.nan, index=X.index)

        return pd.DataFrame(result, index=X.index) if result else pd.DataFrame(index=X.index)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)
        return list(self._feature_names_out)

    @property
    def best_reward(self) -> float:
        return self._best_reward

    @property
    def episode_history(self) -> list[Episode]:
        return list(self._episodes)

    def _evaluate_features(
        self, X_aug: pd.DataFrame, y: pd.Series, estimator: Any, scoring: str,
    ) -> float:
        X_num = X_aug.select_dtypes(include=[np.number]).fillna(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                scores = cross_val_score(estimator, X_num, y, cv=self.cv, scoring=scoring)
                return float(np.mean(scores))
            except Exception:
                return float("-inf")

    @staticmethod
    def _infer_scoring(y: pd.Series) -> str:
        return "accuracy" if y.nunique() <= 20 else "neg_mean_squared_error"

    @staticmethod
    def _get_estimator(y: pd.Series) -> Any:
        if y.nunique() <= 20:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=20, max_depth=4, random_state=0)
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=20, max_depth=4, random_state=0)
