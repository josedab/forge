"""Bayesian and multi-fidelity search strategies for AutoFE.

Extends the search module with:
- BayesianFeatureSearch: Surrogate-model-guided search using UCB acquisition.
- MultiFidelitySearch: Successive halving for efficient evaluation.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from forge.search.grammar import TransformGrammar, TransformNode
from forge.search.search import SearchResult

logger = logging.getLogger(__name__)


class BayesianFeatureSearch:
    """Bayesian search over the feature transformation space.

    Uses a surrogate model (random forest) to predict feature quality
    and an Upper Confidence Bound (UCB) acquisition function to
    balance exploration vs exploitation.

    Args:
        grammar: Transformation grammar.
        n_initial: Number of random initial evaluations.
        n_iterations: Number of Bayesian optimization iterations.
        batch_size: Candidates to evaluate per iteration.
        scoring: sklearn scoring metric.
        cv: Cross-validation folds.
        estimator: sklearn estimator for feature evaluation.
        exploration_weight: UCB exploration parameter (higher = more exploration).
        random_state: Random seed.

    Example:
        >>> search = BayesianFeatureSearch(n_iterations=30, scoring="accuracy")
        >>> search.fit(X, y)
        >>> print(search.best_features_[:5])
    """

    def __init__(
        self,
        grammar: TransformGrammar | None = None,
        n_initial: int = 30,
        n_iterations: int = 50,
        batch_size: int = 5,
        scoring: str | None = None,
        cv: int = 3,
        estimator: Any = None,
        exploration_weight: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.grammar = grammar or TransformGrammar()
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.scoring = scoring
        self.cv = cv
        self.estimator = estimator
        self.exploration_weight = exploration_weight
        self.random_state = random_state
        self.results_: list[SearchResult] = []
        self.best_features_: list[SearchResult] = []
        self.history_: list[dict[str, float]] = []
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BayesianFeatureSearch:  # type: ignore[type-arg]
        """Run Bayesian feature search.

        Args:
            X: Input features.
            y: Target variable.

        Returns:
            Self with results populated.
        """
        rng = np.random.RandomState(self.random_state)
        estimator = self._get_estimator(y)
        scoring = self.scoring or self._infer_scoring(y)

        # Phase 1: random initial sampling
        all_programs = self.grammar.sample_programs(X, n=self.n_initial, rng=rng)
        evaluated: list[tuple[list[TransformNode], np.ndarray, float]] = []

        for prog in all_programs:
            fingerprint, score = self._evaluate_program(prog, X, y, estimator, scoring)
            if fingerprint is not None:
                evaluated.append((prog, fingerprint, score))
                self.results_.append(SearchResult(
                    program=prog,
                    name=self.grammar.program_name(prog),
                    score=score,
                    generation=0,
                ))

        # Phase 2: Bayesian iterations with surrogate model
        for iteration in range(1, self.n_iterations + 1):
            if len(evaluated) < 3:
                # Not enough data for surrogate — do random
                new_progs = self.grammar.sample_programs(X, n=self.batch_size, rng=rng)
            else:
                new_progs = self._acquire_candidates(
                    X, evaluated, self.batch_size, rng
                )

            for prog in new_progs:
                fingerprint, score = self._evaluate_program(
                    prog, X, y, estimator, scoring
                )
                if fingerprint is not None:
                    evaluated.append((prog, fingerprint, score))
                    self.results_.append(SearchResult(
                        program=prog,
                        name=self.grammar.program_name(prog),
                        score=score,
                        generation=iteration,
                    ))

            scores = [s for _, _, s in evaluated if s > float("-inf")]
            self.history_.append({
                "iteration": iteration,
                "best_score": max(scores) if scores else 0.0,
                "mean_score": float(np.mean(scores)) if scores else 0.0,
                "n_evaluated": len(evaluated),
            })

        # Finalize
        self.results_.sort(key=lambda r: r.score, reverse=True)
        seen: set[str] = set()
        unique: list[SearchResult] = []
        for r in self.results_:
            if r.name not in seen:
                seen.add(r.name)
                unique.append(r)
        self.results_ = unique
        self.best_features_ = unique[:max(10, len(unique) // 5)]
        self._is_fitted = True
        return self

    def _acquire_candidates(
        self,
        X: pd.DataFrame,
        evaluated: list[tuple[list[TransformNode], np.ndarray, float]],
        n: int,
        rng: np.random.RandomState,
    ) -> list[list[TransformNode]]:
        """Use UCB acquisition to select promising candidates."""
        from sklearn.ensemble import RandomForestRegressor

        # Build surrogate
        fingerprints = np.array([fp for _, fp, _ in evaluated])
        scores = np.array([s for _, _, s in evaluated])

        # Replace -inf with worst finite score
        finite_mask = np.isfinite(scores)
        if finite_mask.sum() < 3:
            return self.grammar.sample_programs(X, n=n, rng=rng)

        worst_finite = scores[finite_mask].min()
        scores = np.where(finite_mask, scores, worst_finite)

        surrogate = RandomForestRegressor(
            n_estimators=20, max_depth=5, random_state=0
        )
        surrogate.fit(fingerprints, scores)

        # Generate candidates and rank by UCB
        candidates = self.grammar.sample_programs(X, n=n * 10, rng=rng)
        best_candidates: list[tuple[list[TransformNode], float]] = []

        for prog in candidates:
            try:
                feature = self.grammar.evaluate_program(prog, X)
                if feature.isna().all():
                    continue
                fp = self._fingerprint(feature)
                if fp is None:
                    continue

                # UCB: mean + exploration_weight * std
                preds = np.array([
                    t.predict(fp.reshape(1, -1))[0]
                    for t in surrogate.estimators_
                ])
                mean_pred = preds.mean()
                std_pred = preds.std()
                ucb = mean_pred + self.exploration_weight * std_pred
                best_candidates.append((prog, ucb))
            except Exception:
                continue

        best_candidates.sort(key=lambda x: x[1], reverse=True)
        return [prog for prog, _ in best_candidates[:n]]

    def _evaluate_program(
        self,
        prog: list[TransformNode],
        X: pd.DataFrame,
        y: pd.Series,  # type: ignore[type-arg]
        estimator: Any,
        scoring: str,
    ) -> tuple[np.ndarray | None, float]:
        """Evaluate a program and return its fingerprint and score."""
        try:
            feature = self.grammar.evaluate_program(prog, X)
            if feature.isna().all() or feature.std() == 0:
                return None, float("-inf")

            fp = self._fingerprint(feature)
            if fp is None:
                return None, float("-inf")

            X_eval = X.copy()
            X_eval["_candidate"] = feature
            X_eval = X_eval.select_dtypes(include=[np.number]).fillna(0)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(
                    estimator, X_eval, y, cv=self.cv, scoring=scoring
                )
            return fp, float(np.mean(scores))
        except Exception:
            return None, float("-inf")

    def _fingerprint(self, feature: pd.Series) -> np.ndarray | None:  # type: ignore[type-arg]
        """Compute a fixed-size statistical fingerprint of a feature."""
        try:
            vals = feature.dropna().values.astype(float)
            if len(vals) == 0:
                return None
            return np.array([
                np.mean(vals),
                np.std(vals),
                np.min(vals),
                np.max(vals),
                np.median(vals),
                float(np.percentile(vals, 25)),
                float(np.percentile(vals, 75)),
                float(len(vals)),
            ])
        except (ValueError, TypeError):
            return None

    def _get_estimator(self, y: pd.Series) -> Any:  # type: ignore[type-arg]
        if self.estimator is not None:
            return self.estimator
        if y.nunique() <= 20:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=30, max_depth=5, random_state=0)
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=30, max_depth=5, random_state=0)

    def _infer_scoring(self, y: pd.Series) -> str:  # type: ignore[type-arg]
        return "accuracy" if y.nunique() <= 20 else "neg_mean_squared_error"


class MultiFidelitySearch:
    """Multi-fidelity feature search using successive halving.

    Evaluates a large pool of candidates at low fidelity (small CV subset),
    then progressively increases fidelity for the top performers. This is
    much more compute-efficient than full evaluation of all candidates.

    Args:
        grammar: Transformation grammar.
        n_candidates: Initial candidate pool size.
        halving_rounds: Number of halving rounds.
        scoring: sklearn scoring metric.
        estimator: sklearn estimator for evaluation.
        min_fidelity: Minimum fraction of data for lowest fidelity.
        random_state: Random seed.

    Example:
        >>> search = MultiFidelitySearch(n_candidates=500)
        >>> search.fit(X, y)
        >>> print(search.best_features_[:10])
    """

    def __init__(
        self,
        grammar: TransformGrammar | None = None,
        n_candidates: int = 200,
        halving_rounds: int = 3,
        scoring: str | None = None,
        estimator: Any = None,
        min_fidelity: float = 0.2,
        random_state: int | None = None,
    ) -> None:
        self.grammar = grammar or TransformGrammar()
        self.n_candidates = n_candidates
        self.halving_rounds = halving_rounds
        self.scoring = scoring
        self.estimator = estimator
        self.min_fidelity = min_fidelity
        self.random_state = random_state
        self.results_: list[SearchResult] = []
        self.best_features_: list[SearchResult] = []
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MultiFidelitySearch:  # type: ignore[type-arg]
        """Run multi-fidelity feature search.

        Args:
            X: Input features.
            y: Target variable.

        Returns:
            Self with results populated.
        """
        rng = np.random.RandomState(self.random_state)
        estimator = self._get_estimator(y)
        scoring = self.scoring or self._infer_scoring(y)

        programs = self.grammar.sample_programs(X, n=self.n_candidates, rng=rng)
        survivors: list[tuple[list[TransformNode], float]] = [
            (p, 0.0) for p in programs
        ]

        n_rows = len(X)

        for round_idx in range(self.halving_rounds):
            fidelity = self.min_fidelity + (1.0 - self.min_fidelity) * (
                round_idx / max(self.halving_rounds - 1, 1)
            )
            sample_size = max(10, int(n_rows * fidelity))

            # Subsample data
            idx = rng.choice(n_rows, size=min(sample_size, n_rows), replace=False)
            X_sub = X.iloc[idx]
            y_sub = y.iloc[idx]

            scored: list[tuple[list[TransformNode], float]] = []
            for prog, _ in survivors:
                try:
                    feature = self.grammar.evaluate_program(prog, X_sub)
                    if feature.isna().all() or feature.std() == 0:
                        scored.append((prog, float("-inf")))
                        continue

                    X_eval = X_sub.copy()
                    X_eval["_candidate"] = feature.values
                    X_eval = X_eval.select_dtypes(include=[np.number]).fillna(0)

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        cv_folds = min(3, max(2, len(X_eval) // 5))
                        scores = cross_val_score(
                            estimator, X_eval, y_sub, cv=cv_folds, scoring=scoring
                        )
                    scored.append((prog, float(np.mean(scores))))
                except Exception:
                    scored.append((prog, float("-inf")))

            scored.sort(key=lambda x: x[1], reverse=True)
            keep = max(2, len(scored) // 2)
            survivors = scored[:keep]

            logger.info(
                "Round %d (fidelity=%.0f%%): %d → %d survivors",
                round_idx, fidelity * 100, len(scored), keep,
            )

        # Final full evaluation of survivors
        for prog, _ in survivors:
            try:
                feature = self.grammar.evaluate_program(prog, X)
                if feature.isna().all() or feature.std() == 0:
                    continue
                X_eval = X.copy()
                X_eval["_candidate"] = feature.values
                X_eval = X_eval.select_dtypes(include=[np.number]).fillna(0)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scores = cross_val_score(
                        estimator, X_eval, y, cv=3, scoring=scoring
                    )
                name = self.grammar.program_name(prog)
                self.results_.append(SearchResult(
                    program=prog, name=name, score=float(np.mean(scores)),
                ))
            except Exception:
                continue

        self.results_.sort(key=lambda r: r.score, reverse=True)
        self.best_features_ = self.results_[:max(10, len(self.results_) // 5)]
        self._is_fitted = True
        return self

    def _get_estimator(self, y: pd.Series) -> Any:  # type: ignore[type-arg]
        if self.estimator is not None:
            return self.estimator
        if y.nunique() <= 20:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=20, max_depth=4, random_state=0)
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=20, max_depth=4, random_state=0)

    def _infer_scoring(self, y: pd.Series) -> str:  # type: ignore[type-arg]
        return "accuracy" if y.nunique() <= 20 else "neg_mean_squared_error"
