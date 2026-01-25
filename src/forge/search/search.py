"""Search algorithms for automatic feature transformation discovery."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score

from forge.exceptions import ConfigurationError, NotFittedError, ValidationError
from forge.search.grammar import TransformGrammar, TransformNode

if TYPE_CHECKING:
    from typing_extensions import Self

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result of a feature search iteration.

    Args:
        program: The transformation program.
        name: Human-readable feature name.
        score: Cross-validated fitness score.
        generation: The generation/iteration this was found in.
    """

    program: list[TransformNode]
    name: str
    score: float
    generation: int = 0


class RandomSearch:
    """Random search over the transformation grammar.

    Samples random feature programs and evaluates them with
    cross-validation, keeping the top performers.

    Args:
        grammar: Transformation grammar defining the search space.
        n_candidates: Number of random candidates to evaluate.
        scoring: Scoring metric for cross-validation.
        cv: Number of cross-validation folds.
        estimator: sklearn estimator for evaluation. None for auto-detect.
        random_state: Random seed.

    Example:
        >>> search = RandomSearch(n_candidates=200, scoring="accuracy")
        >>> search.fit(X, y)
        >>> print(search.best_features_[:5])
    """

    def __init__(
        self,
        grammar: TransformGrammar | None = None,
        n_candidates: int = 100,
        scoring: str | None = None,
        cv: int = 3,
        estimator: Any = None,
        random_state: int | None = None,
    ) -> None:
        self.grammar = grammar or TransformGrammar()
        self.n_candidates = n_candidates
        self.scoring = scoring
        self.cv = cv
        self.estimator = estimator
        self.random_state = random_state
        self.results_: list[SearchResult] = []
        self.best_features_: list[SearchResult] = []
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RandomSearch:
        """Run random search to find useful feature transformations.

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
        self.results_ = []

        for prog in programs:
            try:
                feature = self.grammar.evaluate_program(prog, X)
                if feature.isna().all() or feature.std() == 0:
                    continue
                name = self.grammar.program_name(prog)
                score = self._evaluate_feature(X, feature, y, estimator, scoring)
                self.results_.append(
                    SearchResult(program=prog, name=name, score=score)
                )
            except Exception:
                continue

        self.results_.sort(key=lambda r: r.score, reverse=True)
        self.best_features_ = self.results_[: max(10, len(self.results_) // 5)]
        self._is_fitted = True
        return self

    def _evaluate_feature(
        self,
        X: pd.DataFrame,
        feature: pd.Series,
        y: pd.Series,
        estimator: Any,
        scoring: str,
    ) -> float:
        """Evaluate a single feature using cross-validation."""
        X_eval = X.copy()
        X_eval["_candidate"] = feature
        X_eval = X_eval.select_dtypes(include=[np.number]).fillna(0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                scores = cross_val_score(
                    estimator, X_eval, y, cv=self.cv, scoring=scoring
                )
                return float(np.mean(scores))
            except Exception:
                return float("-inf")

    def _get_estimator(self, y: pd.Series) -> Any:
        if self.estimator is not None:
            return self.estimator
        if y.nunique() <= 20:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=30, max_depth=5, random_state=0)
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=30, max_depth=5, random_state=0)

    def _infer_scoring(self, y: pd.Series) -> str:
        if y.nunique() <= 20:
            return "accuracy"
        return "neg_mean_squared_error"


class EvolutionarySearch:
    """Evolutionary (genetic programming) search for feature transformations.

    Uses a population of feature programs that evolve through
    selection, crossover, and mutation to discover high-performing features.

    Args:
        grammar: Transformation grammar.
        population_size: Number of individuals per generation.
        n_generations: Number of generations to evolve.
        mutation_rate: Probability of mutating a program.
        crossover_rate: Probability of crossover between programs.
        elite_fraction: Fraction of top performers to keep each generation.
        scoring: Scoring metric.
        cv: Cross-validation folds.
        estimator: sklearn estimator for evaluation.
        random_state: Random seed.

    Example:
        >>> search = EvolutionarySearch(
        ...     population_size=50, n_generations=20, scoring="accuracy"
        ... )
        >>> search.fit(X, y)
        >>> top_features = search.best_features_
    """

    def __init__(
        self,
        grammar: TransformGrammar | None = None,
        population_size: int = 50,
        n_generations: int = 10,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.5,
        elite_fraction: float = 0.2,
        scoring: str | None = None,
        cv: int = 3,
        estimator: Any = None,
        random_state: int | None = None,
    ) -> None:
        self.grammar = grammar or TransformGrammar()
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_fraction = elite_fraction
        self.scoring = scoring
        self.cv = cv
        self.estimator = estimator
        self.random_state = random_state
        self.results_: list[SearchResult] = []
        self.best_features_: list[SearchResult] = []
        self.history_: list[dict[str, float]] = []
        self._is_fitted = False
        self._early_stopping_patience: int | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> EvolutionarySearch:
        """Run evolutionary search.

        Args:
            X: Input features.
            y: Target variable.

        Returns:
            Self with results populated.
        """
        rng = np.random.RandomState(self.random_state)
        estimator = self._get_estimator(y)
        scoring = self.scoring or self._infer_scoring(y)
        numeric_cols: list[str] = list(X.select_dtypes(include=[np.number]).columns)

        if not numeric_cols:
            raise ValidationError("No numeric columns found for feature search.")

        # Initialize population
        population = self.grammar.sample_programs(
            X, n=self.population_size, rng=rng
        )
        all_results: list[SearchResult] = []

        # Set up early stopping if configured
        early_stop_monitor = None
        if self._early_stopping_patience is not None:
            from forge.search.pareto import EarlyStoppingMonitor
            early_stop_monitor = EarlyStoppingMonitor(
                patience=self._early_stopping_patience
            )

        for gen in range(self.n_generations):
            scored: list[tuple[list[TransformNode], float]] = []
            for prog in population:
                try:
                    feature = self.grammar.evaluate_program(prog, X)
                    if feature.isna().all() or feature.std() == 0:
                        scored.append((prog, float("-inf")))
                        continue
                    score = self._evaluate_feature(
                        X, feature, y, estimator, scoring
                    )
                    scored.append((prog, score))
                    name = self.grammar.program_name(prog)
                    all_results.append(
                        SearchResult(program=prog, name=name, score=score, generation=gen)
                    )
                except Exception:
                    scored.append((prog, float("-inf")))

            scored.sort(key=lambda x: x[1], reverse=True)
            valid_scores = [s for _, s in scored if s > float("-inf")]
            self.history_.append({
                "generation": gen,
                "best_score": valid_scores[0] if valid_scores else 0.0,
                "mean_score": float(np.mean(valid_scores)) if valid_scores else 0.0,
            })

            logger.info(
                "Generation %d: best=%.4f, mean=%.4f",
                gen,
                self.history_[-1]["best_score"],
                self.history_[-1]["mean_score"],
            )

            # Check early stopping
            if early_stop_monitor is not None:
                best = self.history_[-1]["best_score"]
                if early_stop_monitor.should_stop(best):
                    logger.info("Early stopping at generation %d", gen)
                    break

            # Selection: keep elites
            n_elite = max(2, int(self.population_size * self.elite_fraction))
            elites = [prog for prog, _ in scored[:n_elite]]

            # Build next generation
            new_population = list(elites)
            while len(new_population) < self.population_size:
                if rng.random() < self.crossover_rate and len(elites) >= 2:
                    p1, p2 = [
                        elites[rng.randint(len(elites))] for _ in range(2)
                    ]
                    child = self._crossover(p1, p2, rng)
                else:
                    parent = elites[rng.randint(len(elites))]
                    child = list(parent)

                if rng.random() < self.mutation_rate:
                    child = self._mutate(child, numeric_cols, rng)
                new_population.append(child)

            population = new_population

        all_results.sort(key=lambda r: r.score, reverse=True)
        # Deduplicate by name
        seen: set[str] = set()
        unique: list[SearchResult] = []
        for r in all_results:
            if r.name not in seen:
                seen.add(r.name)
                unique.append(r)
        self.results_ = unique
        self.best_features_ = unique[: max(10, len(unique) // 5)]
        self._is_fitted = True
        return self

    def _crossover(
        self,
        p1: list[TransformNode],
        p2: list[TransformNode],
        rng: np.random.RandomState,
    ) -> list[TransformNode]:
        """Single-point crossover between two programs."""
        if len(p1) <= 1 and len(p2) <= 1:
            return list(p1) if rng.random() < 0.5 else list(p2)
        cut1 = rng.randint(0, max(len(p1), 1))
        cut2 = rng.randint(0, max(len(p2), 1))
        return list(p1[:cut1]) + list(p2[cut2:]) or list(p1)

    def _mutate(
        self,
        program: list[TransformNode],
        columns: list[str],
        rng: np.random.RandomState,
    ) -> list[TransformNode]:
        """Mutate a program by changing an operation or column."""
        if not program:
            return program
        idx = rng.randint(len(program))
        node = program[idx]

        if rng.random() < 0.5 and node.columns != ["_prev"]:
            # Mutate column reference
            new_cols = list(rng.choice(columns, size=len(node.columns), replace=False))
            program[idx] = TransformNode(op=node.op, columns=new_cols, params=node.params)
        else:
            # Mutate operation
            if node.is_unary or node.columns == ["_prev"]:
                ops = self.grammar.unary_ops
            else:
                ops = self.grammar.binary_ops
            new_op = ops[rng.randint(len(ops))]
            program[idx] = TransformNode(op=new_op, columns=node.columns, params=node.params)

        return program

    def _evaluate_feature(
        self, X: pd.DataFrame, feature: pd.Series, y: pd.Series,
        estimator: Any, scoring: str,
    ) -> float:
        X_eval = X.copy()
        X_eval["_candidate"] = feature
        X_eval = X_eval.select_dtypes(include=[np.number]).fillna(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                scores = cross_val_score(
                    estimator, X_eval, y, cv=self.cv, scoring=scoring
                )
                return float(np.mean(scores))
            except Exception:
                return float("-inf")

    def _get_estimator(self, y: pd.Series) -> Any:
        if self.estimator is not None:
            return self.estimator
        if y.nunique() <= 20:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=30, max_depth=5, random_state=0)
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=30, max_depth=5, random_state=0)

    def _infer_scoring(self, y: pd.Series) -> str:
        if y.nunique() <= 20:
            return "accuracy"
        return "neg_mean_squared_error"


class AutoFeatureSearch(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """sklearn-compatible transformer that uses search to discover features.

    Wraps search algorithms and produces a transformer that generates
    the discovered features. Supports Pareto-optimal feature set selection
    to balance performance vs complexity.

    Args:
        method: Search method ("random", "evolutionary", "bayesian",
            or "multi_fidelity").
        n_features: Maximum number of discovered features to keep.
        n_candidates: Number of candidates to evaluate (random/multi-fidelity).
        population_size: Population size (evolutionary search).
        n_generations: Number of generations (evolutionary search).
        n_iterations: Number of iterations (Bayesian search).
        scoring: Scoring metric.
        cv: Cross-validation folds.
        pareto_select: If True, use Pareto-optimal knee-point selection
            to balance performance vs feature count.
        early_stopping_patience: Generations without improvement before
            stopping. None to disable early stopping.
        random_state: Random seed.

    Example:
        >>> search = AutoFeatureSearch(method="evolutionary", n_features=20)
        >>> X_new = search.fit_transform(X, y)
        >>> # Pareto-optimal selection
        >>> search = AutoFeatureSearch(pareto_select=True, n_features=30)
        >>> X_new = search.fit_transform(X, y)
    """

    def __init__(
        self,
        method: str = "random",
        n_features: int = 20,
        n_candidates: int = 100,
        population_size: int = 50,
        n_generations: int = 10,
        n_iterations: int = 50,
        scoring: str | None = None,
        cv: int = 3,
        pareto_select: bool = False,
        early_stopping_patience: int | None = None,
        random_state: int | None = None,
    ) -> None:
        self.method = method
        self.n_features = n_features
        self.n_candidates = n_candidates
        self.population_size = population_size
        self.n_generations = n_generations
        self.n_iterations = n_iterations
        self.scoring = scoring
        self.cv = cv
        self.pareto_select = pareto_select
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self._is_fitted = False
        self._selected_programs: list[SearchResult] = []
        self._feature_names_out: list[str] = []
        self._grammar = TransformGrammar()
        self.pareto_frontier_: list[Any] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Discover features using the selected search method.

        Args:
            X: Input features.
            y: Target variable (required).

        Returns:
            Self for method chaining.
        """
        if y is None:
            raise ValidationError("Target variable y is required for AutoFeatureSearch.")

        if not isinstance(X, pd.DataFrame):
            raise ValidationError(f"Expected DataFrame, got {type(X).__name__}")

        all_results = self._run_search(X, y)

        # Apply Pareto selection or simple top-N
        if self.pareto_select and all_results:
            from forge.search.pareto import pareto_optimal_sets, select_knee_point

            frontier = pareto_optimal_sets(
                all_results,
                max_features=self.n_features,
            )
            self.pareto_frontier_ = frontier
            knee = select_knee_point(frontier)
            if knee is not None:
                self._selected_programs = knee.features
            else:
                self._selected_programs = all_results[: self.n_features]
        else:
            self._selected_programs = all_results[: self.n_features]

        self._feature_names_out = [r.name for r in self._selected_programs]
        self._is_fitted = True
        return self

    def _run_search(self, X: pd.DataFrame, y: pd.Series) -> list[SearchResult]:
        """Execute the configured search method and return results."""
        if self.method == "random":
            searcher = RandomSearch(
                grammar=self._grammar,
                n_candidates=self.n_candidates,
                scoring=self.scoring,
                cv=self.cv,
                random_state=self.random_state,
            )
            searcher.fit(X, y)
            return searcher.best_features_

        if self.method == "evolutionary":
            searcher_evo = EvolutionarySearch(
                grammar=self._grammar,
                population_size=self.population_size,
                n_generations=self.n_generations,
                scoring=self.scoring,
                cv=self.cv,
                random_state=self.random_state,
            )
            if self.early_stopping_patience is not None:
                searcher_evo._early_stopping_patience = self.early_stopping_patience
            searcher_evo.fit(X, y)
            return searcher_evo.best_features_

        if self.method == "bayesian":
            from forge.search.bayesian import BayesianFeatureSearch

            searcher_bay = BayesianFeatureSearch(
                grammar=self._grammar,
                n_iterations=self.n_iterations,
                scoring=self.scoring,
                cv=self.cv,
                random_state=self.random_state,
            )
            searcher_bay.fit(X, y)
            return searcher_bay.best_features_

        if self.method == "multi_fidelity":
            from forge.search.bayesian import MultiFidelitySearch

            searcher_mf = MultiFidelitySearch(
                grammar=self._grammar,
                n_candidates=self.n_candidates,
                scoring=self.scoring,
                random_state=self.random_state,
            )
            searcher_mf.fit(X, y)
            return searcher_mf.best_features_

        raise ConfigurationError(
            f"Unknown search method: {self.method}. "
            "Use 'random', 'evolutionary', 'bayesian', or 'multi_fidelity'."
        )

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
        for sr in self._selected_programs:
            try:
                feature = self._grammar.evaluate_program(sr.program, X)
                result[sr.name] = feature
            except Exception:
                result[sr.name] = pd.Series(np.nan, index=X.index)

        return pd.DataFrame(result, index=X.index)

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> list[str]:
        """Get names of discovered features."""
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)
        return self._feature_names_out.copy()

    def get_search_results(self) -> list[SearchResult]:
        """Get all search results sorted by score."""
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)
        return list(self._selected_programs)
