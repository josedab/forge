"""AutoML feature selection with Bayesian optimization."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from forge.exceptions import ConfigurationError, NotFittedError, ValidationError
from forge.selectors.base import BaseFeatureSelector

if TYPE_CHECKING:
    from typing_extensions import Self
    from numpy.typing import NDArray


@dataclass
class OptimizationResult:
    """Result from a single optimization iteration."""

    features: list[str]
    score: float
    n_features: int
    iteration: int


@dataclass
class BayesianOptimizationState:
    """State for Bayesian optimization."""

    evaluated_subsets: list[tuple[frozenset[str], float]] = field(default_factory=list)
    best_subset: frozenset[str] | None = None
    best_score: float = float("-inf")
    acquisition_values: dict[frozenset[str], float] = field(default_factory=dict)


class BayesianFeatureSelector(BaseFeatureSelector):
    """Feature selector using Bayesian optimization.

    Uses Bayesian optimization to efficiently search for the optimal
    feature subset by balancing exploration and exploitation.

    Example:
        >>> from forge.selectors import BayesianFeatureSelector
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> selector = BayesianFeatureSelector(
        ...     estimator=RandomForestClassifier(),
        ...     n_iterations=50,
        ...     scoring="accuracy",
        ... )
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        estimator: Any = None,
        n_iterations: int = 50,
        min_features: int = 1,
        max_features: int | float | None = None,
        scoring: str | Callable[..., float] = "accuracy",
        cv: int = 5,
        exploration_weight: float = 1.0,
        n_initial_random: int = 10,
        random_state: int | None = None,
        n_jobs: int = 1,
        verbose: int = 0,
    ) -> None:
        """Initialize the Bayesian feature selector.

        Args:
            estimator: Scikit-learn estimator. If None, uses RandomForest.
            n_iterations: Number of optimization iterations.
            min_features: Minimum number of features to select.
            max_features: Maximum features. If float, fraction of total.
            scoring: Scoring metric for cross-validation.
            cv: Number of cross-validation folds.
            exploration_weight: UCB exploration parameter (higher = more exploration).
            n_initial_random: Initial random evaluations before optimization.
            random_state: Random seed for reproducibility.
            n_jobs: Number of parallel jobs for cross-validation.
            verbose: Verbosity level.
        """
        super().__init__()
        self.estimator = estimator
        self.n_iterations = n_iterations
        self.min_features = min_features
        self.max_features = max_features
        self.scoring = scoring
        self.cv = cv
        self.exploration_weight = exploration_weight
        self.n_initial_random = n_initial_random
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        self._optimization_history: list[OptimizationResult] = []
        self._state: BayesianOptimizationState | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the selector using Bayesian optimization.

        Args:
            X: Input features.
            y: Target variable (required).

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)
        if y is None:
            raise ValidationError("Target variable y is required for BayesianFeatureSelector")

        # Set up estimator
        if self.estimator is None:
            from sklearn.ensemble import RandomForestClassifier
            self._estimator = RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=self.n_jobs
            )
        else:
            from sklearn.base import clone
            self._estimator = clone(self.estimator)

        # Calculate max features
        n_total_features = X.shape[1]
        if self.max_features is None:
            self._max_features = n_total_features
        elif isinstance(self.max_features, float):
            self._max_features = max(1, int(n_total_features * self.max_features))
        else:
            self._max_features = min(self.max_features, n_total_features)

        # Set up random state
        self._rng = np.random.RandomState(self.random_state)

        # Initialize optimization state
        self._state = BayesianOptimizationState()
        self._feature_names_in = list(X.columns)
        all_features = set(self._feature_names_in)

        # Run optimization
        for iteration in range(self.n_iterations):
            if iteration < self.n_initial_random:
                # Random exploration phase
                subset = self._random_subset(all_features)
            else:
                # Bayesian optimization phase
                subset = self._select_next_subset(all_features)

            # Skip if already evaluated
            if frozenset(subset) in {s for s, _ in self._state.evaluated_subsets}:
                subset = self._random_subset(all_features)

            # Evaluate subset
            score = self._evaluate_subset(X, y, subset)
            self._state.evaluated_subsets.append((frozenset(subset), score))

            # Track best
            if score > self._state.best_score:
                self._state.best_score = score
                self._state.best_subset = frozenset(subset)

            # Record history
            self._optimization_history.append(OptimizationResult(
                features=list(subset),
                score=score,
                n_features=len(subset),
                iteration=iteration,
            ))

            if self.verbose > 0:
                print(f"Iteration {iteration + 1}/{self.n_iterations}: "
                      f"score={score:.4f}, n_features={len(subset)}, "
                      f"best={self._state.best_score:.4f}")

        # Set final support mask
        best_features = self._state.best_subset or frozenset(self._feature_names_in[:self.min_features])
        self._support_mask = np.array([
            col in best_features for col in self._feature_names_in
        ], dtype=bool)

        # Compute scores (based on inclusion in top subsets)
        feature_scores = {f: 0.0 for f in self._feature_names_in}
        for subset, score in self._state.evaluated_subsets:
            for f in subset:
                feature_scores[f] += score
        total = sum(feature_scores.values()) or 1.0
        self._scores = np.array([feature_scores[f] / total for f in self._feature_names_in])

        self._finalize_fit(X)
        return self

    def _random_subset(self, all_features: set[str]) -> list[str]:
        """Generate a random feature subset."""
        n_features = self._rng.randint(self.min_features, self._max_features + 1)
        return list(self._rng.choice(list(all_features), size=n_features, replace=False))

    def _select_next_subset(self, all_features: set[str]) -> list[str]:
        """Select next subset using UCB acquisition function."""
        # Build feature importance from evaluated subsets
        feature_scores: dict[str, list[float]] = {f: [] for f in all_features}

        for subset, score in self._state.evaluated_subsets:
            for f in subset:
                feature_scores[f].append(score)

        # Compute mean and uncertainty for each feature
        feature_stats = {}
        for f, scores in feature_scores.items():
            if scores:
                feature_stats[f] = (np.mean(scores), np.std(scores) + 0.1)
            else:
                feature_stats[f] = (0.0, 1.0)  # High uncertainty for unseen

        # UCB-based feature selection
        ucb_scores = {
            f: mean + self.exploration_weight * std
            for f, (mean, std) in feature_stats.items()
        }

        # Select features with highest UCB scores
        n_features = self._rng.randint(self.min_features, self._max_features + 1)
        sorted_features = sorted(ucb_scores.keys(), key=lambda f: ucb_scores[f], reverse=True)

        # Add some randomness to avoid getting stuck
        selected = sorted_features[:n_features]
        if self._rng.random() < 0.2:  # 20% chance of random perturbation
            n_swap = max(1, n_features // 4)
            remaining = [f for f in all_features if f not in selected]
            if remaining:
                to_remove = self._rng.choice(selected, size=min(n_swap, len(selected)), replace=False)
                to_add = self._rng.choice(remaining, size=min(n_swap, len(remaining)), replace=False)
                selected = [f for f in selected if f not in to_remove] + list(to_add)

        return selected

    def _evaluate_subset(self, X: pd.DataFrame, y: pd.Series, subset: list[str]) -> float:
        """Evaluate a feature subset using cross-validation."""
        X_subset = X[subset]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                scores = cross_val_score(
                    self._estimator, X_subset, y,
                    cv=self.cv, scoring=self.scoring, n_jobs=self.n_jobs
                )
                return float(np.mean(scores))
            except Exception:
                return float("-inf")

    def get_optimization_history(self) -> pd.DataFrame:
        """Get the optimization history.

        Returns:
            DataFrame with iteration results.
        """
        self._check_is_fitted()
        return pd.DataFrame([
            {
                "iteration": r.iteration,
                "score": r.score,
                "n_features": r.n_features,
                "features": r.features,
            }
            for r in self._optimization_history
        ])

    def get_best_score(self) -> float:
        """Get the best score achieved.

        Returns:
            Best cross-validation score.
        """
        self._check_is_fitted()
        return self._state.best_score if self._state else 0.0


class SequentialFeatureSelector(BaseFeatureSelector):
    """Sequential forward/backward feature selection.

    Implements SFS (Sequential Forward Selection) and SBS (Sequential
    Backward Selection) with optional floating variants.

    Example:
        >>> selector = SequentialFeatureSelector(
        ...     estimator=LogisticRegression(),
        ...     direction="forward",
        ...     n_features_to_select=10,
        ... )
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        estimator: Any = None,
        direction: Literal["forward", "backward"] = "forward",
        n_features_to_select: int | float = 0.5,
        floating: bool = False,
        scoring: str | Callable[..., float] = "accuracy",
        cv: int = 5,
        n_jobs: int = 1,
        verbose: int = 0,
    ) -> None:
        """Initialize sequential feature selector.

        Args:
            estimator: Scikit-learn estimator.
            direction: "forward" or "backward" selection.
            n_features_to_select: Number or fraction of features.
            floating: Whether to use floating variant (SFFS/SBFS).
            scoring: Scoring metric.
            cv: Cross-validation folds.
            n_jobs: Parallel jobs.
            verbose: Verbosity level.
        """
        super().__init__()
        self.estimator = estimator
        self.direction = direction
        self.n_features_to_select = n_features_to_select
        self.floating = floating
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose

        self._selection_history: list[tuple[str, float, str]] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the sequential selector.

        Args:
            X: Input features.
            y: Target variable (required).

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)
        if y is None:
            raise ValidationError("Target variable y is required")

        # Set up estimator
        if self.estimator is None:
            from sklearn.ensemble import RandomForestClassifier
            self._estimator = RandomForestClassifier(n_estimators=100, n_jobs=self.n_jobs)
        else:
            from sklearn.base import clone
            self._estimator = clone(self.estimator)

        self._feature_names_in = list(X.columns)
        n_total = len(self._feature_names_in)

        # Calculate target number of features
        if isinstance(self.n_features_to_select, float):
            n_target = max(1, int(n_total * self.n_features_to_select))
        else:
            n_target = min(self.n_features_to_select, n_total)

        # Initialize selected set
        if self.direction == "forward":
            selected: set[str] = set()
            remaining: set[str] = set(self._feature_names_in)
        else:
            selected = set(self._feature_names_in)
            remaining = set()

        # Run selection
        while len(selected) != n_target:
            if self.direction == "forward":
                if not remaining:
                    break
                best_feature, best_score = self._find_best_feature_to_add(
                    X, y, selected, remaining
                )
                selected.add(best_feature)
                remaining.remove(best_feature)
                self._selection_history.append((best_feature, best_score, "add"))

                if self.floating and len(selected) > 1:
                    # Try removing features
                    self._floating_removal(X, y, selected, remaining)
            else:
                if len(selected) <= 1:
                    break
                worst_feature, score = self._find_worst_feature_to_remove(X, y, selected)
                selected.remove(worst_feature)
                remaining.add(worst_feature)
                self._selection_history.append((worst_feature, score, "remove"))

                if self.floating and len(remaining) > 0:
                    # Try adding features
                    self._floating_addition(X, y, selected, remaining)

            if self.verbose > 0:
                print(f"Selected features: {len(selected)}/{n_target}, "
                      f"last action: {self._selection_history[-1]}")

        # Set support mask
        self._support_mask = np.array([
            col in selected for col in self._feature_names_in
        ], dtype=bool)

        # Compute scores based on selection order
        self._scores = np.zeros(n_total)
        for i, col in enumerate(self._feature_names_in):
            if col in selected:
                # Earlier selected = higher score
                for j, (feat, score, _) in enumerate(self._selection_history):
                    if feat == col:
                        self._scores[i] = (len(self._selection_history) - j) / len(self._selection_history)
                        break
                else:
                    self._scores[i] = 0.5

        self._finalize_fit(X)
        return self

    def _find_best_feature_to_add(
        self, X: pd.DataFrame, y: pd.Series, selected: set[str], remaining: set[str]
    ) -> tuple[str, float]:
        """Find the best feature to add."""
        best_feature = None
        best_score = float("-inf")

        for feature in remaining:
            subset = list(selected | {feature})
            score = self._evaluate_subset(X, y, subset)
            if score > best_score:
                best_score = score
                best_feature = feature

        return best_feature, best_score

    def _find_worst_feature_to_remove(
        self, X: pd.DataFrame, y: pd.Series, selected: set[str]
    ) -> tuple[str, float]:
        """Find the worst feature to remove."""
        worst_feature = None
        best_score = float("-inf")  # Score without the feature

        for feature in selected:
            subset = list(selected - {feature})
            if not subset:
                continue
            score = self._evaluate_subset(X, y, subset)
            if score > best_score:
                best_score = score
                worst_feature = feature

        return worst_feature, best_score

    def _floating_removal(
        self, X: pd.DataFrame, y: pd.Series, selected: set[str], remaining: set[str]
    ) -> None:
        """Try removing features in floating mode."""
        improved = True
        while improved and len(selected) > 1:
            improved = False
            current_score = self._evaluate_subset(X, y, list(selected))

            for feature in list(selected):
                subset = list(selected - {feature})
                if not subset:
                    continue
                score = self._evaluate_subset(X, y, subset)
                if score > current_score:
                    selected.remove(feature)
                    remaining.add(feature)
                    self._selection_history.append((feature, score, "float_remove"))
                    improved = True
                    break

    def _floating_addition(
        self, X: pd.DataFrame, y: pd.Series, selected: set[str], remaining: set[str]
    ) -> None:
        """Try adding features in floating mode."""
        improved = True
        while improved and remaining:
            improved = False
            current_score = self._evaluate_subset(X, y, list(selected))

            for feature in list(remaining):
                subset = list(selected | {feature})
                score = self._evaluate_subset(X, y, subset)
                if score > current_score:
                    selected.add(feature)
                    remaining.remove(feature)
                    self._selection_history.append((feature, score, "float_add"))
                    improved = True
                    break

    def _evaluate_subset(self, X: pd.DataFrame, y: pd.Series, subset: list[str]) -> float:
        """Evaluate a feature subset."""
        if not subset:
            return float("-inf")

        X_subset = X[subset]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                scores = cross_val_score(
                    self._estimator, X_subset, y,
                    cv=self.cv, scoring=self.scoring, n_jobs=self.n_jobs
                )
                return float(np.mean(scores))
            except Exception:
                return float("-inf")

    def get_selection_history(self) -> list[tuple[str, float, str]]:
        """Get the selection history.

        Returns:
            List of (feature, score, action) tuples.
        """
        self._check_is_fitted()
        return self._selection_history.copy()


class GeneticFeatureSelector(BaseFeatureSelector):
    """Feature selection using genetic algorithms.

    Uses evolutionary optimization to find optimal feature subsets
    through selection, crossover, and mutation operations.

    Example:
        >>> selector = GeneticFeatureSelector(
        ...     estimator=XGBClassifier(),
        ...     population_size=50,
        ...     n_generations=100,
        ... )
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        estimator: Any = None,
        population_size: int = 50,
        n_generations: int = 100,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.1,
        tournament_size: int = 3,
        elitism: int = 2,
        min_features: int = 1,
        max_features: int | float | None = None,
        scoring: str | Callable[..., float] = "accuracy",
        cv: int = 5,
        random_state: int | None = None,
        n_jobs: int = 1,
        verbose: int = 0,
    ) -> None:
        """Initialize genetic feature selector.

        Args:
            estimator: Scikit-learn estimator.
            population_size: Size of population.
            n_generations: Number of generations.
            crossover_prob: Probability of crossover.
            mutation_prob: Probability of mutation per gene.
            tournament_size: Tournament selection size.
            elitism: Number of elite individuals to keep.
            min_features: Minimum features in individual.
            max_features: Maximum features (float = fraction).
            scoring: Scoring metric.
            cv: Cross-validation folds.
            random_state: Random seed.
            n_jobs: Parallel jobs.
            verbose: Verbosity level.
        """
        super().__init__()
        self.estimator = estimator
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.min_features = min_features
        self.max_features = max_features
        self.scoring = scoring
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        self._generation_history: list[dict[str, Any]] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit using genetic algorithm.

        Args:
            X: Input features.
            y: Target variable (required).

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)
        if y is None:
            raise ValidationError("Target variable y is required")

        # Setup
        if self.estimator is None:
            from sklearn.ensemble import RandomForestClassifier
            self._estimator = RandomForestClassifier(n_estimators=100, n_jobs=self.n_jobs)
        else:
            from sklearn.base import clone
            self._estimator = clone(self.estimator)

        self._rng = np.random.RandomState(self.random_state)
        self._feature_names_in = list(X.columns)
        n_features = len(self._feature_names_in)

        # Max features
        if self.max_features is None:
            max_feat = n_features
        elif isinstance(self.max_features, float):
            max_feat = max(1, int(n_features * self.max_features))
        else:
            max_feat = min(self.max_features, n_features)

        # Initialize population (each individual is a binary mask)
        population = self._initialize_population(n_features, max_feat)
        fitness_cache: dict[tuple[bool, ...], float] = {}

        best_individual = None
        best_fitness = float("-inf")

        for gen in range(self.n_generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                key = tuple(individual)
                if key not in fitness_cache:
                    fitness_cache[key] = self._evaluate_individual(X, y, individual)
                fitness_scores.append(fitness_cache[key])

            fitness_scores = np.array(fitness_scores)

            # Track best
            gen_best_idx = np.argmax(fitness_scores)
            if fitness_scores[gen_best_idx] > best_fitness:
                best_fitness = fitness_scores[gen_best_idx]
                best_individual = population[gen_best_idx].copy()

            # Record history
            self._generation_history.append({
                "generation": gen,
                "best_fitness": best_fitness,
                "mean_fitness": float(np.mean(fitness_scores)),
                "n_features_best": int(np.sum(best_individual)) if best_individual is not None else 0,
            })

            if self.verbose > 0:
                print(f"Generation {gen + 1}/{self.n_generations}: "
                      f"best={best_fitness:.4f}, mean={np.mean(fitness_scores):.4f}")

            # Create next generation
            new_population = []

            # Elitism
            elite_indices = np.argsort(fitness_scores)[-self.elitism:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())

            # Fill rest with offspring
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_select(population, fitness_scores)
                parent2 = self._tournament_select(population, fitness_scores)

                # Crossover
                if self._rng.random() < self.crossover_prob:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Mutation
                child1 = self._mutate(child1, max_feat)
                child2 = self._mutate(child2, max_feat)

                new_population.extend([child1, child2])

            population = new_population[:self.population_size]

        # Set support mask from best individual
        if best_individual is None:
            best_individual = np.ones(n_features, dtype=bool)

        self._support_mask = best_individual

        # Scores based on frequency in final population
        feature_freq = np.zeros(n_features)
        for individual in population:
            feature_freq += individual
        self._scores = feature_freq / len(population)

        self._finalize_fit(X)
        return self

    def _initialize_population(self, n_features: int, max_feat: int) -> list[np.ndarray]:
        """Initialize random population."""
        population = []
        for _ in range(self.population_size):
            n_select = self._rng.randint(self.min_features, max_feat + 1)
            individual = np.zeros(n_features, dtype=bool)
            selected_idx = self._rng.choice(n_features, size=n_select, replace=False)
            individual[selected_idx] = True
            population.append(individual)
        return population

    def _evaluate_individual(self, X: pd.DataFrame, y: pd.Series, individual: np.ndarray) -> float:
        """Evaluate fitness of an individual."""
        selected_features = [f for f, s in zip(self._feature_names_in, individual) if s]
        if not selected_features:
            return float("-inf")

        X_subset = X[selected_features]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                scores = cross_val_score(
                    self._estimator, X_subset, y,
                    cv=self.cv, scoring=self.scoring, n_jobs=self.n_jobs
                )
                return float(np.mean(scores))
            except Exception:
                return float("-inf")

    def _tournament_select(
        self, population: list[np.ndarray], fitness_scores: np.ndarray
    ) -> np.ndarray:
        """Select individual via tournament."""
        indices = self._rng.choice(len(population), size=self.tournament_size, replace=False)
        best_idx = indices[np.argmax(fitness_scores[indices])]
        return population[best_idx]

    def _crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform uniform crossover."""
        mask = self._rng.random(len(parent1)) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2

    def _mutate(self, individual: np.ndarray, max_feat: int) -> np.ndarray:
        """Mutate an individual."""
        for i in range(len(individual)):
            if self._rng.random() < self.mutation_prob:
                individual[i] = not individual[i]

        # Ensure constraints
        n_selected = np.sum(individual)
        if n_selected < self.min_features:
            # Add random features
            false_idx = np.where(~individual)[0]
            n_add = self.min_features - n_selected
            to_add = self._rng.choice(false_idx, size=int(min(n_add, len(false_idx))), replace=False)
            individual[to_add] = True
        elif n_selected > max_feat:
            # Remove random features
            true_idx = np.where(individual)[0]
            n_remove = n_selected - max_feat
            to_remove = self._rng.choice(true_idx, size=int(n_remove), replace=False)
            individual[to_remove] = False

        return individual

    def get_generation_history(self) -> pd.DataFrame:
        """Get the generation history.

        Returns:
            DataFrame with generation statistics.
        """
        self._check_is_fitted()
        return pd.DataFrame(self._generation_history)


def auto_select_features(
    X: pd.DataFrame,
    y: pd.Series,
    method: Literal["bayesian", "sequential", "genetic"] = "bayesian",
    n_features: int | float | None = None,
    estimator: Any = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, BaseFeatureSelector]:
    """Convenience function for automatic feature selection.

    Args:
        X: Input features.
        y: Target variable.
        method: Selection method.
        n_features: Target number of features.
        estimator: Estimator for evaluation.
        **kwargs: Additional arguments for selector.

    Returns:
        Tuple of (selected features DataFrame, fitted selector).
    """
    if method == "bayesian":
        selector = BayesianFeatureSelector(estimator=estimator, **kwargs)
    elif method == "sequential":
        if n_features is not None:
            kwargs["n_features_to_select"] = n_features
        selector = SequentialFeatureSelector(estimator=estimator, **kwargs)
    elif method == "genetic":
        selector = GeneticFeatureSelector(estimator=estimator, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

    X_selected = selector.fit_transform(X, y)
    return X_selected, selector
