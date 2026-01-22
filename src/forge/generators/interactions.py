"""Automatic feature interaction discovery."""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from forge.exceptions import NotFittedError, ValidationError
from forge.generators.base import BaseFeatureGenerator

if TYPE_CHECKING:
    from typing_extensions import Self


@dataclass
class InteractionCandidate:
    """A candidate feature interaction."""

    name: str
    features: tuple[str, ...]
    operation: str
    score: float
    description: str = ""


class InteractionDiscoverer(BaseFeatureGenerator):
    """Discovers and generates valuable feature interactions.

    Automatically finds interactions between features that improve
    predictive power, using statistical tests and model-based
    importance scoring.

    Example:
        >>> discoverer = InteractionDiscoverer(
        ...     max_interactions=50,
        ...     interaction_types=["multiply", "divide", "add"],
        ... )
        >>> X_with_interactions = discoverer.fit_transform(X, y)
    """

    INTERACTION_OPERATIONS = {
        "multiply": lambda a, b: a * b,
        "divide": lambda a, b: np.divide(a, b, where=b != 0, out=np.zeros_like(a)),
        "add": lambda a, b: a + b,
        "subtract": lambda a, b: a - b,
        "ratio_diff": lambda a, b: (a - b) / (np.abs(a) + np.abs(b) + 1e-8),
        "geometric_mean": lambda a, b: np.sqrt(np.abs(a * b)) * np.sign(a * b),
        "harmonic_mean": lambda a, b: 2 * a * b / (a + b + 1e-8),
        "max": lambda a, b: np.maximum(a, b),
        "min": lambda a, b: np.minimum(a, b),
    }

    def __init__(
        self,
        max_interactions: int = 50,
        interaction_types: list[str] | None = None,
        min_score_threshold: float = 0.01,
        scoring_method: Literal["mutual_info", "correlation", "tree"] = "tree",
        max_candidates: int = 500,
        include_polynomial: bool = False,
        polynomial_degree: int = 2,
        columns: list[str] | None = None,
        task: str = "auto",
        include_original: bool = True,
        n_jobs: int = -1,
        random_state: int | None = 42,
        verbose: int = 0,
    ) -> None:
        """Initialize interaction discoverer.

        Args:
            max_interactions: Maximum number of interactions to keep.
            interaction_types: Types of interactions to try.
            min_score_threshold: Minimum score to keep interaction.
            scoring_method: Method to score interactions.
            max_candidates: Maximum candidates to evaluate.
            include_polynomial: Include polynomial features.
            polynomial_degree: Degree for polynomial features.
            columns: Columns to use. None for all numeric.
            task: Task type ("classification", "regression", "auto").
            include_original: Include original columns in output.
            n_jobs: Parallel jobs.
            random_state: Random seed.
            verbose: Verbosity level.
        """
        super().__init__()
        self.max_interactions = max_interactions
        self.interaction_types = interaction_types or ["multiply", "divide", "add"]
        self.min_score_threshold = min_score_threshold
        self.scoring_method = scoring_method
        self.max_candidates = max_candidates
        self.include_polynomial = include_polynomial
        self.polynomial_degree = polynomial_degree
        self.columns = columns
        self.task = task
        self.include_original = include_original
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        self._selected_interactions: list[InteractionCandidate] = []
        self._numeric_cols: list[str] = []
        self._task: str = "classification"

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the interaction discoverer.

        Args:
            X: Input features.
            y: Target variable (required for scoring).

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)
        if y is None:
            raise ValidationError("Target variable y is required for interaction discovery")

        # Get numeric columns
        if self.columns is not None:
            self._numeric_cols = [c for c in self.columns if c in X.columns]
        else:
            self._numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if len(self._numeric_cols) < 2:
            raise ValidationError("Need at least 2 numeric columns for interactions")

        self._input_columns = list(X.columns)

        # Determine task
        if self.task == "auto":
            unique_vals = len(np.unique(y))
            self._task = "classification" if unique_vals <= 20 else "regression"
        else:
            self._task = self.task

        # Generate candidates
        candidates = self._generate_candidates(X, y)

        # Score candidates
        scored_candidates = self._score_candidates(X, y, candidates)

        # Select top interactions
        scored_candidates.sort(key=lambda x: x.score, reverse=True)
        self._selected_interactions = scored_candidates[:self.max_interactions]

        # Build feature names
        names = []
        if self.include_original:
            names.extend(X.columns.tolist())
        names.extend([c.name for c in self._selected_interactions])
        self._feature_names_out = names

        if self.verbose > 0:
            print(f"Selected {len(self._selected_interactions)} interactions")

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction features.

        Args:
            X: Input DataFrame.

        Returns:
            DataFrame with interaction features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result = X.copy() if self.include_original else pd.DataFrame(index=X.index)

        # Generate interactions
        for interaction in self._selected_interactions:
            features = interaction.features
            op = interaction.operation

            if len(features) == 2 and op in self.INTERACTION_OPERATIONS:
                a = X[features[0]].values
                b = X[features[1]].values
                result[interaction.name] = self.INTERACTION_OPERATIONS[op](a, b)
            elif op == "polynomial":
                # Handle polynomial terms
                vals = X[list(features)].values
                result[interaction.name] = np.prod(vals, axis=1)

        return result

    def _generate_candidates(
        self, X: pd.DataFrame, y: pd.Series
    ) -> list[InteractionCandidate]:
        """Generate candidate interactions."""
        candidates = []

        # Pairwise interactions
        pairs = list(itertools.combinations(self._numeric_cols, 2))

        # Limit pairs if too many
        if len(pairs) > self.max_candidates // len(self.interaction_types):
            # Prioritize based on correlation with target
            correlations = {}
            for col in self._numeric_cols:
                corr = abs(X[col].corr(y))
                correlations[col] = corr if not np.isnan(corr) else 0

            # Sort pairs by combined correlation
            pairs.sort(
                key=lambda p: correlations.get(p[0], 0) + correlations.get(p[1], 0),
                reverse=True
            )
            pairs = pairs[:self.max_candidates // len(self.interaction_types)]

        for col1, col2 in pairs:
            for op in self.interaction_types:
                if op not in self.INTERACTION_OPERATIONS:
                    continue

                name = f"{col1}_{op}_{col2}"
                candidates.append(InteractionCandidate(
                    name=name,
                    features=(col1, col2),
                    operation=op,
                    score=0.0,
                    description=f"{op} of {col1} and {col2}",
                ))

                if len(candidates) >= self.max_candidates:
                    break
            if len(candidates) >= self.max_candidates:
                break

        # Add polynomial features if enabled
        if self.include_polynomial and len(candidates) < self.max_candidates:
            for r in range(2, min(self.polynomial_degree + 1, 4)):
                for combo in itertools.combinations_with_replacement(self._numeric_cols[:10], r):
                    if len(candidates) >= self.max_candidates:
                        break
                    name = "_x_".join(combo)
                    candidates.append(InteractionCandidate(
                        name=f"poly_{name}",
                        features=combo,
                        operation="polynomial",
                        score=0.0,
                        description=f"Polynomial term: {' * '.join(combo)}",
                    ))

        return candidates

    def _score_candidates(
        self, X: pd.DataFrame, y: pd.Series, candidates: list[InteractionCandidate]
    ) -> list[InteractionCandidate]:
        """Score candidate interactions."""
        scored = []

        # Build interaction values
        interaction_values = {}
        for candidate in candidates:
            try:
                if candidate.operation in self.INTERACTION_OPERATIONS:
                    a = X[candidate.features[0]].values
                    b = X[candidate.features[1]].values
                    values = self.INTERACTION_OPERATIONS[candidate.operation](a, b)
                else:
                    vals = X[list(candidate.features)].values
                    values = np.prod(vals, axis=1)

                # Handle inf/nan
                values = np.nan_to_num(values, nan=0, posinf=0, neginf=0)
                interaction_values[candidate.name] = values
            except Exception:
                continue

        if not interaction_values:
            return []

        # Score based on method
        if self.scoring_method == "correlation":
            scores = self._score_by_correlation(interaction_values, y)
        elif self.scoring_method == "mutual_info":
            scores = self._score_by_mutual_info(interaction_values, y)
        else:  # tree
            scores = self._score_by_tree(X, interaction_values, y)

        # Update candidates with scores
        for candidate in candidates:
            if candidate.name in scores:
                score = scores[candidate.name]
                if score >= self.min_score_threshold:
                    candidate.score = score
                    scored.append(candidate)

        return scored

    def _score_by_correlation(
        self, interaction_values: dict[str, np.ndarray], y: pd.Series
    ) -> dict[str, float]:
        """Score by correlation with target."""
        scores = {}
        y_arr = y.values

        for name, values in interaction_values.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr = np.abs(np.corrcoef(values, y_arr)[0, 1])
                scores[name] = corr if not np.isnan(corr) else 0.0

        return scores

    def _score_by_mutual_info(
        self, interaction_values: dict[str, np.ndarray], y: pd.Series
    ) -> dict[str, float]:
        """Score by mutual information with target."""
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

        # Build feature matrix
        names = list(interaction_values.keys())
        X_int = np.column_stack([interaction_values[n] for n in names])

        if self._task == "classification":
            mi = mutual_info_classif(X_int, y, random_state=self.random_state)
        else:
            mi = mutual_info_regression(X_int, y, random_state=self.random_state)

        # Normalize
        mi_max = mi.max() if mi.max() > 0 else 1.0
        return {name: float(mi[i] / mi_max) for i, name in enumerate(names)}

    def _score_by_tree(
        self, X: pd.DataFrame, interaction_values: dict[str, np.ndarray], y: pd.Series
    ) -> dict[str, float]:
        """Score by tree-based importance."""
        # Build combined feature matrix using pd.concat to avoid fragmentation
        interaction_df = pd.DataFrame(interaction_values, index=X.index)
        X_combined = pd.concat([X[self._numeric_cols], interaction_df], axis=1)

        X_clean = X_combined.fillna(0).values

        if self._task == "classification":
            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_clean, y)

        importances = model.feature_importances_
        feature_names = list(X_combined.columns)

        # Extract interaction scores
        scores = {}
        for name in interaction_values:
            if name in feature_names:
                idx = feature_names.index(name)
                scores[name] = float(importances[idx])

        return scores

    def get_interactions(self) -> pd.DataFrame:
        """Get discovered interactions.

        Returns:
            DataFrame with interaction details.
        """
        self._check_is_fitted()

        if not self._selected_interactions:
            return pd.DataFrame(columns=["name", "features", "operation", "score", "description"])

        return pd.DataFrame([
            {
                "name": i.name,
                "features": i.features,
                "operation": i.operation,
                "score": i.score,
                "description": i.description,
            }
            for i in self._selected_interactions
        ]).sort_values("score", ascending=False)


class PolynomialInteractionGenerator(BaseFeatureGenerator):
    """Generates polynomial and interaction features.

    Wrapper around sklearn's PolynomialFeatures with additional
    filtering and naming capabilities.

    Example:
        >>> generator = PolynomialInteractionGenerator(
        ...     degree=2,
        ...     interaction_only=True,
        ... )
        >>> X_poly = generator.fit_transform(X)
    """

    def __init__(
        self,
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = False,
        columns: list[str] | None = None,
        max_features: int | None = None,
        include_original: bool = True,
    ) -> None:
        """Initialize polynomial generator.

        Args:
            degree: Maximum polynomial degree.
            interaction_only: Only include interaction terms.
            include_bias: Include bias (constant) term.
            columns: Columns to use. None for all numeric.
            max_features: Maximum features to generate.
            include_original: Include original columns.
        """
        super().__init__()
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.columns = columns
        self.max_features = max_features
        self.include_original = include_original

        self._poly: PolynomialFeatures | None = None
        self._numeric_cols: list[str] = []
        self._poly_feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the polynomial generator.

        Args:
            X: Input features.
            y: Ignored.

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)
        self._input_columns = list(X.columns)

        # Get numeric columns
        if self.columns is not None:
            self._numeric_cols = [c for c in self.columns if c in X.columns]
        else:
            self._numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if not self._numeric_cols:
            raise ValidationError("No numeric columns found")

        # Fit polynomial features
        self._poly = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias,
        )

        X_numeric = X[self._numeric_cols].fillna(0)
        self._poly.fit(X_numeric)

        # Get feature names
        poly_names = self._poly.get_feature_names_out(self._numeric_cols)
        self._poly_feature_names = [
            name.replace(" ", "_").replace("^", "_pow_")
            for name in poly_names
        ]

        # Limit features if needed
        if self.max_features and len(self._poly_feature_names) > self.max_features:
            self._poly_feature_names = self._poly_feature_names[:self.max_features]

        # Build output names
        names = []
        if self.include_original:
            names.extend(X.columns.tolist())
        names.extend([
            f"poly_{name}" for name in self._poly_feature_names
            if name not in self._numeric_cols  # Skip original features
        ])
        self._feature_names_out = names

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate polynomial features.

        Args:
            X: Input DataFrame.

        Returns:
            DataFrame with polynomial features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result = X.copy() if self.include_original else pd.DataFrame(index=X.index)

        X_numeric = X[self._numeric_cols].fillna(0)
        poly_values = self._poly.transform(X_numeric)

        # Add polynomial features
        for i, name in enumerate(self._poly_feature_names):
            if name not in self._numeric_cols:  # Skip original
                col_name = f"poly_{name}"
                if i < poly_values.shape[1]:
                    result[col_name] = poly_values[:, i]

        return result


class GroupedInteractionGenerator(BaseFeatureGenerator):
    """Generates interactions within and between feature groups.

    Useful for domain-driven feature engineering where features
    are organized into logical groups.

    Example:
        >>> generator = GroupedInteractionGenerator(
        ...     groups={
        ...         "financial": ["income", "expenses", "debt"],
        ...         "demographic": ["age", "tenure"],
        ...     },
        ...     within_group=True,
        ...     between_groups=True,
        ... )
        >>> X_interactions = generator.fit_transform(X)
    """

    def __init__(
        self,
        groups: dict[str, list[str]],
        within_group: bool = True,
        between_groups: bool = True,
        operations: list[str] | None = None,
        max_per_group: int = 20,
        include_original: bool = True,
    ) -> None:
        """Initialize grouped interaction generator.

        Args:
            groups: Dict mapping group names to column lists.
            within_group: Generate within-group interactions.
            between_groups: Generate between-group interactions.
            operations: Interaction operations to use.
            max_per_group: Maximum interactions per group pair.
            include_original: Include original columns.
        """
        super().__init__()
        self.groups = groups
        self.within_group = within_group
        self.between_groups = between_groups
        self.operations = operations or ["multiply", "ratio"]
        self.max_per_group = max_per_group
        self.include_original = include_original

        self._interactions: list[tuple[str, str, str, str]] = []  # (col1, col2, op, name)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the generator.

        Args:
            X: Input features.
            y: Ignored.

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)
        self._input_columns = list(X.columns)

        # Validate groups
        validated_groups = {}
        for group_name, cols in self.groups.items():
            valid_cols = [c for c in cols if c in X.columns]
            if valid_cols:
                validated_groups[group_name] = valid_cols

        if not validated_groups:
            raise ValidationError("No valid columns found in any group")

        # Generate interaction specifications
        self._interactions = []

        # Within-group interactions
        if self.within_group:
            for group_name, cols in validated_groups.items():
                count = 0
                for col1, col2 in itertools.combinations(cols, 2):
                    for op in self.operations:
                        if count >= self.max_per_group:
                            break
                        name = f"{group_name}_{col1}_{op}_{col2}"
                        self._interactions.append((col1, col2, op, name))
                        count += 1

        # Between-group interactions
        if self.between_groups:
            group_names = list(validated_groups.keys())
            for g1, g2 in itertools.combinations(group_names, 2):
                count = 0
                for col1 in validated_groups[g1]:
                    for col2 in validated_groups[g2]:
                        for op in self.operations:
                            if count >= self.max_per_group:
                                break
                            name = f"{g1}_{col1}_{op}_{g2}_{col2}"
                            self._interactions.append((col1, col2, op, name))
                            count += 1
                        if count >= self.max_per_group:
                            break
                    if count >= self.max_per_group:
                        break

        # Build feature names
        names = []
        if self.include_original:
            names.extend(X.columns.tolist())
        names.extend([i[3] for i in self._interactions])
        self._feature_names_out = names

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate grouped interactions.

        Args:
            X: Input DataFrame.

        Returns:
            DataFrame with grouped interactions.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result = X.copy() if self.include_original else pd.DataFrame(index=X.index)

        operations = {
            "multiply": lambda a, b: a * b,
            "ratio": lambda a, b: a / (b + 1e-8),
            "add": lambda a, b: a + b,
            "subtract": lambda a, b: a - b,
            "diff_ratio": lambda a, b: (a - b) / (a + b + 1e-8),
        }

        for col1, col2, op, name in self._interactions:
            if col1 in X.columns and col2 in X.columns and op in operations:
                a = X[col1].values
                b = X[col2].values
                result[name] = operations[op](a, b)

        return result


def discover_interactions(
    X: pd.DataFrame,
    y: pd.Series,
    max_interactions: int = 50,
    **kwargs: Any,
) -> tuple[pd.DataFrame, InteractionDiscoverer]:
    """Convenience function for interaction discovery.

    Args:
        X: Input features.
        y: Target variable.
        max_interactions: Maximum interactions to keep.
        **kwargs: Additional arguments.

    Returns:
        Tuple of (features with interactions, fitted discoverer).
    """
    discoverer = InteractionDiscoverer(
        max_interactions=max_interactions,
        **kwargs,
    )
    X_with_interactions = discoverer.fit_transform(X, y)
    return X_with_interactions, discoverer
