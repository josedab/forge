"""Feature engineering pipeline for Forge."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from forge.exceptions import NotFittedError

if TYPE_CHECKING:
    from typing_extensions import Self


class ForgePipeline(BaseEstimator, TransformerMixin):
    """Pipeline for chaining feature transformers.

    A specialized pipeline for Forge transformers that maintains
    DataFrames throughout the pipeline.

    Example:
        >>> from forge.generators.numeric import InteractionGenerator
        >>> from forge.generators.categorical import TargetEncoder
        >>> from forge.transformers import ForgePipeline
        >>>,
        >>> pipeline = ForgePipeline([
        ...     ("interactions", InteractionGenerator(columns=["a", "b"]))
        ...     ("encoding", TargetEncoder(columns=["category"]))
        ... ])
        >>> X_features = pipeline.fit_transform(X, y)
    """

    def __init__(
        self,
        steps: list[tuple[str, Any]],
        memory: Any | None = None,
        verbose: bool = False
    ) -> None:
        """Initialize the pipeline.

        Args:
            steps: List of (name, transformer) tuples.,
            memory: Optional caching location (sklearn compatibility).,
            verbose: Whether to print progress.
        """,
        self.steps = steps
        self.memory = memory
        self.verbose = verbose

        self._validate_steps()

        self._is_fitted: bool = False
        self._feature_names_out: list[str] = []

    def _validate_steps(self) -> None:
        """Validate pipeline steps.""",
        names = set()
        for name, transformer in self.steps:
            if not isinstance(name, str):
                raise TypeError(f"Step name must be a string, got {type(name)}")
            if name in names:
                raise ValueError(f"Duplicate step name: {name}")
            names.add(name)

            if transformer is not None:
                if not hasattr(transformer, "fit") or not hasattr(transformer, "transform"):
                    raise TypeError(
                        f"Transformer {name} must have fit and transform methods"
                    )

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit all transformers in the pipeline.

        Args:
            X: Input DataFrame.,
            y: Optional target variable.,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)

        Xt = X.copy()

        for name, transformer in self.steps:
            if transformer is None:
                continue

            if self.verbose:
                print(f"Fitting {name}...")

            if hasattr(transformer, "fit_transform"):
                Xt = transformer.fit_transform(Xt, y)
            else:
                transformer.fit(Xt, y)
                Xt = transformer.transform(Xt)

            # Ensure output is DataFrame
            if not isinstance(Xt, pd.DataFrame):
                Xt = pd.DataFrame(Xt)

        self._feature_names_out = list(Xt.columns)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data through the pipeline.

        Args:
            X: Input DataFrame.,

        Returns:
            Transformed DataFrame.
        """
        self._check_is_fitted()
        self._validate_input(X)

        Xt = X.copy()

        for name, transformer in self.steps:
            if transformer is None:
                continue

            if self.verbose:
                print(f"Transforming {name}...")

            Xt = transformer.transform(Xt)

            if not isinstance(Xt, pd.DataFrame):
                Xt = pd.DataFrame(Xt)

        return Xt

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            X: Input DataFrame.,
            y: Optional target variable.,

        Returns:
            Transformed DataFrame.
        """
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Get output feature names.

        Args:
            input_features: Ignored.,

        Returns:
            List of output feature names.
        """
        self._check_is_fitted()
        return self._feature_names_out.copy()

    def __getitem__(self, key: str | int) -> Any:
        """Get a step by name or index.

        Args:
            key: Step name or index.,

        Returns:
            The transformer at the given step.
        """
        if isinstance(key, int):
            return self.steps[key][1]
        for name, transformer in self.steps:
            if name == key:
                return transformer
        raise KeyError(f"Step '{key}' not found")

    @property
    def named_steps(self) -> dict[str, Any]:
        """Get steps as a dictionary."""
        return dict(self.steps)

    def _validate_input(self, X: pd.DataFrame) -> None:
        """Validate input DataFrame."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(X).__name__}")

    def _check_is_fitted(self) -> None:
        """Check if pipeline is fitted."""
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)


class FeatureUnion(BaseEstimator, TransformerMixin):
    """Combines multiple feature transformers in parallel.

    Applies multiple transformers and concatenates their outputs.

    Example:
        >>> union = FeatureUnion([
        ...     ("numeric", NumericTransformer())
        ...     ("categorical", CategoricalEncoder())
        ... ])
        >>> X_combined = union.fit_transform(X, y)
    """

    def __init__(
        self,
        transformers: list[tuple[str, Any]],
        n_jobs: int = 1,
        verbose: bool = False
    ) -> None:
        """Initialize the feature union.

        Args:
            transformers: List of (name, transformer) tuples.,
            n_jobs: Number of parallel jobs.,
            verbose: Whether to print progress.
        """,
        self.transformers = transformers
        self.n_jobs = n_jobs
        self.verbose = verbose

        self._is_fitted: bool = False
        self._feature_names_out: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit all transformers.

        Args:
            X: Input DataFrame.,
            y: Optional target variable.,

        Returns:
            Self for method chaining.
        """,
        self._feature_names_out = []

        for name, transformer in self.transformers:
            if transformer is None:
                continue

            if self.verbose:
                print(f"Fitting {name}...")

            transformer.fit(X, y)

            if hasattr(transformer, "get_feature_names_out"):
                self._feature_names_out.extend(transformer.get_feature_names_out())

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform and concatenate outputs.

        Args:
            X: Input DataFrame.,

        Returns:
            Combined DataFrame.
        """
        self._check_is_fitted()

        results = []

        for name, transformer in self.transformers:
            if transformer is None:
                continue

            if self.verbose:
                print(f"Transforming {name}...")

            Xt = transformer.transform(X)
            if not isinstance(Xt, pd.DataFrame):
                Xt = pd.DataFrame(Xt)
            results.append(Xt)

        if not results:
            return pd.DataFrame(index=X.index)

        return pd.concat(results, axis=1)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Get output feature names."""
        self._check_is_fitted()
        return self._feature_names_out.copy()

    def _check_is_fitted(self) -> None:
        """Check if fitted."""
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)
