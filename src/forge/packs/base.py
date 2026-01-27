"""Base classes for domain-specific feature packs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from forge.exceptions import ConfigurationError, NotFittedError, ValidationError

if TYPE_CHECKING:
    from typing_extensions import Self


class FeaturePack(BaseEstimator, TransformerMixin, ABC):  # type: ignore[misc]
    """Abstract base class for domain-specific feature packs.

    Feature packs are sklearn-compatible transformers that generate
    domain-specific features from raw data. Each pack encapsulates
    domain knowledge as reusable feature engineering recipes.

    Subclasses must implement ``_generate_features()``.

    Args:
        features: List of feature names to generate. None for all.
        prefix: Prefix for generated feature names.
    """

    domain: str = "base"

    def __init__(
        self,
        features: list[str] | None = None,
        prefix: str | None = None,
    ) -> None:
        self.features = features
        self.prefix = prefix or f"{self.domain}_"
        self._is_fitted: bool = False
        self._feature_names_out: list[str] = []
        self._input_columns: list[str] = []

    @abstractmethod
    def _generate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate domain-specific features.

        Args:
            X: Input DataFrame.

        Returns:
            DataFrame with generated features.
        """

    @abstractmethod
    def available_features(self) -> list[str]:
        """Return list of all available feature names for this pack."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the feature pack to data.

        Args:
            X: Input DataFrame.
            y: Ignored.

        Returns:
            Self for method chaining.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValidationError(f"Expected DataFrame, got {type(X).__name__}")
        self._input_columns = list(X.columns)
        result = self._generate_features(X)
        self._feature_names_out = list(result.columns)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using the feature pack.

        Args:
            X: Input DataFrame.

        Returns:
            DataFrame with generated features.
        """
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)
        return self._generate_features(X)

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> list[str]:
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)
        return self._feature_names_out.copy()


class FeaturePackRegistry:
    """Registry for discovering and creating feature packs.

    Example:
        >>> registry = FeaturePackRegistry()
        >>> registry.register("finance", FinanceFeaturePack)
        >>> pack = registry.create("finance")
    """

    def __init__(self) -> None:
        self._packs: dict[str, type[FeaturePack]] = {}

    def register(self, name: str, pack_class: type[FeaturePack]) -> None:
        """Register a feature pack."""
        self._packs[name.lower()] = pack_class

    def create(self, name: str, **kwargs: Any) -> FeaturePack:
        """Create a feature pack instance."""
        name_lower = name.lower()
        if name_lower not in self._packs:
            raise ConfigurationError(
                f"Unknown pack: {name!r}. Available: {list(self._packs.keys())}"
            )
        return self._packs[name_lower](**kwargs)

    def list_packs(self) -> list[str]:
        """List all registered pack names."""
        return list(self._packs.keys())


# Global registry
_global_registry = FeaturePackRegistry()


def get_pack_registry() -> FeaturePackRegistry:
    """Get the global feature pack registry."""
    return _global_registry
