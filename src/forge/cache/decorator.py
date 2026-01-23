"""Caching decorator for sklearn transform methods."""

from __future__ import annotations

import functools
import logging
from typing import Any

import pandas as pd

from forge.cache.cache import FeatureCache

logger = logging.getLogger(__name__)

# Module-level default cache
_default_cache: FeatureCache | None = None


def get_default_cache() -> FeatureCache:
    """Get or create the default module-level feature cache."""
    global _default_cache
    if _default_cache is None:
        _default_cache = FeatureCache()
    return _default_cache


def set_default_cache(cache: FeatureCache) -> None:
    """Set the default module-level feature cache."""
    global _default_cache
    _default_cache = cache


def cached_transform(
    cache: FeatureCache | None = None,
) -> Any:
    """Decorator that caches transform() results.

    Wraps an sklearn-compatible transformer's transform method
    to check the cache before computing. Caches the result
    after computation.

    Args:
        cache: FeatureCache to use. None for the default cache.

    Example:
        >>> @cached_transform()
        ... class MyTransformer(BaseEstimator, TransformerMixin):
        ...     def fit(self, X, y=None):
        ...         self._is_fitted = True
        ...         return self
        ...     def transform(self, X):
        ...         # This will be cached!
        ...         return expensive_computation(X)
    """
    def decorator(cls: type) -> type:
        original_transform = cls.transform  # type: ignore[attr-defined]

        @functools.wraps(original_transform)
        def cached_transform_method(self: Any, X: pd.DataFrame) -> pd.DataFrame:
            c = cache or get_default_cache()
            key = c.compute_key(X, transformer=self)

            cached = c.get(key)
            if cached is not None:
                logger.debug(
                    "Cache hit for %s (key=%s...)", type(self).__name__, key[:12]
                )
                return cached

            result = original_transform(self, X)
            c.put(key, result, transformer_name=type(self).__name__)
            logger.debug(
                "Cached result for %s (key=%s...)", type(self).__name__, key[:12]
            )
            return result  # type: ignore[no-any-return]

        cls.transform = cached_transform_method  # type: ignore[attr-defined]
        return cls

    return decorator
