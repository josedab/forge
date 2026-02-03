#!/usr/bin/env python
"""Feature caching example for Forge.

This example demonstrates FeatureCache, CacheConfig, and the
cached_transform decorator for avoiding redundant computation
in feature engineering pipelines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from forge.cache import CacheConfig, FeatureCache, cached_transform


def create_sample_data(n: int = 500) -> pd.DataFrame:
    """Create sample data for caching demonstrations."""
    np.random.seed(42)
    return pd.DataFrame({
        "feature_a": np.random.normal(0, 1, n),
        "feature_b": np.random.exponential(2, n),
        "feature_c": np.random.uniform(-1, 1, n),
    })


def example_basic_caching():
    """Example: Basic cache usage with put/get/has."""
    print("=" * 60)
    print("Example: Basic Feature Caching")
    print("=" * 60)

    cache = FeatureCache(config=CacheConfig(max_memory_items=50))
    X = create_sample_data()

    # Compute a cache key from input data
    key = cache.compute_key(X, params={"operation": "square"})
    print(f"\nCache key (first 20 chars): {key[:20]}...")
    print(f"Cache has key: {cache.has(key)}")

    # Simulate an expensive transformation
    print("\nPerforming 'expensive' transformation...")
    result = X ** 2
    result.columns = [f"{c}_squared" for c in X.columns]

    # Store result
    cache.put(key, result, transformer_name="SquareTransform")
    print(f"Stored in cache. Size: {cache.size}")

    # Retrieve from cache
    cached_result = cache.get(key)
    print(f"Retrieved from cache: {cached_result is not None}")
    print(f"Shapes match: {cached_result.shape == result.shape}")

    # Check stats
    stats = cache.stats
    print(f"\nCache stats: items={stats['memory_items']}, "
          f"hits={stats['hits']}, misses={stats['misses']}, rate={stats['hit_rate']:.2%}")


def example_cache_with_transformer():
    """Example: Cache keyed by transformer parameters."""
    print("\n" + "=" * 60)
    print("Example: Cache with Transformer Parameters")
    print("=" * 60)

    cache = FeatureCache()
    X = create_sample_data(200)

    # Different params produce different cache keys
    key1 = cache.compute_key(X, params={"degree": 2, "method": "poly"})
    key2 = cache.compute_key(X, params={"degree": 3, "method": "poly"})
    key3 = cache.compute_key(X, params={"degree": 2, "method": "poly"})

    print(f"\nKeys for different parameters:")
    print(f"  degree=2: {key1[:20]}...")
    print(f"  degree=3: {key2[:20]}...")
    print(f"  degree=2 (again): {key3[:20]}...")
    print(f"  Same params produce same key: {key1 == key3}")
    print(f"  Different params produce different key: {key1 != key2}")

    # Store results for both
    cache.put(key1, X ** 2, transformer_name="Poly")
    cache.put(key2, X ** 3, transformer_name="Poly")

    # Retrieve the right one
    result = cache.get(key1)
    print(f"\n  Retrieved degree=2 result shape: {result.shape}")
    print(f"  Cache size: {cache.size}")

    # Invalidate one key
    cache.invalidate(key2)
    print(f"  After invalidating degree=3: cache size = {cache.size}")
    print(f"  degree=3 still in cache: {cache.has(key2)}")


def example_cached_transform_decorator():
    """Example: Use @cached_transform to auto-cache transform results."""
    print("\n" + "=" * 60)
    print("Example: @cached_transform Decorator")
    print("=" * 60)

    cache = FeatureCache()

    @cached_transform(cache=cache)
    class ExpensiveTransformer(BaseEstimator, TransformerMixin):
        """A transformer whose transform() is automatically cached."""

        def __init__(self, multiplier: float = 2.0):
            self.multiplier = multiplier

        def fit(self, X, y=None):
            self._is_fitted = True
            return self

        def transform(self, X):
            print("    [Computing transform -- this is the expensive part]")
            result = X * self.multiplier
            result.columns = [f"{c}_x{self.multiplier}" for c in X.columns]
            return result

    X = create_sample_data(100)
    transformer = ExpensiveTransformer(multiplier=3.0)
    transformer.fit(X)

    print("\nFirst call (cache miss):")
    result1 = transformer.transform(X)
    print(f"  Result shape: {result1.shape}")

    print("\nSecond call with same data (cache hit):")
    result2 = transformer.transform(X)
    print(f"  Result shape: {result2.shape}")

    print(f"\nCache stats after two calls:")
    stats = cache.stats
    print(f"  Hits:   {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")


if __name__ == "__main__":
    example_basic_caching()
    example_cache_with_transformer()
    example_cached_transform_decorator()

    print("\n" + "=" * 60)
    print("All caching examples completed!")
    print("=" * 60)
