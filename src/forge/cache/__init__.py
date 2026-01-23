"""Intelligent feature caching and reuse.

Content-addressable cache that detects when features can be reused
across experiments. Stores computed features with metadata hashes,
enabling instant retrieval for repeated transformations.
"""

from __future__ import annotations

from forge.cache.cache import CacheConfig, FeatureCache
from forge.cache.decorator import cached_transform
from forge.cache.hasher import DataFrameHasher

__all__ = [
    "CacheConfig",
    "DataFrameHasher",
    "FeatureCache",
    "cached_transform",
]
