"""Feature cache with LRU eviction and disk persistence."""

from __future__ import annotations

import logging
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from forge.cache.hasher import DataFrameHasher

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for the feature cache.

    Args:
        max_memory_items: Maximum number of items to hold in memory.
        disk_path: Path for disk-based persistence. None for memory-only.
        max_disk_mb: Maximum disk cache size in MB.
        ttl_seconds: Time-to-live for cache entries. 0 for no expiry.
    """

    max_memory_items: int = 100
    disk_path: str | None = None
    max_disk_mb: float = 1000.0
    ttl_seconds: int = 0


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""

    key: str
    data: pd.DataFrame
    created_at: float
    transformer_name: str = ""
    params_hash: str = ""
    hits: int = 0
    size_bytes: int = 0

    @property
    def is_expired(self) -> bool:
        """Always returns False; TTL is checked by the cache itself."""
        return False


class FeatureCache:
    """Content-addressable feature cache with LRU eviction.

    Caches computed features keyed by a hash of (input data +
    transformer parameters). Supports both memory and disk backends.

    Args:
        config: Cache configuration.
        hasher: DataFrameHasher instance.

    Example:
        >>> cache = FeatureCache()
        >>> key = cache.compute_key(X, transformer)
        >>> if cache.has(key):
        ...     result = cache.get(key)
        ... else:
        ...     result = transformer.fit_transform(X, y)
        ...     cache.put(key, result)
    """

    def __init__(
        self,
        config: CacheConfig | None = None,
        hasher: DataFrameHasher | None = None,
    ) -> None:
        self.config = config or CacheConfig()
        self.hasher = hasher or DataFrameHasher()
        self._memory: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

        if self.config.disk_path:
            self._disk_path = Path(self.config.disk_path)
            self._disk_path.mkdir(parents=True, exist_ok=True)
        else:
            self._disk_path: Path | None = None

    def compute_key(
        self,
        X: pd.DataFrame,
        transformer: Any = None,
        params: dict[str, Any] | None = None,
    ) -> str:
        """Compute cache key for given data and transformer.

        Args:
            X: Input DataFrame.
            transformer: sklearn transformer (uses get_params).
            params: Explicit parameters (overrides transformer.get_params).

        Returns:
            Cache key string.
        """
        t_name = ""
        t_params = params or {}
        if transformer is not None:
            t_name = type(transformer).__name__
            if not t_params and hasattr(transformer, "get_params"):
                t_params = transformer.get_params(deep=False)
        return self.hasher.hash_key(X, params=t_params, transformer_name=t_name)

    def has(self, key: str) -> bool:
        """Check if a key exists in cache (memory or disk)."""
        if key in self._memory:
            entry = self._memory[key]
            if self._is_expired(entry):
                self._memory.pop(key)
                return False
            return True
        if self._disk_path:
            return (self._disk_path / f"{key}.pkl").exists()
        return False

    def get(self, key: str) -> pd.DataFrame | None:
        """Retrieve cached features.

        Args:
            key: Cache key.

        Returns:
            Cached DataFrame or None if not found.
        """
        # Check memory first
        if key in self._memory:
            entry = self._memory[key]
            if self._is_expired(entry):
                self._memory.pop(key)
            else:
                self._memory.move_to_end(key)
                entry.hits += 1
                self._stats["hits"] += 1
                return entry.data

        # Check disk
        if self._disk_path:
            disk_file = self._disk_path / f"{key}.pkl"
            if disk_file.exists():
                try:
                    with open(disk_file, "rb") as f:
                        entry = pickle.load(f)  # noqa: S301
                    if not self._is_expired(entry):
                        self._put_memory(key, entry)
                        self._stats["hits"] += 1
                        return entry.data  # type: ignore[no-any-return]
                    else:
                        disk_file.unlink(missing_ok=True)
                except Exception:
                    disk_file.unlink(missing_ok=True)

        self._stats["misses"] += 1
        return None

    def put(
        self,
        key: str,
        data: pd.DataFrame,
        transformer_name: str = "",
    ) -> None:
        """Store features in the cache.

        Args:
            key: Cache key.
            data: DataFrame to cache.
            transformer_name: Name of the transformer that produced the data.
        """
        entry = CacheEntry(
            key=key,
            data=data,
            created_at=time.time(),
            transformer_name=transformer_name,
            size_bytes=data.memory_usage(deep=True).sum(),
        )

        self._put_memory(key, entry)

        # Persist to disk
        if self._disk_path:
            disk_file = self._disk_path / f"{key}.pkl"
            try:
                with open(disk_file, "wb") as f:
                    pickle.dump(entry, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                logger.warning("Failed to write cache to disk: %s", e)

    def _put_memory(self, key: str, entry: CacheEntry) -> None:
        """Put entry in memory with LRU eviction."""
        self._memory[key] = entry
        self._memory.move_to_end(key)

        while len(self._memory) > self.config.max_memory_items:
            evicted_key, _ = self._memory.popitem(last=False)
            self._stats["evictions"] += 1
            logger.debug("Evicted cache entry: %s", evicted_key)

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if an entry has expired."""
        if self.config.ttl_seconds <= 0:
            return False
        return (time.time() - entry.created_at) > self.config.ttl_seconds

    def invalidate(self, key: str) -> bool:
        """Remove a specific entry from the cache.

        Args:
            key: Cache key to invalidate.

        Returns:
            True if entry was found and removed.
        """
        removed = False
        if key in self._memory:
            self._memory.pop(key)
            removed = True
        if self._disk_path:
            disk_file = self._disk_path / f"{key}.pkl"
            if disk_file.exists():
                disk_file.unlink()
                removed = True
        return removed

    def clear(self) -> None:
        """Clear all cache entries."""
        self._memory.clear()
        if self._disk_path:
            for f in self._disk_path.glob("*.pkl"):
                f.unlink()
        logger.info("Cache cleared")

    @property
    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0.0
        return {
            "memory_items": len(self._memory),
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "evictions": self._stats["evictions"],
            "hit_rate": hit_rate,
        }

    @property
    def size(self) -> int:
        """Return number of items in memory cache."""
        return len(self._memory)
