"""Feature server for real-time feature computation and serving.

Provides low-latency feature lookup, on-demand computation, and
caching with TTL for online inference workloads.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ServingConfig:
    """Configuration for the feature server.

    Attributes:
    ----------
    cache_enabled : bool
        Whether to enable feature caching.
    cache_ttl : float
        Cache time-to-live in seconds.
    cache_max_size : int
        Maximum number of cached entries.
    compute_timeout : float
        Timeout for on-demand computation in seconds.
    """

    cache_enabled: bool = True
    cache_ttl: float = 300.0
    cache_max_size: int = 10000
    compute_timeout: float = 30.0


@dataclass
class FeatureRequest:
    """Request for feature values.

    Attributes:
    ----------
    entity_id : str
        Entity identifier (e.g., user_id, item_id).
    feature_names : list[str] | None
        Specific features requested. None means all.
    data : dict[str, Any] | None
        Raw data for on-demand computation.
    """

    entity_id: str
    feature_names: list[str] | None = None
    data: dict[str, Any] | None = None


@dataclass
class FeatureResponse:
    """Response containing feature values.

    Attributes:
    ----------
    entity_id : str
        Entity identifier.
    features : dict[str, Any]
        Feature name -> value mapping.
    metadata : dict[str, Any]
        Response metadata (latency, cache hit, etc.).
    success : bool
        Whether the request succeeded.
    error : str | None
        Error message if request failed.
    """

    entity_id: str
    features: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str | None = None


class _LRUCache:
    """Simple LRU cache with TTL support."""

    def __init__(self, max_size: int, ttl: float) -> None:
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """Get a value from cache. Returns None if not found or expired."""
        if key not in self._cache:
            self._misses += 1
            return None

        value, timestamp = self._cache[key]
        if time.time() - timestamp > self.ttl:
            del self._cache[key]
            self._misses += 1
            return None

        self._cache.move_to_end(key)
        self._hits += 1
        return value

    def put(self, key: str, value: Any) -> None:
        """Put a value into cache."""
        if key in self._cache:
            self._cache.move_to_end(key)
        elif len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)

        self._cache[key] = (value, time.time())

    def invalidate(self, key: str) -> None:
        """Remove a specific key from cache."""
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear entire cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> dict[str, Any]:
        """Cache statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }


class FeatureServer:
    """Real-time feature server with caching and on-demand computation.

    Supports three modes of feature serving:
    1. **Pre-computed lookup**: Look up features from a materialized store.
    2. **On-demand computation**: Compute features in real-time from raw data.
    3. **Hybrid**: Combine pre-computed and on-demand features.

    Parameters
    ----------
    transformer : Any | None
        A fitted sklearn-compatible transformer for on-demand computation.
    config : ServingConfig | None
        Server configuration. Uses defaults if None.

    Examples:
    --------
    >>> from forge.serving import FeatureServer, FeatureRequest, ServingConfig
    >>>
    >>> server = FeatureServer(transformer=fitted_transformer)
    >>> server.materialize(X_precomputed, entity_column="user_id")
    >>>
    >>> request = FeatureRequest(entity_id="user_123")
    >>> response = server.serve(request)
    >>> print(response.features)
    """

    def __init__(
        self,
        transformer: Any | None = None,
        config: ServingConfig | None = None,
    ) -> None:
        self.transformer = transformer
        self.config = config or ServingConfig()

        self._cache = _LRUCache(
            max_size=self.config.cache_max_size,
            ttl=self.config.cache_ttl,
        )
        self._feature_store: dict[str, dict[str, Any]] = {}
        self._feature_names: list[str] = []
        self._is_initialized = False

    def materialize(
        self,
        X: pd.DataFrame,
        entity_column: str,
        feature_columns: list[str] | None = None,
    ) -> None:
        """Materialize pre-computed features into the serving store.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame with entity IDs and pre-computed features.
        entity_column : str
            Column name containing entity identifiers.
        feature_columns : list[str] | None
            Feature columns to store. None uses all except entity_column.
        """
        if entity_column not in X.columns:
            raise ValueError(f"Entity column '{entity_column}' not found.")

        if feature_columns is None:
            feature_columns = [c for c in X.columns if c != entity_column]

        self._feature_names = feature_columns
        self._feature_store = {}

        for _, row in X.iterrows():
            entity_id = str(row[entity_column])
            features = {}
            for col in feature_columns:
                val = row[col]
                if isinstance(val, (np.integer,)):
                    val = int(val)
                elif isinstance(val, (np.floating,)):
                    val = float(val)
                features[col] = val
            self._feature_store[entity_id] = features

        self._is_initialized = True
        logger.info(
            "Materialized %d entities with %d features.",
            len(self._feature_store), len(feature_columns),
        )

    def serve(self, request: FeatureRequest) -> FeatureResponse:
        """Serve a feature request.

        Parameters
        ----------
        request : FeatureRequest
            Feature request.

        Returns:
        -------
        FeatureResponse
            Feature response with values and metadata.
        """
        start_time = time.time()

        try:
            features: dict[str, Any] = {}
            cache_hit = False

            # Try cache first
            if self.config.cache_enabled:
                cache_key = self._cache_key(request)
                cached = self._cache.get(cache_key)
                if cached is not None:
                    features = cached
                    cache_hit = True

            if not cache_hit:
                # Try pre-computed store
                if request.entity_id in self._feature_store:
                    features = dict(self._feature_store[request.entity_id])

                # On-demand computation if data provided
                if request.data is not None and self.transformer is not None:
                    computed = self._compute_on_demand(request.data)
                    features.update(computed)

                # Cache the result
                if self.config.cache_enabled and features:
                    self._cache.put(self._cache_key(request), features)

            # Filter to requested features
            if request.feature_names:
                features = {
                    k: v for k, v in features.items()
                    if k in request.feature_names
                }

            latency = time.time() - start_time

            return FeatureResponse(
                entity_id=request.entity_id,
                features=features,
                metadata={
                    "latency_ms": latency * 1000,
                    "cache_hit": cache_hit,
                    "feature_count": len(features),
                },
                success=True,
            )

        except Exception as e:
            latency = time.time() - start_time
            logger.exception("Error serving request for %s", request.entity_id)
            return FeatureResponse(
                entity_id=request.entity_id,
                metadata={"latency_ms": latency * 1000},
                success=False,
                error=str(e),
            )

    def serve_batch(self, requests: list[FeatureRequest]) -> list[FeatureResponse]:
        """Serve multiple feature requests.

        Parameters
        ----------
        requests : list[FeatureRequest]
            Batch of feature requests.

        Returns:
        -------
        list[FeatureResponse]
            Batch of feature responses.
        """
        return [self.serve(req) for req in requests]

    def _compute_on_demand(self, data: dict[str, Any]) -> dict[str, Any]:
        """Compute features on-demand from raw data."""
        import pandas as pd

        row_df = pd.DataFrame([data])

        try:
            result = self.transformer.transform(row_df)
            if isinstance(result, pd.DataFrame):
                return result.iloc[0].to_dict()
            return dict(enumerate(result[0]))
        except Exception as e:
            logger.warning("On-demand computation failed: %s", e)
            return {}

    @staticmethod
    def _cache_key(request: FeatureRequest) -> str:
        """Generate cache key from request."""
        key_parts = [request.entity_id]
        if request.feature_names:
            key_parts.extend(sorted(request.feature_names))
        return hashlib.md5(
            "|".join(key_parts).encode()
        ).hexdigest()

    def get_health(self) -> dict[str, Any]:
        """Get server health status.

        Returns:
        -------
        dict
            Health information including cache stats and store size.
        """
        return {
            "status": "healthy" if self._is_initialized or self.transformer else "not_initialized",
            "store_size": len(self._feature_store),
            "feature_count": len(self._feature_names),
            "cache_stats": self._cache.stats,
            "has_transformer": self.transformer is not None,
        }

    def invalidate_entity(self, entity_id: str) -> None:
        """Invalidate cached features for an entity."""
        self._feature_store.pop(entity_id, None)
        # Cache key depends on request, so we can't invalidate precisely
        # Clear entire cache for safety
        self._cache.clear()

    def clear_cache(self) -> None:
        """Clear the feature cache."""
        self._cache.clear()
