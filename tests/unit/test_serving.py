"""Tests for real-time feature server."""

from __future__ import annotations

import time

import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from forge.serving import (
    FeatureRequest,
    FeatureServer,
    ServingConfig,
)


@pytest.fixture
def precomputed_df():
    return pd.DataFrame({
        "user_id": ["u1", "u2", "u3", "u4"],
        "age": [25, 30, 35, 40],
        "income": [50000, 60000, 70000, 80000],
        "score": [0.8, 0.6, 0.9, 0.7],
    })


@pytest.fixture
def simple_transformer():
    scaler = StandardScaler()
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    scaler.fit(X)
    return scaler


class TestFeatureServer:
    """Tests for FeatureServer."""

    def test_materialize_basic(self, precomputed_df):
        server = FeatureServer()
        server.materialize(precomputed_df, entity_column="user_id")
        assert server._is_initialized
        assert len(server._feature_store) == 4

    def test_materialize_specific_columns(self, precomputed_df):
        server = FeatureServer()
        server.materialize(
            precomputed_df, entity_column="user_id",
            feature_columns=["age", "score"],
        )
        assert "income" not in server._feature_store["u1"]

    def test_materialize_invalid_entity_column(self, precomputed_df):
        server = FeatureServer()
        with pytest.raises(ValueError, match="not found"):
            server.materialize(precomputed_df, entity_column="nonexistent")

    def test_serve_precomputed(self, precomputed_df):
        server = FeatureServer()
        server.materialize(precomputed_df, entity_column="user_id")
        response = server.serve(FeatureRequest(entity_id="u1"))
        assert response.success
        assert response.features["age"] == 25
        assert response.features["income"] == 50000

    def test_serve_specific_features(self, precomputed_df):
        server = FeatureServer()
        server.materialize(precomputed_df, entity_column="user_id")
        response = server.serve(
            FeatureRequest(entity_id="u1", feature_names=["age"])
        )
        assert "age" in response.features
        assert "income" not in response.features

    def test_serve_unknown_entity(self, precomputed_df):
        server = FeatureServer()
        server.materialize(precomputed_df, entity_column="user_id")
        response = server.serve(FeatureRequest(entity_id="unknown"))
        assert response.success
        assert len(response.features) == 0

    def test_serve_with_metadata(self, precomputed_df):
        server = FeatureServer()
        server.materialize(precomputed_df, entity_column="user_id")
        response = server.serve(FeatureRequest(entity_id="u1"))
        assert "latency_ms" in response.metadata
        assert "cache_hit" in response.metadata

    def test_serve_batch(self, precomputed_df):
        server = FeatureServer()
        server.materialize(precomputed_df, entity_column="user_id")
        requests = [
            FeatureRequest(entity_id="u1"),
            FeatureRequest(entity_id="u2"),
        ]
        responses = server.serve_batch(requests)
        assert len(responses) == 2
        assert all(r.success for r in responses)

    def test_cache_hit(self, precomputed_df):
        config = ServingConfig(cache_enabled=True, cache_ttl=60.0)
        server = FeatureServer(config=config)
        server.materialize(precomputed_df, entity_column="user_id")

        req = FeatureRequest(entity_id="u1")
        resp1 = server.serve(req)
        assert resp1.metadata.get("cache_hit") is False

        resp2 = server.serve(req)
        assert resp2.metadata.get("cache_hit") is True

    def test_cache_disabled(self, precomputed_df):
        config = ServingConfig(cache_enabled=False)
        server = FeatureServer(config=config)
        server.materialize(precomputed_df, entity_column="user_id")

        req = FeatureRequest(entity_id="u1")
        server.serve(req)
        resp = server.serve(req)
        assert resp.metadata.get("cache_hit") is False

    def test_on_demand_computation(self, simple_transformer):
        server = FeatureServer(transformer=simple_transformer)
        req = FeatureRequest(
            entity_id="new_user",
            data={"a": 2.0, "b": 5.0},
        )
        response = server.serve(req)
        assert response.success
        assert len(response.features) > 0

    def test_get_health(self, precomputed_df):
        server = FeatureServer()
        health = server.get_health()
        assert health["status"] == "not_initialized"

        server.materialize(precomputed_df, entity_column="user_id")
        health = server.get_health()
        assert health["status"] == "healthy"
        assert health["store_size"] == 4

    def test_invalidate_entity(self, precomputed_df):
        server = FeatureServer()
        server.materialize(precomputed_df, entity_column="user_id")
        server.invalidate_entity("u1")
        assert "u1" not in server._feature_store

    def test_clear_cache(self, precomputed_df):
        server = FeatureServer()
        server.materialize(precomputed_df, entity_column="user_id")
        server.serve(FeatureRequest(entity_id="u1"))
        server.clear_cache()
        assert server._cache.stats["size"] == 0


class TestLRUCache:
    """Tests for the internal LRU cache."""

    def test_basic_get_put(self):
        from forge.serving.server import _LRUCache
        cache = _LRUCache(max_size=10, ttl=60)
        cache.put("key1", {"a": 1})
        assert cache.get("key1") == {"a": 1}

    def test_ttl_expiry(self):
        from forge.serving.server import _LRUCache
        cache = _LRUCache(max_size=10, ttl=0.01)
        cache.put("key1", {"a": 1})
        time.sleep(0.02)
        assert cache.get("key1") is None

    def test_max_size_eviction(self):
        from forge.serving.server import _LRUCache
        cache = _LRUCache(max_size=2, ttl=60)
        cache.put("k1", 1)
        cache.put("k2", 2)
        cache.put("k3", 3)
        assert cache.get("k1") is None
        assert cache.get("k3") == 3

    def test_stats(self):
        from forge.serving.server import _LRUCache
        cache = _LRUCache(max_size=10, ttl=60)
        cache.put("k1", 1)
        cache.get("k1")
        cache.get("missing")
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


class TestServingConfig:
    """Tests for ServingConfig."""

    def test_defaults(self):
        config = ServingConfig()
        assert config.cache_enabled is True
        assert config.cache_ttl == 300.0
        assert config.cache_max_size == 10000

    def test_custom(self):
        config = ServingConfig(cache_enabled=False, cache_ttl=60.0)
        assert config.cache_enabled is False
        assert config.cache_ttl == 60.0
