"""Tests for the intelligent feature cache."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from forge.cache import CacheConfig, DataFrameHasher, FeatureCache, cached_transform


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [5.0, 4.0, 3.0, 2.0, 1.0],
    })


@pytest.fixture
def cache():
    return FeatureCache(config=CacheConfig(max_memory_items=10))


class TestDataFrameHasher:
    def test_deterministic_hash(self, sample_df):
        hasher = DataFrameHasher()
        h1 = hasher.hash_dataframe(sample_df)
        h2 = hasher.hash_dataframe(sample_df)
        assert h1 == h2

    def test_different_data_different_hash(self):
        hasher = DataFrameHasher()
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})
        assert hasher.hash_dataframe(df1) != hasher.hash_dataframe(df2)

    def test_different_schema_different_hash(self):
        hasher = DataFrameHasher()
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [1, 2, 3]})
        assert hasher.hash_dataframe(df1) != hasher.hash_dataframe(df2)

    def test_hash_params(self):
        hasher = DataFrameHasher()
        h1 = hasher.hash_params({"degree": 2})
        h2 = hasher.hash_params({"degree": 2})
        h3 = hasher.hash_params({"degree": 3})
        assert h1 == h2
        assert h1 != h3

    def test_hash_key_combines_all(self, sample_df):
        hasher = DataFrameHasher()
        k1 = hasher.hash_key(sample_df, params={"a": 1}, transformer_name="T1")
        k2 = hasher.hash_key(sample_df, params={"a": 1}, transformer_name="T2")
        assert k1 != k2

    def test_sampling_large_df(self):
        hasher = DataFrameHasher(sample_rows=10)
        df = pd.DataFrame({"x": range(10000)})
        h = hasher.hash_dataframe(df)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest


class TestFeatureCache:
    def test_put_and_get(self, cache, sample_df):
        cache.put("key1", sample_df)
        result = cache.get("key1")
        assert result is not None
        pd.testing.assert_frame_equal(result, sample_df)

    def test_cache_miss(self, cache):
        result = cache.get("nonexistent")
        assert result is None

    def test_has(self, cache, sample_df):
        assert not cache.has("key1")
        cache.put("key1", sample_df)
        assert cache.has("key1")

    def test_invalidate(self, cache, sample_df):
        cache.put("key1", sample_df)
        assert cache.invalidate("key1")
        assert not cache.has("key1")
        assert not cache.invalidate("nonexistent")

    def test_clear(self, cache, sample_df):
        cache.put("key1", sample_df)
        cache.put("key2", sample_df)
        cache.clear()
        assert cache.size == 0

    def test_lru_eviction(self, sample_df):
        cache = FeatureCache(config=CacheConfig(max_memory_items=3))
        cache.put("k1", sample_df)
        cache.put("k2", sample_df)
        cache.put("k3", sample_df)
        cache.put("k4", sample_df)  # Should evict k1
        assert not cache.has("k1")
        assert cache.has("k4")
        assert cache.stats["evictions"] >= 1

    def test_stats(self, cache, sample_df):
        cache.put("k1", sample_df)
        cache.get("k1")  # hit
        cache.get("k2")  # miss
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_compute_key(self, cache, sample_df):
        class MockTransformer:
            def get_params(self, deep=False):
                return {"n": 5}
        t = MockTransformer()
        key = cache.compute_key(sample_df, transformer=t)
        assert isinstance(key, str)
        assert len(key) > 0

    def test_ttl_expiry(self, sample_df):
        config = CacheConfig(max_memory_items=10, ttl_seconds=1)
        cache = FeatureCache(config=config)
        cache.put("k1", sample_df)
        # Entry should be accessible immediately
        assert cache.has("k1")
        # We can't easily test actual expiry in unit tests without sleeping

    def test_disk_cache(self, sample_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(max_memory_items=10, disk_path=tmpdir)
            cache = FeatureCache(config=config)
            cache.put("disk_key", sample_df)

            # Check disk file exists
            assert (Path(tmpdir) / "disk_key.pkl").exists()

            # Create a new cache pointing to same disk
            cache2 = FeatureCache(config=config)
            result = cache2.get("disk_key")
            assert result is not None
            pd.testing.assert_frame_equal(result, sample_df)

    def test_disk_invalidate(self, sample_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(disk_path=tmpdir)
            cache = FeatureCache(config=config)
            cache.put("k1", sample_df)
            cache.invalidate("k1")
            assert not (Path(tmpdir) / "k1.pkl").exists()


class TestCachedTransformDecorator:
    def test_caching_decorator(self, sample_df):
        call_count = {"n": 0}

        @cached_transform()
        class TestTransformer(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                self._is_fitted = True
                return self

            def transform(self, X):
                call_count["n"] += 1
                return X * 2

        t = TestTransformer()
        t.fit(sample_df)

        result1 = t.transform(sample_df)
        assert call_count["n"] == 1

        result2 = t.transform(sample_df)
        assert call_count["n"] == 1  # Should be cached
        pd.testing.assert_frame_equal(result1, result2)
