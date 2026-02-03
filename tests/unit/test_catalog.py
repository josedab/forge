"""Tests for the feature catalog store and API."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from forge.catalog.store import CatalogEntry, CatalogStore, FeatureCatalogStore


class TestCatalogEntry:
    def test_creation(self):
        entry = CatalogEntry(name="age", description="Customer age in years")
        assert entry.name == "age"
        assert entry.created_at != ""
        assert not entry.deprecated

    def test_to_dict(self):
        entry = CatalogEntry(name="x", tags={"team": "ml"})
        d = entry.to_dict()
        assert d["name"] == "x"
        assert d["tags"]["team"] == "ml"

    def test_from_dict(self):
        data = {"name": "y", "description": "test", "dtype": "int64"}
        entry = CatalogEntry.from_dict(data)
        assert entry.name == "y"
        assert entry.dtype == "int64"


class TestCatalogStore:
    @pytest.fixture
    def store(self):
        return CatalogStore()

    @pytest.fixture
    def populated_store(self, store):
        store.add(CatalogEntry(name="age", description="Customer age", owner="team-a", tags={"domain": "customer"}))
        store.add(CatalogEntry(name="income", description="Annual income", owner="team-a", tags={"domain": "finance"}))
        store.add(CatalogEntry(name="score", description="Credit score", owner="team-b", tags={"domain": "finance"}))
        return store

    def test_add_and_get(self, store):
        entry = CatalogEntry(name="test", description="Test feature")
        store.add(entry)
        result = store.get("test")
        assert result is not None
        assert result.name == "test"

    def test_get_nonexistent(self, store):
        assert store.get("nonexistent") is None

    def test_delete(self, store):
        store.add(CatalogEntry(name="temp"))
        assert store.delete("temp")
        assert store.get("temp") is None
        assert not store.delete("nonexistent")

    def test_list_all(self, populated_store):
        assert len(populated_store.list_all()) == 3

    def test_search_by_text(self, populated_store):
        results = populated_store.search("income")
        assert len(results) == 1
        assert results[0].name == "income"

    def test_search_by_description(self, populated_store):
        results = populated_store.search("credit")
        assert len(results) == 1

    def test_search_case_insensitive(self, populated_store):
        results = populated_store.search("CUSTOMER")
        assert len(results) == 1

    def test_search_by_tags(self, populated_store):
        results = populated_store.search(tags={"domain": "finance"})
        assert len(results) == 2

    def test_search_by_owner(self, populated_store):
        results = populated_store.search(owner="team-b")
        assert len(results) == 1

    def test_search_excludes_deprecated(self, populated_store):
        populated_store.deprecate("age", "too old")
        results = populated_store.search()
        assert all(r.name != "age" for r in results)

    def test_search_includes_deprecated(self, populated_store):
        populated_store.deprecate("age")
        results = populated_store.search(include_deprecated=True)
        assert any(r.name == "age" for r in results)

    def test_deprecate(self, populated_store):
        assert populated_store.deprecate("age", "not needed")
        entry = populated_store.get("age")
        assert entry is not None
        assert entry.deprecated
        assert entry.deprecation_reason == "not needed"

    def test_tag(self, populated_store):
        assert populated_store.tag("age", {"importance": "high"})
        entry = populated_store.get("age")
        assert entry is not None
        assert entry.tags["importance"] == "high"

    def test_statistics(self, populated_store):
        stats = populated_store.get_statistics()
        assert stats["total_features"] == 3
        assert stats["unique_owners"] == 2
        assert "domain" in stats["tag_counts"]

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "catalog.json")
            store1 = CatalogStore(persist_path=path)
            store1.add(CatalogEntry(name="x", description="test"))

            store2 = CatalogStore(persist_path=path)
            assert store2.get("x") is not None

    def test_export_import_json(self, populated_store):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "export.json"
            populated_store.export_json(path)
            assert path.exists()

            new_store = CatalogStore()
            count = new_store.import_json(path)
            assert count == 3
            assert new_store.get("age") is not None

    def test_size(self, populated_store):
        assert populated_store.size == 3


class TestFeatureCatalogStore:
    def test_add_from_transformer(self):
        class MockTransformer:
            def get_feature_names_out(self):
                return ["feat_a", "feat_b", "feat_c"]

        store = FeatureCatalogStore()
        count = store.add_from_transformer(MockTransformer(), owner="ml-team")
        assert count == 3
        assert store.get("feat_a") is not None
        assert store.get("feat_a").owner == "ml-team"

    def test_add_from_unfitted_transformer(self):
        class BadTransformer:
            def get_feature_names_out(self):
                raise RuntimeError("Not fitted")

        store = FeatureCatalogStore()
        count = store.add_from_transformer(BadTransformer())
        assert count == 0
