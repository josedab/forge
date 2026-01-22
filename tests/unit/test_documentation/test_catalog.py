"""Tests for feature catalog."""

import json
import tempfile
from pathlib import Path

import pytest

from forge.documentation.generator import FeatureDoc, DatasetDoc
from forge.documentation.catalog import (
    FeatureCatalog,
    CatalogEntry,
    create_catalog,
)


@pytest.fixture
def sample_features():
    """Create sample feature docs."""
    return [
        FeatureDoc(
            name="user_age",
            dtype="int64",
            description="User age in years",
            category="numeric",
            tags=["demographic", "user"],
        ),
        FeatureDoc(
            name="purchase_amount",
            dtype="float64",
            description="Total purchase amount",
            category="numeric",
            tags=["financial", "transaction"],
        ),
        FeatureDoc(
            name="product_category",
            dtype="object",
            description="Product category",
            category="categorical",
            tags=["product"],
        ),
    ]


@pytest.fixture
def sample_dataset(sample_features):
    """Create sample dataset doc."""
    return DatasetDoc(
        name="ecommerce_data",
        description="E-commerce transaction data",
        features=sample_features,
        n_samples=1000,
        n_features=3,
    )


@pytest.fixture
def second_dataset():
    """Create second dataset for multi-dataset tests."""
    features = [
        FeatureDoc(
            name="click_count",
            dtype="int64",
            description="Number of clicks",
            category="numeric",
            tags=["behavioral"],
        ),
        FeatureDoc(
            name="session_duration",
            dtype="float64",
            description="Session duration in seconds",
            category="numeric",
            tags=["behavioral", "time"],
        ),
    ]
    return DatasetDoc(
        name="clickstream_data",
        description="User clickstream data",
        features=features,
        n_samples=5000,
        n_features=2,
    )


class TestCatalogEntry:
    """Tests for CatalogEntry."""

    def test_creation(self, sample_features):
        """Test CatalogEntry creation."""
        entry = CatalogEntry(
            feature_doc=sample_features[0],
            dataset_name="test_dataset",
            status="active",
            owner="data_team",
        )

        assert entry.feature_doc.name == "user_age"
        assert entry.dataset_name == "test_dataset"
        assert entry.status == "active"
        assert entry.owner == "data_team"

    def test_to_dict(self, sample_features):
        """Test to_dict method."""
        entry = CatalogEntry(
            feature_doc=sample_features[0],
            dataset_name="test",
            status="active",
        )

        d = entry.to_dict()

        assert d["dataset_name"] == "test"
        assert d["status"] == "active"
        assert "feature" in d

    def test_from_dict(self, sample_features):
        """Test from_dict method."""
        data = {
            "feature": sample_features[0].to_dict(),
            "dataset_name": "test",
            "status": "deprecated",
            "owner": "ml_team",
        }

        entry = CatalogEntry.from_dict(data)

        assert entry.dataset_name == "test"
        assert entry.status == "deprecated"
        assert entry.owner == "ml_team"


class TestFeatureCatalog:
    """Tests for FeatureCatalog."""

    def test_empty_catalog(self):
        """Test empty catalog creation."""
        catalog = FeatureCatalog()

        assert len(catalog) == 0
        assert catalog.list_datasets() == []

    def test_add_dataset(self, sample_dataset):
        """Test adding a dataset."""
        catalog = FeatureCatalog()
        count = catalog.add_dataset(sample_dataset, owner="data_team")

        assert count == 3
        assert len(catalog) == 3
        assert "ecommerce_data" in catalog.list_datasets()

    def test_add_multiple_datasets(self, sample_dataset, second_dataset):
        """Test adding multiple datasets."""
        catalog = FeatureCatalog()
        catalog.add_dataset(sample_dataset)
        catalog.add_dataset(second_dataset)

        assert len(catalog) == 5
        assert len(catalog.list_datasets()) == 2

    def test_get_feature(self, sample_dataset):
        """Test getting a specific feature."""
        catalog = FeatureCatalog()
        catalog.add_dataset(sample_dataset)

        entry = catalog.get_feature("ecommerce_data", "user_age")

        assert entry is not None
        assert entry.feature_doc.name == "user_age"

    def test_get_nonexistent_feature(self, sample_dataset):
        """Test getting nonexistent feature."""
        catalog = FeatureCatalog()
        catalog.add_dataset(sample_dataset)

        entry = catalog.get_feature("ecommerce_data", "nonexistent")

        assert entry is None

    def test_search_by_query(self, sample_dataset):
        """Test text search."""
        catalog = FeatureCatalog()
        catalog.add_dataset(sample_dataset)

        results = catalog.search(query="purchase")

        assert len(results) == 1
        assert results[0].feature_doc.name == "purchase_amount"

    def test_search_by_category(self, sample_dataset):
        """Test search by category."""
        catalog = FeatureCatalog()
        catalog.add_dataset(sample_dataset)

        results = catalog.search(category="numeric")

        assert len(results) == 2

    def test_search_by_tags(self, sample_dataset):
        """Test search by tags."""
        catalog = FeatureCatalog()
        catalog.add_dataset(sample_dataset)

        results = catalog.search(tags=["financial"])

        assert len(results) == 1
        assert results[0].feature_doc.name == "purchase_amount"

    def test_search_by_dataset(self, sample_dataset, second_dataset):
        """Test search by dataset."""
        catalog = FeatureCatalog()
        catalog.add_dataset(sample_dataset)
        catalog.add_dataset(second_dataset)

        results = catalog.search(dataset="clickstream_data")

        assert len(results) == 2

    def test_search_by_status(self, sample_dataset):
        """Test search by status."""
        catalog = FeatureCatalog()
        catalog.add_dataset(sample_dataset, status="experimental")

        results = catalog.search(status="experimental")

        assert len(results) == 3

    def test_search_combined_filters(self, sample_dataset, second_dataset):
        """Test search with combined filters."""
        catalog = FeatureCatalog()
        catalog.add_dataset(sample_dataset)
        catalog.add_dataset(second_dataset)

        results = catalog.search(
            category="numeric",
            dataset="ecommerce_data",
        )

        assert len(results) == 2

    def test_list_categories(self, sample_dataset):
        """Test list_categories method."""
        catalog = FeatureCatalog()
        catalog.add_dataset(sample_dataset)

        categories = catalog.list_categories()

        assert "numeric" in categories
        assert "categorical" in categories

    def test_list_tags(self, sample_dataset):
        """Test list_tags method."""
        catalog = FeatureCatalog()
        catalog.add_dataset(sample_dataset)

        tags = catalog.list_tags()

        assert "demographic" in tags
        assert "financial" in tags

    def test_get_statistics(self, sample_dataset, second_dataset):
        """Test get_statistics method."""
        catalog = FeatureCatalog()
        catalog.add_dataset(sample_dataset)
        catalog.add_dataset(second_dataset)

        stats = catalog.get_statistics()

        assert stats["total_features"] == 5
        assert stats["total_datasets"] == 2
        assert "categories" in stats

    def test_update_status(self, sample_dataset):
        """Test updating feature status."""
        catalog = FeatureCatalog()
        catalog.add_dataset(sample_dataset)

        result = catalog.update_status(
            "ecommerce_data", "user_age", "deprecated"
        )

        assert result is True
        entry = catalog.get_feature("ecommerce_data", "user_age")
        assert entry.status == "deprecated"

    def test_deprecate_feature(self, sample_dataset):
        """Test deprecating a feature."""
        catalog = FeatureCatalog()
        catalog.add_dataset(sample_dataset)

        result = catalog.deprecate_feature(
            "ecommerce_data",
            "purchase_amount",
            reason="Replaced by normalized version",
        )

        assert result is True
        entry = catalog.get_feature("ecommerce_data", "purchase_amount")
        assert entry.status == "deprecated"
        assert any("Deprecated" in note for note in entry.feature_doc.quality_notes)

    def test_remove_dataset(self, sample_dataset, second_dataset):
        """Test removing a dataset."""
        catalog = FeatureCatalog()
        catalog.add_dataset(sample_dataset)
        catalog.add_dataset(second_dataset)

        count = catalog.remove_dataset("ecommerce_data")

        assert count == 3
        assert len(catalog) == 2
        assert "ecommerce_data" not in catalog.list_datasets()

    def test_iteration(self, sample_dataset):
        """Test iteration over catalog."""
        catalog = FeatureCatalog()
        catalog.add_dataset(sample_dataset)

        entries = list(catalog)

        assert len(entries) == 3
        assert all(isinstance(e, CatalogEntry) for e in entries)

    def test_contains(self, sample_dataset):
        """Test contains check."""
        catalog = FeatureCatalog()
        catalog.add_dataset(sample_dataset)

        assert "ecommerce_data:user_age" in catalog
        assert "ecommerce_data:nonexistent" not in catalog

    def test_save_and_load(self, sample_dataset):
        """Test save and load."""
        catalog = FeatureCatalog()
        catalog.add_dataset(sample_dataset, owner="test_team")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            catalog.save(path)

            loaded = FeatureCatalog.load(path)

            assert len(loaded) == len(catalog)
            assert loaded.list_datasets() == catalog.list_datasets()

            entry = loaded.get_feature("ecommerce_data", "user_age")
            assert entry.owner == "test_team"
        finally:
            path.unlink()


class TestCreateCatalogFunction:
    """Tests for create_catalog convenience function."""

    def test_create_empty(self):
        """Test creating empty catalog."""
        catalog = create_catalog()

        assert isinstance(catalog, FeatureCatalog)
        assert len(catalog) == 0

    def test_create_with_datasets(self, sample_dataset, second_dataset):
        """Test creating with datasets."""
        catalog = create_catalog(datasets=[sample_dataset, second_dataset])

        assert len(catalog) == 5

    def test_load_from_path(self, sample_dataset):
        """Test loading from path."""
        # Create and save a catalog
        original = FeatureCatalog()
        original.add_dataset(sample_dataset)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            original.save(path)

            # Load using create_catalog
            loaded = create_catalog(path=path)

            assert len(loaded) == len(original)
        finally:
            path.unlink()
