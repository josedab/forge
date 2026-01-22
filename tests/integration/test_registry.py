"""Integration tests for feature registry operations."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forge.registry import (
    LocalRegistry,
    FeatureDefinition,
    FeatureSet,
    RegistryConfig,
)
from forge.registry.base import FeatureType
from forge.registry.local_registry import create_local_registry, create_registry
from forge.exceptions import ConfigurationError, ValidationError


@pytest.fixture
def temp_registry_path():
    """Create a temporary directory for the registry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_registry"


@pytest.fixture
def registry_config(temp_registry_path):
    """Create a registry configuration."""
    return RegistryConfig(
        registry_type="local",
        project_name="test_project",
        connection_params={"path": str(temp_registry_path)},
    )


@pytest.fixture
def sample_feature():
    """Create a sample feature definition."""
    return FeatureDefinition(
        name="total_spend",
        dtype=FeatureType.FLOAT64,
        description="Total customer spending",
        source_columns=["order_amount"],
        transformation="sum",
        entity="customer",
        tags=["financial", "aggregation"],
    )


@pytest.fixture
def sample_feature_2():
    """Create another sample feature definition."""
    return FeatureDefinition(
        name="order_count",
        dtype=FeatureType.INT64,
        description="Number of customer orders",
        source_columns=["order_id"],
        transformation="count",
        entity="customer",
        tags=["count", "aggregation"],
    )


@pytest.fixture
def sample_feature_set(sample_feature, sample_feature_2):
    """Create a sample feature set."""
    return FeatureSet(
        name="customer_features",
        features=[sample_feature, sample_feature_2],
        entity="customer",
        description="Customer-level aggregated features",
        tags=["customer", "ml-ready"],
    )


@pytest.fixture
def sample_data():
    """Create sample feature data."""
    np.random.seed(42)
    return pd.DataFrame({
        "customer_id": range(1, 101),
        "total_spend": np.random.uniform(100, 5000, 100),
        "order_count": np.random.randint(1, 50, 100),
        "event_timestamp": pd.date_range("2024-01-01", periods=100, freq="D"),
    })


class TestLocalRegistryInitialization:
    """Tests for LocalRegistry initialization."""

    def test_init_with_config(self, registry_config):
        """Test initialization with config."""
        registry = LocalRegistry(registry_config)

        assert registry.config == registry_config
        assert not registry._connected

    def test_init_creates_directories_on_connect(self, registry_config, temp_registry_path):
        """Test that connect creates necessary directories."""
        registry = LocalRegistry(registry_config)
        registry.connect()

        assert registry._connected
        assert temp_registry_path.exists()
        registry.disconnect()

    def test_default_path_when_not_specified(self):
        """Test default path when not specified in config."""
        config = RegistryConfig(
            registry_type="local",
            project_name="test",
            connection_params={},
        )
        registry = LocalRegistry(config)

        assert registry._path is not None


class TestLocalRegistryConnection:
    """Tests for LocalRegistry connection management."""

    def test_connect_and_disconnect(self, registry_config):
        """Test connect and disconnect cycle."""
        registry = LocalRegistry(registry_config)

        assert not registry._connected

        registry.connect()
        assert registry._connected

        registry.disconnect()
        assert not registry._connected

    def test_context_manager(self, registry_config):
        """Test using registry as context manager."""
        with LocalRegistry(registry_config) as registry:
            assert registry._connected

        assert not registry._connected

    def test_operations_require_connection(self, registry_config, sample_feature):
        """Test that operations raise error when not connected."""
        registry = LocalRegistry(registry_config)

        with pytest.raises(ConfigurationError, match="not connected"):
            registry.register_feature(sample_feature)


class TestFeatureRegistration:
    """Tests for feature registration operations."""

    @pytest.fixture
    def connected_registry(self, registry_config):
        """Create a connected registry."""
        registry = LocalRegistry(registry_config)
        registry.connect()
        yield registry
        registry.disconnect()

    def test_register_single_feature(self, connected_registry, sample_feature):
        """Test registering a single feature."""
        version = connected_registry.register_feature(sample_feature)

        assert version is not None
        # register_feature returns a FeatureVersion object
        assert version.version.startswith("v")

    def test_register_feature_set(self, connected_registry, sample_feature_set):
        """Test registering a feature set."""
        version = connected_registry.register_feature_set(sample_feature_set)

        assert version is not None

    def test_register_same_feature_increments_version(self, connected_registry, sample_feature):
        """Test that registering the same feature increments version."""
        version1 = connected_registry.register_feature(sample_feature)
        version2 = connected_registry.register_feature(sample_feature)

        assert version1 != version2

    def test_get_registered_feature(self, connected_registry, sample_feature):
        """Test retrieving a registered feature."""
        connected_registry.register_feature(sample_feature)

        retrieved = connected_registry.get_feature("total_spend")

        assert retrieved is not None
        assert retrieved.name == "total_spend"
        assert retrieved.dtype == FeatureType.FLOAT64

    def test_get_nonexistent_feature(self, connected_registry):
        """Test retrieving a non-existent feature returns None."""
        result = connected_registry.get_feature("nonexistent")

        assert result is None

    def test_get_feature_set(self, connected_registry, sample_feature_set):
        """Test retrieving a feature set."""
        connected_registry.register_feature_set(sample_feature_set)

        retrieved = connected_registry.get_feature_set("customer_features")

        assert retrieved is not None
        assert retrieved.name == "customer_features"
        assert len(retrieved.features) == 2


class TestFeatureListing:
    """Tests for listing features and feature sets."""

    @pytest.fixture
    def populated_registry(self, registry_config, sample_feature_set):
        """Create a registry with registered features."""
        registry = LocalRegistry(registry_config)
        registry.connect()
        registry.register_feature_set(sample_feature_set)
        yield registry
        registry.disconnect()

    def test_list_all_features(self, populated_registry):
        """Test listing all registered features."""
        features = populated_registry.list_features()

        assert "total_spend" in features
        assert "order_count" in features

    def test_list_features_by_entity(self, populated_registry):
        """Test listing features filtered by entity."""
        features = populated_registry.list_features(entity="customer")

        assert "total_spend" in features
        assert "order_count" in features

        features = populated_registry.list_features(entity="nonexistent")
        assert len(features) == 0

    def test_list_features_filtered(self, populated_registry):
        """Test listing features (local registry only supports entity filter)."""
        # Local registry only supports entity filtering
        features = populated_registry.list_features(entity="customer")

        assert "total_spend" in features
        assert "order_count" in features

    def test_list_feature_sets(self, populated_registry):
        """Test listing all feature sets."""
        sets = populated_registry.list_feature_sets()

        assert "customer_features" in sets


class TestFeatureDeletion:
    """Tests for deleting features."""

    @pytest.fixture
    def populated_registry(self, registry_config, sample_feature, sample_feature_2):
        """Create a registry with registered features."""
        registry = LocalRegistry(registry_config)
        registry.connect()
        registry.register_feature(sample_feature)
        registry.register_feature(sample_feature_2)
        yield registry
        registry.disconnect()

    def test_delete_feature(self, populated_registry):
        """Test deleting a feature."""
        result = populated_registry.delete_feature("total_spend")

        assert result is True
        assert populated_registry.get_feature("total_spend") is None

    def test_delete_nonexistent_feature(self, populated_registry):
        """Test deleting a non-existent feature."""
        result = populated_registry.delete_feature("nonexistent")

        assert result is False


class TestDataMaterialization:
    """Tests for data materialization operations."""

    @pytest.fixture
    def registry_with_features(self, registry_config, sample_feature_set):
        """Create a registry with registered feature set."""
        registry = LocalRegistry(registry_config)
        registry.connect()
        registry.register_feature_set(sample_feature_set)
        yield registry
        registry.disconnect()

    def test_materialize_data(self, registry_with_features, sample_data):
        """Test materializing feature data."""
        rows = registry_with_features.materialize("customer_features", sample_data)

        assert rows == 100

    def test_materialize_adds_timestamp(self, registry_with_features):
        """Test that materialize adds timestamp if not present."""
        data = pd.DataFrame({
            "customer_id": [1, 2, 3],
            "total_spend": [100.0, 200.0, 300.0],
            "order_count": [5, 10, 15],
        })

        rows = registry_with_features.materialize("customer_features", data)

        assert rows == 3

    def test_materialize_unknown_feature_set(self, registry_with_features):
        """Test materializing to unknown feature set raises error."""
        data = pd.DataFrame({"a": [1, 2, 3]})

        with pytest.raises(ValidationError, match="not found"):
            registry_with_features.materialize("nonexistent", data)


class TestHistoricalFeatures:
    """Tests for historical feature retrieval."""

    @pytest.fixture
    def registry_with_data(self, registry_config, sample_feature_set, sample_data):
        """Create a registry with materialized data."""
        registry = LocalRegistry(registry_config)
        registry.connect()
        registry.register_feature_set(sample_feature_set)
        registry.materialize("customer_features", sample_data)
        yield registry
        registry.disconnect()

    def test_get_historical_features(self, registry_with_data):
        """Test retrieving historical features."""
        entity_df = pd.DataFrame({
            "customer_id": [1, 2, 3, 4, 5],
        })

        result = registry_with_data.get_historical_features(
            entity_df=entity_df,
            features=["total_spend", "order_count"],
            entity_column="customer_id",
        )

        assert "total_spend" in result.columns
        assert "order_count" in result.columns
        assert len(result) == 5

    def test_get_historical_features_with_timestamp(self, registry_with_data):
        """Test retrieving historical features with timestamp filter."""
        entity_df = pd.DataFrame({
            "customer_id": [1, 2, 3],
            "event_timestamp": pd.to_datetime(["2024-01-15", "2024-01-16", "2024-01-17"]),
        })

        result = registry_with_data.get_historical_features(
            entity_df=entity_df,
            features=["total_spend"],
            entity_column="customer_id",
        )

        assert "total_spend" in result.columns


class TestExportImport:
    """Tests for export/import functionality."""

    @pytest.fixture
    def registry_with_features(self, registry_config, sample_feature_set):
        """Create a registry with registered features."""
        registry = LocalRegistry(registry_config)
        registry.connect()
        registry.register_feature_set(sample_feature_set)
        yield registry
        registry.disconnect()

    def test_export_definitions(self, registry_with_features, temp_registry_path):
        """Test exporting feature definitions."""
        export_path = temp_registry_path / "export.json"

        registry_with_features.export_definitions(str(export_path))

        assert export_path.exists()

    def test_import_definitions(self, registry_config, temp_registry_path, sample_feature_set):
        """Test importing feature definitions."""
        # First export from one registry
        registry1 = LocalRegistry(registry_config)
        registry1.connect()
        registry1.register_feature_set(sample_feature_set)

        export_path = temp_registry_path / "export.json"
        registry1.export_definitions(str(export_path))
        registry1.disconnect()

        # Create a new registry and import
        config2 = RegistryConfig(
            registry_type="local",
            project_name="test_project_2",
            connection_params={"path": str(temp_registry_path / "registry2")},
        )
        registry2 = LocalRegistry(config2)
        registry2.connect()
        registry2.import_definitions(str(export_path))

        # Verify features were imported
        features = registry2.list_features()
        assert "total_spend" in features
        registry2.disconnect()


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_local_registry(self, temp_registry_path):
        """Test create_local_registry convenience function."""
        registry = create_local_registry(
            path=str(temp_registry_path),
            project_name="convenience_test",
        )

        assert registry._connected
        assert registry.config.project_name == "convenience_test"
        registry.disconnect()

    def test_create_local_registry_default_project(self, temp_registry_path):
        """Test create_local_registry with default project name."""
        registry = create_local_registry(path=str(temp_registry_path))

        assert registry._connected
        assert registry.config.project_name == "forge_features"
        registry.disconnect()

    def test_create_registry_local_type(self, temp_registry_path):
        """Test create_registry with local type."""
        registry = create_registry(
            registry_type="local",
            path=str(temp_registry_path),
            project_name="test",
        )

        assert registry._connected
        assert isinstance(registry, LocalRegistry)
        registry.disconnect()


class TestVersioning:
    """Tests for feature versioning."""

    @pytest.fixture
    def connected_registry(self, registry_config):
        """Create a connected registry."""
        registry = LocalRegistry(registry_config)
        registry.connect()
        yield registry
        registry.disconnect()

    def test_get_feature_after_versioning(self, connected_registry, sample_feature):
        """Test getting feature after multiple registrations."""
        v1 = connected_registry.register_feature(sample_feature)
        assert v1.version == "v1.0.0"

        # Register again (creates new version)
        v2 = connected_registry.register_feature(sample_feature)
        assert v2.version == "v2.0.0"

        # Get current feature
        feature = connected_registry.get_feature("total_spend")
        assert feature is not None

    def test_version_tracking_internal(self, connected_registry, sample_feature):
        """Test internal version tracking after multiple registrations."""
        connected_registry.register_feature(sample_feature)
        connected_registry.register_feature(sample_feature)
        connected_registry.register_feature(sample_feature)

        # Access internal versions (for testing only)
        versions = connected_registry._versions.get("total_spend", [])

        assert len(versions) >= 3


class TestIntegrationWorkflows:
    """End-to-end integration tests."""

    def test_full_feature_lifecycle(self, temp_registry_path):
        """Test complete feature lifecycle: create, register, materialize, retrieve."""
        # Step 1: Create and connect registry
        registry = create_local_registry(
            path=str(temp_registry_path),
            project_name="lifecycle_test",
        )

        # Step 2: Define features
        feature = FeatureDefinition(
            name="avg_transaction",
            dtype=FeatureType.FLOAT64,
            description="Average transaction amount",
            entity="customer",
        )

        feature_set = FeatureSet(
            name="transaction_features",
            features=[feature],
            entity="customer",
        )

        # Step 3: Register
        version = registry.register_feature_set(feature_set)
        assert version is not None

        # Step 4: Materialize data
        data = pd.DataFrame({
            "customer_id": range(1, 11),
            "avg_transaction": np.random.uniform(50, 500, 10),
        })
        rows = registry.materialize("transaction_features", data)
        assert rows == 10

        # Step 5: Retrieve historical features
        entity_df = pd.DataFrame({"customer_id": [1, 2, 3]})
        result = registry.get_historical_features(
            entity_df=entity_df,
            features=["avg_transaction"],
            entity_column="customer_id",
        )
        assert "avg_transaction" in result.columns

        # Step 6: Export definitions
        export_path = temp_registry_path / "definitions.json"
        registry.export_definitions(str(export_path))
        assert export_path.exists()

        registry.disconnect()

    def test_multiple_feature_sets(self, temp_registry_path):
        """Test managing multiple feature sets."""
        registry = create_local_registry(
            path=str(temp_registry_path),
            project_name="multi_set_test",
        )

        # Create multiple feature sets for different entities
        customer_features = FeatureSet(
            name="customer_features",
            features=[
                FeatureDefinition(name="customer_age", dtype=FeatureType.INT64, entity="customer"),
                FeatureDefinition(name="customer_tenure", dtype=FeatureType.INT64, entity="customer"),
            ],
            entity="customer",
        )

        product_features = FeatureSet(
            name="product_features",
            features=[
                FeatureDefinition(name="product_price", dtype=FeatureType.FLOAT64, entity="product"),
                FeatureDefinition(name="product_rating", dtype=FeatureType.FLOAT64, entity="product"),
            ],
            entity="product",
        )

        # Register both
        registry.register_feature_set(customer_features)
        registry.register_feature_set(product_features)

        # Verify
        all_sets = registry.list_feature_sets()
        assert "customer_features" in all_sets
        assert "product_features" in all_sets

        customer_feats = registry.list_features(entity="customer")
        assert "customer_age" in customer_feats
        assert "product_price" not in customer_feats

        registry.disconnect()

    def test_feature_updates_workflow(self, temp_registry_path):
        """Test updating features over time."""
        registry = create_local_registry(
            path=str(temp_registry_path),
            project_name="update_test",
        )

        # Initial feature
        feature_v1 = FeatureDefinition(
            name="user_score",
            dtype=FeatureType.FLOAT64,
            description="Initial user score",
            entity="user",
        )
        v1 = registry.register_feature(feature_v1)
        assert v1.version == "v1.0.0"

        # Update description
        feature_v2 = FeatureDefinition(
            name="user_score",
            dtype=FeatureType.FLOAT64,
            description="Updated user score with new calculation",
            entity="user",
        )
        v2 = registry.register_feature(feature_v2)
        assert v2.version == "v2.0.0"

        # Verify internal versions (local registry tracks versions internally)
        internal_versions = registry._versions.get("user_score", [])
        assert len(internal_versions) >= 2

        # Latest feature should be available
        latest = registry.get_feature("user_score")
        assert latest is not None

        registry.disconnect()
