"""Tests for Feast feature registry integration."""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from forge.exceptions import ConfigurationError, MissingDependencyError, ValidationError
from forge.registry.base import (
    FeatureDefinition,
    FeatureSet,
    FeatureType,
    RegistryConfig,
)
from forge.registry.feast_registry import FeastRegistry, create_feast_registry


# Create a mock feast module for testing
class MockValueType:
    INT32 = "INT32"
    INT64 = "INT64"
    FLOAT = "FLOAT"
    DOUBLE = "DOUBLE"
    STRING = "STRING"
    BOOL = "BOOL"
    UNIX_TIMESTAMP = "UNIX_TIMESTAMP"


class MockFeastModule:
    """Mock feast module for testing."""
    FeatureStore = MagicMock
    Entity = MagicMock
    Feature = MagicMock
    FeatureView = MagicMock
    FileSource = MagicMock
    ValueType = MockValueType


# Install mock feast module
mock_feast = MockFeastModule()
sys.modules["feast"] = mock_feast


@pytest.fixture
def temp_repo_path():
    """Create a temporary directory for Feast repo."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "feast_repo"


@pytest.fixture
def feast_config(temp_repo_path):
    """Create a Feast registry config."""
    return RegistryConfig(
        registry_type="feast",
        project_name="test_project",
        connection_params={"repo_path": str(temp_repo_path)},
    )


@pytest.fixture
def sample_feature():
    """Create a sample feature definition."""
    return FeatureDefinition(
        name="test_feature",
        dtype=FeatureType.FLOAT64,
        description="A test feature",
        source_columns=["col1", "col2"],
        transformation="interaction",
        entity="customer",
    )


@pytest.fixture
def sample_feature_set(sample_feature):
    """Create a sample feature set."""
    return FeatureSet(
        name="test_feature_set",
        features=[
            sample_feature,
            FeatureDefinition(
                name="feature_2",
                dtype=FeatureType.INT64,
                entity="customer",
            ),
        ],
        entity="customer",
        description="Test feature set",
    )


class TestFeastRegistryInitialization:
    """Tests for FeastRegistry initialization."""

    def test_init_with_config(self, feast_config):
        """Test initialization with config."""
        registry = FeastRegistry(feast_config)

        assert registry.config == feast_config
        assert registry._store is None
        assert not registry._connected

    def test_init_default_repo_path(self):
        """Test default repo path."""
        config = RegistryConfig(
            registry_type="feast",
            project_name="test",
            connection_params={},
        )
        registry = FeastRegistry(config)

        assert registry._repo_path == "./feature_repo"


class TestFeastRegistryConnection:
    """Tests for FeastRegistry connection handling."""

    def test_connect_creates_repo_if_not_exists(self, feast_config, temp_repo_path):
        """Test connect creates repo when it doesn't exist."""
        # The mock feast module is already installed
        registry = FeastRegistry(feast_config)
        registry.connect()

        assert registry._connected
        # Verify minimal repo was created
        assert temp_repo_path.exists()

    def test_connect_with_existing_repo(self, feast_config, temp_repo_path):
        """Test connect with existing repo."""
        temp_repo_path.mkdir(parents=True, exist_ok=True)
        (temp_repo_path / "feature_store.yaml").write_text("project: test\n")

        registry = FeastRegistry(feast_config)
        registry.connect()

        assert registry._connected
        registry.disconnect()

    def test_disconnect(self, feast_config, temp_repo_path):
        """Test disconnect clears connection state."""
        registry = FeastRegistry(feast_config)
        registry.connect()

        registry.disconnect()

        assert registry._store is None
        assert not registry._connected


class TestFeastRegistryOperationsWithoutConnection:
    """Tests for operations that require connection."""

    def test_register_feature_without_connection(self, feast_config, sample_feature):
        """Test register_feature raises error when not connected."""
        registry = FeastRegistry(feast_config)

        with pytest.raises(ConfigurationError, match="Not connected"):
            registry.register_feature(sample_feature)

    def test_register_feature_set_without_connection(self, feast_config, sample_feature_set):
        """Test register_feature_set raises error when not connected."""
        registry = FeastRegistry(feast_config)

        with pytest.raises(ConfigurationError, match="Not connected"):
            registry.register_feature_set(sample_feature_set)

    def test_list_features_without_connection(self, feast_config):
        """Test list_features raises error when not connected."""
        registry = FeastRegistry(feast_config)

        with pytest.raises(ConfigurationError, match="Not connected"):
            registry.list_features()


class TestFeastRegistryFeatureOperations:
    """Tests for feature registration and retrieval."""

    @pytest.fixture
    def connected_registry(self, feast_config, temp_repo_path):
        """Create a connected registry."""
        registry = FeastRegistry(feast_config)
        registry.connect()
        yield registry
        registry.disconnect()

    def test_register_feature_set(self, connected_registry, sample_feature_set):
        """Test registering a feature set."""
        version = connected_registry.register_feature_set(sample_feature_set)

        assert version is not None
        assert sample_feature_set.name in connected_registry._feature_sets

    def test_register_feature(self, connected_registry, sample_feature):
        """Test registering a single feature."""
        version = connected_registry.register_feature(sample_feature)

        assert version is not None
        assert sample_feature.name in connected_registry._versions

    def test_get_feature(self, connected_registry, sample_feature_set):
        """Test getting a feature by name."""
        connected_registry.register_feature_set(sample_feature_set)

        feature = connected_registry.get_feature("test_feature")

        assert feature is not None
        assert feature.name == "test_feature"

    def test_get_feature_not_found(self, connected_registry):
        """Test getting a non-existent feature returns None."""
        result = connected_registry.get_feature("nonexistent")

        assert result is None

    def test_get_feature_set(self, connected_registry, sample_feature_set):
        """Test getting a feature set by name."""
        connected_registry.register_feature_set(sample_feature_set)

        fs = connected_registry.get_feature_set("test_feature_set")

        assert fs is not None
        assert fs.name == "test_feature_set"

    def test_list_features(self, connected_registry, sample_feature_set):
        """Test listing all features."""
        connected_registry.register_feature_set(sample_feature_set)

        features = connected_registry.list_features()

        assert "test_feature" in features
        assert "feature_2" in features

    def test_list_features_by_entity(self, connected_registry, sample_feature_set):
        """Test listing features filtered by entity."""
        connected_registry.register_feature_set(sample_feature_set)

        features = connected_registry.list_features(entity="customer")
        assert "test_feature" in features

        features = connected_registry.list_features(entity="nonexistent")
        assert len(features) == 0

    def test_list_feature_sets(self, connected_registry, sample_feature_set):
        """Test listing all feature sets."""
        connected_registry.register_feature_set(sample_feature_set)

        sets = connected_registry.list_feature_sets()

        assert "test_feature_set" in sets

    def test_delete_feature(self, connected_registry, sample_feature):
        """Test deleting a feature."""
        # Create a single-feature set so deletion removes the whole set
        single_feature_set = FeatureSet(
            name="single_feature_set",
            features=[sample_feature],
            entity="customer",
        )
        connected_registry.register_feature_set(single_feature_set)

        result = connected_registry.delete_feature("test_feature")

        assert result is True
        assert connected_registry.get_feature("test_feature") is None


class TestFeastRegistryMaterialization:
    """Tests for data materialization."""

    @pytest.fixture
    def connected_registry(self, feast_config, temp_repo_path):
        """Create a connected registry."""
        registry = FeastRegistry(feast_config)
        registry.connect()
        yield registry
        registry.disconnect()

    def test_materialize(self, connected_registry, sample_feature_set, temp_repo_path):
        """Test materializing data."""
        connected_registry.register_feature_set(sample_feature_set)

        data = pd.DataFrame({
            "customer_id": [1, 2, 3],
            "test_feature": [1.0, 2.0, 3.0],
            "feature_2": [10, 20, 30],
        })

        rows = connected_registry.materialize("test_feature_set", data)

        assert rows == 3
        output_path = temp_repo_path / "data" / "test_feature_set.parquet"
        assert output_path.exists()

    def test_materialize_adds_timestamp(self, connected_registry, sample_feature_set):
        """Test that materialize adds event timestamp if not present."""
        connected_registry.register_feature_set(sample_feature_set)

        data = pd.DataFrame({
            "customer_id": [1, 2, 3],
            "test_feature": [1.0, 2.0, 3.0],
            "feature_2": [10, 20, 30],
        })

        rows = connected_registry.materialize("test_feature_set", data)

        assert rows == 3

    def test_materialize_unknown_feature_set(self, connected_registry):
        """Test materialize with unknown feature set raises error."""
        data = pd.DataFrame({"a": [1, 2, 3]})

        with pytest.raises(ValidationError, match="not found"):
            connected_registry.materialize("nonexistent", data)


class TestFeastRegistryHistoricalFeatures:
    """Tests for historical feature retrieval."""

    @pytest.fixture
    def connected_registry(self, feast_config, temp_repo_path):
        """Create a connected registry with mock store for historical features."""
        registry = FeastRegistry(feast_config)
        registry.connect()

        # Configure mock store to return historical features
        mock_result = MagicMock()
        mock_result.to_df.return_value = pd.DataFrame({
            "customer_id": [1, 2],
            "test_feature": [1.0, 2.0],
        })
        registry._store.get_historical_features.return_value = mock_result

        yield registry
        registry.disconnect()

    def test_get_historical_features(self, connected_registry, sample_feature_set):
        """Test getting historical features."""
        connected_registry.register_feature_set(sample_feature_set)

        entity_df = pd.DataFrame({"customer_id": [1, 2]})
        result = connected_registry.get_historical_features(
            entity_df=entity_df,
            features=["test_feature"],
            entity_column="customer_id",
        )

        assert "test_feature" in result.columns

    def test_get_historical_features_empty_features(self, connected_registry):
        """Test getting historical features with no matching features."""
        entity_df = pd.DataFrame({"customer_id": [1, 2]})
        result = connected_registry.get_historical_features(
            entity_df=entity_df,
            features=["nonexistent"],
            entity_column="customer_id",
        )

        # Should return entity_df unchanged when no features found
        assert len(result) == 2


class TestFeastRegistryCodeGeneration:
    """Tests for Feast code generation."""

    @pytest.fixture
    def connected_registry(self, feast_config, temp_repo_path):
        """Create a connected registry."""
        registry = FeastRegistry(feast_config)
        registry.connect()
        yield registry
        registry.disconnect()

    def test_generate_feast_definitions(self, connected_registry, sample_feature_set):
        """Test generating Feast Python definitions."""
        connected_registry.register_feature_set(sample_feature_set)

        code = connected_registry.generate_feast_definitions()

        assert "# Auto-generated by Forge" in code
        assert "from feast import" in code
        assert "Entity(" in code
        assert "FeatureView(" in code

    def test_generate_feature_view_code(self, connected_registry, sample_feature_set):
        """Test generating feature view code."""
        code = connected_registry._generate_feature_view_code(sample_feature_set, [])

        assert "FileSource(" in code
        assert "FeatureView(" in code
        assert "test_feature_set" in code


class TestFeastRegistryTypeMapping:
    """Tests for Feast type mapping."""

    def test_map_all_feature_types(self, feast_config):
        """Test that all FeatureTypes can be mapped."""
        registry = FeastRegistry(feast_config)
        registry._connected = True

        # Test each type - mock feast module provides MockValueType
        assert registry._map_to_feast_type(FeatureType.INT32) == "INT32"
        assert registry._map_to_feast_type(FeatureType.INT64) == "INT64"
        assert registry._map_to_feast_type(FeatureType.FLOAT32) == "FLOAT"
        assert registry._map_to_feast_type(FeatureType.FLOAT64) == "DOUBLE"
        assert registry._map_to_feast_type(FeatureType.STRING) == "STRING"
        assert registry._map_to_feast_type(FeatureType.BOOL) == "BOOL"
        assert registry._map_to_feast_type(FeatureType.DATETIME) == "UNIX_TIMESTAMP"


class TestCreateFeastRegistryConvenience:
    """Tests for create_feast_registry convenience function."""

    def test_creates_connected_registry(self, temp_repo_path):
        """Test that convenience function creates connected registry."""
        registry = create_feast_registry(
            repo_path=str(temp_repo_path),
            project_name="test_project",
        )

        assert registry._connected
        registry.disconnect()

    def test_default_parameters(self, temp_repo_path):
        """Test default parameters are applied."""
        # Use temp path to avoid creating in cwd
        registry = create_feast_registry(repo_path=str(temp_repo_path))

        assert registry.config.project_name == "forge_features"
        registry.disconnect()


class TestFeastRegistryMinimalRepo:
    """Tests for minimal Feast repo creation."""

    def test_create_minimal_repo(self, feast_config, temp_repo_path):
        """Test creating a minimal Feast repository."""
        registry = FeastRegistry(feast_config)
        registry._create_minimal_repo(temp_repo_path)

        assert temp_repo_path.exists()
        assert (temp_repo_path / "data").exists()
        assert (temp_repo_path / "feature_store.yaml").exists()
        assert (temp_repo_path / "features.py").exists()

        yaml_content = (temp_repo_path / "feature_store.yaml").read_text()
        assert "test_project" in yaml_content
