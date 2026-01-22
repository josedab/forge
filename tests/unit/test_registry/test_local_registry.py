"""Tests for local feature registry."""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from forge.registry.base import (
    FeatureDefinition,
    FeatureSet,
    FeatureType,
    RegistryConfig,
    infer_feature_type,
    create_feature_definitions_from_dataframe,
)
from forge.registry.local_registry import LocalRegistry, create_local_registry


@pytest.fixture
def temp_registry_path():
    """Create a temporary directory for registry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_registry"


@pytest.fixture
def local_registry(temp_registry_path):
    """Create a local registry for testing."""
    config = RegistryConfig(
        registry_type="local",
        project_name="test_project",
        connection_params={"path": str(temp_registry_path)},
    )
    registry = LocalRegistry(config)
    registry.connect()
    yield registry
    registry.disconnect()


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


class TestFeatureDefinition:
    """Tests for FeatureDefinition class."""

    def test_to_dict(self, sample_feature):
        """Test conversion to dictionary."""
        data = sample_feature.to_dict()

        assert data["name"] == "test_feature"
        assert data["dtype"] == "float64"
        assert "col1" in data["source_columns"]

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "name": "test",
            "dtype": "float64",
            "description": "Test",
            "source_columns": ["a"],
            "transformation": "log",
            "tags": {},
            "owner": "",
            "entity": "item",
            "created_at": "2024-01-01T00:00:00",
        }

        feature = FeatureDefinition.from_dict(data)
        assert feature.name == "test"
        assert feature.dtype == FeatureType.FLOAT64

    def test_compute_hash(self, sample_feature):
        """Test hash computation."""
        hash1 = sample_feature.compute_hash()
        hash2 = sample_feature.compute_hash()

        assert hash1 == hash2
        assert len(hash1) == 12


class TestFeatureSet:
    """Tests for FeatureSet class."""

    def test_to_dict(self, sample_feature_set):
        """Test conversion to dictionary."""
        data = sample_feature_set.to_dict()

        assert data["name"] == "test_feature_set"
        assert len(data["features"]) == 2

    def test_add_feature(self, sample_feature_set):
        """Test adding a feature."""
        new_feature = FeatureDefinition(
            name="new_feature",
            dtype=FeatureType.STRING,
        )
        sample_feature_set.add_feature(new_feature)

        assert len(sample_feature_set.features) == 3

    def test_get_feature(self, sample_feature_set):
        """Test getting a feature by name."""
        feature = sample_feature_set.get_feature("test_feature")
        assert feature is not None
        assert feature.name == "test_feature"

    def test_feature_names(self, sample_feature_set):
        """Test getting feature names."""
        names = sample_feature_set.feature_names
        assert "test_feature" in names
        assert "feature_2" in names


class TestLocalRegistry:
    """Tests for LocalRegistry class."""

    def test_connect(self, temp_registry_path):
        """Test connection creates directory structure."""
        config = RegistryConfig(
            registry_type="local",
            connection_params={"path": str(temp_registry_path)},
        )
        registry = LocalRegistry(config)
        registry.connect()

        assert (temp_registry_path / "features").exists()
        assert (temp_registry_path / "feature_sets").exists()
        assert (temp_registry_path / "data").exists()

        registry.disconnect()

    def test_register_feature(self, local_registry, sample_feature):
        """Test registering a feature."""
        version = local_registry.register_feature(sample_feature)

        assert version.version == "v1.0.0"
        assert local_registry.get_feature("test_feature") is not None

    def test_register_feature_set(self, local_registry, sample_feature_set):
        """Test registering a feature set."""
        version = local_registry.register_feature_set(sample_feature_set)

        assert version is not None
        assert local_registry.get_feature_set("test_feature_set") is not None

    def test_list_features(self, local_registry, sample_feature):
        """Test listing features."""
        local_registry.register_feature(sample_feature)
        features = local_registry.list_features()

        assert "test_feature" in features

    def test_list_features_by_entity(self, local_registry, sample_feature):
        """Test listing features filtered by entity."""
        local_registry.register_feature(sample_feature)
        features = local_registry.list_features(entity="customer")

        assert "test_feature" in features

        features = local_registry.list_features(entity="nonexistent")
        assert len(features) == 0

    def test_delete_feature(self, local_registry, sample_feature):
        """Test deleting a feature."""
        local_registry.register_feature(sample_feature)
        result = local_registry.delete_feature("test_feature")

        assert result is True
        assert local_registry.get_feature("test_feature") is None

    def test_materialize(self, local_registry, sample_feature_set):
        """Test materializing data."""
        local_registry.register_feature_set(sample_feature_set)

        data = pd.DataFrame({
            "customer_id": [1, 2, 3],
            "test_feature": [1.0, 2.0, 3.0],
            "feature_2": [10, 20, 30],
        })

        rows = local_registry.materialize("test_feature_set", data)
        assert rows == 3

    def test_get_historical_features(self, local_registry, sample_feature_set):
        """Test getting historical features."""
        local_registry.register_feature_set(sample_feature_set)

        data = pd.DataFrame({
            "customer_id": [1, 2, 3],
            "test_feature": [1.0, 2.0, 3.0],
            "feature_2": [10, 20, 30],
        })
        local_registry.materialize("test_feature_set", data)

        entity_df = pd.DataFrame({"customer_id": [1, 2]})
        result = local_registry.get_historical_features(
            entity_df=entity_df,
            features=["test_feature"],
            entity_column="customer_id",
        )

        assert "test_feature" in result.columns
        assert len(result) == 2

    def test_export_import_definitions(self, local_registry, sample_feature_set):
        """Test exporting and importing definitions."""
        local_registry.register_feature_set(sample_feature_set)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            export_path = f.name

        local_registry.export_definitions(export_path)

        # Create new registry and import
        with tempfile.TemporaryDirectory() as tmpdir:
            new_config = RegistryConfig(
                registry_type="local",
                connection_params={"path": tmpdir},
            )
            new_registry = LocalRegistry(new_config)
            new_registry.connect()

            count = new_registry.import_definitions(export_path)
            assert count > 0

            new_registry.disconnect()


class TestInferFeatureType:
    """Tests for infer_feature_type function."""

    def test_infer_int64(self):
        """Test inferring int64 type."""
        series = pd.Series([1, 2, 3], dtype="int64")
        assert infer_feature_type(series) == FeatureType.INT64

    def test_infer_float64(self):
        """Test inferring float64 type."""
        series = pd.Series([1.0, 2.0, 3.0], dtype="float64")
        assert infer_feature_type(series) == FeatureType.FLOAT64

    def test_infer_string(self):
        """Test inferring string type."""
        series = pd.Series(["a", "b", "c"])
        assert infer_feature_type(series) == FeatureType.STRING

    def test_infer_bool(self):
        """Test inferring bool type."""
        series = pd.Series([True, False, True], dtype="bool")
        assert infer_feature_type(series) == FeatureType.BOOL

    def test_infer_datetime(self):
        """Test inferring datetime type."""
        series = pd.to_datetime(pd.Series(["2024-01-01", "2024-01-02"]))
        assert infer_feature_type(series) == FeatureType.DATETIME


class TestCreateFeatureDefinitionsFromDataFrame:
    """Tests for create_feature_definitions_from_dataframe function."""

    def test_basic_creation(self):
        """Test basic definition creation."""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [1.0, 2.0, 3.0],
            "c": ["x", "y", "z"],
        })

        definitions = create_feature_definitions_from_dataframe(df)

        assert len(definitions) == 3
        names = [d.name for d in definitions]
        assert "a" in names
        assert "b" in names
        assert "c" in names

    def test_exclude_columns(self):
        """Test excluding columns."""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [1.0, 2.0, 3.0],
            "id": [1, 2, 3],
        })

        definitions = create_feature_definitions_from_dataframe(
            df, exclude_columns=["id"]
        )

        names = [d.name for d in definitions]
        assert "id" not in names


class TestCreateLocalRegistry:
    """Tests for create_local_registry convenience function."""

    def test_creates_connected_registry(self, temp_registry_path):
        """Test that function creates a connected registry."""
        registry = create_local_registry(path=str(temp_registry_path))

        assert registry._connected
        registry.disconnect()
