"""Tests for the Feature Store Integration Hub."""

from __future__ import annotations

import pytest

from forge.registry.base import FeatureDefinition, FeatureSet, FeatureType
from forge.registry.hub import (
    DatabricksRegistry,
    HopsworksRegistry,
    SageMakerRegistry,
    TectonRegistry,
    create_feature_store,
)


@pytest.fixture
def sample_feature():
    return FeatureDefinition(
        name="test_feature",
        dtype=FeatureType.FLOAT64,
        description="A test feature",
        source_columns=["col_a"],
        transformation="mean(col_a)",
        tags={"team": "ml"},
        owner="test",
        entity="user",
    )


@pytest.fixture
def sample_feature_set(sample_feature):
    return FeatureSet(
        name="test_set",
        features=[sample_feature],
        description="Test feature set",
        entity="user",
    )


class TestTectonRegistry:
    def test_register_and_retrieve(self, sample_feature):
        registry = TectonRegistry(workspace="test")
        registry._is_connected = True
        ver = registry.register_feature(sample_feature)
        assert ver.version == "1.0.0"
        assert registry.get_feature("test_feature") is sample_feature

    def test_register_feature_set(self, sample_feature_set):
        registry = TectonRegistry()
        registry._is_connected = True
        registry.register_feature_set(sample_feature_set)
        assert registry.get_feature_set("test_set") is not None
        assert len(registry.list_features()) == 1

    def test_list_with_tags(self, sample_feature):
        registry = TectonRegistry()
        registry._is_connected = True
        registry.register_feature(sample_feature)
        assert len(registry.list_features(tags={"team": "ml"})) == 1
        assert len(registry.list_features(tags={"team": "other"})) == 0

    def test_delete_feature(self, sample_feature):
        registry = TectonRegistry()
        registry._is_connected = True
        registry.register_feature(sample_feature)
        assert registry.delete_feature("test_feature")
        assert registry.get_feature("test_feature") is None
        assert not registry.delete_feature("nonexistent")


class TestHopsworksRegistry:
    def test_register_and_list(self, sample_feature, sample_feature_set):
        registry = HopsworksRegistry(project_name="test")
        registry._is_connected = True
        registry.register_feature_set(sample_feature_set)
        assert len(registry.list_feature_sets()) == 1
        assert len(registry.list_features()) == 1

    def test_disconnect(self):
        registry = HopsworksRegistry()
        registry._is_connected = True
        registry.disconnect()
        assert not registry._is_connected


class TestDatabricksRegistry:
    def test_register_and_retrieve(self, sample_feature):
        registry = DatabricksRegistry(catalog="test_catalog", schema="test_schema")
        registry._is_connected = True
        ver = registry.register_feature(sample_feature)
        assert ver.version == "1.0.0"
        result = registry.get_feature("test_feature")
        assert result is not None
        assert result.name == "test_feature"


class TestSageMakerRegistry:
    def test_register_and_list(self, sample_feature):
        registry = SageMakerRegistry(region="us-west-2")
        registry._is_connected = True
        registry.register_feature(sample_feature)
        features = registry.list_features()
        assert len(features) == 1

    def test_delete(self, sample_feature):
        registry = SageMakerRegistry()
        registry._is_connected = True
        registry.register_feature(sample_feature)
        assert registry.delete_feature("test_feature")
        assert len(registry.list_features()) == 0


class TestCreateFeatureStore:
    def test_create_tecton(self):
        store = create_feature_store("tecton", workspace="prod")
        assert isinstance(store, TectonRegistry)

    def test_create_hopsworks(self):
        store = create_feature_store("hopsworks", project_name="test")
        assert isinstance(store, HopsworksRegistry)

    def test_create_databricks(self):
        store = create_feature_store("databricks", catalog="ml")
        assert isinstance(store, DatabricksRegistry)

    def test_create_sagemaker(self):
        store = create_feature_store("sagemaker", region="eu-west-1")
        assert isinstance(store, SageMakerRegistry)

    def test_unknown_backend(self):
        with pytest.raises(Exception, match="Unknown feature store"):
            create_feature_store("unknown_store")

    def test_case_insensitive(self):
        store = create_feature_store("TECTON", workspace="test")
        assert isinstance(store, TectonRegistry)
