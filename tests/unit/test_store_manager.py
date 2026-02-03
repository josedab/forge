"""Tests for unified feature store integration manager."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from forge.exceptions import ConfigurationError
from forge.registry.base import FeatureDefinition, FeatureSet, FeatureType, FeatureVersion
from forge.registry.store_manager import (
    FeatureStoreManager,
    MigrationResult,
)


def _make_mock_registry(
    features: dict[str, FeatureDefinition] | None = None,
    feature_sets: dict[str, FeatureSet] | None = None,
) -> MagicMock:
    """Create a mock FeatureRegistry."""
    mock = MagicMock()
    features = features or {}
    feature_sets = feature_sets or {}

    mock.list_features.return_value = list(features.keys())
    mock.list_feature_sets.return_value = list(feature_sets.keys())
    mock.get_feature.side_effect = lambda name, version=None: features.get(name)
    mock.get_feature_set.side_effect = lambda name, version=None: feature_sets.get(name)
    mock.register_feature.return_value = FeatureVersion(
        version="v1", feature_hash="abc"
    )
    return mock


@pytest.fixture
def sample_features() -> dict[str, FeatureDefinition]:
    return {
        "log_price": FeatureDefinition(
            name="log_price", dtype=FeatureType.FLOAT64, description="Log of price"
        ),
        "age_bin": FeatureDefinition(
            name="age_bin", dtype=FeatureType.INT32, description="Age binned"
        ),
    }


class TestFeatureStoreManager:
    def test_register_and_get_store(self) -> None:
        manager = FeatureStoreManager()
        mock = _make_mock_registry()
        manager.register_store("dev", mock)
        assert manager.get_store("dev") is mock

    def test_default_store(self) -> None:
        manager = FeatureStoreManager()
        mock = _make_mock_registry()
        manager.register_store("first", mock)
        assert manager.get_store() is mock

    def test_get_missing_store_raises(self) -> None:
        manager = FeatureStoreManager()
        with pytest.raises(ConfigurationError):
            manager.get_store("missing")

    def test_store_names(self) -> None:
        manager = FeatureStoreManager()
        manager.register_store("a", _make_mock_registry())
        manager.register_store("b", _make_mock_registry())
        assert sorted(manager.store_names) == ["a", "b"]

    def test_discover_all(self, sample_features: dict[str, FeatureDefinition]) -> None:
        manager = FeatureStoreManager()
        manager.register_store("dev", _make_mock_registry(features=sample_features))
        manager.register_store("prod", _make_mock_registry())
        results = manager.discover_all()
        assert len(results) == 2
        dev_result = [r for r in results if r.store_name == "dev"][0]
        assert sorted(dev_result.features) == ["age_bin", "log_price"]

    def test_push_features(self, sample_features: dict[str, FeatureDefinition]) -> None:
        source = _make_mock_registry(features=sample_features)
        target = _make_mock_registry()
        manager = FeatureStoreManager()
        manager.register_store("dev", source)
        manager.register_store("prod", target)
        result = manager.push("dev", "prod")
        assert result.pushed == 2
        assert result.errors == 0

    def test_push_skips_existing(self, sample_features: dict[str, FeatureDefinition]) -> None:
        source = _make_mock_registry(features=sample_features)
        target = _make_mock_registry(features=sample_features)
        manager = FeatureStoreManager()
        manager.register_store("dev", source)
        manager.register_store("prod", target)
        result = manager.push("dev", "prod", overwrite=False)
        assert result.skipped == 2
        assert result.pushed == 0

    def test_push_overwrite(self, sample_features: dict[str, FeatureDefinition]) -> None:
        source = _make_mock_registry(features=sample_features)
        target = _make_mock_registry(features=sample_features)
        manager = FeatureStoreManager()
        manager.register_store("dev", source)
        manager.register_store("prod", target)
        result = manager.push("dev", "prod", overwrite=True)
        assert result.pushed == 2

    def test_push_specific_features(self, sample_features: dict[str, FeatureDefinition]) -> None:
        source = _make_mock_registry(features=sample_features)
        target = _make_mock_registry()
        manager = FeatureStoreManager()
        manager.register_store("dev", source)
        manager.register_store("prod", target)
        result = manager.push("dev", "prod", features=["log_price"])
        assert result.pushed == 1

    def test_diff(self, sample_features: dict[str, FeatureDefinition]) -> None:
        only_dev = {"dev_only": FeatureDefinition(name="dev_only", dtype=FeatureType.FLOAT64)}
        dev_features = {**sample_features, **only_dev}
        manager = FeatureStoreManager()
        manager.register_store("dev", _make_mock_registry(features=dev_features))
        manager.register_store("prod", _make_mock_registry(features=sample_features))
        diff = manager.diff("dev", "prod")
        assert "dev_only" in diff["only_in_a"]
        assert diff["only_in_b"] == []
        assert sorted(diff["in_both"]) == ["age_bin", "log_price"]


class TestMigrationResult:
    def test_summary(self) -> None:
        result = MigrationResult(source_store="dev", target_store="prod")
        s = result.summary()
        assert "dev" in s
        assert "prod" in s

    def test_empty_result(self) -> None:
        result = MigrationResult()
        assert result.pushed == 0
        assert result.skipped == 0
        assert result.errors == 0
