"""Tests for feature store sync layer."""

from __future__ import annotations

import pandas as pd
import pytest

from forge.registry.base import (
    FeatureDefinition,
    FeatureSet,
    FeatureType,
    RegistryConfig,
)
from forge.registry.local_registry import LocalRegistry
from forge.registry.sync import (
    ConflictStrategy,
    FeatureStoreSync,
    SyncConfig,
    SyncDirection,
    check_type_compatible,
)


@pytest.fixture
def source_registry(tmp_path):
    """Create a source local registry."""
    config = RegistryConfig(
        registry_type="local",
        project_name="source_project",
        connection_params={"path": str(tmp_path / "source")},
    )
    reg = LocalRegistry(config)
    reg.connect()
    return reg


@pytest.fixture
def target_registry(tmp_path):
    """Create a target local registry."""
    config = RegistryConfig(
        registry_type="local",
        project_name="target_project",
        connection_params={"path": str(tmp_path / "target")},
    )
    reg = LocalRegistry(config)
    reg.connect()
    return reg


@pytest.fixture
def sample_features():
    """Create sample feature definitions."""
    return [
        FeatureDefinition(
            name="user_age",
            dtype=FeatureType.INT64,
            description="User age in years",
            source_columns=["birth_date"],
            transformation="age_from_birthdate",
            entity="user",
        ),
        FeatureDefinition(
            name="total_spend",
            dtype=FeatureType.FLOAT64,
            description="Total user spending",
            source_columns=["transactions"],
            transformation="sum",
            entity="user",
        ),
    ]


@pytest.fixture
def sample_feature_set(sample_features):
    """Create a sample feature set."""
    return FeatureSet(
        name="user_features",
        features=sample_features,
        entity="user",
        description="User feature set",
    )


class TestTypeCompatibility:
    """Tests for type compatibility checking."""

    def test_same_type_compatible(self):
        assert check_type_compatible(FeatureType.FLOAT64, FeatureType.FLOAT64)

    def test_int32_to_float64_compatible(self):
        assert check_type_compatible(FeatureType.INT32, FeatureType.FLOAT64)

    def test_float64_to_int32_incompatible(self):
        assert not check_type_compatible(FeatureType.FLOAT64, FeatureType.INT32)

    def test_bool_to_int_compatible(self):
        assert check_type_compatible(FeatureType.BOOL, FeatureType.INT32)

    def test_string_to_float_incompatible(self):
        assert not check_type_compatible(FeatureType.STRING, FeatureType.FLOAT64)


class TestFeatureStoreSync:
    """Tests for FeatureStoreSync."""

    def test_init_different_registries(self, source_registry, target_registry):
        sync = FeatureStoreSync(source=source_registry, target=target_registry)
        assert sync.source is source_registry
        assert sync.target is target_registry

    def test_init_same_registry_raises(self, source_registry):
        with pytest.raises(Exception):
            FeatureStoreSync(source=source_registry, target=source_registry)

    def test_push_features(
        self, source_registry, target_registry, sample_features, sample_feature_set
    ):
        # Register in source
        source_registry.register_feature_set(sample_feature_set)
        for f in sample_features:
            source_registry.register_feature(f)

        sync = FeatureStoreSync(source=source_registry, target=target_registry)
        result = sync.push(feature_set_name="user_features")

        assert result.success
        assert result.created_count == 2
        assert result.direction == SyncDirection.PUSH

        # Verify in target
        assert target_registry.get_feature("user_age") is not None
        assert target_registry.get_feature("total_spend") is not None

    def test_push_idempotent(
        self, source_registry, target_registry, sample_features, sample_feature_set
    ):
        source_registry.register_feature_set(sample_feature_set)
        for f in sample_features:
            source_registry.register_feature(f)

        sync = FeatureStoreSync(source=source_registry, target=target_registry)

        result1 = sync.push(feature_set_name="user_features")
        assert result1.created_count == 2

        result2 = sync.push(feature_set_name="user_features")
        assert result2.skipped_count == 2
        assert result2.created_count == 0

    def test_push_with_conflict_skip(
        self, source_registry, target_registry, sample_features, sample_feature_set
    ):
        source_registry.register_feature_set(sample_feature_set)
        for f in sample_features:
            source_registry.register_feature(f)

        # Register a different version in target
        modified = FeatureDefinition(
            name="user_age",
            dtype=FeatureType.FLOAT64,
            description="Modified",
            entity="user",
        )
        target_registry.register_feature(modified)

        config = SyncConfig(conflict_strategy=ConflictStrategy.SKIP)
        sync = FeatureStoreSync(
            source=source_registry, target=target_registry, config=config
        )
        result = sync.push(feature_set_name="user_features")

        # user_age skipped, total_spend created
        assert result.skipped_count >= 1
        assert result.created_count >= 1

    def test_push_with_conflict_error(
        self, source_registry, target_registry, sample_features, sample_feature_set
    ):
        source_registry.register_feature_set(sample_feature_set)
        for f in sample_features:
            source_registry.register_feature(f)

        modified = FeatureDefinition(
            name="user_age",
            dtype=FeatureType.FLOAT64,
            description="Modified",
            entity="user",
        )
        target_registry.register_feature(modified)

        config = SyncConfig(conflict_strategy=ConflictStrategy.ERROR)
        sync = FeatureStoreSync(
            source=source_registry, target=target_registry, config=config
        )
        result = sync.push(feature_set_name="user_features")

        assert result.error_count >= 1

    def test_push_with_conflict_overwrite(
        self, source_registry, target_registry, sample_features, sample_feature_set
    ):
        source_registry.register_feature_set(sample_feature_set)
        for f in sample_features:
            source_registry.register_feature(f)

        modified = FeatureDefinition(
            name="user_age",
            dtype=FeatureType.FLOAT64,
            description="Modified",
            entity="user",
        )
        target_registry.register_feature(modified)

        config = SyncConfig(conflict_strategy=ConflictStrategy.OVERWRITE)
        sync = FeatureStoreSync(
            source=source_registry, target=target_registry, config=config
        )
        result = sync.push(feature_set_name="user_features")

        assert result.error_count == 0

    def test_dry_run(
        self, source_registry, target_registry, sample_features, sample_feature_set
    ):
        source_registry.register_feature_set(sample_feature_set)
        for f in sample_features:
            source_registry.register_feature(f)

        config = SyncConfig(dry_run=True)
        sync = FeatureStoreSync(
            source=source_registry, target=target_registry, config=config
        )
        result = sync.push(feature_set_name="user_features")

        assert result.created_count == 2
        # Target should still be empty
        assert len(target_registry.list_features()) == 0

    def test_pull_features(
        self, source_registry, target_registry, sample_features, sample_feature_set
    ):
        # Register in target
        target_registry.register_feature_set(sample_feature_set)
        for f in sample_features:
            target_registry.register_feature(f)

        sync = FeatureStoreSync(source=source_registry, target=target_registry)
        result = sync.pull(feature_set_name="user_features")

        assert result.success
        assert result.created_count == 2
        assert result.direction == SyncDirection.PULL

    def test_diff_identifies_differences(
        self, source_registry, target_registry, sample_features, sample_feature_set
    ):
        source_registry.register_feature_set(sample_feature_set)
        for f in sample_features:
            source_registry.register_feature(f)

        # Only register one in target
        target_registry.register_feature(sample_features[0])

        sync = FeatureStoreSync(source=source_registry, target=target_registry)
        diffs = sync.diff(feature_set_name="user_features")

        assert len(diffs) >= 1
        source_only = [d for d in diffs if d["status"] == "source_only"]
        assert len(source_only) >= 1

    def test_register_from_dataframe(self, source_registry, target_registry):
        data = pd.DataFrame({
            "entity_id": ["a", "b", "c"],
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [4, 5, 6],
        })

        sync = FeatureStoreSync(source=source_registry, target=target_registry)
        result = sync.register_from_dataframe(
            data=data,
            feature_set_name="test_features",
            entity_column="entity_id",
        )

        assert result.success
        assert result.created_count == 2  # feature1, feature2

    def test_lineage_tracking(
        self, source_registry, target_registry, sample_features, sample_feature_set
    ):
        source_registry.register_feature_set(sample_feature_set)
        for f in sample_features:
            source_registry.register_feature(f)

        sync = FeatureStoreSync(source=source_registry, target=target_registry)
        sync.push(feature_set_name="user_features")

        lineage = sync.get_lineage()
        assert len(lineage) == 2

        age_lineage = sync.get_lineage("user_age")
        assert len(age_lineage) == 1
        assert age_lineage[0].source_registry == "local"

    def test_validate_schema_compatibility(
        self, source_registry, target_registry, sample_features, sample_feature_set
    ):
        source_registry.register_feature_set(sample_feature_set)
        for f in sample_features:
            source_registry.register_feature(f)

        # Register incompatible type in target
        incompatible = FeatureDefinition(
            name="user_age",
            dtype=FeatureType.STRING,
            entity="user",
        )
        target_registry.register_feature(incompatible)

        sync = FeatureStoreSync(source=source_registry, target=target_registry)
        errors = sync.validate_schema_compatibility("user_features")

        assert len(errors) >= 1
        assert "user_age" in errors[0]

    def test_sync_result_summary(self):
        from forge.registry.sync import SyncResult

        result = SyncResult(direction=SyncDirection.PUSH)
        summary = result.summary()
        assert "push" in summary.lower() or "Sync" in summary

    def test_push_with_data_materialization(
        self, source_registry, target_registry, sample_features, sample_feature_set
    ):
        source_registry.register_feature_set(sample_feature_set)
        for f in sample_features:
            source_registry.register_feature(f)

        data = pd.DataFrame({
            "user_age": [25, 30, 35],
            "total_spend": [100.0, 200.0, 300.0],
        })

        sync = FeatureStoreSync(source=source_registry, target=target_registry)
        result = sync.push(
            feature_set_name="user_features",
            data=data,
        )

        assert result.success
        assert result.rows_synced == 3
