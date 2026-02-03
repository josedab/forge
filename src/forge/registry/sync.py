"""Unified feature store synchronization layer.

Provides bi-directional sync between Forge-generated features and
production feature stores, with schema mapping, lineage tracking,
and conflict resolution.

Example:
    >>> from forge.registry.sync import FeatureStoreSync, SyncConfig
    >>> sync = FeatureStoreSync(source=local_registry, target=feast_registry)
    >>> result = sync.push(feature_set_name="user_features", data=X_train)
    >>> print(result.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd

from forge.exceptions import ConfigurationError, ValidationError
from forge.registry.base import (
    FeatureDefinition,
    FeatureRegistry,
    FeatureSet,
    FeatureType,
    infer_feature_type,
)

logger = logging.getLogger(__name__)


class ConflictStrategy(str, Enum):
    """Strategy for handling schema conflicts during sync."""

    OVERWRITE = "overwrite"
    SKIP = "skip"
    RENAME = "rename"
    ERROR = "error"


class SyncDirection(str, Enum):
    """Direction of feature synchronization."""

    PUSH = "push"
    PULL = "pull"


@dataclass
class SyncConfig:
    """Configuration for feature store synchronization.

    Args:
        conflict_strategy: How to handle conflicts during sync.
        dry_run: If True, report what would change without applying.
        include_data: Whether to sync materialized data (not just definitions).
        batch_size: Number of features to sync at once.
        validate_schemas: Whether to validate schema compatibility before sync.
    """

    conflict_strategy: ConflictStrategy = ConflictStrategy.ERROR
    dry_run: bool = False
    include_data: bool = True
    batch_size: int = 100
    validate_schemas: bool = True


@dataclass
class SyncAction:
    """A single action taken during synchronization."""

    feature_name: str
    action: str  # "created", "updated", "skipped", "renamed", "error"
    direction: SyncDirection = SyncDirection.PUSH
    details: str = ""
    old_version: str | None = None
    new_version: str | None = None


@dataclass
class LineageRecord:
    """Tracks provenance of a feature across stores."""

    feature_name: str
    source_registry: str
    target_registry: str
    source_columns: list[str] = field(default_factory=list)
    transformation: str = ""
    synced_at: datetime = field(default_factory=datetime.now)
    sync_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "feature_name": self.feature_name,
            "source_registry": self.source_registry,
            "target_registry": self.target_registry,
            "source_columns": self.source_columns,
            "transformation": self.transformation,
            "synced_at": self.synced_at.isoformat(),
            "sync_version": self.sync_version,
        }


@dataclass
class SyncResult:
    """Result of a synchronization operation."""

    direction: SyncDirection
    actions: list[SyncAction] = field(default_factory=list)
    lineage_records: list[LineageRecord] = field(default_factory=list)
    rows_synced: int = 0
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def created_count(self) -> int:
        """Number of features created."""
        return sum(1 for a in self.actions if a.action == "created")

    @property
    def updated_count(self) -> int:
        """Number of features updated."""
        return sum(1 for a in self.actions if a.action == "updated")

    @property
    def skipped_count(self) -> int:
        """Number of features skipped."""
        return sum(1 for a in self.actions if a.action == "skipped")

    @property
    def error_count(self) -> int:
        """Number of errors during sync."""
        return sum(1 for a in self.actions if a.action == "error")

    @property
    def success(self) -> bool:
        """Whether the sync completed without errors."""
        return self.error_count == 0

    def summary(self) -> str:
        """Human-readable summary of the sync result."""
        lines = [
            f"Sync {self.direction.value}: "
            f"{self.created_count} created, "
            f"{self.updated_count} updated, "
            f"{self.skipped_count} skipped, "
            f"{self.error_count} errors",
        ]
        if self.rows_synced > 0:
            lines.append(f"Data: {self.rows_synced} rows synced")
        if self.errors:
            lines.append(f"Errors: {'; '.join(self.errors[:5])}")
        return "\n".join(lines)


# Schema mapping between different feature store type systems
_TYPE_COMPATIBILITY: dict[FeatureType, set[FeatureType]] = {
    FeatureType.INT32: {FeatureType.INT32, FeatureType.INT64, FeatureType.FLOAT32, FeatureType.FLOAT64},
    FeatureType.INT64: {FeatureType.INT64, FeatureType.FLOAT64},
    FeatureType.FLOAT32: {FeatureType.FLOAT32, FeatureType.FLOAT64},
    FeatureType.FLOAT64: {FeatureType.FLOAT64},
    FeatureType.STRING: {FeatureType.STRING},
    FeatureType.BOOL: {FeatureType.BOOL, FeatureType.INT32, FeatureType.INT64},
    FeatureType.DATETIME: {FeatureType.DATETIME, FeatureType.STRING},
    FeatureType.ARRAY: {FeatureType.ARRAY, FeatureType.STRING},
}


def check_type_compatible(source: FeatureType, target: FeatureType) -> bool:
    """Check if a source type can be safely mapped to a target type."""
    compatible = _TYPE_COMPATIBILITY.get(source, set())
    return target in compatible


class FeatureStoreSync:
    """Bi-directional sync between feature registries.

    Enables pushing features from a source registry to a target registry,
    pulling features in the reverse direction, and tracking lineage across
    stores.

    Example:
        >>> sync = FeatureStoreSync(
        ...     source=local_registry,
        ...     target=feast_registry,
        ...     config=SyncConfig(conflict_strategy=ConflictStrategy.OVERWRITE),
        ... )
        >>> result = sync.push("user_features")
        >>> print(result.summary())
    """

    def __init__(
        self,
        source: FeatureRegistry,
        target: FeatureRegistry,
        config: SyncConfig | None = None,
    ) -> None:
        """Initialize the sync layer.

        Args:
            source: Source feature registry.
            target: Target feature registry.
            config: Sync configuration. Uses defaults if None.
        """
        if source is target:
            raise ConfigurationError("Source and target registries must be different")
        self.source = source
        self.target = target
        self.config = config or SyncConfig()
        self._lineage: list[LineageRecord] = []

    def push(
        self,
        feature_set_name: str | None = None,
        feature_names: list[str] | None = None,
        data: pd.DataFrame | None = None,
        timestamp_column: str | None = None,
    ) -> SyncResult:
        """Push features from source to target registry.

        Args:
            feature_set_name: Name of the feature set to push.
            feature_names: Specific feature names to push (if None, push all).
            data: Optional DataFrame to materialize in target.
            timestamp_column: Timestamp column for materialization.

        Returns:
            SyncResult with details of the operation.
        """
        result = SyncResult(direction=SyncDirection.PUSH)

        try:
            features_to_sync = self._resolve_features(
                self.source, feature_set_name, feature_names
            )
        except Exception as e:
            result.errors.append(f"Failed to resolve features: {e}")
            result.completed_at = datetime.now()
            return result

        if not features_to_sync:
            result.completed_at = datetime.now()
            return result

        # Sync definitions
        for feature_def in features_to_sync:
            action = self._sync_feature_definition(
                feature_def, self.target, SyncDirection.PUSH
            )
            result.actions.append(action)

            if action.action not in ("error", "skipped"):
                lineage = LineageRecord(
                    feature_name=feature_def.name,
                    source_registry=self.source.config.registry_type,
                    target_registry=self.target.config.registry_type,
                    source_columns=feature_def.source_columns,
                    transformation=feature_def.transformation,
                    sync_version=action.new_version or "",
                )
                result.lineage_records.append(lineage)
                self._lineage.append(lineage)

        # Sync feature set definition if provided
        if feature_set_name and not self.config.dry_run:
            source_set = self.source.get_feature_set(feature_set_name)
            if source_set is not None:
                try:
                    self.target.register_feature_set(source_set)
                except Exception:
                    pass  # Set may already exist

        # Materialize data if provided
        if data is not None and self.config.include_data and not self.config.dry_run:
            try:
                target_set = feature_set_name or "synced_features"
                rows = self.target.materialize(
                    target_set, data, timestamp_column
                )
                result.rows_synced = rows
            except Exception as e:
                result.errors.append(f"Materialization failed: {e}")

        result.completed_at = datetime.now()
        return result

    def pull(
        self,
        feature_set_name: str | None = None,
        feature_names: list[str] | None = None,
        entity_df: pd.DataFrame | None = None,
        entity_column: str | None = None,
        timestamp_column: str | None = None,
    ) -> SyncResult:
        """Pull features from target to source registry.

        Args:
            feature_set_name: Name of the feature set to pull.
            feature_names: Specific feature names to pull.
            entity_df: Optional entity DataFrame for historical feature retrieval.
            entity_column: Entity key column name.
            timestamp_column: Timestamp column for point-in-time join.

        Returns:
            SyncResult with details of the operation.
        """
        result = SyncResult(direction=SyncDirection.PULL)

        try:
            features_to_sync = self._resolve_features(
                self.target, feature_set_name, feature_names
            )
        except Exception as e:
            result.errors.append(f"Failed to resolve features: {e}")
            result.completed_at = datetime.now()
            return result

        if not features_to_sync:
            result.completed_at = datetime.now()
            return result

        # Sync definitions
        for feature_def in features_to_sync:
            action = self._sync_feature_definition(
                feature_def, self.source, SyncDirection.PULL
            )
            result.actions.append(action)

        # Pull data if entity_df is provided
        if (
            entity_df is not None
            and entity_column
            and self.config.include_data
            and not self.config.dry_run
        ):
            try:
                names = [f.name for f in features_to_sync]
                data = self.target.get_historical_features(
                    entity_df=entity_df,
                    features=names,
                    entity_column=entity_column,
                    timestamp_column=timestamp_column,
                )
                result.rows_synced = len(data)

                # Materialize pulled data into source
                target_set = feature_set_name or "pulled_features"
                self.source.materialize(
                    target_set, data, timestamp_column
                )
            except Exception as e:
                result.errors.append(f"Data pull failed: {e}")

        result.completed_at = datetime.now()
        return result

    def diff(
        self,
        feature_set_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Compare features between source and target registries.

        Returns a list of differences found between the registries.

        Args:
            feature_set_name: Optional feature set to compare.

        Returns:
            List of difference dictionaries with keys: feature, status, details.
        """
        diffs: list[dict[str, Any]] = []

        source_features = self._resolve_features(
            self.source, feature_set_name, None
        )
        target_names = set(self.target.list_features())

        source_map = {f.name: f for f in source_features}

        for name, src_def in source_map.items():
            if name not in target_names:
                diffs.append({
                    "feature": name,
                    "status": "source_only",
                    "details": "Exists in source but not target",
                })
            else:
                tgt_def = self.target.get_feature(name)
                if tgt_def is not None and tgt_def.dtype != src_def.dtype:
                    diffs.append({
                        "feature": name,
                        "status": "type_mismatch",
                        "details": f"Source: {src_def.dtype.value}, Target: {tgt_def.dtype.value}",
                    })
                elif tgt_def is not None and tgt_def.compute_hash() != src_def.compute_hash():
                    diffs.append({
                        "feature": name,
                        "status": "definition_changed",
                        "details": "Feature definition differs between registries",
                    })

        for target_name in target_names:
            if target_name not in source_map:
                diffs.append({
                    "feature": target_name,
                    "status": "target_only",
                    "details": "Exists in target but not source",
                })

        return diffs

    def register_from_dataframe(
        self,
        data: pd.DataFrame,
        feature_set_name: str,
        entity: str = "item",
        entity_column: str | None = None,
        timestamp_column: str | None = None,
        exclude_columns: list[str] | None = None,
    ) -> SyncResult:
        """Register features from a DataFrame and sync to target.

        Convenience method that creates feature definitions from a DataFrame,
        registers them in the source registry, and pushes to the target.

        Args:
            data: DataFrame with features.
            feature_set_name: Name for the feature set.
            entity: Entity name.
            entity_column: Column containing entity keys (excluded from features).
            timestamp_column: Column containing timestamps (excluded from features).
            exclude_columns: Additional columns to exclude.

        Returns:
            SyncResult from the push operation.
        """
        exclude = list(exclude_columns or [])
        if entity_column and entity_column not in exclude:
            exclude.append(entity_column)
        if timestamp_column and timestamp_column not in exclude:
            exclude.append(timestamp_column)

        features = []
        for col in data.columns:
            if col in exclude:
                continue
            feature = FeatureDefinition(
                name=col,
                dtype=infer_feature_type(data[col]),
                entity=entity,
                owner=self.source.config.default_owner,
            )
            features.append(feature)

        feature_set = FeatureSet(
            name=feature_set_name,
            features=features,
            entity=entity,
        )

        if not self.config.dry_run:
            self.source.register_feature_set(feature_set)
            for f in features:
                self.source.register_feature(f)

        return self.push(
            feature_set_name=feature_set_name,
            data=data,
            timestamp_column=timestamp_column,
        )

    def get_lineage(self, feature_name: str | None = None) -> list[LineageRecord]:
        """Get lineage records for synced features.

        Args:
            feature_name: Optional filter by feature name.

        Returns:
            List of lineage records.
        """
        if feature_name is None:
            return list(self._lineage)
        return [r for r in self._lineage if r.feature_name == feature_name]

    def validate_schema_compatibility(
        self,
        feature_set_name: str | None = None,
    ) -> list[str]:
        """Validate that source features are compatible with target schema.

        Args:
            feature_set_name: Optional feature set to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []
        source_features = self._resolve_features(
            self.source, feature_set_name, None
        )

        for feature_def in source_features:
            existing = self.target.get_feature(feature_def.name)
            if existing is not None:
                if not check_type_compatible(feature_def.dtype, existing.dtype):
                    errors.append(
                        f"Feature '{feature_def.name}': type {feature_def.dtype.value} "
                        f"is not compatible with target type {existing.dtype.value}"
                    )

        return errors

    def _resolve_features(
        self,
        registry: FeatureRegistry,
        feature_set_name: str | None,
        feature_names: list[str] | None,
    ) -> list[FeatureDefinition]:
        """Resolve which features to sync."""
        features: list[FeatureDefinition] = []

        if feature_set_name:
            fs = registry.get_feature_set(feature_set_name)
            if fs is not None:
                features = fs.features
            else:
                raise ValidationError(
                    f"Feature set '{feature_set_name}' not found in registry"
                )
        elif feature_names:
            for name in feature_names:
                f = registry.get_feature(name)
                if f is not None:
                    features.append(f)
                else:
                    logger.warning("Feature '%s' not found, skipping", name)
        else:
            for name in registry.list_features():
                f = registry.get_feature(name)
                if f is not None:
                    features.append(f)

        return features

    def _sync_feature_definition(
        self,
        feature_def: FeatureDefinition,
        target: FeatureRegistry,
        direction: SyncDirection,
    ) -> SyncAction:
        """Sync a single feature definition to a target registry."""
        existing = target.get_feature(feature_def.name)

        if existing is not None:
            if existing.compute_hash() == feature_def.compute_hash():
                return SyncAction(
                    feature_name=feature_def.name,
                    action="skipped",
                    direction=direction,
                    details="Already up to date",
                )

            # Handle conflict
            if self.config.conflict_strategy == ConflictStrategy.SKIP:
                return SyncAction(
                    feature_name=feature_def.name,
                    action="skipped",
                    direction=direction,
                    details="Conflict: skipped per strategy",
                )
            elif self.config.conflict_strategy == ConflictStrategy.ERROR:
                return SyncAction(
                    feature_name=feature_def.name,
                    action="error",
                    direction=direction,
                    details=f"Conflict: feature exists with different definition "
                    f"(hash {existing.compute_hash()} vs {feature_def.compute_hash()})",
                )
            elif self.config.conflict_strategy == ConflictStrategy.RENAME:
                suffix = feature_def.compute_hash()[:6]
                renamed = FeatureDefinition(
                    name=f"{feature_def.name}_{suffix}",
                    dtype=feature_def.dtype,
                    description=feature_def.description,
                    source_columns=feature_def.source_columns,
                    transformation=feature_def.transformation,
                    tags=feature_def.tags,
                    owner=feature_def.owner,
                    entity=feature_def.entity,
                )
                if not self.config.dry_run:
                    version = target.register_feature(renamed)
                    return SyncAction(
                        feature_name=renamed.name,
                        action="renamed",
                        direction=direction,
                        details=f"Renamed from {feature_def.name}",
                        new_version=version.version,
                    )
                return SyncAction(
                    feature_name=renamed.name,
                    action="renamed",
                    direction=direction,
                    details=f"[dry-run] Would rename from {feature_def.name}",
                )
            # OVERWRITE falls through

        # Create or update
        if self.config.dry_run:
            action_type = "updated" if existing else "created"
            return SyncAction(
                feature_name=feature_def.name,
                action=action_type,
                direction=direction,
                details=f"[dry-run] Would {action_type}",
            )

        try:
            if self.config.validate_schemas and existing is not None:
                if not check_type_compatible(feature_def.dtype, existing.dtype):
                    return SyncAction(
                        feature_name=feature_def.name,
                        action="error",
                        direction=direction,
                        details=f"Incompatible types: {feature_def.dtype.value} → {existing.dtype.value}",
                    )

            version = target.register_feature(feature_def)
            action_type = "updated" if existing else "created"
            return SyncAction(
                feature_name=feature_def.name,
                action=action_type,
                direction=direction,
                new_version=version.version,
            )
        except Exception as e:
            return SyncAction(
                feature_name=feature_def.name,
                action="error",
                direction=direction,
                details=str(e),
            )
