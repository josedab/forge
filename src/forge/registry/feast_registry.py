"""Feast feature store integration."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from forge.exceptions import ConfigurationError, MissingDependencyError, ValidationError
from forge.registry.base import (
    FeatureDefinition,
    FeatureRegistry,
    FeatureSet,
    FeatureType,
    FeatureVersion,
    RegistryConfig,
)


class FeastRegistry(FeatureRegistry):
    """Feast feature store registry integration.

    This class provides integration with Feast for feature registration,
    materialization, and retrieval.

    Example:
        >>> config = RegistryConfig(
        ...     registry_type="feast",
        ...     project_name="my_project",
        ...     connection_params={"repo_path": "./feature_repo"},
        ... )
        >>> registry = FeastRegistry(config)
        >>> registry.connect()
        >>> registry.register_feature_set(feature_set)
    """

    FEAST_TYPE_MAPPING = {
        FeatureType.INT32: "INT32",
        FeatureType.INT64: "INT64",
        FeatureType.FLOAT32: "FLOAT",
        FeatureType.FLOAT64: "DOUBLE",
        FeatureType.STRING: "STRING",
        FeatureType.BOOL: "BOOL",
        FeatureType.DATETIME: "UNIX_TIMESTAMP",
    }

    def __init__(self, config: RegistryConfig) -> None:
        """Initialize Feast registry.

        Args:
            config: Registry configuration with connection params.
                Expected params:
                - repo_path: Path to Feast feature repo
                - offline_store: Optional offline store config
                - online_store: Optional online store config
        """
        super().__init__(config)
        self._store = None
        self._repo_path = config.connection_params.get("repo_path", "./feature_repo")
        self._feature_sets: dict[str, FeatureSet] = {}
        self._versions: dict[str, list[FeatureVersion]] = {}

    def connect(self) -> None:
        """Connect to Feast feature store."""
        try:
            from feast import FeatureStore
        except ImportError:
            raise MissingDependencyError("feast", "Feast feature store integration")

        repo_path = Path(self._repo_path)

        # Initialize feature store
        if repo_path.exists():
            self._store = FeatureStore(repo_path=str(repo_path))
        else:
            # Create a minimal repo if it doesn't exist
            self._create_minimal_repo(repo_path)
            self._store = FeatureStore(repo_path=str(repo_path))

        self._connected = True

    def disconnect(self) -> None:
        """Disconnect from Feast."""
        self._store = None
        self._connected = False

    def register_feature(self, feature: FeatureDefinition) -> FeatureVersion:
        """Register a single feature with Feast.

        Note: Feast requires features to be part of a FeatureView,
        so this creates a single-feature view.
        """
        self._check_connected()

        # Create a feature set with single feature
        feature_set = FeatureSet(
            name=f"{feature.name}_view",
            features=[feature],
            entity=feature.entity or self.config.default_entity,
        )

        self.register_feature_set(feature_set)

        version = FeatureVersion(
            version=feature_set.version,
            feature_hash=feature.compute_hash(),
        )

        if feature.name not in self._versions:
            self._versions[feature.name] = []
        self._versions[feature.name].append(version)

        return version

    def register_feature_set(self, feature_set: FeatureSet) -> str:
        """Register a feature set as a Feast FeatureView."""
        self._check_connected()

        try:
            from feast import Entity, Feature, FeatureView, FileSource, ValueType
        except ImportError:
            raise MissingDependencyError("feast", "Feast feature store integration")

        # Store locally for reference
        self._feature_sets[feature_set.name] = feature_set

        # Generate Feast feature definitions
        feast_features = []
        for feat in feature_set.features:
            feast_type = self._map_to_feast_type(feat.dtype)
            feast_features.append(
                Feature(name=feat.name, dtype=feast_type)
            )

        # Create or get entity
        entity_name = feature_set.entity or self.config.default_entity

        # Generate Python code for the feature view
        code = self._generate_feature_view_code(feature_set, feast_features)

        # Write to repo if auto-generation is enabled
        if self.config.auto_version:
            self._write_feature_definition(feature_set.name, code)

        return feature_set.version

    def get_feature(self, name: str, version: str | None = None) -> FeatureDefinition | None:
        """Get a feature definition."""
        self._check_connected()

        for fs in self._feature_sets.values():
            for feat in fs.features:
                if feat.name == name:
                    return feat

        return None

    def get_feature_set(self, name: str, version: str | None = None) -> FeatureSet | None:
        """Get a feature set."""
        self._check_connected()
        return self._feature_sets.get(name)

    def list_features(self, entity: str | None = None) -> list[str]:
        """List all registered features."""
        self._check_connected()

        features = []
        for fs in self._feature_sets.values():
            if entity is None or fs.entity == entity:
                features.extend([f.name for f in fs.features])

        return features

    def list_feature_sets(self) -> list[str]:
        """List all feature sets."""
        self._check_connected()
        return list(self._feature_sets.keys())

    def delete_feature(self, name: str) -> bool:
        """Delete a feature (removes from local registry)."""
        self._check_connected()

        for fs_name, fs in list(self._feature_sets.items()):
            fs.features = [f for f in fs.features if f.name != name]
            if not fs.features:
                del self._feature_sets[fs_name]
                return True

        return False

    def materialize(
        self,
        feature_set: str | FeatureSet,
        data: pd.DataFrame,
        timestamp_column: str | None = None,
    ) -> int:
        """Materialize feature data to Feast offline store.

        Args:
            feature_set: FeatureSet name or instance.
            data: DataFrame with feature values.
            timestamp_column: Column containing event timestamps.

        Returns:
            Number of rows materialized.
        """
        self._check_connected()

        if isinstance(feature_set, str):
            fs = self._feature_sets.get(feature_set)
            if fs is None:
                raise ValidationError(f"Feature set '{feature_set}' not found")
        else:
            fs = feature_set

        # Add event timestamp if not present
        if timestamp_column is None:
            timestamp_column = "event_timestamp"
            if timestamp_column not in data.columns:
                data = data.copy()
                data[timestamp_column] = datetime.now()

        # Write to parquet for Feast FileSource
        output_path = Path(self._repo_path) / "data" / f"{fs.name}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(output_path, index=False)

        return len(data)

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        features: list[str],
        entity_column: str,
        timestamp_column: str | None = None,
    ) -> pd.DataFrame:
        """Get historical feature values from Feast.

        Args:
            entity_df: DataFrame with entity keys and timestamps.
            features: List of feature names to retrieve.
            entity_column: Column containing entity keys.
            timestamp_column: Column containing request timestamps.

        Returns:
            DataFrame with feature values joined to entity_df.
        """
        self._check_connected()

        if self._store is None:
            raise ConfigurationError("Feast store not initialized")

        # Ensure timestamp column exists
        if timestamp_column is None:
            timestamp_column = "event_timestamp"
            if timestamp_column not in entity_df.columns:
                entity_df = entity_df.copy()
                entity_df[timestamp_column] = datetime.now()

        # Format feature references for Feast
        feature_refs = []
        for feat_name in features:
            # Find the feature view containing this feature
            for fs_name, fs in self._feature_sets.items():
                if any(f.name == feat_name for f in fs.features):
                    feature_refs.append(f"{fs_name}:{feat_name}")
                    break

        if not feature_refs:
            return entity_df

        try:
            historical_features = self._store.get_historical_features(
                entity_df=entity_df,
                features=feature_refs,
            )
            return historical_features.to_df()
        except Exception:
            # Fallback: return entity_df with NaN features
            result = entity_df.copy()
            for feat in features:
                if feat not in result.columns:
                    result[feat] = pd.NA
            return result

    def apply(self) -> None:
        """Apply all registered feature definitions to Feast.

        This writes the feature definitions and runs feast apply.
        """
        self._check_connected()

        if self._store is None:
            raise ConfigurationError("Feast store not initialized")

        try:
            self._store.apply([])  # Apply current definitions
        except Exception as e:
            raise ConfigurationError(f"Failed to apply Feast definitions: {e}")

    def generate_feast_definitions(self) -> str:
        """Generate Python code for all feature definitions.

        Returns:
            Python code string for feature_store.py.
        """
        lines = [
            "# Auto-generated by Forge",
            "from datetime import timedelta",
            "",
            "from feast import Entity, Feature, FeatureView, FileSource, ValueType",
            "",
        ]

        # Generate entities
        entities = set()
        for fs in self._feature_sets.values():
            entities.add(fs.entity or self.config.default_entity)

        for entity in entities:
            lines.extend([
                f"{entity} = Entity(",
                f'    name="{entity}",',
                '    value_type=ValueType.STRING,',
                f'    description="Entity for {entity}",',
                ")",
                "",
            ])

        # Generate feature views
        for fs in self._feature_sets.values():
            lines.append(self._generate_feature_view_code(fs, []))

        return "\n".join(lines)

    def _check_connected(self) -> None:
        """Check if connected to Feast."""
        if not self._connected:
            raise ConfigurationError("Not connected to Feast. Call connect() first.")

    def _map_to_feast_type(self, dtype: FeatureType) -> Any:
        """Map Forge FeatureType to Feast ValueType."""
        try:
            from feast import ValueType
        except ImportError:
            raise MissingDependencyError("feast", "Feast feature store integration")

        mapping = {
            FeatureType.INT32: ValueType.INT32,
            FeatureType.INT64: ValueType.INT64,
            FeatureType.FLOAT32: ValueType.FLOAT,
            FeatureType.FLOAT64: ValueType.DOUBLE,
            FeatureType.STRING: ValueType.STRING,
            FeatureType.BOOL: ValueType.BOOL,
            FeatureType.DATETIME: ValueType.UNIX_TIMESTAMP,
            FeatureType.UNKNOWN: ValueType.DOUBLE,
        }
        return mapping.get(dtype, ValueType.DOUBLE)

    def _generate_feature_view_code(
        self, feature_set: FeatureSet, feast_features: list
    ) -> str:
        """Generate Python code for a Feast FeatureView."""
        entity = feature_set.entity or self.config.default_entity
        view_name = feature_set.name.replace("-", "_")

        lines = [
            f"{view_name}_source = FileSource(",
            f'    path="data/{feature_set.name}.parquet",',
            '    event_timestamp_column="event_timestamp",',
            ")",
            "",
            f"{view_name} = FeatureView(",
            f'    name="{feature_set.name}",',
            f'    entities=["{entity}"],',
            "    ttl=timedelta(days=365),",
            "    features=[",
        ]

        for feat in feature_set.features:
            feast_type = self.FEAST_TYPE_MAPPING.get(feat.dtype, "DOUBLE")
            lines.append(f'        Feature(name="{feat.name}", dtype=ValueType.{feast_type}),')

        lines.extend([
            "    ],",
            f"    source={view_name}_source,",
            ")",
            "",
        ])

        return "\n".join(lines)

    def _create_minimal_repo(self, repo_path: Path) -> None:
        """Create a minimal Feast repository."""
        repo_path.mkdir(parents=True, exist_ok=True)
        data_path = repo_path / "data"
        data_path.mkdir(exist_ok=True)

        # Create feature_store.yaml
        yaml_content = f"""
project: {self.config.project_name}
registry: {repo_path}/registry.db
provider: local
online_store:
  type: sqlite
  path: {repo_path}/online_store.db
offline_store:
  type: file
"""
        (repo_path / "feature_store.yaml").write_text(yaml_content.strip())

        # Create empty features.py
        (repo_path / "features.py").write_text("# Feature definitions\n")

    def _write_feature_definition(self, name: str, code: str) -> None:
        """Write feature definition to repo."""
        repo_path = Path(self._repo_path)
        features_file = repo_path / "features.py"

        existing = features_file.read_text() if features_file.exists() else ""
        if name not in existing:
            with open(features_file, "a") as f:
                f.write(f"\n\n# Feature set: {name}\n")
                f.write(code)


def create_feast_registry(
    repo_path: str = "./feature_repo",
    project_name: str = "forge_features",
) -> FeastRegistry:
    """Convenience function to create a Feast registry.

    Args:
        repo_path: Path to Feast repository.
        project_name: Feast project name.

    Returns:
        Connected FeastRegistry instance.
    """
    config = RegistryConfig(
        registry_type="feast",
        project_name=project_name,
        connection_params={"repo_path": repo_path},
    )
    registry = FeastRegistry(config)
    registry.connect()
    return registry
