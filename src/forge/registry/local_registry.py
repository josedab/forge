"""Local file-based feature registry for development and testing."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from forge.exceptions import ConfigurationError, ValidationError
from forge.registry.base import (
    FeatureDefinition,
    FeatureRegistry,
    FeatureSet,
    FeatureVersion,
    RegistryConfig,
)


class LocalRegistry(FeatureRegistry):
    """Local file-based feature registry.

    This registry stores feature definitions and data locally in a
    directory structure. Useful for development, testing, and
    single-machine deployments.

    Directory structure:
        registry_path/
        ├── registry.json          # Main registry metadata
        ├── features/              # Individual feature definitions
        │   └── {feature_name}.json
        ├── feature_sets/          # Feature set definitions
        │   └── {set_name}.json
        ├── versions/              # Version history
        │   └── {feature_name}/
        │       └── {version}.json
        └── data/                  # Materialized data
            └── {feature_set}/
                └── {timestamp}.parquet

    Example:
        >>> config = RegistryConfig(
        ...     registry_type="local",
        ...     project_name="my_project",
        ...     connection_params={"path": "./feature_registry"},
        ... )
        >>> registry = LocalRegistry(config)
        >>> registry.connect()
        >>> registry.register_feature(feature)
    """

    def __init__(self, config: RegistryConfig) -> None:
        """Initialize local registry.

        Args:
            config: Registry configuration.
                Expected connection_params:
                - path: Path to registry directory
        """
        super().__init__(config)
        self._path = Path(config.connection_params.get("path", "./feature_registry"))
        self._features: dict[str, FeatureDefinition] = {}
        self._feature_sets: dict[str, FeatureSet] = {}
        self._versions: dict[str, list[FeatureVersion]] = {}
        self._data: dict[str, pd.DataFrame] = {}

    def connect(self) -> None:
        """Initialize the local registry directory structure."""
        self._path.mkdir(parents=True, exist_ok=True)
        (self._path / "features").mkdir(exist_ok=True)
        (self._path / "feature_sets").mkdir(exist_ok=True)
        (self._path / "versions").mkdir(exist_ok=True)
        (self._path / "data").mkdir(exist_ok=True)

        # Load existing registry
        self._load_registry()
        self._connected = True

    def disconnect(self) -> None:
        """Save and close the registry."""
        if self._connected:
            self._save_registry()
        self._connected = False

    def register_feature(self, feature: FeatureDefinition) -> FeatureVersion:
        """Register a single feature."""
        self._check_connected()

        # Check for existing feature
        existing = self._features.get(feature.name)
        feature_hash = feature.compute_hash()

        # Determine version
        if existing:
            existing_versions = self._versions.get(feature.name, [])
            version_num = len(existing_versions) + 1
        else:
            version_num = 1

        version = FeatureVersion(
            version=f"v{version_num}.0.0",
            feature_hash=feature_hash,
            description=f"Version {version_num}",
        )

        # Store feature
        self._features[feature.name] = feature
        if feature.name not in self._versions:
            self._versions[feature.name] = []
        self._versions[feature.name].append(version)

        # Persist
        self._save_feature(feature)
        self._save_version(feature.name, version)

        return version

    def register_feature_set(self, feature_set: FeatureSet) -> str:
        """Register a feature set."""
        self._check_connected()

        # Register individual features
        for feature in feature_set.features:
            if feature.name not in self._features:
                self.register_feature(feature)

        # Store feature set
        self._feature_sets[feature_set.name] = feature_set

        # Persist
        self._save_feature_set(feature_set)

        return feature_set.version

    def get_feature(self, name: str, version: str | None = None) -> FeatureDefinition | None:
        """Get a feature definition."""
        self._check_connected()
        return self._features.get(name)

    def get_feature_set(self, name: str, version: str | None = None) -> FeatureSet | None:
        """Get a feature set."""
        self._check_connected()
        return self._feature_sets.get(name)

    def list_features(self, entity: str | None = None) -> list[str]:
        """List all registered features."""
        self._check_connected()

        if entity is None:
            return list(self._features.keys())

        return [
            name for name, feat in self._features.items()
            if feat.entity == entity
        ]

    def list_feature_sets(self) -> list[str]:
        """List all feature sets."""
        self._check_connected()
        return list(self._feature_sets.keys())

    def delete_feature(self, name: str) -> bool:
        """Delete a feature."""
        self._check_connected()

        if name not in self._features:
            return False

        del self._features[name]
        self._versions.pop(name, None)

        # Remove file
        feature_file = self._path / "features" / f"{name}.json"
        if feature_file.exists():
            feature_file.unlink()

        return True

    def delete_feature_set(self, name: str) -> bool:
        """Delete a feature set (but not its features)."""
        self._check_connected()

        if name not in self._feature_sets:
            return False

        del self._feature_sets[name]

        # Remove file
        fs_file = self._path / "feature_sets" / f"{name}.json"
        if fs_file.exists():
            fs_file.unlink()

        return True

    def materialize(
        self,
        feature_set: str | FeatureSet,
        data: pd.DataFrame,
        timestamp_column: str | None = None,
    ) -> int:
        """Materialize feature data to local storage."""
        self._check_connected()

        if isinstance(feature_set, str):
            fs = self._feature_sets.get(feature_set)
            if fs is None:
                raise ValidationError(f"Feature set '{feature_set}' not found")
            fs_name = feature_set
        else:
            fs = feature_set
            fs_name = fs.name

        # Add timestamp if not present
        if timestamp_column is None:
            timestamp_column = "event_timestamp"
            if timestamp_column not in data.columns:
                data = data.copy()
                data[timestamp_column] = datetime.now()

        # Store in memory and persist
        self._data[fs_name] = data

        # Save to parquet
        data_dir = self._path / "data" / fs_name
        data_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_file = data_dir / f"{timestamp}.parquet"
        data.to_parquet(data_file, index=False)

        return len(data)

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        features: list[str],
        entity_column: str,
        timestamp_column: str | None = None,
    ) -> pd.DataFrame:
        """Get historical feature values."""
        self._check_connected()

        result = entity_df.copy()

        # Find features in materialized data
        for feat_name in features:
            feature_values = self._find_feature_values(
                feat_name, entity_df, entity_column
            )
            if feature_values is not None:
                result[feat_name] = feature_values
            else:
                result[feat_name] = pd.NA

        return result

    def get_online_features(
        self,
        entity_ids: list[Any],
        features: list[str],
        entity_column: str = "id",
    ) -> pd.DataFrame:
        """Get latest feature values for online serving."""
        self._check_connected()

        # Create entity DataFrame with current timestamp
        entity_df = pd.DataFrame({entity_column: entity_ids})
        return self.get_historical_features(
            entity_df, features, entity_column
        )

    def get_feature_statistics(self, feature_name: str) -> dict[str, Any]:
        """Get statistics for a feature from materialized data."""
        self._check_connected()

        for fs_name, data in self._data.items():
            if feature_name in data.columns:
                series = data[feature_name]
                stats = {
                    "count": int(series.count()),
                    "null_count": int(series.isna().sum()),
                    "unique_count": int(series.nunique()),
                }

                if pd.api.types.is_numeric_dtype(series.dtype):
                    stats.update({
                        "mean": float(series.mean()),
                        "std": float(series.std()),
                        "min": float(series.min()),
                        "max": float(series.max()),
                    })

                return stats

        return {}

    def export_definitions(self, output_path: str | Path) -> None:
        """Export all feature definitions to a single JSON file."""
        self._check_connected()

        output = {
            "project": self.config.project_name,
            "exported_at": datetime.now().isoformat(),
            "features": {
                name: feat.to_dict()
                for name, feat in self._features.items()
            },
            "feature_sets": {
                name: fs.to_dict()
                for name, fs in self._feature_sets.items()
            },
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

    def import_definitions(self, input_path: str | Path) -> int:
        """Import feature definitions from a JSON file."""
        self._check_connected()

        with open(input_path) as f:
            data = json.load(f)

        count = 0

        for name, feat_data in data.get("features", {}).items():
            feature = FeatureDefinition.from_dict(feat_data)
            self.register_feature(feature)
            count += 1

        for name, fs_data in data.get("feature_sets", {}).items():
            fs = FeatureSet.from_dict(fs_data)
            self._feature_sets[name] = fs
            self._save_feature_set(fs)

        return count

    def _check_connected(self) -> None:
        """Check if connected."""
        if not self._connected:
            raise ConfigurationError("Registry not connected. Call connect() first.")

    def _load_registry(self) -> None:
        """Load existing registry from disk."""
        # Load features
        features_dir = self._path / "features"
        for feat_file in features_dir.glob("*.json"):
            with open(feat_file) as f:
                data = json.load(f)
                feature = FeatureDefinition.from_dict(data)
                self._features[feature.name] = feature

        # Load feature sets
        fs_dir = self._path / "feature_sets"
        for fs_file in fs_dir.glob("*.json"):
            with open(fs_file) as f:
                data = json.load(f)
                fs = FeatureSet.from_dict(data)
                self._feature_sets[fs.name] = fs

        # Load versions
        versions_dir = self._path / "versions"
        for feat_dir in versions_dir.iterdir():
            if feat_dir.is_dir():
                feat_name = feat_dir.name
                self._versions[feat_name] = []
                for ver_file in sorted(feat_dir.glob("*.json")):
                    with open(ver_file) as f:
                        data = json.load(f)
                        version = FeatureVersion(**data)
                        self._versions[feat_name].append(version)

        # Load latest data files
        data_dir = self._path / "data"
        for fs_dir in data_dir.iterdir():
            if fs_dir.is_dir():
                parquet_files = sorted(fs_dir.glob("*.parquet"))
                if parquet_files:
                    latest = parquet_files[-1]
                    self._data[fs_dir.name] = pd.read_parquet(latest)

    def _save_registry(self) -> None:
        """Save registry metadata."""
        registry_file = self._path / "registry.json"
        metadata = {
            "project": self.config.project_name,
            "updated_at": datetime.now().isoformat(),
            "feature_count": len(self._features),
            "feature_set_count": len(self._feature_sets),
        }
        with open(registry_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def _save_feature(self, feature: FeatureDefinition) -> None:
        """Save feature definition to disk."""
        feat_file = self._path / "features" / f"{feature.name}.json"
        with open(feat_file, "w") as f:
            json.dump(feature.to_dict(), f, indent=2, default=str)

    def _save_feature_set(self, fs: FeatureSet) -> None:
        """Save feature set to disk."""
        fs_file = self._path / "feature_sets" / f"{fs.name}.json"
        with open(fs_file, "w") as f:
            json.dump(fs.to_dict(), f, indent=2, default=str)

    def _save_version(self, feature_name: str, version: FeatureVersion) -> None:
        """Save version information."""
        ver_dir = self._path / "versions" / feature_name
        ver_dir.mkdir(parents=True, exist_ok=True)

        ver_file = ver_dir / f"{version.version}.json"
        with open(ver_file, "w") as f:
            json.dump(version.to_dict(), f, indent=2, default=str)

    def _find_feature_values(
        self,
        feature_name: str,
        entity_df: pd.DataFrame,
        entity_column: str,
    ) -> pd.Series | None:
        """Find feature values in materialized data."""
        for fs_name, data in self._data.items():
            if feature_name in data.columns and entity_column in data.columns:
                # Create mapping and join
                mapping = data.set_index(entity_column)[feature_name].to_dict()
                return entity_df[entity_column].map(mapping)

        return None


def create_local_registry(
    path: str = "./feature_registry",
    project_name: str = "forge_features",
) -> LocalRegistry:
    """Convenience function to create a local registry.

    Args:
        path: Path to registry directory.
        project_name: Project name.

    Returns:
        Connected LocalRegistry instance.
    """
    config = RegistryConfig(
        registry_type="local",
        project_name=project_name,
        connection_params={"path": path},
    )
    registry = LocalRegistry(config)
    registry.connect()
    return registry


def create_registry(
    registry_type: str = "local",
    **kwargs: Any,
) -> FeatureRegistry:
    """Factory function to create a feature registry.

    Args:
        registry_type: Type of registry ("local", "feast").
        **kwargs: Additional arguments for registry configuration.

    Returns:
        Connected FeatureRegistry instance.

    Example:
        >>> registry = create_registry("local", path="./my_registry")
        >>> registry = create_registry("feast", repo_path="./feast_repo")
    """
    if registry_type == "local":
        return create_local_registry(
            path=kwargs.get("path", "./feature_registry"),
            project_name=kwargs.get("project_name", "forge_features"),
        )
    elif registry_type == "feast":
        from forge.registry.feast_registry import create_feast_registry
        return create_feast_registry(
            repo_path=kwargs.get("repo_path", "./feature_repo"),
            project_name=kwargs.get("project_name", "forge_features"),
        )
    else:
        raise ValueError(f"Unknown registry type: {registry_type}")
