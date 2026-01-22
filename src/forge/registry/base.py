"""Base classes and protocols for feature store integration."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


class FeatureType(str, Enum):
    """Data types for features."""

    INT32 = "int32"
    INT64 = "int64"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    STRING = "string"
    BOOL = "bool"
    DATETIME = "datetime"
    ARRAY = "array"
    UNKNOWN = "unknown"


@dataclass
class FeatureDefinition:
    """Definition of a feature for registration.

    This class contains all metadata needed to register a feature
    with a feature store, including its name, type, and lineage.
    """

    name: str
    dtype: FeatureType
    description: str = ""
    source_columns: list[str] = field(default_factory=list)
    transformation: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    owner: str = ""
    entity: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "dtype": self.dtype.value,
            "description": self.description,
            "source_columns": self.source_columns,
            "transformation": self.transformation,
            "tags": self.tags,
            "owner": self.owner,
            "entity": self.entity,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureDefinition:
        """Create from dictionary."""
        data = data.copy()
        data["dtype"] = FeatureType(data["dtype"])
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)

    def compute_hash(self) -> str:
        """Compute a hash for version tracking."""
        content = json.dumps(
            {
                "name": self.name,
                "dtype": self.dtype.value,
                "source_columns": sorted(self.source_columns),
                "transformation": self.transformation,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:12]


@dataclass
class FeatureVersion:
    """Version information for a feature."""

    version: str
    feature_hash: str
    created_at: datetime = field(default_factory=datetime.now)
    description: str = ""
    is_active: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "feature_hash": self.feature_hash,
            "created_at": self.created_at.isoformat(),
            "description": self.description,
            "is_active": self.is_active,
        }


@dataclass
class FeatureSet:
    """Collection of related features.

    A FeatureSet groups related features together, typically those
    that are computed from the same source data or serve the same purpose.
    """

    name: str
    features: list[FeatureDefinition]
    description: str = ""
    entity: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "features": [f.to_dict() for f in self.features],
            "description": self.description,
            "entity": self.entity,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureSet:
        """Create from dictionary."""
        data = data.copy()
        data["features"] = [FeatureDefinition.from_dict(f) for f in data["features"]]
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)

    def add_feature(self, feature: FeatureDefinition) -> None:
        """Add a feature to the set."""
        self.features.append(feature)

    def get_feature(self, name: str) -> FeatureDefinition | None:
        """Get a feature by name."""
        for f in self.features:
            if f.name == name:
                return f
        return None

    @property
    def feature_names(self) -> list[str]:
        """Get list of feature names."""
        return [f.name for f in self.features]


@dataclass
class RegistryConfig:
    """Configuration for feature registry connection."""

    registry_type: str  # "feast", "tecton", "hopsworks", "local"
    project_name: str = "forge_features"
    connection_params: dict[str, Any] = field(default_factory=dict)
    default_entity: str = "item"
    default_owner: str = ""
    auto_version: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "registry_type": self.registry_type,
            "project_name": self.project_name,
            "connection_params": self.connection_params,
            "default_entity": self.default_entity,
            "default_owner": self.default_owner,
            "auto_version": self.auto_version,
        }


class FeatureRegistry(ABC):
    """Abstract base class for feature store registries.

    This class defines the interface for interacting with feature stores.
    Concrete implementations provide connectivity to specific stores.
    """

    def __init__(self, config: RegistryConfig) -> None:
        """Initialize the registry.

        Args:
            config: Registry configuration.
        """
        self.config = config
        self._connected = False

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the feature store."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the feature store."""
        pass

    @abstractmethod
    def register_feature(self, feature: FeatureDefinition) -> FeatureVersion:
        """Register a single feature.

        Args:
            feature: Feature definition to register.

        Returns:
            FeatureVersion with version information.
        """
        pass

    @abstractmethod
    def register_feature_set(self, feature_set: FeatureSet) -> str:
        """Register a feature set.

        Args:
            feature_set: FeatureSet to register.

        Returns:
            Registration ID or version string.
        """
        pass

    @abstractmethod
    def get_feature(self, name: str, version: str | None = None) -> FeatureDefinition | None:
        """Get a feature definition.

        Args:
            name: Feature name.
            version: Optional specific version.

        Returns:
            FeatureDefinition or None if not found.
        """
        pass

    @abstractmethod
    def get_feature_set(self, name: str, version: str | None = None) -> FeatureSet | None:
        """Get a feature set.

        Args:
            name: FeatureSet name.
            version: Optional specific version.

        Returns:
            FeatureSet or None if not found.
        """
        pass

    @abstractmethod
    def list_features(self, entity: str | None = None) -> list[str]:
        """List registered feature names.

        Args:
            entity: Optional entity to filter by.

        Returns:
            List of feature names.
        """
        pass

    @abstractmethod
    def list_feature_sets(self) -> list[str]:
        """List registered feature set names.

        Returns:
            List of feature set names.
        """
        pass

    @abstractmethod
    def delete_feature(self, name: str) -> bool:
        """Delete a feature.

        Args:
            name: Feature name to delete.

        Returns:
            True if deleted, False if not found.
        """
        pass

    @abstractmethod
    def materialize(
        self,
        feature_set: str | FeatureSet,
        data: pd.DataFrame,
        timestamp_column: str | None = None,
    ) -> int:
        """Materialize feature data to the store.

        Args:
            feature_set: FeatureSet name or instance.
            data: DataFrame with feature values.
            timestamp_column: Column containing event timestamps.

        Returns:
            Number of rows materialized.
        """
        pass

    @abstractmethod
    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        features: list[str],
        entity_column: str,
        timestamp_column: str | None = None,
    ) -> pd.DataFrame:
        """Get historical feature values.

        Args:
            entity_df: DataFrame with entity keys and timestamps.
            features: List of feature names to retrieve.
            entity_column: Column containing entity keys.
            timestamp_column: Column containing request timestamps.

        Returns:
            DataFrame with feature values joined to entity_df.
        """
        pass

    def register_from_transformer(
        self,
        transformer: BaseEstimator,
        feature_set_name: str | None = None,
        entity: str | None = None,
    ) -> FeatureSet:
        """Register features from a Forge transformer.

        Args:
            transformer: Fitted Forge transformer.
            feature_set_name: Name for the feature set.
            entity: Entity name for features.

        Returns:
            Created FeatureSet.
        """
        # Get feature names from transformer
        if hasattr(transformer, "get_feature_names_out"):
            feature_names = transformer.get_feature_names_out()
        else:
            raise ValueError("Transformer must have get_feature_names_out method")

        entity = entity or self.config.default_entity
        feature_set_name = feature_set_name or f"{entity}_features"

        features = []
        for name in feature_names:
            feature = FeatureDefinition(
                name=name,
                dtype=FeatureType.FLOAT64,  # Default type
                description=f"Feature generated by {transformer.__class__.__name__}",
                entity=entity,
                owner=self.config.default_owner,
            )
            features.append(feature)

        feature_set = FeatureSet(
            name=feature_set_name,
            features=features,
            entity=entity,
            description=f"Features from {transformer.__class__.__name__}",
        )

        self.register_feature_set(feature_set)
        return feature_set

    def __enter__(self) -> FeatureRegistry:
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.disconnect()


def infer_feature_type(series: pd.Series) -> FeatureType:
    """Infer FeatureType from a pandas Series.

    Args:
        series: Pandas Series to analyze.

    Returns:
        Inferred FeatureType.
    """
    dtype = series.dtype

    if pd.api.types.is_integer_dtype(dtype):
        if dtype == "int32":
            return FeatureType.INT32
        return FeatureType.INT64
    elif pd.api.types.is_float_dtype(dtype):
        if dtype == "float32":
            return FeatureType.FLOAT32
        return FeatureType.FLOAT64
    elif pd.api.types.is_bool_dtype(dtype):
        return FeatureType.BOOL
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return FeatureType.DATETIME
    elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
        return FeatureType.STRING
    else:
        return FeatureType.UNKNOWN


def create_feature_definitions_from_dataframe(
    df: pd.DataFrame,
    entity: str = "item",
    exclude_columns: list[str] | None = None,
) -> list[FeatureDefinition]:
    """Create FeatureDefinitions from a DataFrame.

    Args:
        df: DataFrame to analyze.
        entity: Entity name for features.
        exclude_columns: Columns to exclude.

    Returns:
        List of FeatureDefinition objects.
    """
    exclude = set(exclude_columns or [])
    definitions = []

    for col in df.columns:
        if col in exclude:
            continue

        feature = FeatureDefinition(
            name=col,
            dtype=infer_feature_type(df[col]),
            entity=entity,
        )
        definitions.append(feature)

    return definitions
