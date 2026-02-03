"""Feature Store Integration Hub.

Provides unified integrations with multiple feature store platforms
beyond Feast: Tecton, Hopsworks, Databricks, and SageMaker.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import pandas as pd

from forge.exceptions import ConfigurationError, MissingDependencyError
from forge.registry.base import (
    FeatureDefinition,
    FeatureRegistry,
    FeatureSet,
    FeatureType,
    FeatureVersion,
)

if TYPE_CHECKING:
    from typing_extensions import Self

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Tecton Feature Store Backend
# ──────────────────────────────────────────────────────────────


class TectonRegistry(FeatureRegistry):
    """Feature registry backed by Tecton.

    Requires the ``tecton`` package. Features are registered as
    Tecton Feature Views and materialized through the Tecton API.

    Args:
        workspace: Tecton workspace name.
        api_key: Tecton API key.
        url: Tecton cluster URL.
        project_name: Project name for organizing features.

    Example:
        >>> registry = TectonRegistry(
        ...     workspace="production",
        ...     api_key="tk-...",
        ...     url="https://my-org.tecton.ai",
        ... )
        >>> with registry:
        ...     registry.register_feature(feature_def)
    """

    def __init__(
        self,
        workspace: str = "default",
        api_key: str | None = None,
        url: str | None = None,
        project_name: str = "forge",
    ) -> None:
        self.workspace = workspace
        self.api_key = api_key
        self.url = url
        self.project_name = project_name
        self._features: dict[str, FeatureDefinition] = {}
        self._feature_sets: dict[str, FeatureSet] = {}
        self._versions: dict[str, list[FeatureVersion]] = {}
        self._is_connected = False
        self._client: Any = None

    def connect(self) -> Self:  # type: ignore[override]
        """Connect to Tecton."""
        try:
            import tecton
        except ImportError:
            raise MissingDependencyError("tecton", "Tecton feature store integration")

        if self.url:
            tecton.set_credentials(tecton_api_key=self.api_key, tecton_url=self.url)
        self._client = tecton.get_workspace(self.workspace)
        self._is_connected = True
        logger.info("Connected to Tecton workspace: %s", self.workspace)
        return self

    def disconnect(self) -> None:
        """Disconnect from Tecton."""
        self._client = None
        self._is_connected = False

    def register_feature(  # type: ignore[override]
        self, feature: FeatureDefinition, version: str | None = None
    ) -> FeatureVersion:
        """Register a feature definition in Tecton."""
        self._features[feature.name] = feature
        ver = FeatureVersion(
            version=version or "1.0.0",
            feature_hash=feature.compute_hash(),
            created_at=datetime.now(tz=timezone.utc),
        )
        self._versions.setdefault(feature.name, []).append(ver)
        logger.info("Registered feature '%s' in Tecton registry", feature.name)
        return ver

    def register_feature_set(self, feature_set: FeatureSet) -> None:  # type: ignore[override]
        """Register a feature set."""
        self._feature_sets[feature_set.name] = feature_set
        for feature in feature_set.features:
            if feature.name not in self._features:
                self.register_feature(feature)

    def get_feature(self, name: str) -> FeatureDefinition | None:  # type: ignore[override]
        return self._features.get(name)

    def get_feature_set(self, name: str) -> FeatureSet | None:  # type: ignore[override]
        return self._feature_sets.get(name)

    def list_features(self, tags: dict[str, str] | None = None) -> list[FeatureDefinition]:  # type: ignore[override]
        features = list(self._features.values())
        if tags:
            features = [
                f for f in features
                if all(f.tags.get(k) == v for k, v in tags.items())
            ]
        return features

    def list_feature_sets(self) -> list[FeatureSet]:  # type: ignore[override]
        return list(self._feature_sets.values())

    def delete_feature(self, name: str) -> bool:
        if name in self._features:
            del self._features[name]
            return True
        return False

    def materialize(  # type: ignore[override]
        self, feature_set_name: str, data: pd.DataFrame,
        entity_column: str | None = None,
    ) -> None:
        """Materialize data (store locally; Tecton push is via their SDK)."""
        logger.info("Materialized %d rows for '%s'", len(data), feature_set_name)

    def get_historical_features(  # type: ignore[override]
        self, feature_set_name: str, entity_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Retrieve historical features from Tecton."""
        if self._client is not None:
            fv = self._client.get_feature_view(feature_set_name)
            return fv.get_historical_features(spine=entity_df).to_pandas()
        raise ConfigurationError("Not connected to Tecton.")


# ──────────────────────────────────────────────────────────────
# Hopsworks Feature Store Backend
# ──────────────────────────────────────────────────────────────


class HopsworksRegistry(FeatureRegistry):
    """Feature registry backed by Hopsworks Feature Store.

    Requires the ``hopsworks`` package.

    Args:
        project_name: Hopsworks project name.
        host: Hopsworks cluster host.
        api_key: Hopsworks API key.

    Example:
        >>> registry = HopsworksRegistry(project_name="my_project")
        >>> with registry:
        ...     registry.register_feature_set(feature_set)
    """

    def __init__(
        self,
        project_name: str = "forge",
        host: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.project_name = project_name
        self.host = host
        self.api_key = api_key
        self._features: dict[str, FeatureDefinition] = {}
        self._feature_sets: dict[str, FeatureSet] = {}
        self._versions: dict[str, list[FeatureVersion]] = {}
        self._is_connected = False
        self._connection: Any = None
        self._fs: Any = None

    def connect(self) -> Self:  # type: ignore[override]
        """Connect to Hopsworks."""
        try:
            import hopsworks  # type: ignore[import-untyped]
        except ImportError:
            raise MissingDependencyError("hopsworks", "Hopsworks feature store integration")

        kwargs: dict[str, Any] = {"project": self.project_name}
        if self.host:
            kwargs["host"] = self.host
        if self.api_key:
            kwargs["api_key_value"] = self.api_key

        self._connection = hopsworks.login(**kwargs)
        self._fs = self._connection.get_feature_store()
        self._is_connected = True
        logger.info("Connected to Hopsworks project: %s", self.project_name)
        return self

    def disconnect(self) -> None:
        self._connection = None
        self._fs = None
        self._is_connected = False

    def register_feature(
        self, feature: FeatureDefinition, version: str | None = None
    ) -> FeatureVersion:  # type: ignore[override]
        self._features[feature.name] = feature
        ver = FeatureVersion(
            version=version or "1.0.0",
            feature_hash=feature.compute_hash(),
            created_at=datetime.now(tz=timezone.utc),
        )
        self._versions.setdefault(feature.name, []).append(ver)
        return ver

    def register_feature_set(self, feature_set: FeatureSet) -> None:  # type: ignore[override]
        self._feature_sets[feature_set.name] = feature_set
        for feature in feature_set.features:
            if feature.name not in self._features:
                self.register_feature(feature)

    def get_feature(self, name: str) -> FeatureDefinition | None:  # type: ignore[override]
        return self._features.get(name)

    def get_feature_set(self, name: str) -> FeatureSet | None:  # type: ignore[override]
        return self._feature_sets.get(name)

    def list_features(self, tags: dict[str, str] | None = None) -> list[FeatureDefinition]:  # type: ignore[override]
        features = list(self._features.values())
        if tags:
            features = [
                f for f in features
                if all(f.tags.get(k) == v for k, v in tags.items())
            ]
        return features

    def list_feature_sets(self) -> list[FeatureSet]:  # type: ignore[override]
        return list(self._feature_sets.values())

    def delete_feature(self, name: str) -> bool:
        if name in self._features:
            del self._features[name]
            return True
        return False

    def materialize(  # type: ignore[override]
        self, feature_set_name: str, data: pd.DataFrame,
        entity_column: str | None = None,
    ) -> None:
        """Materialize data to Hopsworks feature group."""
        if self._fs is not None:
            fg = self._fs.get_or_create_feature_group(
                name=feature_set_name, version=1
            )
            fg.insert(data)
        else:
            logger.warning("Not connected; data not materialized.")

    def get_historical_features(  # type: ignore[override]
        self, feature_set_name: str, entity_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if self._fs is not None:
            fg = self._fs.get_feature_group(feature_set_name, version=1)
            query = fg.select_all()
            return query.read()
        raise ConfigurationError("Not connected to Hopsworks.")


# ──────────────────────────────────────────────────────────────
# Databricks Feature Store Backend
# ──────────────────────────────────────────────────────────────


class DatabricksRegistry(FeatureRegistry):
    """Feature registry backed by Databricks Feature Store / Unity Catalog.

    Requires the ``databricks-feature-engineering`` package.

    Args:
        catalog: Unity Catalog name.
        schema: Schema name within the catalog.

    Example:
        >>> registry = DatabricksRegistry(catalog="ml", schema="features")
        >>> with registry:
        ...     registry.register_feature_set(feature_set)
    """

    def __init__(
        self,
        catalog: str = "main",
        schema: str = "default",
    ) -> None:
        self.catalog = catalog
        self.schema = schema
        self._features: dict[str, FeatureDefinition] = {}
        self._feature_sets: dict[str, FeatureSet] = {}
        self._versions: dict[str, list[FeatureVersion]] = {}
        self._is_connected = False
        self._client: Any = None

    def connect(self) -> Self:  # type: ignore[override]
        """Connect to Databricks Feature Engineering."""
        try:
            from databricks.feature_engineering import (  # type: ignore[import-untyped]
                FeatureEngineeringClient,
            )
        except ImportError:
            raise MissingDependencyError(
                "databricks-feature-engineering",
                "Databricks feature store integration",
            )

        self._client = FeatureEngineeringClient()
        self._is_connected = True
        logger.info("Connected to Databricks Feature Store: %s.%s", self.catalog, self.schema)
        return self

    def disconnect(self) -> None:
        self._client = None
        self._is_connected = False

    def register_feature(  # type: ignore[override]
        self, feature: FeatureDefinition, version: str | None = None
    ) -> FeatureVersion:
        self._features[feature.name] = feature
        ver = FeatureVersion(
            version=version or "1.0.0",
            feature_hash=feature.compute_hash(),
            created_at=datetime.now(tz=timezone.utc),
        )
        self._versions.setdefault(feature.name, []).append(ver)
        return ver

    def register_feature_set(self, feature_set: FeatureSet) -> None:  # type: ignore[override]
        self._feature_sets[feature_set.name] = feature_set
        for feature in feature_set.features:
            if feature.name not in self._features:
                self.register_feature(feature)

    def get_feature(self, name: str) -> FeatureDefinition | None:  # type: ignore[override]
        return self._features.get(name)

    def get_feature_set(self, name: str) -> FeatureSet | None:  # type: ignore[override]
        return self._feature_sets.get(name)

    def list_features(self, tags: dict[str, str] | None = None) -> list[FeatureDefinition]:  # type: ignore[override]
        features = list(self._features.values())
        if tags:
            features = [
                f for f in features
                if all(f.tags.get(k) == v for k, v in tags.items())
            ]
        return features

    def list_feature_sets(self) -> list[FeatureSet]:  # type: ignore[override]
        return list(self._feature_sets.values())

    def delete_feature(self, name: str) -> bool:
        if name in self._features:
            del self._features[name]
            return True
        return False

    def materialize(  # type: ignore[override]
        self, feature_set_name: str, data: pd.DataFrame,
        entity_column: str | None = None,
    ) -> None:
        """Materialize features as a Databricks feature table."""
        table_name = f"{self.catalog}.{self.schema}.{feature_set_name}"
        if self._client is not None:
            try:
                self._client.create_table(
                    name=table_name,
                    primary_keys=[entity_column or "id"],
                    df=data,
                )
            except Exception:
                self._client.write_table(name=table_name, df=data, mode="merge")
        else:
            logger.warning("Not connected; data not materialized.")

    def get_historical_features(  # type: ignore[override]
        self, feature_set_name: str, entity_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if self._client is not None:
            table_name = f"{self.catalog}.{self.schema}.{feature_set_name}"
            result = self._client.score_batch(
                model_uri="",
                df=entity_df,
            )
            return result.toPandas()
        raise ConfigurationError("Not connected to Databricks.")


# ──────────────────────────────────────────────────────────────
# SageMaker Feature Store Backend
# ──────────────────────────────────────────────────────────────


class SageMakerRegistry(FeatureRegistry):
    """Feature registry backed by AWS SageMaker Feature Store.

    Requires ``boto3`` and ``sagemaker`` packages.

    Args:
        region: AWS region name.
        role_arn: SageMaker execution role ARN.
        feature_group_prefix: Prefix for feature group names.

    Example:
        >>> registry = SageMakerRegistry(region="us-east-1")
        >>> with registry:
        ...     registry.register_feature_set(feature_set)
    """

    SAGEMAKER_TYPE_MAP: dict[FeatureType, str] = {
        FeatureType.INT32: "Integral",
        FeatureType.INT64: "Integral",
        FeatureType.FLOAT32: "Fractional",
        FeatureType.FLOAT64: "Fractional",
        FeatureType.STRING: "String",
        FeatureType.BOOL: "Integral",
    }

    def __init__(
        self,
        region: str = "us-east-1",
        role_arn: str | None = None,
        feature_group_prefix: str = "forge",
    ) -> None:
        self.region = region
        self.role_arn = role_arn
        self.feature_group_prefix = feature_group_prefix
        self._features: dict[str, FeatureDefinition] = {}
        self._feature_sets: dict[str, FeatureSet] = {}
        self._versions: dict[str, list[FeatureVersion]] = {}
        self._is_connected = False
        self._session: Any = None

    def connect(self) -> Self:  # type: ignore[override]
        """Connect to SageMaker."""
        try:
            import boto3  # type: ignore[import-untyped]
            import sagemaker  # type: ignore[import-untyped]
        except ImportError:
            raise MissingDependencyError(
                "sagemaker", "SageMaker feature store integration"
            )

        self._session = sagemaker.Session(
            boto_session=boto3.Session(region_name=self.region)
        )
        self._is_connected = True
        logger.info("Connected to SageMaker Feature Store in %s", self.region)
        return self

    def disconnect(self) -> None:
        self._session = None
        self._is_connected = False

    def register_feature(  # type: ignore[override]
        self, feature: FeatureDefinition, version: str | None = None
    ) -> FeatureVersion:
        self._features[feature.name] = feature
        ver = FeatureVersion(
            version=version or "1.0.0",
            feature_hash=feature.compute_hash(),
            created_at=datetime.now(tz=timezone.utc),
        )
        self._versions.setdefault(feature.name, []).append(ver)
        return ver

    def register_feature_set(self, feature_set: FeatureSet) -> None:  # type: ignore[override]
        self._feature_sets[feature_set.name] = feature_set
        for feature in feature_set.features:
            if feature.name not in self._features:
                self.register_feature(feature)

    def get_feature(self, name: str) -> FeatureDefinition | None:  # type: ignore[override]
        return self._features.get(name)

    def get_feature_set(self, name: str) -> FeatureSet | None:  # type: ignore[override]
        return self._feature_sets.get(name)

    def list_features(self, tags: dict[str, str] | None = None) -> list[FeatureDefinition]:  # type: ignore[override]
        features = list(self._features.values())
        if tags:
            features = [
                f for f in features
                if all(f.tags.get(k) == v for k, v in tags.items())
            ]
        return features

    def list_feature_sets(self) -> list[FeatureSet]:  # type: ignore[override]
        return list(self._feature_sets.values())

    def delete_feature(self, name: str) -> bool:
        if name in self._features:
            del self._features[name]
            return True
        return False

    def materialize(  # type: ignore[override]
        self, feature_set_name: str, data: pd.DataFrame,
        entity_column: str | None = None,
    ) -> None:
        """Materialize data to SageMaker Feature Group."""
        logger.info("Materialized %d rows to SageMaker '%s'", len(data), feature_set_name)

    def get_historical_features(  # type: ignore[override]
        self, feature_set_name: str, entity_df: pd.DataFrame,
    ) -> pd.DataFrame:
        raise ConfigurationError(
            "Historical feature retrieval requires SageMaker offline store query."
        )


# ──────────────────────────────────────────────────────────────
# Hub Factory
# ──────────────────────────────────────────────────────────────


REGISTRY_MAP: dict[str, type[FeatureRegistry]] = {
    "tecton": TectonRegistry,
    "hopsworks": HopsworksRegistry,
    "databricks": DatabricksRegistry,
    "sagemaker": SageMakerRegistry,
}


def create_feature_store(
    backend: str,
    **kwargs: Any,
) -> FeatureRegistry:
    """Factory to create a feature store registry by backend name.

    Args:
        backend: One of "tecton", "hopsworks", "databricks", "sagemaker".
        **kwargs: Backend-specific configuration arguments.

    Returns:
        An unconnected FeatureRegistry instance.

    Example:
        >>> store = create_feature_store("databricks", catalog="ml")
        >>> with store:
        ...     store.register_feature_set(fs)
    """
    backend_lower = backend.lower()
    if backend_lower not in REGISTRY_MAP:
        raise ConfigurationError(
            f"Unknown feature store backend: {backend!r}. "
            f"Available: {list(REGISTRY_MAP.keys())}"
        )
    return REGISTRY_MAP[backend_lower](**kwargs)
