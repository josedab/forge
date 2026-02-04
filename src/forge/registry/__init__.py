"""Feature store registry integration module.

This module provides connectors for popular feature stores like Feast,
Tecton, and Hopsworks, enabling seamless feature registration and serving.
"""

from __future__ import annotations

from forge.registry.base import (
    FeatureDefinition,
    FeatureRegistry,
    FeatureSet,
    FeatureVersion,
    RegistryConfig,
)
from forge.registry.feast_registry import FeastRegistry
from forge.registry.local_registry import LocalRegistry
from forge.registry.sync import (
    ConflictStrategy,
    FeatureStoreSync,
    LineageRecord,
    SyncAction,
    SyncConfig,
    SyncDirection,
    SyncResult,
)

__all__ = [
    "ConflictStrategy",
    "FeastRegistry",
    "FeatureDefinition",
    "FeatureRegistry",
    "FeatureSet",
    "FeatureStoreSync",
    "FeatureVersion",
    "LineageRecord",
    "LocalRegistry",
    "RegistryConfig",
    "SyncAction",
    "SyncConfig",
    "SyncDirection",
    "SyncResult",
]
