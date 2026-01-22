"""Feature store registry integration module.

This module provides connectors for popular feature stores like Feast,
Tecton, and Hopsworks, enabling seamless feature registration and serving.
"""

from __future__ import annotations

from forge.registry.base import (
    FeatureRegistry,
    FeatureDefinition,
    FeatureVersion,
    FeatureSet,
    RegistryConfig,
)
from forge.registry.feast_registry import FeastRegistry
from forge.registry.local_registry import LocalRegistry

__all__ = [
    "FeatureRegistry",
    "FeatureDefinition",
    "FeatureVersion",
    "FeatureSet",
    "RegistryConfig",
    "FeastRegistry",
    "LocalRegistry",
]
