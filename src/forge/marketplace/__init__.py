"""Feature Marketplace and Hub for community feature packs.

Provides a registry for discovering, publishing, and installing
feature packs and transformation recipes, plus compatibility checking
and manifest export.
"""

from __future__ import annotations

from forge.marketplace.compatibility import (
    CompatibilityReport,
    check_compatibility,
    check_version_constraint,
    export_manifest,
)
from forge.marketplace.hub import (
    FeatureHub,
    HubConfig,
    PackMetadata,
    PackVersion,
)
from forge.marketplace.pack_manager import (
    InstallResult,
    PackDefinition,
    PackManager,
    ValidationResult,
)

__all__ = [
    "CompatibilityReport",
    "FeatureHub",
    "HubConfig",
    "InstallResult",
    "PackDefinition",
    "PackManager",
    "PackMetadata",
    "PackVersion",
    "ValidationResult",
    "check_compatibility",
    "check_version_constraint",
    "export_manifest",
]
