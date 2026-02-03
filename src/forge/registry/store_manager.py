"""Unified feature store integration manager.

Provides a single interface for working with multiple feature store
backends, enabling bi-directional sync, feature discovery across stores,
and migration between stores.

Example:
    >>> from forge.registry.store_manager import FeatureStoreManager
    >>> manager = FeatureStoreManager()
    >>> manager.register_store("local", local_registry)
    >>> manager.register_store("production", feast_registry)
    >>> manager.push("local", "production", feature_set="user_features")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from forge.exceptions import ConfigurationError, ValidationError
from forge.registry.base import (
    FeatureRegistry,
)

logger = logging.getLogger(__name__)


@dataclass
class MigrationAction:
    """A single action during feature migration between stores.

    Attributes:
        feature_name: Feature being migrated.
        action: What happened ("pushed", "pulled", "skipped", "error").
        source_store: Where the feature came from.
        target_store: Where the feature went.
        details: Additional context.
    """

    feature_name: str
    action: str
    source_store: str
    target_store: str
    details: str = ""


@dataclass
class MigrationResult:
    """Result of a feature migration operation.

    Attributes:
        actions: List of actions taken.
        source_store: Source store name.
        target_store: Target store name.
        timestamp: When the migration occurred.
    """

    actions: list[MigrationAction] = field(default_factory=list)
    source_store: str = ""
    target_store: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    @property
    def pushed(self) -> int:
        """Number of features successfully pushed."""
        return sum(1 for a in self.actions if a.action == "pushed")

    @property
    def skipped(self) -> int:
        """Number of features skipped."""
        return sum(1 for a in self.actions if a.action == "skipped")

    @property
    def errors(self) -> int:
        """Number of features that failed."""
        return sum(1 for a in self.actions if a.action == "error")

    def summary(self) -> str:
        """Human-readable migration summary."""
        return (
            f"Migration {self.source_store} → {self.target_store}: "
            f"{self.pushed} pushed, {self.skipped} skipped, {self.errors} errors"
        )


@dataclass
class StoreDiscoveryResult:
    """Result of discovering features across stores.

    Attributes:
        store_name: Name of the store.
        features: List of feature names found.
        feature_sets: List of feature set names found.
    """

    store_name: str
    features: list[str] = field(default_factory=list)
    feature_sets: list[str] = field(default_factory=list)


class FeatureStoreManager:
    """Unified manager for multiple feature store backends.

    Provides a single point of access to register, discover, and
    migrate features across multiple store backends.

    Args:
        default_store: Name of the default store to use.

    Example:
        >>> manager = FeatureStoreManager()
        >>> manager.register_store("dev", local_registry)
        >>> manager.register_store("prod", feast_registry)
        >>> # Discover features across all stores
        >>> results = manager.discover_all()
        >>> # Push features from dev to prod
        >>> result = manager.push("dev", "prod", features=["log_price"])
    """

    def __init__(self, default_store: str | None = None) -> None:
        self._stores: dict[str, FeatureRegistry] = {}
        self._default_store = default_store

    def register_store(self, name: str, registry: FeatureRegistry) -> None:
        """Register a feature store backend.

        Args:
            name: Identifier for this store.
            registry: FeatureRegistry implementation.
        """
        self._stores[name] = registry
        if self._default_store is None:
            self._default_store = name
        logger.info("Registered feature store: %s (%s)", name, type(registry).__name__)

    def get_store(self, name: str | None = None) -> FeatureRegistry:
        """Get a registered store by name.

        Args:
            name: Store name. None for default.

        Returns:
            The FeatureRegistry instance.

        Raises:
            ConfigurationError: If store not found.
        """
        store_name = name or self._default_store
        if store_name is None or store_name not in self._stores:
            raise ConfigurationError(
                f"Store '{store_name}' not found. "
                f"Available: {list(self._stores.keys())}"
            )
        return self._stores[store_name]

    @property
    def store_names(self) -> list[str]:
        """List all registered store names."""
        return list(self._stores.keys())

    def discover_all(self) -> list[StoreDiscoveryResult]:
        """Discover features across all registered stores.

        Returns:
            List of discovery results, one per store.
        """
        results = []
        for name, store in self._stores.items():
            try:
                features = store.list_features()
                feature_sets = store.list_feature_sets()
                results.append(StoreDiscoveryResult(
                    store_name=name,
                    features=features,
                    feature_sets=feature_sets,
                ))
            except Exception as exc:
                logger.warning("Failed to discover features in store '%s': %s", name, exc)
                results.append(StoreDiscoveryResult(store_name=name))
        return results

    def push(
        self,
        source: str,
        target: str,
        features: list[str] | None = None,
        feature_set: str | None = None,
        overwrite: bool = False,
    ) -> MigrationResult:
        """Push features from source store to target store.

        Args:
            source: Source store name.
            target: Target store name.
            features: Specific feature names to push. None for all.
            feature_set: Feature set name to push.
            overwrite: Whether to overwrite existing features.

        Returns:
            MigrationResult with action details.
        """
        source_store = self.get_store(source)
        target_store = self.get_store(target)

        result = MigrationResult(source_store=source, target_store=target)

        if feature_set:
            fs = source_store.get_feature_set(feature_set)
            if fs is None:
                raise ValidationError(
                    f"Feature set '{feature_set}' not found in store '{source}'"
                )
            feature_names = [fd.name for fd in fs.features]
        elif features:
            feature_names = features
        else:
            feature_names = source_store.list_features()

        for fname in feature_names:
            try:
                feat_def = source_store.get_feature(fname)
                if feat_def is None:
                    result.actions.append(MigrationAction(
                        feature_name=fname,
                        action="skipped",
                        source_store=source,
                        target_store=target,
                        details="Not found in source",
                    ))
                    continue

                existing = target_store.get_feature(fname)
                if existing is not None and not overwrite:
                    result.actions.append(MigrationAction(
                        feature_name=fname,
                        action="skipped",
                        source_store=source,
                        target_store=target,
                        details="Already exists in target",
                    ))
                    continue

                target_store.register_feature(feat_def)
                result.actions.append(MigrationAction(
                    feature_name=fname,
                    action="pushed",
                    source_store=source,
                    target_store=target,
                ))
            except Exception as exc:
                result.actions.append(MigrationAction(
                    feature_name=fname,
                    action="error",
                    source_store=source,
                    target_store=target,
                    details=str(exc),
                ))

        logger.info(result.summary())
        return result

    def pull(
        self,
        source: str,
        target: str,
        features: list[str] | None = None,
        overwrite: bool = False,
    ) -> MigrationResult:
        """Pull features from source store into target store.

        Convenience wrapper around push with reversed arguments.

        Args:
            source: Remote store to pull from.
            target: Local store to pull into.
            features: Specific feature names. None for all.
            overwrite: Whether to overwrite existing.

        Returns:
            MigrationResult with action details.
        """
        return self.push(source, target, features=features, overwrite=overwrite)

    def diff(self, store_a: str, store_b: str) -> dict[str, list[str]]:
        """Compare features between two stores.

        Args:
            store_a: First store name.
            store_b: Second store name.

        Returns:
            Dict with "only_in_a", "only_in_b", and "in_both" keys.
        """
        features_a = set(self.get_store(store_a).list_features())
        features_b = set(self.get_store(store_b).list_features())

        return {
            "only_in_a": sorted(features_a - features_b),
            "only_in_b": sorted(features_b - features_a),
            "in_both": sorted(features_a & features_b),
        }
