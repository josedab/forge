"""Feature lineage versioning — snapshot, diff, and merge for feature sets.

Provides git-like version control for feature lineage graphs, enabling
teams to track how feature sets evolve over time, compare versions,
and resolve conflicts in collaborative feature development.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LineageEvent:
    """A recorded lineage event from a generator or selector.

    Attributes:
        feature_name: Name of the feature produced/affected.
        event_type: One of "create", "select", "drop", "transform".
        source_columns: Input columns used.
        operation: Name of the transformation/operation.
        parameters: Operation parameters.
        timestamp: When the event occurred.
        generator_class: Class name of the generator.
    """

    feature_name: str
    event_type: str
    source_columns: list[str] = field(default_factory=list)
    operation: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    generator_class: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "feature_name": self.feature_name,
            "event_type": self.event_type,
            "source_columns": self.source_columns,
            "operation": self.operation,
            "parameters": self.parameters,
            "timestamp": self.timestamp,
            "generator_class": self.generator_class,
        }


class LineageEventRecorder:
    """Records lineage events emitted by generators and selectors.

    Provides a central log of all feature transformations that can
    be used to build lineage graphs and audit trails.

    Example:
        >>> recorder = LineageEventRecorder()
        >>> recorder.record_create("log_price", ["price"], "LogTransform")
        >>> recorder.record_select("log_price", importance=0.85)
        >>> print(len(recorder.events))
        2
    """

    def __init__(self) -> None:
        self._events: list[LineageEvent] = []

    def record_create(
        self,
        feature_name: str,
        source_columns: list[str],
        operation: str = "",
        generator_class: str = "",
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Record a feature creation event."""
        self._events.append(LineageEvent(
            feature_name=feature_name,
            event_type="create",
            source_columns=source_columns,
            operation=operation,
            generator_class=generator_class,
            parameters=parameters or {},
        ))

    def record_select(
        self,
        feature_name: str,
        importance: float | None = None,
    ) -> None:
        """Record a feature selection event."""
        params: dict[str, Any] = {}
        if importance is not None:
            params["importance"] = importance
        self._events.append(LineageEvent(
            feature_name=feature_name,
            event_type="select",
            parameters=params,
        ))

    def record_drop(
        self, feature_name: str, reason: str = ""
    ) -> None:
        """Record a feature drop event."""
        self._events.append(LineageEvent(
            feature_name=feature_name,
            event_type="drop",
            parameters={"reason": reason},
        ))

    @property
    def events(self) -> list[LineageEvent]:
        """All recorded events."""
        return list(self._events)

    def clear(self) -> None:
        """Clear all recorded events."""
        self._events.clear()


@dataclass
class LineageSnapshot:
    """A versioned snapshot of a feature lineage state.

    Attributes:
        version: Snapshot version identifier.
        features: Map of feature name → metadata at snapshot time.
        timestamp: When the snapshot was created.
        description: Human-readable description of this version.
        parent_version: Previous version (for diffing).
    """

    version: str
    features: dict[str, dict[str, Any]]
    timestamp: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    description: str = ""
    parent_version: str | None = None

    @property
    def feature_names(self) -> list[str]:
        """Sorted list of feature names in this snapshot."""
        return sorted(self.features.keys())

    def checksum(self) -> str:
        """Compute a hash of the feature set for comparison."""
        content = json.dumps(
            sorted(self.features.items()), sort_keys=True, default=str
        )
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "version": self.version,
            "features": self.features,
            "timestamp": self.timestamp,
            "description": self.description,
            "parent_version": self.parent_version,
            "checksum": self.checksum(),
        }


@dataclass
class LineageDiff:
    """Difference between two lineage snapshots.

    Attributes:
        from_version: Source version.
        to_version: Target version.
        added: Features added in target.
        removed: Features removed from source.
        modified: Features whose metadata changed.
    """

    from_version: str
    to_version: str
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        """Whether there are any differences."""
        return bool(self.added or self.removed or self.modified)

    def summary(self) -> str:
        """Human-readable diff summary."""
        lines = [f"Diff: {self.from_version} → {self.to_version}"]
        if self.added:
            lines.append(f"  + Added ({len(self.added)}): {', '.join(self.added[:5])}")
            if len(self.added) > 5:
                lines.append(f"    ... and {len(self.added) - 5} more")
        if self.removed:
            lines.append(f"  - Removed ({len(self.removed)}): {', '.join(self.removed[:5])}")
            if len(self.removed) > 5:
                lines.append(f"    ... and {len(self.removed) - 5} more")
        if self.modified:
            lines.append(f"  ~ Modified ({len(self.modified)}): {', '.join(self.modified[:5])}")
        if not self.has_changes:
            lines.append("  No changes")
        return "\n".join(lines)


class LineageVersionStore:
    """Version store for feature lineage snapshots.

    Provides git-like versioning: create snapshots, diff between versions,
    and list version history.

    Example:
        >>> store = LineageVersionStore()
        >>> store.snapshot("v1", {"price": {"type": "source"}, "log_price": {"type": "generated"}})
        >>> store.snapshot("v2", {"price": {"type": "source"}, "sqrt_price": {"type": "generated"}})
        >>> diff = store.diff("v1", "v2")
        >>> print(diff.summary())
    """

    def __init__(self) -> None:
        self._versions: dict[str, LineageSnapshot] = {}
        self._version_order: list[str] = []

    def snapshot(
        self,
        version: str,
        features: dict[str, dict[str, Any]],
        description: str = "",
    ) -> LineageSnapshot:
        """Create a versioned snapshot of the current feature state.

        Args:
            version: Version identifier (e.g., "v1", "2024-01-15").
            features: Map of feature name → metadata dict.
            description: Human-readable description.

        Returns:
            The created snapshot.
        """
        parent = self._version_order[-1] if self._version_order else None
        snap = LineageSnapshot(
            version=version,
            features=dict(features),
            description=description,
            parent_version=parent,
        )
        self._versions[version] = snap
        self._version_order.append(version)
        logger.info("Created lineage snapshot: %s (%d features)", version, len(features))
        return snap

    def get(self, version: str) -> LineageSnapshot | None:
        """Retrieve a snapshot by version."""
        return self._versions.get(version)

    @property
    def latest(self) -> LineageSnapshot | None:
        """Get the most recent snapshot."""
        if not self._version_order:
            return None
        return self._versions[self._version_order[-1]]

    @property
    def versions(self) -> list[str]:
        """List all version identifiers in order."""
        return list(self._version_order)

    def diff(self, from_version: str, to_version: str) -> LineageDiff:
        """Compute the diff between two snapshots.

        Args:
            from_version: Source version.
            to_version: Target version.

        Returns:
            LineageDiff describing the changes.

        Raises:
            KeyError: If either version is not found.
        """
        v_from = self._versions.get(from_version)
        v_to = self._versions.get(to_version)
        if v_from is None:
            raise KeyError(f"Version '{from_version}' not found")
        if v_to is None:
            raise KeyError(f"Version '{to_version}' not found")

        from_names = set(v_from.features.keys())
        to_names = set(v_to.features.keys())

        added = sorted(to_names - from_names)
        removed = sorted(from_names - to_names)

        modified: list[str] = []
        for name in sorted(from_names & to_names):
            if v_from.features[name] != v_to.features[name]:
                modified.append(name)

        return LineageDiff(
            from_version=from_version,
            to_version=to_version,
            added=added,
            removed=removed,
            modified=modified,
        )

    def history(self) -> list[dict[str, Any]]:
        """Get version history as a list of dicts."""
        result = []
        for v in self._version_order:
            snap = self._versions[v]
            result.append({
                "version": v,
                "n_features": len(snap.features),
                "timestamp": snap.timestamp,
                "description": snap.description,
                "checksum": snap.checksum(),
            })
        return result
