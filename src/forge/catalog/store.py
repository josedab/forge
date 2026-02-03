"""Feature catalog store with search, versioning, and CRUD operations."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CatalogEntry:
    """A single entry in the feature catalog.

    Args:
        name: Feature name (unique identifier).
        description: Human-readable description.
        dtype: Data type string.
        source_columns: Original columns this feature is derived from.
        transformation: Description of the transformation applied.
        owner: Person or team owning this feature.
        tags: Key-value tags for categorization.
        version: Version string.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
        statistics: Summary statistics (mean, std, min, max, etc.).
        deprecated: Whether this feature is deprecated.
        deprecation_reason: Reason for deprecation.
    """

    name: str
    description: str = ""
    dtype: str = "float64"
    source_columns: list[str] = field(default_factory=list)
    transformation: str = ""
    owner: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    version: str = "1.0.0"
    created_at: str = ""
    updated_at: str = ""
    statistics: dict[str, Any] = field(default_factory=dict)
    deprecated: bool = False
    deprecation_reason: str = ""

    def __post_init__(self) -> None:
        now = datetime.now(tz=timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CatalogEntry:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class CatalogStore:
    """In-memory feature catalog with search and CRUD operations.

    Provides full-text search, tag filtering, versioning, and
    persistence to JSON files.

    Args:
        persist_path: Path for JSON persistence. None for memory-only.

    Example:
        >>> store = CatalogStore()
        >>> store.add(CatalogEntry(name="age", description="Customer age"))
        >>> results = store.search("customer")
    """

    def __init__(self, persist_path: str | None = None) -> None:
        self._entries: dict[str, CatalogEntry] = {}
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path and self._persist_path.exists():
            self._load()

    def add(self, entry: CatalogEntry) -> CatalogEntry:
        """Add or update a catalog entry.

        Args:
            entry: Catalog entry to add.

        Returns:
            The stored entry.
        """
        if entry.name in self._entries:
            entry.updated_at = datetime.now(tz=timezone.utc).isoformat()
        self._entries[entry.name] = entry
        self._persist()
        return entry

    def get(self, name: str) -> CatalogEntry | None:
        """Get an entry by name."""
        return self._entries.get(name)

    def delete(self, name: str) -> bool:
        """Delete an entry by name."""
        if name in self._entries:
            del self._entries[name]
            self._persist()
            return True
        return False

    def list_all(self) -> list[CatalogEntry]:
        """List all entries."""
        return list(self._entries.values())

    def search(
        self,
        query: str = "",
        tags: dict[str, str] | None = None,
        owner: str | None = None,
        include_deprecated: bool = False,
    ) -> list[CatalogEntry]:
        """Search the catalog with full-text query and filters.

        Args:
            query: Text to search in name, description, transformation.
            tags: Filter by tags (all must match).
            owner: Filter by owner.
            include_deprecated: Whether to include deprecated features.

        Returns:
            List of matching entries.
        """
        results = list(self._entries.values())

        if not include_deprecated:
            results = [e for e in results if not e.deprecated]

        if query:
            pattern = re.compile(re.escape(query), re.IGNORECASE)
            results = [
                e for e in results
                if pattern.search(e.name)
                or pattern.search(e.description)
                or pattern.search(e.transformation)
                or pattern.search(e.owner)
            ]

        if tags:
            results = [
                e for e in results
                if all(e.tags.get(k) == v for k, v in tags.items())
            ]

        if owner:
            results = [e for e in results if e.owner == owner]

        return results

    def deprecate(self, name: str, reason: str = "") -> bool:
        """Mark a feature as deprecated."""
        entry = self._entries.get(name)
        if entry is None:
            return False
        entry.deprecated = True
        entry.deprecation_reason = reason
        entry.updated_at = datetime.now(tz=timezone.utc).isoformat()
        self._persist()
        return True

    def tag(self, name: str, tags: dict[str, str]) -> bool:
        """Add tags to an entry."""
        entry = self._entries.get(name)
        if entry is None:
            return False
        entry.tags.update(tags)
        entry.updated_at = datetime.now(tz=timezone.utc).isoformat()
        self._persist()
        return True

    def get_statistics(self) -> dict[str, Any]:
        """Get catalog statistics."""
        total = len(self._entries)
        deprecated = sum(1 for e in self._entries.values() if e.deprecated)
        owners = {e.owner for e in self._entries.values() if e.owner}
        all_tags: dict[str, int] = {}
        for e in self._entries.values():
            for k in e.tags:
                all_tags[k] = all_tags.get(k, 0) + 1
        return {
            "total_features": total,
            "active_features": total - deprecated,
            "deprecated_features": deprecated,
            "unique_owners": len(owners),
            "tag_counts": all_tags,
        }

    def export_json(self, path: str | Path) -> Path:
        """Export catalog to a JSON file."""
        path = Path(path)
        data = [e.to_dict() for e in self._entries.values()]
        path.write_text(json.dumps(data, indent=2))
        return path

    def import_json(self, path: str | Path) -> int:
        """Import entries from a JSON file. Returns count imported."""
        path = Path(path)
        data = json.loads(path.read_text())
        count = 0
        for item in data:
            entry = CatalogEntry.from_dict(item)
            self.add(entry)
            count += 1
        return count

    def _persist(self) -> None:
        """Save to disk if persist path is configured."""
        if self._persist_path:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = [e.to_dict() for e in self._entries.values()]
            self._persist_path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        """Load from disk."""
        if self._persist_path and self._persist_path.exists():
            data = json.loads(self._persist_path.read_text())
            for item in data:
                entry = CatalogEntry.from_dict(item)
                self._entries[entry.name] = entry

    @property
    def size(self) -> int:
        return len(self._entries)


class FeatureCatalogStore(CatalogStore):
    """Extended catalog with transformer integration.

    Adds convenience methods for populating the catalog
    from fitted Forge transformers.

    Example:
        >>> store = FeatureCatalogStore()
        >>> store.add_from_transformer(transformer, owner="ml-team")
        >>> store.search("price")
    """

    def add_from_transformer(
        self,
        transformer: Any,
        owner: str = "",
        tags: dict[str, str] | None = None,
    ) -> int:
        """Populate catalog from a fitted transformer.

        Args:
            transformer: Fitted sklearn transformer with get_feature_names_out.
            owner: Owner to assign.
            tags: Tags to apply.

        Returns:
            Number of entries added.
        """
        names: list[str] = []
        if hasattr(transformer, "get_feature_names_out"):
            try:
                names = transformer.get_feature_names_out()
            except Exception:
                return 0

        count = 0
        for name in names:
            entry = CatalogEntry(
                name=name,
                owner=owner,
                tags=tags or {},
            )
            self.add(entry)
            count += 1

        return count
