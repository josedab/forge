"""Feature catalog for organizing and searching documentation."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from forge.documentation.generator import DatasetDoc, FeatureDoc


@dataclass
class CatalogEntry:
    """An entry in the feature catalog."""

    feature_doc: FeatureDoc
    dataset_name: str
    dataset_version: str = "1.0.0"
    status: Literal["active", "deprecated", "experimental"] = "active"
    owner: str = ""
    last_updated: str = ""
    usage_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature": self.feature_doc.to_dict(),
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "status": self.status,
            "owner": self.owner,
            "last_updated": self.last_updated,
            "usage_count": self.usage_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CatalogEntry:
        """Create from dictionary."""
        return cls(
            feature_doc=FeatureDoc.from_dict(data["feature"]),
            dataset_name=data.get("dataset_name", ""),
            dataset_version=data.get("dataset_version", "1.0.0"),
            status=data.get("status", "active"),
            owner=data.get("owner", ""),
            last_updated=data.get("last_updated", ""),
            usage_count=data.get("usage_count", 0),
        )


class FeatureCatalog:
    """A searchable catalog of feature documentation.

    Provides organization, search, and management capabilities
    for feature documentation across multiple datasets.

    Example:
        >>> catalog = FeatureCatalog()
        >>> catalog.add_dataset(dataset_doc)
        >>> results = catalog.search("price", category="financial")
        >>> catalog.save("feature_catalog.json")
    """

    def __init__(self) -> None:
        """Initialize an empty catalog."""
        self._entries: dict[str, CatalogEntry] = {}
        self._datasets: dict[str, DatasetDoc] = {}
        self._tags_index: dict[str, set[str]] = {}
        self._category_index: dict[str, set[str]] = {}

    def add_dataset(
        self,
        dataset_doc: DatasetDoc,
        owner: str = "",
        status: Literal["active", "deprecated", "experimental"] = "active",
    ) -> int:
        """Add a dataset's features to the catalog.

        Args:
            dataset_doc: Dataset documentation.
            owner: Owner of the features.
            status: Status of the features.

        Returns:
            Number of features added.
        """
        self._datasets[dataset_doc.name] = dataset_doc
        count = 0

        for feature in dataset_doc.features:
            entry_id = f"{dataset_doc.name}:{feature.name}"
            entry = CatalogEntry(
                feature_doc=feature,
                dataset_name=dataset_doc.name,
                status=status,
                owner=owner,
                last_updated=datetime.now().isoformat(),
            )
            self._entries[entry_id] = entry
            self._index_entry(entry_id, entry)
            count += 1

        return count

    def _index_entry(self, entry_id: str, entry: CatalogEntry) -> None:
        """Index an entry for searching."""
        # Index by tags
        for tag in entry.feature_doc.tags:
            if tag not in self._tags_index:
                self._tags_index[tag] = set()
            self._tags_index[tag].add(entry_id)

        # Index by category
        category = entry.feature_doc.category
        if category not in self._category_index:
            self._category_index[category] = set()
        self._category_index[category].add(entry_id)

    def get_feature(
        self, dataset_name: str, feature_name: str
    ) -> CatalogEntry | None:
        """Get a specific feature entry.

        Args:
            dataset_name: Name of the dataset.
            feature_name: Name of the feature.

        Returns:
            CatalogEntry or None if not found.
        """
        entry_id = f"{dataset_name}:{feature_name}"
        return self._entries.get(entry_id)

    def search(
        self,
        query: str | None = None,
        category: str | None = None,
        tags: list[str] | None = None,
        dataset: str | None = None,
        status: str | None = None,
        dtype: str | None = None,
    ) -> list[CatalogEntry]:
        """Search for features in the catalog.

        Args:
            query: Text search in name and description.
            category: Filter by category.
            tags: Filter by tags (any match).
            dataset: Filter by dataset name.
            status: Filter by status.
            dtype: Filter by data type.

        Returns:
            List of matching entries.
        """
        results = set(self._entries.keys())

        # Filter by category
        if category and category in self._category_index:
            results &= self._category_index[category]
        elif category:
            results = set()

        # Filter by tags
        if tags:
            tag_matches = set()
            for tag in tags:
                if tag in self._tags_index:
                    tag_matches |= self._tags_index[tag]
            results &= tag_matches

        # Filter by dataset
        if dataset:
            results = {r for r in results if r.startswith(f"{dataset}:")}

        # Filter by status
        if status:
            results = {r for r in results if self._entries[r].status == status}

        # Filter by dtype
        if dtype:
            results = {r for r in results if self._entries[r].feature_doc.dtype == dtype}

        # Text search
        if query:
            query_lower = query.lower()
            text_matches = set()
            for entry_id in results:
                entry = self._entries[entry_id]
                if (query_lower in entry.feature_doc.name.lower() or
                    query_lower in entry.feature_doc.description.lower()):
                    text_matches.add(entry_id)
            results = text_matches

        return [self._entries[r] for r in results]

    def list_categories(self) -> list[str]:
        """List all categories in the catalog."""
        return list(self._category_index.keys())

    def list_tags(self) -> list[str]:
        """List all tags in the catalog."""
        return list(self._tags_index.keys())

    def list_datasets(self) -> list[str]:
        """List all datasets in the catalog."""
        return list(self._datasets.keys())

    def get_statistics(self) -> dict[str, Any]:
        """Get catalog statistics."""
        return {
            "total_features": len(self._entries),
            "total_datasets": len(self._datasets),
            "categories": {cat: len(ids) for cat, ids in self._category_index.items()},
            "status_counts": self._count_by_status(),
            "dtype_counts": self._count_by_dtype(),
        }

    def _count_by_status(self) -> dict[str, int]:
        """Count entries by status."""
        counts: dict[str, int] = {}
        for entry in self._entries.values():
            counts[entry.status] = counts.get(entry.status, 0) + 1
        return counts

    def _count_by_dtype(self) -> dict[str, int]:
        """Count entries by dtype."""
        counts: dict[str, int] = {}
        for entry in self._entries.values():
            dtype = entry.feature_doc.dtype
            counts[dtype] = counts.get(dtype, 0) + 1
        return counts

    def update_status(
        self,
        dataset_name: str,
        feature_name: str,
        status: Literal["active", "deprecated", "experimental"],
    ) -> bool:
        """Update the status of a feature.

        Args:
            dataset_name: Dataset name.
            feature_name: Feature name.
            status: New status.

        Returns:
            True if updated, False if not found.
        """
        entry = self.get_feature(dataset_name, feature_name)
        if entry:
            entry.status = status
            entry.last_updated = datetime.now().isoformat()
            return True
        return False

    def deprecate_feature(
        self, dataset_name: str, feature_name: str, reason: str = ""
    ) -> bool:
        """Deprecate a feature.

        Args:
            dataset_name: Dataset name.
            feature_name: Feature name.
            reason: Reason for deprecation.

        Returns:
            True if deprecated, False if not found.
        """
        entry = self.get_feature(dataset_name, feature_name)
        if entry:
            entry.status = "deprecated"
            entry.last_updated = datetime.now().isoformat()
            if reason:
                entry.feature_doc.quality_notes.append(f"Deprecated: {reason}")
            return True
        return False

    def remove_dataset(self, dataset_name: str) -> int:
        """Remove a dataset from the catalog.

        Args:
            dataset_name: Name of dataset to remove.

        Returns:
            Number of features removed.
        """
        if dataset_name not in self._datasets:
            return 0

        # Remove entries
        to_remove = [k for k in self._entries if k.startswith(f"{dataset_name}:")]
        for entry_id in to_remove:
            entry = self._entries.pop(entry_id)
            # Remove from indices
            for tag in entry.feature_doc.tags:
                if tag in self._tags_index:
                    self._tags_index[tag].discard(entry_id)
            category = entry.feature_doc.category
            if category in self._category_index:
                self._category_index[category].discard(entry_id)

        del self._datasets[dataset_name]
        return len(to_remove)

    def save(self, path: str | Path) -> None:
        """Save the catalog to a JSON file.

        Args:
            path: Path to save to.
        """
        data = {
            "entries": {k: v.to_dict() for k, v in self._entries.items()},
            "datasets": {k: v.to_dict() for k, v in self._datasets.items()},
            "saved_at": datetime.now().isoformat(),
        }
        Path(path).write_text(json.dumps(data, indent=2, default=str))

    @classmethod
    def load(cls, path: str | Path) -> FeatureCatalog:
        """Load a catalog from a JSON file.

        Args:
            path: Path to load from.

        Returns:
            Loaded FeatureCatalog.
        """
        data = json.loads(Path(path).read_text())

        catalog = cls()
        for entry_id, entry_data in data.get("entries", {}).items():
            entry = CatalogEntry.from_dict(entry_data)
            catalog._entries[entry_id] = entry
            catalog._index_entry(entry_id, entry)

        for name, dataset_data in data.get("datasets", {}).items():
            catalog._datasets[name] = DatasetDoc.from_dict(dataset_data)

        return catalog

    def __len__(self) -> int:
        """Return number of features in catalog."""
        return len(self._entries)

    def __iter__(self) -> Iterator[CatalogEntry]:
        """Iterate over catalog entries."""
        return iter(self._entries.values())

    def __contains__(self, key: str) -> bool:
        """Check if entry exists."""
        return key in self._entries


def create_catalog(
    datasets: list[DatasetDoc] | None = None,
    path: str | Path | None = None,
) -> FeatureCatalog:
    """Create a feature catalog.

    Args:
        datasets: Optional list of datasets to add.
        path: Optional path to load existing catalog.

    Returns:
        FeatureCatalog instance.
    """
    if path and Path(path).exists():
        catalog = FeatureCatalog.load(path)
    else:
        catalog = FeatureCatalog()

    if datasets:
        for dataset in datasets:
            catalog.add_dataset(dataset)

    return catalog
