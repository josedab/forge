"""Pipeline persistence: save / load / version-track studio pipelines.

Provides JSON-based serialisation so visual studio pipelines can be
saved, reloaded, versioned, and shared.

Example:
    >>> from forge.studio.persistence import PipelineStore
    >>> store = PipelineStore("./pipelines")
    >>> store.save(studio, name="churn_v1")
    >>> restored = store.load("churn_v1")
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PipelineVersion:
    """A versioned snapshot of a pipeline configuration.

    Attributes:
        name: Pipeline name.
        version: Semantic version string.
        steps: Serialised step descriptors.
        created_at: ISO timestamp.
        description: Optional description.
        metadata: Arbitrary key-value metadata.
        content_hash: Hash of step content for dedup.
    """

    name: str
    version: str
    steps: list[dict[str, Any]]
    created_at: str = ""
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.content_hash:
            self.content_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        blob = json.dumps(self.steps, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "steps": self.steps,
            "created_at": self.created_at,
            "description": self.description,
            "metadata": self.metadata,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineVersion:
        return cls(
            name=data["name"],
            version=data.get("version", "0.0.0"),
            steps=data.get("steps", []),
            created_at=data.get("created_at", ""),
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
            content_hash=data.get("content_hash", ""),
        )


class PipelineStore:
    """Persistent store for feature-engineering pipeline configurations.

    Stores pipelines as JSON files in a directory, with automatic
    versioning and content-hash deduplication.

    Parameters
    ----------
    storage_dir : str | Path | None
        Directory for pipeline files. ``None`` for in-memory-only.
    """

    def __init__(self, storage_dir: str | Path | None = None) -> None:
        self._storage_dir = Path(storage_dir) if storage_dir else None
        self._pipelines: dict[str, list[PipelineVersion]] = {}
        if self._storage_dir:
            self._storage_dir.mkdir(parents=True, exist_ok=True)
            self._load_all()

    # ---- public API --------------------------------------------------

    def save(
        self,
        steps: list[dict[str, Any]],
        name: str,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> PipelineVersion:
        """Save a pipeline configuration.

        Automatically increments the version number.

        Args:
            steps: Serialised step descriptors.
            name: Pipeline name.
            description: Optional description.
            metadata: Extra metadata.

        Returns:
            The created PipelineVersion.
        """
        versions = self._pipelines.get(name, [])
        next_ver = f"0.{len(versions) + 1}.0"

        pv = PipelineVersion(
            name=name,
            version=next_ver,
            steps=steps,
            description=description,
            metadata=metadata or {},
        )
        versions.append(pv)
        self._pipelines[name] = versions

        if self._storage_dir:
            self._persist(name)

        return pv

    def load(
        self, name: str, version: str | None = None,
    ) -> PipelineVersion | None:
        """Load a pipeline by name and optional version.

        Args:
            name: Pipeline name.
            version: Specific version. ``None`` loads latest.

        Returns:
            PipelineVersion or ``None`` if not found.
        """
        versions = self._pipelines.get(name, [])
        if not versions:
            return None
        if version is None:
            return versions[-1]
        for pv in versions:
            if pv.version == version:
                return pv
        return None

    def list_pipelines(self) -> list[str]:
        """List all pipeline names."""
        return sorted(self._pipelines.keys())

    def list_versions(self, name: str) -> list[PipelineVersion]:
        """List all versions of a pipeline."""
        return list(self._pipelines.get(name, []))

    def delete(self, name: str) -> bool:
        """Delete a pipeline and all its versions."""
        if name in self._pipelines:
            del self._pipelines[name]
            if self._storage_dir:
                path = self._storage_dir / f"{name}.json"
                path.unlink(missing_ok=True)
            return True
        return False

    @property
    def size(self) -> int:
        """Total number of pipeline versions stored."""
        return sum(len(v) for v in self._pipelines.values())

    # ---- persistence -------------------------------------------------

    def _persist(self, name: str) -> None:
        if self._storage_dir is None:
            return
        versions = self._pipelines.get(name, [])
        path = self._storage_dir / f"{name}.json"
        data = [pv.to_dict() for pv in versions]
        path.write_text(json.dumps(data, indent=2, default=str))

    def _load_all(self) -> None:
        if self._storage_dir is None:
            return
        for path in self._storage_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                name = path.stem
                self._pipelines[name] = [
                    PipelineVersion.from_dict(d) for d in data
                ]
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to load %s: %s", path, e)
