"""Feature Hub: marketplace for feature packs and recipes.

Provides a local registry for publishing, discovering, versioning,
and installing feature packs. Supports both local file-based storage
and remote hub connectivity (future).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HubConfig:
    """Configuration for the Feature Hub.

    Attributes:
    ----------
    storage_path : str
        Local directory for pack storage.
    auto_validate : bool
        Validate packs on install.
    """

    storage_path: str = "~/.forge/hub"
    auto_validate: bool = True


@dataclass
class PackVersion:
    """Version information for a feature pack.

    Attributes:
    ----------
    version : str
        Semantic version string.
    published_at : float
        Unix timestamp of publication.
    checksum : str
        SHA256 checksum of pack contents.
    changelog : str
        What changed in this version.
    """

    version: str
    published_at: float = 0.0
    checksum: str = ""
    changelog: str = ""

    def __post_init__(self) -> None:
        if self.published_at == 0.0:
            self.published_at = time.time()


@dataclass
class PackMetadata:
    """Metadata for a published feature pack.

    Attributes:
    ----------
    name : str
        Unique pack name.
    description : str
        Human-readable description.
    author : str
        Author name.
    tags : list[str]
        Searchable tags.
    domain : str
        Domain category (e.g., 'finance', 'healthcare').
    versions : list[PackVersion]
        Available versions.
    dependencies : list[str]
        Required Python packages.
    downloads : int
        Download count.
    rating : float
        Average rating (0-5).
    compatible_forge_version : str
        Minimum compatible Forge version.
    """

    name: str
    description: str = ""
    author: str = ""
    tags: list[str] = field(default_factory=list)
    domain: str = "general"
    versions: list[PackVersion] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    downloads: int = 0
    rating: float = 0.0
    compatible_forge_version: str = ">=0.1.0"

    @property
    def latest_version(self) -> str:
        """Get latest version string."""
        if not self.versions:
            return "0.0.0"
        return self.versions[-1].version

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "tags": self.tags,
            "domain": self.domain,
            "versions": [
                {
                    "version": v.version,
                    "published_at": v.published_at,
                    "checksum": v.checksum,
                    "changelog": v.changelog,
                }
                for v in self.versions
            ],
            "dependencies": self.dependencies,
            "downloads": self.downloads,
            "rating": self.rating,
            "compatible_forge_version": self.compatible_forge_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PackMetadata:
        """Deserialize from dict."""
        versions = [
            PackVersion(**v) for v in data.get("versions", [])
        ]
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            author=data.get("author", ""),
            tags=data.get("tags", []),
            domain=data.get("domain", "general"),
            versions=versions,
            dependencies=data.get("dependencies", []),
            downloads=data.get("downloads", 0),
            rating=data.get("rating", 0.0),
            compatible_forge_version=data.get("compatible_forge_version", ">=0.1.0"),
        )


class FeatureHub:
    """Local feature hub for managing feature packs.

    Supports publishing, discovering, versioning, and installing
    feature packs from a local registry.

    Parameters
    ----------
    config : HubConfig | None
        Hub configuration. Uses defaults if None.

    Examples:
    --------
    >>> from forge.marketplace import FeatureHub, PackMetadata, PackVersion
    >>>
    >>> hub = FeatureHub()
    >>> pack = PackMetadata(
    ...     name="finance-basics",
    ...     description="Basic finance feature pack",
    ...     author="forge-team",
    ...     domain="finance",
    ...     tags=["finance", "ratios"],
    ... )
    >>> hub.publish(pack, pack_content={"module": "..."})
    >>> found = hub.search("finance")
    >>> installed = hub.install("finance-basics")
    """

    def __init__(self, config: HubConfig | None = None) -> None:
        self.config = config or HubConfig()
        self._storage_path = Path(os.path.expanduser(self.config.storage_path))
        self._registry: dict[str, PackMetadata] = {}
        self._pack_contents: dict[str, dict[str, Any]] = {}
        self._installed: dict[str, str] = {}  # name -> version

    def publish(
        self,
        metadata: PackMetadata,
        pack_content: dict[str, Any] | None = None,
        version: str = "1.0.0",
        changelog: str = "",
    ) -> PackMetadata:
        """Publish a feature pack to the hub.

        Parameters
        ----------
        metadata : PackMetadata
            Pack metadata.
        pack_content : dict | None
            Pack contents (code, config, etc.).
        version : str
            Version to publish.
        changelog : str
            Changelog for this version.

        Returns:
        -------
        PackMetadata
            Updated metadata with new version.
        """
        content_str = json.dumps(pack_content or {}, sort_keys=True)
        checksum = hashlib.sha256(content_str.encode()).hexdigest()

        new_version = PackVersion(
            version=version,
            checksum=checksum,
            changelog=changelog,
        )

        if metadata.name in self._registry:
            existing = self._registry[metadata.name]
            existing_versions = {v.version for v in existing.versions}
            if version in existing_versions:
                raise ValueError(
                    f"Version {version} already exists for '{metadata.name}'."
                )
            existing.versions.append(new_version)
            existing.description = metadata.description or existing.description
            existing.tags = metadata.tags or existing.tags
            metadata = existing
        else:
            metadata.versions.append(new_version)
            self._registry[metadata.name] = metadata

        self._pack_contents[f"{metadata.name}@{version}"] = pack_content or {}

        logger.info("Published %s@%s", metadata.name, version)
        return metadata

    def search(
        self,
        query: str = "",
        domain: str | None = None,
        tags: list[str] | None = None,
    ) -> list[PackMetadata]:
        """Search for feature packs.

        Parameters
        ----------
        query : str
            Text search query (matches name and description).
        domain : str | None
            Filter by domain.
        tags : list[str] | None
            Filter by tags (any match).

        Returns:
        -------
        list[PackMetadata]
            Matching packs.
        """
        results: list[PackMetadata] = []

        for pack in self._registry.values():
            if query:
                q = query.lower()
                if q not in pack.name.lower() and q not in pack.description.lower():
                    continue

            if domain and pack.domain != domain:
                continue

            if tags and not any(t in pack.tags for t in tags):
                continue

            results.append(pack)

        return results

    def install(self, name: str, version: str | None = None) -> dict[str, Any]:
        """Install a feature pack.

        Parameters
        ----------
        name : str
            Pack name.
        version : str | None
            Version to install. None installs latest.

        Returns:
        -------
        dict
            Pack contents.

        Raises:
        ------
        ValueError
            If pack or version not found.
        """
        if name not in self._registry:
            raise ValueError(f"Pack '{name}' not found in hub.")

        pack = self._registry[name]
        if version is None:
            version = pack.latest_version

        key = f"{name}@{version}"
        if key not in self._pack_contents:
            raise ValueError(f"Version {version} of '{name}' not found.")

        self._installed[name] = version
        pack.downloads += 1

        logger.info("Installed %s@%s", name, version)
        return self._pack_contents[key]

    def uninstall(self, name: str) -> bool:
        """Uninstall a feature pack.

        Parameters
        ----------
        name : str
            Pack name to uninstall.

        Returns:
        -------
        bool
            True if pack was installed and removed.
        """
        if name in self._installed:
            del self._installed[name]
            return True
        return False

    def get_pack(self, name: str) -> PackMetadata | None:
        """Get pack metadata by name.

        Parameters
        ----------
        name : str
            Pack name.

        Returns:
        -------
        PackMetadata | None
            Pack metadata if found.
        """
        return self._registry.get(name)

    def list_installed(self) -> dict[str, str]:
        """List all installed packs.

        Returns:
        -------
        dict
            Installed pack name -> version mapping.
        """
        return dict(self._installed)

    def list_all(self) -> list[PackMetadata]:
        """List all available packs.

        Returns:
        -------
        list[PackMetadata]
            All registered packs.
        """
        return list(self._registry.values())

    def rate(self, name: str, rating: float) -> None:
        """Rate a feature pack.

        Parameters
        ----------
        name : str
            Pack name.
        rating : float
            Rating (0-5).
        """
        if name not in self._registry:
            raise ValueError(f"Pack '{name}' not found.")
        if not 0 <= rating <= 5:
            raise ValueError("Rating must be between 0 and 5.")

        pack = self._registry[name]
        # Simple rolling average (for demo purposes)
        if pack.rating == 0:
            pack.rating = rating
        else:
            pack.rating = (pack.rating + rating) / 2

    def get_stats(self) -> dict[str, Any]:
        """Get hub statistics.

        Returns:
        -------
        dict
            Hub statistics.
        """
        total_packs = len(self._registry)
        total_versions = sum(len(p.versions) for p in self._registry.values())
        total_downloads = sum(p.downloads for p in self._registry.values())
        domains = set(p.domain for p in self._registry.values())

        return {
            "total_packs": total_packs,
            "total_versions": total_versions,
            "total_downloads": total_downloads,
            "domains": sorted(domains),
            "installed": len(self._installed),
        }
