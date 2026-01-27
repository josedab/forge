"""Pack manager: create, validate, install with dependency resolution.

Extends the FeatureHub with a CLI-like interface for managing
feature packs, quality gates, and dependency resolution.

Example:
    >>> from forge.marketplace.pack_manager import PackManager
    >>> pm = PackManager()
    >>> pm.create_pack("my-pack", author="me", domain="finance",
    ...     generators=["InteractionGenerator"], description="Finance features")
    >>> validation = pm.validate_pack("my-pack")
    >>> pm.install_pack("my-pack")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from forge.marketplace.hub import FeatureHub, PackMetadata

logger = logging.getLogger(__name__)


@dataclass
class PackDefinition:
    """Full pack definition with code and configuration."""

    name: str
    description: str = ""
    author: str = ""
    domain: str = "general"
    tags: list[str] = field(default_factory=list)
    generators: list[str] = field(default_factory=list)
    selectors: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "domain": self.domain,
            "tags": self.tags,
            "generators": self.generators,
            "selectors": self.selectors,
            "dependencies": self.dependencies,
            "config": self.config,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PackDefinition:
        """Deserialize from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ValidationResult:
    """Result of pack validation."""

    is_valid: bool
    pack_name: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    quality_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "is_valid": self.is_valid,
            "pack_name": self.pack_name,
            "errors": self.errors,
            "warnings": self.warnings,
            "quality_score": self.quality_score,
        }


@dataclass
class InstallResult:
    """Result of pack installation."""

    success: bool
    pack_name: str
    version: str = ""
    installed_deps: list[str] = field(default_factory=list)
    message: str = ""


class PackManager:
    """Manages feature pack lifecycle: create, validate, install.

    Wraps FeatureHub with higher-level operations including
    quality gates, dependency resolution, and pack creation.

    Parameters:
        hub: Optional existing FeatureHub. Creates one if None.
        storage_path: Path for storing pack definitions.
    """

    def __init__(
        self,
        hub: FeatureHub | None = None,
        storage_path: str | Path | None = None,
    ) -> None:
        self.hub = hub or FeatureHub()
        self.storage_path = Path(storage_path) if storage_path else None
        self._pack_definitions: dict[str, PackDefinition] = {}

    def create_pack(
        self,
        name: str,
        author: str = "",
        domain: str = "general",
        tags: list[str] | None = None,
        generators: list[str] | None = None,
        selectors: list[str] | None = None,
        dependencies: list[str] | None = None,
        description: str = "",
        config: dict[str, Any] | None = None,
        version: str = "1.0.0",
    ) -> PackDefinition:
        """Create a new feature pack definition.

        Args:
            name: Pack name (unique identifier).
            author: Author name.
            domain: Domain category.
            tags: Searchable tags.
            generators: Generator class names.
            selectors: Selector class names.
            dependencies: Required pack names.
            description: Human-readable description.
            config: Additional configuration.
            version: Version string.

        Returns:
            Created PackDefinition.
        """
        pack = PackDefinition(
            name=name,
            description=description,
            author=author,
            domain=domain,
            tags=tags or [],
            generators=generators or [],
            selectors=selectors or [],
            dependencies=dependencies or [],
            config=config or {},
            version=version,
        )
        self._pack_definitions[name] = pack
        return pack

    def validate_pack(self, name: str) -> ValidationResult:
        """Validate a pack meets quality gates.

        Checks:
        - Pack exists and has required fields
        - Description is non-empty
        - At least one generator or selector
        - Dependencies are resolvable
        - No circular dependencies

        Args:
            name: Pack name to validate.

        Returns:
            ValidationResult with errors and warnings.
        """
        errors: list[str] = []
        warnings: list[str] = []
        score = 100.0

        pack = self._pack_definitions.get(name)
        if pack is None:
            return ValidationResult(
                is_valid=False,
                pack_name=name,
                errors=[f"Pack '{name}' not found"],
                quality_score=0.0,
            )

        # Required fields
        if not pack.name:
            errors.append("Pack name is required")
            score -= 30

        if not pack.description:
            warnings.append("Pack should have a description")
            score -= 10

        if not pack.author:
            warnings.append("Pack should have an author")
            score -= 5

        # Must have at least one generator or selector
        if not pack.generators and not pack.selectors:
            errors.append("Pack must include at least one generator or selector")
            score -= 30

        # Check dependencies are resolvable
        for dep in pack.dependencies:
            if dep not in self._pack_definitions and self.hub.get_pack(dep) is None:
                errors.append(f"Dependency '{dep}' not found")
                score -= 15

        # Check for circular dependencies
        if self._has_circular_deps(name):
            errors.append("Circular dependency detected")
            score -= 30

        # Tags check
        if not pack.tags:
            warnings.append("Pack should have at least one tag for discoverability")
            score -= 5

        # Domain check
        if pack.domain == "general":
            warnings.append("Consider specifying a domain for better categorization")
            score -= 5

        score = max(0.0, score) / 100.0

        return ValidationResult(
            is_valid=len(errors) == 0,
            pack_name=name,
            errors=errors,
            warnings=warnings,
            quality_score=score,
        )

    def _has_circular_deps(
        self,
        name: str,
        visited: set[str] | None = None,
    ) -> bool:
        """Check for circular dependencies."""
        if visited is None:
            visited = set()
        if name in visited:
            return True
        visited.add(name)

        pack = self._pack_definitions.get(name)
        if pack is None:
            return False
        for dep in pack.dependencies:
            if self._has_circular_deps(dep, visited.copy()):
                return True
        return False

    def publish_pack(
        self,
        name: str,
        changelog: str = "",
    ) -> PackMetadata:
        """Publish a pack to the hub after validation.

        Args:
            name: Pack name.
            changelog: Changelog for this version.

        Returns:
            Published PackMetadata.

        Raises:
            ValueError: If pack not found or validation fails.
        """
        pack = self._pack_definitions.get(name)
        if pack is None:
            raise ValueError(f"Pack '{name}' not found")

        validation = self.validate_pack(name)
        if not validation.is_valid:
            raise ValueError(
                f"Pack '{name}' failed validation: {'; '.join(validation.errors)}"
            )

        metadata = PackMetadata(
            name=pack.name,
            description=pack.description,
            author=pack.author,
            tags=pack.tags,
            domain=pack.domain,
            dependencies=pack.dependencies,
        )

        return self.hub.publish(
            metadata,
            pack_content=pack.to_dict(),
            version=pack.version,
            changelog=changelog,
        )

    def install_pack(
        self,
        name: str,
        version: str | None = None,
        install_deps: bool = True,
    ) -> InstallResult:
        """Install a pack with dependency resolution.

        Args:
            name: Pack name to install.
            version: Specific version, or None for latest.
            install_deps: Whether to install dependencies.

        Returns:
            InstallResult with installation details.
        """
        installed_deps: list[str] = []

        pack = self.hub.get_pack(name)
        if pack is None:
            return InstallResult(
                success=False,
                pack_name=name,
                message=f"Pack '{name}' not found in hub",
            )

        # Resolve and install dependencies first
        if install_deps:
            order = self._resolve_install_order(name)
            for dep in order:
                if dep == name:
                    continue
                try:
                    self.hub.install(dep)
                    installed_deps.append(dep)
                except ValueError as e:
                    return InstallResult(
                        success=False,
                        pack_name=name,
                        message=f"Failed to install dependency '{dep}': {e}",
                    )

        try:
            self.hub.install(name, version=version)
        except ValueError as e:
            return InstallResult(
                success=False,
                pack_name=name,
                message=str(e),
            )

        return InstallResult(
            success=True,
            pack_name=name,
            version=version or (pack.latest_version if pack else ""),
            installed_deps=installed_deps,
            message="Installation successful",
        )

    def _resolve_install_order(self, name: str) -> list[str]:
        """Resolve dependency installation order (topological sort)."""
        order: list[str] = []
        visited: set[str] = set()

        def visit(n: str) -> None:
            if n in visited:
                return
            visited.add(n)
            pack = self.hub.get_pack(n)
            if pack:
                for dep in pack.dependencies:
                    visit(dep)
            order.append(n)

        visit(name)
        return order

    def list_packs(
        self,
        domain: str | None = None,
        installed_only: bool = False,
    ) -> list[str]:
        """List available pack names.

        Args:
            domain: Filter by domain.
            installed_only: Only show installed packs.

        Returns:
            List of pack names.
        """
        if installed_only:
            return list(self.hub.list_installed().keys())

        packs = self.hub.search(domain=domain) if domain else self.hub.list_all()
        return [p.name for p in packs]

    def get_pack_info(self, name: str) -> dict[str, Any] | None:
        """Get detailed info about a pack.

        Args:
            name: Pack name.

        Returns:
            Pack info dictionary, or None.
        """
        meta = self.hub.get_pack(name)
        if meta is None:
            return None

        installed = self.hub.list_installed()
        return {
            "name": meta.name,
            "description": meta.description,
            "author": meta.author,
            "domain": meta.domain,
            "tags": meta.tags,
            "latest_version": meta.latest_version,
            "downloads": meta.downloads,
            "rating": meta.rating,
            "installed": name in installed,
            "installed_version": installed.get(name),
        }

    def save(self) -> None:
        """Persist pack definitions to disk."""
        if self.storage_path is None:
            return
        self.storage_path.mkdir(parents=True, exist_ok=True)
        data_path = self.storage_path / "packs.json"
        data = {k: v.to_dict() for k, v in self._pack_definitions.items()}
        data_path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        """Load pack definitions from disk."""
        if self.storage_path is None:
            return
        data_path = self.storage_path / "packs.json"
        if not data_path.exists():
            return
        try:
            raw = json.loads(data_path.read_text())
            for name, data in raw.items():
                self._pack_definitions[name] = PackDefinition.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load pack definitions: %s", e)
