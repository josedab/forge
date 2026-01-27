"""Compatibility checker and manifest utilities for feature packs.

Provides version compatibility checking and YAML/JSON manifest export
for the feature marketplace.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CompatibilityReport:
    """Result of a compatibility check between a pack and the environment."""

    pack_name: str
    is_compatible: bool
    forge_version_ok: bool
    python_version_ok: bool
    missing_packages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "pack_name": self.pack_name,
            "is_compatible": self.is_compatible,
            "forge_version_ok": self.forge_version_ok,
            "python_version_ok": self.python_version_ok,
            "missing_packages": self.missing_packages,
            "warnings": self.warnings,
        }


def check_version_constraint(current: str, constraint: str) -> bool:
    """Check if a version satisfies a constraint like '>=0.1.0'.

    Args:
        current: Current version string (e.g., '0.1.0').
        constraint: Constraint string (e.g., '>=0.1.0', '==1.0.0').

    Returns:
        True if the current version satisfies the constraint.
    """
    match = re.match(r"^(>=|<=|==|!=|>|<)?\s*(\d+\.\d+\.\d+)$", constraint.strip())
    if not match:
        return True  # unparseable constraint treated as satisfied

    op, target = match.group(1) or ">=", match.group(2)

    def _parse(v: str) -> tuple[int, ...]:
        return tuple(int(x) for x in v.split("."))

    cur = _parse(current)
    tgt = _parse(target)

    ops = {
        ">=": cur >= tgt,
        "<=": cur <= tgt,
        "==": cur == tgt,
        "!=": cur != tgt,
        ">": cur > tgt,
        "<": cur < tgt,
    }
    return ops.get(op, True)


def check_compatibility(
    pack_name: str,
    required_forge_version: str = ">=0.1.0",
    required_python_version: str = ">=3.9.0",
    dependencies: list[str] | None = None,
) -> CompatibilityReport:
    """Check if the current environment is compatible with a pack.

    Args:
        pack_name: Pack identifier.
        required_forge_version: Forge version constraint.
        required_python_version: Python version constraint.
        dependencies: List of required Python package names.

    Returns:
        CompatibilityReport with detailed results.
    """
    import sys

    from forge._version import __version__ as forge_version

    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    forge_ok = check_version_constraint(forge_version, required_forge_version)
    python_ok = check_version_constraint(py_version, required_python_version)

    missing: list[str] = []
    warnings: list[str] = []

    for dep in dependencies or []:
        try:
            __import__(dep.split("[")[0].replace("-", "_"))
        except ImportError:
            missing.append(dep)

    if not forge_ok:
        warnings.append(
            f"Forge version {forge_version} does not satisfy {required_forge_version}"
        )
    if not python_ok:
        warnings.append(
            f"Python version {py_version} does not satisfy {required_python_version}"
        )

    return CompatibilityReport(
        pack_name=pack_name,
        is_compatible=forge_ok and python_ok and len(missing) == 0,
        forge_version_ok=forge_ok,
        python_version_ok=python_ok,
        missing_packages=missing,
        warnings=warnings,
    )


def export_manifest(
    pack_name: str,
    description: str = "",
    author: str = "",
    version: str = "1.0.0",
    domain: str = "general",
    tags: list[str] | None = None,
    generators: list[str] | None = None,
    selectors: list[str] | None = None,
    dependencies: list[str] | None = None,
    forge_version: str = ">=0.1.0",
    format: str = "json",
) -> str:
    """Export a pack manifest in JSON or YAML format.

    Args:
        pack_name: Pack identifier.
        description: Pack description.
        author: Author name.
        version: Version string.
        domain: Domain category.
        tags: Searchable tags.
        generators: Generator class names.
        selectors: Selector class names.
        dependencies: Required packages.
        forge_version: Forge version constraint.
        format: 'json' or 'yaml'.

    Returns:
        Serialized manifest string.
    """
    manifest = {
        "name": pack_name,
        "version": version,
        "description": description,
        "author": author,
        "domain": domain,
        "tags": tags or [],
        "generators": generators or [],
        "selectors": selectors or [],
        "dependencies": dependencies or [],
        "forge_version": forge_version,
    }

    if format == "yaml":
        lines = ["# Forge Feature Pack Manifest"]
        for key, val in manifest.items():
            if isinstance(val, list):
                lines.append(f"{key}:")
                for item in val:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"{key}: {val}")
        return "\n".join(lines) + "\n"

    return json.dumps(manifest, indent=2)
