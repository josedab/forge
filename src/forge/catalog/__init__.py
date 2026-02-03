"""Automated feature documentation and catalog API.

Provides a REST API and catalog store for feature discovery,
search, versioning, and documentation management.
"""

from __future__ import annotations

from forge.catalog.store import CatalogEntry, CatalogStore, FeatureCatalogStore

__all__ = [
    "CatalogEntry",
    "CatalogStore",
    "FeatureCatalogStore",
]
