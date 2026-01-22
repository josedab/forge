"""Automated feature documentation module."""

from __future__ import annotations

from forge.documentation.generator import (
    FeatureDocumentationGenerator,
    FeatureDoc,
    DatasetDoc,
    generate_feature_docs,
)
from forge.documentation.catalog import (
    FeatureCatalog,
    CatalogEntry,
    create_catalog,
)
from forge.documentation.export import (
    MarkdownExporter,
    HTMLExporter,
    JSONExporter,
    export_documentation,
)

__all__ = [
    # Generator
    "FeatureDocumentationGenerator",
    "FeatureDoc",
    "DatasetDoc",
    "generate_feature_docs",
    # Catalog
    "FeatureCatalog",
    "CatalogEntry",
    "create_catalog",
    # Export
    "MarkdownExporter",
    "HTMLExporter",
    "JSONExporter",
    "export_documentation",
]
