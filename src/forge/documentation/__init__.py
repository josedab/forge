"""Automated feature documentation module."""

from __future__ import annotations

from forge.documentation.catalog import (
    CatalogEntry,
    FeatureCatalog,
    create_catalog,
)
from forge.documentation.export import (
    HTMLExporter,
    JSONExporter,
    MarkdownExporter,
    export_documentation,
)
from forge.documentation.generator import (
    DatasetDoc,
    FeatureDoc,
    FeatureDocumentationGenerator,
    generate_feature_docs,
)
from forge.documentation.governance import (
    ComplianceReport,
    FeatureGovernance,
    GovernanceManager,
    SLAConfig,
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
    # Governance
    "ComplianceReport",
    "FeatureGovernance",
    "GovernanceManager",
    "SLAConfig",
]
