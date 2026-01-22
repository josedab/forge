"""Multi-table feature synthesis module.

This module provides feature engineering across related tables,
enabling automatic join detection and deep feature synthesis.
"""

from __future__ import annotations

from forge.multitable.relationships import (
    Relationship,
    RelationshipType,
    RelationshipGraph,
    detect_relationships,
)
from forge.multitable.synthesis import (
    DeepFeatureSynthesis,
    AggregationPrimitive,
    TransformPrimitive,
)
from forge.multitable.transformer import MultiTableTransformer

__all__ = [
    "Relationship",
    "RelationshipType",
    "RelationshipGraph",
    "detect_relationships",
    "DeepFeatureSynthesis",
    "AggregationPrimitive",
    "TransformPrimitive",
    "MultiTableTransformer",
]
