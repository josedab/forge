"""Multi-table feature synthesis module.

This module provides feature engineering across related tables,
enabling automatic join detection and deep feature synthesis.
"""

from __future__ import annotations

from forge.multitable.join_optimizer import (
    CrossTableAggregator,
    JoinPath,
    JoinPathFinder,
)
from forge.multitable.relationships import (
    Relationship,
    RelationshipGraph,
    RelationshipType,
    detect_relationships,
)
from forge.multitable.synthesis import (
    AggregationPrimitive,
    DeepFeatureSynthesis,
    TransformPrimitive,
)
from forge.multitable.transformer import MultiTableTransformer

__all__ = [
    "AggregationPrimitive",
    "CrossTableAggregator",
    "DeepFeatureSynthesis",
    "JoinPath",
    "JoinPathFinder",
    "MultiTableTransformer",
    "Relationship",
    "RelationshipGraph",
    "RelationshipType",
    "TransformPrimitive",
    "detect_relationships",
]
