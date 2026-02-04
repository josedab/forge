"""Feature generation module for Forge."""

from __future__ import annotations

from forge.generators.base import BaseFeatureGenerator
from forge.generators.interactions import (
    GroupedInteractionGenerator,
    InteractionCandidate,
    InteractionDiscoverer,
    PolynomialInteractionGenerator,
    discover_interactions,
)
from forge.generators.registry import GeneratorRegistry, get_registry
from forge.generators.timeseries import (
    DatetimeConfig,
    FourierFeatureGenerator,
    LagConfig,
    RollingConfig,
    SeasonalDecomposer,
    TimeSeriesFeatureGenerator,
    generate_timeseries_features,
)

__all__ = [
    "BaseFeatureGenerator",
    "GeneratorRegistry",
    "get_registry",
    # Time-series
    "TimeSeriesFeatureGenerator",
    "SeasonalDecomposer",
    "FourierFeatureGenerator",
    "LagConfig",
    "RollingConfig",
    "DatetimeConfig",
    "generate_timeseries_features",
    # Interactions
    "InteractionDiscoverer",
    "PolynomialInteractionGenerator",
    "GroupedInteractionGenerator",
    "InteractionCandidate",
    "discover_interactions",
]
