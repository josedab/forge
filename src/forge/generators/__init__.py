"""Feature generation module for Forge."""

from __future__ import annotations

from forge.generators.base import BaseFeatureGenerator
from forge.generators.registry import GeneratorRegistry, get_registry
from forge.generators.timeseries import (
    TimeSeriesFeatureGenerator,
    SeasonalDecomposer,
    FourierFeatureGenerator,
    LagConfig,
    RollingConfig,
    DatetimeConfig,
    generate_timeseries_features,
)
from forge.generators.interactions import (
    InteractionDiscoverer,
    PolynomialInteractionGenerator,
    GroupedInteractionGenerator,
    InteractionCandidate,
    discover_interactions,
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
