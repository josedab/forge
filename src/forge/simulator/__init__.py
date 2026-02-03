"""Feature impact simulator for what-if analysis.

Provides tools to estimate the impact of adding/removing features
on model performance without full retraining.
"""

from __future__ import annotations

from forge.simulator.engine import FeatureImpactSimulator, ImpactReport
from forge.simulator.recommender import (
    CombinationResult,
    FeatureCombinationSimulator,
    FeatureRecommender,
    Recommendation,
)

__all__ = [
    "CombinationResult",
    "FeatureCombinationSimulator",
    "FeatureImpactSimulator",
    "FeatureRecommender",
    "ImpactReport",
    "Recommendation",
]
