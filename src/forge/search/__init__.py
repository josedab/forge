"""AutoFeature search for automatic feature transformation discovery.

This module provides evolutionary, random, Bayesian, and multi-fidelity
search algorithms to discover optimal feature transformations automatically.
Includes Pareto-optimal selection and early stopping.
"""

from __future__ import annotations

from forge.search.bayesian import (
    BayesianFeatureSearch,
    MultiFidelitySearch,
)
from forge.search.grammar import TransformGrammar, TransformNode, TransformOp
from forge.search.pareto import (
    EarlyStoppingMonitor,
    ParetoPoint,
    pareto_optimal_sets,
    select_knee_point,
)
from forge.search.search import (
    AutoFeatureSearch,
    EvolutionarySearch,
    RandomSearch,
    SearchResult,
)

__all__ = [
    "AutoFeatureSearch",
    "BayesianFeatureSearch",
    "EarlyStoppingMonitor",
    "EvolutionarySearch",
    "MultiFidelitySearch",
    "ParetoPoint",
    "RandomSearch",
    "SearchResult",
    "TransformGrammar",
    "TransformNode",
    "TransformOp",
    "pareto_optimal_sets",
    "select_knee_point",
]
