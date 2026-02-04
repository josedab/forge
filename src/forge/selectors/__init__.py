"""Feature selection module for Forge."""

from __future__ import annotations

from forge.selectors.automl import (
    BayesianFeatureSelector,
    GeneticFeatureSelector,
    SequentialFeatureSelector,
    auto_select_features,
)
from forge.selectors.base import BaseFeatureSelector
from forge.selectors.correlation import CorrelationSelector
from forge.selectors.ensemble import (
    EnsembleImportanceSelector,
    ImportanceMethod,
    StabilitySelector,
    ensemble_importance,
)
from forge.selectors.importance import ImportanceSelector
from forge.selectors.shap_selector import ShapSelector
from forge.selectors.statistical import StatisticalSelector
from forge.selectors.variance import VarianceSelector

__all__ = [
    "BaseFeatureSelector",
    "StatisticalSelector",
    "ImportanceSelector",
    "CorrelationSelector",
    "VarianceSelector",
    "ShapSelector",
    # AutoML selectors
    "BayesianFeatureSelector",
    "SequentialFeatureSelector",
    "GeneticFeatureSelector",
    "auto_select_features",
    # Ensemble selectors
    "EnsembleImportanceSelector",
    "StabilitySelector",
    "ImportanceMethod",
    "ensemble_importance",
]
