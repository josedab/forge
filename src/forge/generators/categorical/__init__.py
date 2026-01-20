"""Categorical feature generators for Forge."""

from __future__ import annotations

from forge.generators.categorical.advanced_encoders import (
    CatBoostEncoder,
    HashingEncoder,
    LeaveOneOutEncoder,
    WoEEncoder,
)
from forge.generators.categorical.combinations import CategoryCombiner
from forge.generators.categorical.encoders import (
    FrequencyEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    TargetEncoder,
)
from forge.generators.categorical.statistics import CategoryStatistics

__all__ = [
    # Basic encoders
    "OneHotEncoder",
    "TargetEncoder",
    "FrequencyEncoder",
    "OrdinalEncoder",
    # Advanced encoders
    "WoEEncoder",
    "CatBoostEncoder",
    "LeaveOneOutEncoder",
    "HashingEncoder",
    # Other generators
    "CategoryCombiner",
    "CategoryStatistics",
]
