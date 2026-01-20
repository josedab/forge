"""Forge - Automated Feature Engineering Platform.

Forge provides automatic feature generation, intelligent selection,
and seamless scikit-learn compatibility to reduce the time data
scientists spend on feature engineering.

Example:
    >>> from forge import AutoFeatureTransformer
    >>> transformer = AutoFeatureTransformer(max_features=100)
    >>> X_engineered = transformer.fit_transform(X, y)
"""

from __future__ import annotations

from forge._version import __version__, __version_info__
from forge.exceptions import (
    ColumnNotFoundError,
    ConfigurationError,
    FeatureGenerationError,
    FeatureSelectionError,
    ForgeError,
    InvalidColumnTypeError,
    MissingDependencyError,
    NotFittedError,
    ValidationError,
)
from forge.types import (
    AnalysisReport,
    ColumnInfo,
    ColumnType,
    EncodingStrategy,
    FeatureInfo,
    ImputationStrategy,
    QualityIssue,
    SelectionMethod,
)

__all__ = [
    # Version
    "__version__",
    "__version_info__",
    # Main classes (imported lazily to avoid circular imports)
    "AutoFeatureTransformer",
    "DataAnalyzer",
    "ForgePipeline",
    # Outlier handlers
    "Winsorizer",
    "IQRCapper",
    "ArbitraryCapper",
    "Trimmer",
    # Feature descriptions
    "FeatureDescription",
    "FeatureDescriber",
    "FeatureLineage",
    # Monitoring
    "PSICalculator",
    "DriftDetector",
    "DriftReport",
    "calculate_psi",
    # Types
    "AnalysisReport",
    "ColumnInfo",
    "ColumnType",
    "EncodingStrategy",
    "FeatureInfo",
    "ImputationStrategy",
    "QualityIssue",
    "SelectionMethod",
    # Exceptions
    "ForgeError",
    "NotFittedError",
    "ValidationError",
    "ColumnNotFoundError",
    "InvalidColumnTypeError",
    "ConfigurationError",
    "FeatureGenerationError",
    "FeatureSelectionError",
    "MissingDependencyError",
]


def __getattr__(name: str) -> object:
    """Lazy import of main classes to avoid circular imports."""
    if name == "AutoFeatureTransformer":
        from forge.transformers.auto_transformer import AutoFeatureTransformer

        return AutoFeatureTransformer

    if name == "DataAnalyzer":
        from forge.analyzer.base import DataAnalyzer

        return DataAnalyzer

    if name == "ForgePipeline":
        from forge.transformers.feature_pipeline import ForgePipeline

        return ForgePipeline

    # Outlier handlers
    if name == "Winsorizer":
        from forge.outliers import Winsorizer

        return Winsorizer

    if name == "IQRCapper":
        from forge.outliers import IQRCapper

        return IQRCapper

    if name == "ArbitraryCapper":
        from forge.outliers import ArbitraryCapper

        return ArbitraryCapper

    if name == "Trimmer":
        from forge.outliers import Trimmer

        return Trimmer

    # Feature descriptions
    if name == "FeatureDescription":
        from forge.transformers.descriptions import FeatureDescription

        return FeatureDescription

    if name == "FeatureDescriber":
        from forge.transformers.descriptions import FeatureDescriber

        return FeatureDescriber

    if name == "FeatureLineage":
        from forge.transformers.descriptions import FeatureLineage

        return FeatureLineage

    # Monitoring
    if name == "PSICalculator":
        from forge.monitoring import PSICalculator

        return PSICalculator

    if name == "DriftDetector":
        from forge.monitoring import DriftDetector

        return DriftDetector

    if name == "DriftReport":
        from forge.monitoring import DriftReport

        return DriftReport

    if name == "calculate_psi":
        from forge.monitoring import calculate_psi

        return calculate_psi

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
