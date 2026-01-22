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
    # LLM-Powered Feature Discovery
    "LLMFeatureGenerator",
    "FeatureSuggester",
    "MetadataExtractor",
    "FeatureExplainer",
    # Multi-Table Feature Synthesis
    "MultiTableTransformer",
    "RelationshipGraph",
    "DeepFeatureSynthesis",
    "detect_relationships",
    "multi_table_features",
    # Feature Registry
    "FeatureRegistry",
    "LocalRegistry",
    "FeastRegistry",
    "FeatureDefinition",
    "FeatureSet",
    "create_registry",
    "create_local_registry",
    # AutoML Feature Selection
    "BayesianFeatureSelector",
    "SequentialFeatureSelector",
    "GeneticFeatureSelector",
    "auto_select_features",
    # Feature Importance Ensemble
    "EnsembleImportanceSelector",
    "StabilitySelector",
    "ImportanceMethod",
    "ensemble_importance",
    # Time-Series Feature Engineering
    "TimeSeriesFeatureGenerator",
    "SeasonalDecomposer",
    "FourierFeatureGenerator",
    "generate_timeseries_features",
    # Feature Interaction Discovery
    "InteractionDiscoverer",
    "PolynomialInteractionGenerator",
    "GroupedInteractionGenerator",
    "discover_interactions",
    # Automated Feature Documentation
    "FeatureDocumentationGenerator",
    "FeatureDoc",
    "DatasetDoc",
    "FeatureCatalog",
    "MarkdownExporter",
    "HTMLExporter",
    "JSONExporter",
    "export_documentation",
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

    # LLM-Powered Feature Discovery
    if name == "LLMFeatureGenerator":
        from forge.llm.generator import LLMFeatureGenerator

        return LLMFeatureGenerator

    if name == "FeatureSuggester":
        from forge.llm.suggestions import FeatureSuggester

        return FeatureSuggester

    if name == "MetadataExtractor":
        from forge.llm.metadata import MetadataExtractor

        return MetadataExtractor

    if name == "FeatureExplainer":
        from forge.llm.explainer import FeatureExplainer

        return FeatureExplainer

    # Multi-Table Feature Synthesis
    if name == "MultiTableTransformer":
        from forge.multitable.transformer import MultiTableTransformer

        return MultiTableTransformer

    if name == "RelationshipGraph":
        from forge.multitable.relationships import RelationshipGraph

        return RelationshipGraph

    if name == "DeepFeatureSynthesis":
        from forge.multitable.synthesis import DeepFeatureSynthesis

        return DeepFeatureSynthesis

    if name == "detect_relationships":
        from forge.multitable.relationships import detect_relationships

        return detect_relationships

    if name == "multi_table_features":
        from forge.multitable.transformer import multi_table_features

        return multi_table_features

    # Feature Registry
    if name == "FeatureRegistry":
        from forge.registry.base import FeatureRegistry

        return FeatureRegistry

    if name == "LocalRegistry":
        from forge.registry.local_registry import LocalRegistry

        return LocalRegistry

    if name == "FeastRegistry":
        from forge.registry.feast_registry import FeastRegistry

        return FeastRegistry

    if name == "FeatureDefinition":
        from forge.registry.base import FeatureDefinition

        return FeatureDefinition

    if name == "FeatureSet":
        from forge.registry.base import FeatureSet

        return FeatureSet

    if name == "create_registry":
        from forge.registry.local_registry import create_registry

        return create_registry

    if name == "create_local_registry":
        from forge.registry.local_registry import create_local_registry

        return create_local_registry

    # AutoML Feature Selection
    if name == "BayesianFeatureSelector":
        from forge.selectors.automl import BayesianFeatureSelector

        return BayesianFeatureSelector

    if name == "SequentialFeatureSelector":
        from forge.selectors.automl import SequentialFeatureSelector

        return SequentialFeatureSelector

    if name == "GeneticFeatureSelector":
        from forge.selectors.automl import GeneticFeatureSelector

        return GeneticFeatureSelector

    if name == "auto_select_features":
        from forge.selectors.automl import auto_select_features

        return auto_select_features

    # Feature Importance Ensemble
    if name == "EnsembleImportanceSelector":
        from forge.selectors.ensemble import EnsembleImportanceSelector

        return EnsembleImportanceSelector

    if name == "StabilitySelector":
        from forge.selectors.ensemble import StabilitySelector

        return StabilitySelector

    if name == "ImportanceMethod":
        from forge.selectors.ensemble import ImportanceMethod

        return ImportanceMethod

    if name == "ensemble_importance":
        from forge.selectors.ensemble import ensemble_importance

        return ensemble_importance

    # Time-Series Feature Engineering
    if name == "TimeSeriesFeatureGenerator":
        from forge.generators.timeseries import TimeSeriesFeatureGenerator

        return TimeSeriesFeatureGenerator

    if name == "SeasonalDecomposer":
        from forge.generators.timeseries import SeasonalDecomposer

        return SeasonalDecomposer

    if name == "FourierFeatureGenerator":
        from forge.generators.timeseries import FourierFeatureGenerator

        return FourierFeatureGenerator

    if name == "generate_timeseries_features":
        from forge.generators.timeseries import generate_timeseries_features

        return generate_timeseries_features

    # Feature Interaction Discovery
    if name == "InteractionDiscoverer":
        from forge.generators.interactions import InteractionDiscoverer

        return InteractionDiscoverer

    if name == "PolynomialInteractionGenerator":
        from forge.generators.interactions import PolynomialInteractionGenerator

        return PolynomialInteractionGenerator

    if name == "GroupedInteractionGenerator":
        from forge.generators.interactions import GroupedInteractionGenerator

        return GroupedInteractionGenerator

    if name == "discover_interactions":
        from forge.generators.interactions import discover_interactions

        return discover_interactions

    # Automated Feature Documentation
    if name == "FeatureDocumentationGenerator":
        from forge.documentation.generator import FeatureDocumentationGenerator

        return FeatureDocumentationGenerator

    if name == "FeatureDoc":
        from forge.documentation.generator import FeatureDoc

        return FeatureDoc

    if name == "DatasetDoc":
        from forge.documentation.generator import DatasetDoc

        return DatasetDoc

    if name == "FeatureCatalog":
        from forge.documentation.catalog import FeatureCatalog

        return FeatureCatalog

    if name == "MarkdownExporter":
        from forge.documentation.export import MarkdownExporter

        return MarkdownExporter

    if name == "HTMLExporter":
        from forge.documentation.export import HTMLExporter

        return HTMLExporter

    if name == "JSONExporter":
        from forge.documentation.export import JSONExporter

        return JSONExporter

    if name == "export_documentation":
        from forge.documentation.export import export_documentation

        return export_documentation

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
