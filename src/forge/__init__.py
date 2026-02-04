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
    "JoinPathFinder",
    "CrossTableAggregator",
    "JoinPath",
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
    # Streaming Feature Engine
    "StreamingEngine",
    "StreamingFeatureGenerator",
    "IncrementalAggregator",
    # AutoFeature Search
    "AutoFeatureSearch",
    "EvolutionarySearch",
    "RandomSearch",
    # Feature Store Hub
    "create_feature_store",
    "TectonRegistry",
    "HopsworksRegistry",
    "DatabricksRegistry",
    "SageMakerRegistry",
    # GPU Backends
    "ComputeBackend",
    "CPUBackend",
    "GPUBackend",
    "HybridDispatcher",
    "HybridConfig",
    "ExecutionStats",
    "PolarsBackend",
    "DuckDBBackend",
    "get_backend",
    # Feature Lineage Dashboard
    "FeatureLineageGraph",
    "LineageEventRecorder",
    "LineageVersionStore",
    "LineageDiff",
    "LineageSnapshot",
    # Domain Feature Packs
    "FinanceFeaturePack",
    "HealthcareFeaturePack",
    "EcommerceFeaturePack",
    "GeospatialFeaturePack",
    "IoTFeaturePack",
    "MarketingFeaturePack",
    # Feature Cache
    "FeatureCache",
    "CacheConfig",
    # Graph Features
    "GraphFeatureGenerator",
    "EntityGraph",
    "GraphBuilder",
    # Feature Catalog
    "CatalogStore",
    "FeatureCatalogStore",
    "CatalogEntry",
    # Distributed Backends
    "DaskBackend",
    "RayBackend",
    "SparkBackend",
    "get_distributed_backend",
    # Feature Observability Platform
    "FeatureObserver",
    "AlertRule",
    "Alert",
    "AlertSeverity",
    "MetricType",
    "FeatureMetrics",
    "SLADefinition",
    "SLAReport",
    # Declarative Feature DSL
    "DSLParser",
    "DSLCompiler",
    "FeatureDSLError",
    "FeatureSpec",
    "PipelineSpec",
    # Neural Feature Architecture
    "NeuralFeatureCross",
    "NeuralFeatureGenerator",
    "NeuralEmbeddingGenerator",
    # Feature Impact Simulator
    "FeatureImpactSimulator",
    "FeatureCombinationSimulator",
    "FeatureRecommender",
    "CombinationResult",
    "Recommendation",
    "ImpactReport",
    # Cross-Dataset Feature Transfer
    "FeatureTransferEngine",
    "ColumnFingerprint",
    "DatasetFingerprint",
    "TransferResult",
    # Real-Time Feature Server
    "FeatureServer",
    "FeatureRequest",
    "FeatureResponse",
    "ServingConfig",
    "CompiledPipeline",
    "ServingMetrics",
    # Interactive Feature Studio
    "FeatureStudio",
    "PipelineStep",
    "PreviewResult",
    "ComparisonResult",
    # Compliance & Fairness
    "BiasDetector",
    "BiasReport",
    "FairnessMetric",
    "FairFeatureGenerator",
    # Multi-Modal Feature Fusion
    "ImageFeatureExtractor",
    "SignalFeatureExtractor",
    "MultiModalFusion",
    "ModalityType",
    # Feature Marketplace & Hub
    "FeatureHub",
    "PackMetadata",
    "PackVersion",
    "HubConfig",
    "PackManager",
    "PackDefinition",
    "ValidationResult",
    "InstallResult",
    # Feature Store Sync
    "FeatureStoreSync",
    "SyncConfig",
    "SyncResult",
    "ConflictStrategy",
    # Feature Recommendation Engine
    "FeatureRecommender",
    "DatasetProfiler",
    "DatasetProfile",
    "Recommendation",
    # Metrics Store & Governance
    "MetricsStore",
    "MetricPoint",
    "HealthStatus",
    "GovernanceReport",
    "GovernanceManager",
    "FeatureGovernance",
    "ComplianceReport",
    "SLAConfig",
    # Online Statistics & Streaming
    "OnlineStatistics",
    "TDigest",
    "ExponentialMovingAverage",
    "FrequencyCounter",
    "AdaptiveFeatureEngine",
    "InMemoryStreamConnector",
    "StreamEvent",
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
    # v2: Benchmark Suite
    "BenchmarkHarness",
    "ForgeBaseline",
    "ManualBaseline",
    "PassthroughBaseline",
    # v2: Notebook Experience
    "ColumnProfiler",
    "PipelineBuilder",
    "DatasetExplorer",
    # v2: Freshness & Backfill
    "FreshnessPolicy",
    "BackfillEngine",
    # v2: Data Contracts
    "ContractGenerator",
    "ContractValidator",
    "DataContract",
    # v2: NL Feature Synthesis
    "NLFeatureSynthesizer",
    # v2: Spark Backend
    "SparkNativeBackend",
    # v2: CLI
    "ForgeCLI",
    # v2: Experiments / A/B Testing
    "ExperimentManager",
    "FeatureVariant",
    "EvaluationResult",
    # v2: Privacy
    "PrivacyBudget",
    "DPMean",
    "DPHistogram",
    "DPSum",
    "KAnonymizer",
    # v2: Core Engine
    "Engine",
    # v2: Enhanced modules
    "DatasetProfiler",
    "BayesianFeatureSearch",
    "MultiFidelitySearch",
    "LLMFeatureSynthesizer",
    "GuardrailExperimentManager",
    "DPFeatureTransformer",
    "PrivacyAuditLog",
    "PolarsLazyPipeline",
    "PolarsFeatureGenerator",
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

    if name == "JoinPathFinder":
        from forge.multitable.join_optimizer import JoinPathFinder

        return JoinPathFinder

    if name == "CrossTableAggregator":
        from forge.multitable.join_optimizer import CrossTableAggregator

        return CrossTableAggregator

    if name == "JoinPath":
        from forge.multitable.join_optimizer import JoinPath

        return JoinPath

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

    # Streaming Feature Engine
    if name == "StreamingEngine":
        from forge.streaming.engine import StreamingEngine

        return StreamingEngine

    if name == "StreamingFeatureGenerator":
        from forge.streaming.generators import StreamingFeatureGenerator

        return StreamingFeatureGenerator

    if name == "IncrementalAggregator":
        from forge.streaming.generators import IncrementalAggregator

        return IncrementalAggregator

    # AutoFeature Search
    if name == "AutoFeatureSearch":
        from forge.search.search import AutoFeatureSearch

        return AutoFeatureSearch

    if name == "EvolutionarySearch":
        from forge.search.search import EvolutionarySearch

        return EvolutionarySearch

    if name == "RandomSearch":
        from forge.search.search import RandomSearch

        return RandomSearch

    # Feature Store Hub
    if name == "create_feature_store":
        from forge.registry.hub import create_feature_store

        return create_feature_store

    if name == "TectonRegistry":
        from forge.registry.hub import TectonRegistry

        return TectonRegistry

    if name == "HopsworksRegistry":
        from forge.registry.hub import HopsworksRegistry

        return HopsworksRegistry

    if name == "DatabricksRegistry":
        from forge.registry.hub import DatabricksRegistry

        return DatabricksRegistry

    if name == "SageMakerRegistry":
        from forge.registry.hub import SageMakerRegistry

        return SageMakerRegistry

    # GPU Backends
    if name == "ComputeBackend":
        from forge.backends.base import ComputeBackend

        return ComputeBackend

    if name == "CPUBackend":
        from forge.backends.cpu import CPUBackend

        return CPUBackend

    if name == "GPUBackend":
        from forge.backends.gpu import GPUBackend

        return GPUBackend

    if name == "HybridDispatcher":
        from forge.backends.hybrid import HybridDispatcher

        return HybridDispatcher

    if name == "HybridConfig":
        from forge.backends.hybrid import HybridConfig

        return HybridConfig

    if name == "ExecutionStats":
        from forge.backends.hybrid import ExecutionStats

        return ExecutionStats

    if name == "PolarsBackend":
        from forge.backends.polars_backend import PolarsBackend

        return PolarsBackend

    if name == "DuckDBBackend":
        from forge.backends.duckdb_backend import DuckDBBackend

        return DuckDBBackend

    if name == "get_backend":
        from forge.backends.dispatch import get_backend

        return get_backend

    # Feature Lineage Dashboard
    if name == "FeatureLineageGraph":
        from forge.dashboard.lineage import FeatureLineageGraph

        return FeatureLineageGraph

    if name == "LineageEventRecorder":
        from forge.dashboard.versioning import LineageEventRecorder

        return LineageEventRecorder

    if name == "LineageVersionStore":
        from forge.dashboard.versioning import LineageVersionStore

        return LineageVersionStore

    if name == "LineageDiff":
        from forge.dashboard.versioning import LineageDiff

        return LineageDiff

    if name == "LineageSnapshot":
        from forge.dashboard.versioning import LineageSnapshot

        return LineageSnapshot

    # Domain Feature Packs
    if name == "FinanceFeaturePack":
        from forge.packs.finance import FinanceFeaturePack

        return FinanceFeaturePack

    if name == "HealthcareFeaturePack":
        from forge.packs.healthcare import HealthcareFeaturePack

        return HealthcareFeaturePack

    if name == "EcommerceFeaturePack":
        from forge.packs.ecommerce import EcommerceFeaturePack

        return EcommerceFeaturePack

    if name == "GeospatialFeaturePack":
        from forge.packs.geospatial import GeospatialFeaturePack

        return GeospatialFeaturePack

    if name == "IoTFeaturePack":
        from forge.packs.iot import IoTFeaturePack

        return IoTFeaturePack

    if name == "MarketingFeaturePack":
        from forge.packs.iot import MarketingFeaturePack

        return MarketingFeaturePack

    # Feature Cache
    if name == "FeatureCache":
        from forge.cache.cache import FeatureCache

        return FeatureCache

    if name == "CacheConfig":
        from forge.cache.cache import CacheConfig

        return CacheConfig

    # Graph Features
    if name == "GraphFeatureGenerator":
        from forge.graph.features import GraphFeatureGenerator

        return GraphFeatureGenerator

    if name == "EntityGraph":
        from forge.graph.builder import EntityGraph

        return EntityGraph

    if name == "GraphBuilder":
        from forge.graph.builder import GraphBuilder

        return GraphBuilder

    # Feature Catalog
    if name == "CatalogStore":
        from forge.catalog.store import CatalogStore

        return CatalogStore

    if name == "FeatureCatalogStore":
        from forge.catalog.store import FeatureCatalogStore

        return FeatureCatalogStore

    if name == "CatalogEntry":
        from forge.catalog.store import CatalogEntry

        return CatalogEntry

    # Distributed Backends
    if name == "DaskBackend":
        from forge.distributed.dask_backend import DaskBackend

        return DaskBackend

    if name == "RayBackend":
        from forge.distributed.ray_backend import RayBackend

        return RayBackend

    if name == "SparkBackend":
        from forge.distributed.spark_backend import SparkBackend

        return SparkBackend

    if name == "get_distributed_backend":
        from forge.distributed.dispatch import get_distributed_backend

        return get_distributed_backend

    # Feature Observability Platform
    if name == "FeatureObserver":
        from forge.monitoring.observability import FeatureObserver

        return FeatureObserver

    if name == "AlertRule":
        from forge.monitoring.observability import AlertRule

        return AlertRule

    if name == "Alert":
        from forge.monitoring.observability import Alert

        return Alert

    if name == "AlertSeverity":
        from forge.monitoring.observability import AlertSeverity

        return AlertSeverity

    if name == "MetricType":
        from forge.monitoring.observability import MetricType

        return MetricType

    if name == "FeatureMetrics":
        from forge.monitoring.observability import FeatureMetrics

        return FeatureMetrics

    if name == "SLADefinition":
        from forge.monitoring.observability import SLADefinition

        return SLADefinition

    if name == "SLAReport":
        from forge.monitoring.observability import SLAReport

        return SLAReport

    # Declarative Feature DSL
    if name == "DSLParser":
        from forge.dsl.parser import DSLParser

        return DSLParser

    if name == "DSLCompiler":
        from forge.dsl.compiler import DSLCompiler

        return DSLCompiler

    if name == "FeatureDSLError":
        from forge.dsl.parser import FeatureDSLError

        return FeatureDSLError

    if name == "FeatureSpec":
        from forge.dsl.parser import FeatureSpec

        return FeatureSpec

    if name == "PipelineSpec":
        from forge.dsl.parser import PipelineSpec

        return PipelineSpec

    # Neural Feature Architecture
    if name == "NeuralFeatureCross":
        from forge.neural.features import NeuralFeatureCross

        return NeuralFeatureCross

    if name == "NeuralFeatureGenerator":
        from forge.neural.features import NeuralFeatureGenerator

        return NeuralFeatureGenerator

    if name == "NeuralEmbeddingGenerator":
        from forge.neural.features import NeuralEmbeddingGenerator

        return NeuralEmbeddingGenerator

    # Feature Impact Simulator
    if name == "FeatureImpactSimulator":
        from forge.simulator.engine import FeatureImpactSimulator

        return FeatureImpactSimulator

    if name == "ImpactReport":
        from forge.simulator.engine import ImpactReport

        return ImpactReport

    if name == "FeatureCombinationSimulator":
        from forge.simulator.recommender import FeatureCombinationSimulator

        return FeatureCombinationSimulator

    if name == "FeatureRecommender":
        from forge.simulator.recommender import FeatureRecommender

        return FeatureRecommender

    if name == "CombinationResult":
        from forge.simulator.recommender import CombinationResult

        return CombinationResult

    if name == "Recommendation":
        from forge.simulator.recommender import Recommendation

        return Recommendation

    # Cross-Dataset Feature Transfer
    if name == "FeatureTransferEngine":
        from forge.transfer.engine import FeatureTransferEngine

        return FeatureTransferEngine

    if name == "ColumnFingerprint":
        from forge.transfer.engine import ColumnFingerprint

        return ColumnFingerprint

    if name == "DatasetFingerprint":
        from forge.transfer.engine import DatasetFingerprint

        return DatasetFingerprint

    if name == "TransferResult":
        from forge.transfer.engine import TransferResult

        return TransferResult

    # Real-Time Feature Server
    if name == "FeatureServer":
        from forge.serving.server import FeatureServer

        return FeatureServer

    if name == "FeatureRequest":
        from forge.serving.server import FeatureRequest

        return FeatureRequest

    if name == "FeatureResponse":
        from forge.serving.server import FeatureResponse

        return FeatureResponse

    if name == "ServingConfig":
        from forge.serving.server import ServingConfig

        return ServingConfig

    if name == "CompiledPipeline":
        from forge.serving.compiled import CompiledPipeline

        return CompiledPipeline

    if name == "ServingMetrics":
        from forge.serving.compiled import ServingMetrics

        return ServingMetrics

    # Interactive Feature Studio
    if name == "FeatureStudio":
        from forge.studio.builder import FeatureStudio

        return FeatureStudio

    if name == "PipelineStep":
        from forge.studio.builder import PipelineStep

        return PipelineStep

    if name == "PreviewResult":
        from forge.studio.builder import PreviewResult

        return PreviewResult

    if name == "ComparisonResult":
        from forge.studio.builder import ComparisonResult

        return ComparisonResult

    # Compliance & Fairness
    if name == "BiasDetector":
        from forge.fairness.engine import BiasDetector

        return BiasDetector

    if name == "BiasReport":
        from forge.fairness.engine import BiasReport

        return BiasReport

    if name == "FairnessMetric":
        from forge.fairness.engine import FairnessMetric

        return FairnessMetric

    if name == "FairFeatureGenerator":
        from forge.fairness.engine import FairFeatureGenerator

        return FairFeatureGenerator

    # Multi-Modal Feature Fusion
    if name == "ImageFeatureExtractor":
        from forge.multimodal.features import ImageFeatureExtractor

        return ImageFeatureExtractor

    if name == "SignalFeatureExtractor":
        from forge.multimodal.features import SignalFeatureExtractor

        return SignalFeatureExtractor

    if name == "MultiModalFusion":
        from forge.multimodal.features import MultiModalFusion

        return MultiModalFusion

    if name == "ModalityType":
        from forge.multimodal.features import ModalityType

        return ModalityType

    # Feature Marketplace & Hub
    if name == "FeatureHub":
        from forge.marketplace.hub import FeatureHub

        return FeatureHub

    # Feature Store Sync
    if name == "FeatureStoreSync":
        from forge.registry.sync import FeatureStoreSync

        return FeatureStoreSync

    if name == "SyncConfig":
        from forge.registry.sync import SyncConfig

        return SyncConfig

    if name == "SyncResult":
        from forge.registry.sync import SyncResult

        return SyncResult

    if name == "ConflictStrategy":
        from forge.registry.sync import ConflictStrategy

        return ConflictStrategy

    # Feature Recommendation Engine
    if name == "FeatureRecommender":
        from forge.recommend import FeatureRecommender

        return FeatureRecommender

    if name == "DatasetProfiler":
        from forge.recommend import DatasetProfiler

        return DatasetProfiler

    if name == "DatasetProfile":
        from forge.recommend import DatasetProfile

        return DatasetProfile

    if name == "Recommendation":
        from forge.recommend import Recommendation

        return Recommendation

    if name == "PackMetadata":
        from forge.marketplace.hub import PackMetadata

        return PackMetadata

    if name == "PackVersion":
        from forge.marketplace.hub import PackVersion

        return PackVersion

    if name == "HubConfig":
        from forge.marketplace.hub import HubConfig

        return HubConfig

    # Pack Manager
    if name == "PackManager":
        from forge.marketplace.pack_manager import PackManager

        return PackManager

    if name == "PackDefinition":
        from forge.marketplace.pack_manager import PackDefinition

        return PackDefinition

    if name == "ValidationResult":
        from forge.marketplace.pack_manager import ValidationResult

        return ValidationResult

    if name == "InstallResult":
        from forge.marketplace.pack_manager import InstallResult

        return InstallResult

    # Metrics Store
    if name == "MetricsStore":
        from forge.monitoring.metrics_store import MetricsStore

        return MetricsStore

    if name == "MetricPoint":
        from forge.monitoring.metrics_store import MetricPoint

        return MetricPoint

    if name == "HealthStatus":
        from forge.monitoring.metrics_store import HealthStatus

        return HealthStatus

    if name == "GovernanceReport":
        from forge.monitoring.metrics_store import GovernanceReport

        return GovernanceReport

    # Governance Manager
    if name == "GovernanceManager":
        from forge.documentation.governance import GovernanceManager

        return GovernanceManager

    if name == "FeatureGovernance":
        from forge.documentation.governance import FeatureGovernance

        return FeatureGovernance

    if name == "ComplianceReport":
        from forge.documentation.governance import ComplianceReport

        return ComplianceReport

    if name == "SLAConfig":
        from forge.documentation.governance import SLAConfig

        return SLAConfig

    # Online Statistics
    if name == "OnlineStatistics":
        from forge.streaming.online import OnlineStatistics

        return OnlineStatistics

    if name == "TDigest":
        from forge.streaming.online import TDigest

        return TDigest

    if name == "ExponentialMovingAverage":
        from forge.streaming.online import ExponentialMovingAverage

        return ExponentialMovingAverage

    if name == "FrequencyCounter":
        from forge.streaming.online import FrequencyCounter

        return FrequencyCounter

    if name == "AdaptiveFeatureEngine":
        from forge.streaming.online import AdaptiveFeatureEngine

        return AdaptiveFeatureEngine

    if name == "InMemoryStreamConnector":
        from forge.streaming.online import InMemoryStreamConnector

        return InMemoryStreamConnector

    if name == "StreamEvent":
        from forge.streaming.online import StreamEvent

        return StreamEvent

    # v2: Benchmark Suite
    if name == "BenchmarkHarness":
        from forge.benchmarks import BenchmarkHarness

        return BenchmarkHarness

    if name == "ForgeBaseline":
        from forge.benchmarks import ForgeBaseline

        return ForgeBaseline

    if name == "ManualBaseline":
        from forge.benchmarks import ManualBaseline

        return ManualBaseline

    if name == "PassthroughBaseline":
        from forge.benchmarks import PassthroughBaseline

        return PassthroughBaseline

    # v2: Notebook Experience
    if name == "ColumnProfiler":
        from forge.notebook import ColumnProfiler

        return ColumnProfiler

    if name == "PipelineBuilder":
        from forge.notebook import PipelineBuilder

        return PipelineBuilder

    if name == "DatasetExplorer":
        from forge.notebook import DatasetExplorer

        return DatasetExplorer

    # v2: Freshness & Backfill
    if name == "FreshnessPolicy":
        from forge.freshness import FreshnessPolicy

        return FreshnessPolicy

    if name == "BackfillEngine":
        from forge.freshness import BackfillEngine

        return BackfillEngine

    # v2: Data Contracts
    if name == "ContractGenerator":
        from forge.contracts import ContractGenerator

        return ContractGenerator

    if name == "ContractValidator":
        from forge.contracts import ContractValidator

        return ContractValidator

    if name == "DataContract":
        from forge.contracts import DataContract

        return DataContract

    # v2: NL Feature Synthesis
    if name == "NLFeatureSynthesizer":
        from forge.nlgen import NLFeatureSynthesizer

        return NLFeatureSynthesizer

    # v2: Spark Backend
    if name == "SparkNativeBackend":
        from forge.backends.spark_native import SparkNativeBackend

        return SparkNativeBackend

    # v2: CLI
    if name == "ForgeCLI":
        from forge.cli import ForgeCLI

        return ForgeCLI

    # v2: Experiments / A/B Testing
    if name == "ExperimentManager":
        from forge.experiments import ExperimentManager

        return ExperimentManager

    if name == "FeatureVariant":
        from forge.experiments import FeatureVariant

        return FeatureVariant

    if name == "EvaluationResult":
        from forge.experiments import EvaluationResult

        return EvaluationResult

    # v2: Privacy
    if name == "PrivacyBudget":
        from forge.privacy import PrivacyBudget

        return PrivacyBudget

    if name == "DPMean":
        from forge.privacy import DPMean

        return DPMean

    if name == "DPHistogram":
        from forge.privacy import DPHistogram

        return DPHistogram

    if name == "DPSum":
        from forge.privacy import DPSum

        return DPSum

    if name == "KAnonymizer":
        from forge.privacy import KAnonymizer

        return KAnonymizer

    # v2: Core Engine
    if name == "Engine":
        from forge.core_engine import Engine

        return Engine

    # v2: Enhanced modules
    if name == "DatasetProfiler":
        from forge.dashboard.profiler import DatasetProfiler

        return DatasetProfiler

    if name == "BayesianFeatureSearch":
        from forge.search.bayesian import BayesianFeatureSearch

        return BayesianFeatureSearch

    if name == "MultiFidelitySearch":
        from forge.search.bayesian import MultiFidelitySearch

        return MultiFidelitySearch

    if name == "LLMFeatureSynthesizer":
        from forge.nlgen.llm_synthesizer import LLMFeatureSynthesizer

        return LLMFeatureSynthesizer

    if name == "GuardrailExperimentManager":
        from forge.experiments.guardrails import GuardrailExperimentManager

        return GuardrailExperimentManager

    if name == "DPFeatureTransformer":
        from forge.privacy.advanced import DPFeatureTransformer

        return DPFeatureTransformer

    if name == "PrivacyAuditLog":
        from forge.privacy.advanced import PrivacyAuditLog

        return PrivacyAuditLog

    if name == "PolarsLazyPipeline":
        from forge.backends.polars_lazy import PolarsLazyPipeline

        return PolarsLazyPipeline

    if name == "PolarsFeatureGenerator":
        from forge.backends.polars_lazy import PolarsFeatureGenerator

        return PolarsFeatureGenerator

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
