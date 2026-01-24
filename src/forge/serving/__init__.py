"""Real-time feature serving for online inference.

Provides a feature serving layer with pre-computed feature lookup,
on-demand computation, caching with TTL support, circuit breaking,
request batching, and a REST API.
"""

from __future__ import annotations

from forge.serving.api import (
    APIConfig,
    APIMetrics,
    create_app,
)
from forge.serving.compiled import (
    CompiledGraph,
    CompiledPipeline,
    ServingMetrics,
    TransformStep,
)
from forge.serving.realtime import (
    CircuitBreaker,
    CircuitState,
    ComputationEngine,
    ComputationResult,
    TTLCache,
)
from forge.serving.server import (
    FeatureRequest,
    FeatureResponse,
    FeatureServer,
    ServingConfig,
)

__all__ = [
    "APIConfig",
    "APIMetrics",
    "CircuitBreaker",
    "CircuitState",
    "CompiledGraph",
    "CompiledPipeline",
    "ComputationEngine",
    "ComputationResult",
    "FeatureRequest",
    "FeatureResponse",
    "FeatureServer",
    "ServingConfig",
    "ServingMetrics",
    "TTLCache",
    "TransformStep",
    "create_app",
]
