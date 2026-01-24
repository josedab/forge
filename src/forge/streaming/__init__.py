"""Streaming feature engine for real-time feature computation.

This module provides incremental/online feature generators that support
partial_fit semantics and windowed aggregations for streaming data.
"""

from __future__ import annotations

from forge.streaming.engine import StreamingEngine
from forge.streaming.generators import (
    IncrementalAggregator,
    IncrementalInteractionGenerator,
    StreamingFeatureGenerator,
    StreamingWindowAggregator,
)
from forge.streaming.online import (
    AdaptiveFeatureEngine,
    ExponentialMovingAverage,
    FrequencyCounter,
    InMemoryStreamConnector,
    OnlineStatistics,
    StreamConnector,
    StreamEvent,
    TDigest,
    WelfordState,
)
from forge.streaming.window import SlidingWindow, TumblingWindow, WindowConfig

__all__ = [
    "AdaptiveFeatureEngine",
    "ExponentialMovingAverage",
    "FrequencyCounter",
    "InMemoryStreamConnector",
    "IncrementalAggregator",
    "IncrementalInteractionGenerator",
    "OnlineStatistics",
    "SlidingWindow",
    "StreamConnector",
    "StreamEvent",
    "StreamingEngine",
    "StreamingFeatureGenerator",
    "StreamingWindowAggregator",
    "TDigest",
    "TumblingWindow",
    "WelfordState",
    "WindowConfig",
]
