#!/usr/bin/env python
"""Streaming feature engineering example for Forge.

This example demonstrates how to use the streaming module for
real-time, incremental feature computation using StreamingEngine,
IncrementalAggregator, and SlidingWindow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forge.streaming import (
    IncrementalAggregator,
    SlidingWindow,
    StreamingEngine,
    StreamingWindowAggregator,
)


def create_sensor_stream(n_batches: int = 5, batch_size: int = 50) -> list[pd.DataFrame]:
    """Create simulated sensor data arriving in batches."""
    np.random.seed(42)
    batches = []
    for i in range(n_batches):
        data = {
            "temperature": np.random.normal(22 + i * 0.5, 2, batch_size),
            "humidity": np.random.normal(55, 10, batch_size),
            "pressure": np.random.normal(1013, 5, batch_size),
        }
        batches.append(pd.DataFrame(data))
    return batches


def example_streaming_engine():
    """Example: Process a data stream with StreamingEngine."""
    print("=" * 60)
    print("Example: StreamingEngine for Real-Time Features")
    print("=" * 60)

    batches = create_sensor_stream()

    engine = StreamingEngine(
        generators=[
            IncrementalAggregator(stats=["mean", "std", "min", "max"]),
            StreamingWindowAggregator(window_size=100, stats=["mean", "std"]),
        ],
        passthrough=True,
    )

    print(f"\nProcessing {len(batches)} batches through the engine...\n")
    for i, batch in enumerate(batches):
        features = engine.process(batch)
        print(f"  Batch {i + 1}: {batch.shape[0]} rows -> {features.shape[1]} feature columns")

    stats = engine.stats
    print(f"\nEngine statistics:")
    print(f"  Batches processed: {stats['n_batches_processed']}")
    print(f"  Events processed:  {stats['n_events_processed']}")
    print(f"  Total features:    {stats['n_features']}")

    print(f"\nGenerated feature names (first 8):")
    for name in engine.get_feature_names_out()[:8]:
        print(f"  - {name}")


def example_incremental_aggregator():
    """Example: Running statistics with IncrementalAggregator."""
    print("\n" + "=" * 60)
    print("Example: IncrementalAggregator Running Statistics")
    print("=" * 60)

    batches = create_sensor_stream(n_batches=3, batch_size=100)

    agg = IncrementalAggregator(
        columns=["temperature", "humidity"],
        stats=["mean", "std", "min", "max"],
    )

    print("\nUpdating running stats with each batch:\n")
    for i, batch in enumerate(batches):
        agg.partial_fit(batch)
        features = agg.partial_transform(batch)

        temp_mean = features["temperature_running_mean"].iloc[0]
        temp_std = features["temperature_running_std"].iloc[0]
        print(f"  After batch {i + 1}:")
        print(f"    Temperature: mean={temp_mean:.2f}, std={temp_std:.2f}")
        print(f"    Events seen: {agg._n_events_seen}")

    print(f"\nFinal feature columns: {agg.get_feature_names_out()}")


def example_sliding_window():
    """Example: Low-level SlidingWindow for efficient streaming stats."""
    print("\n" + "=" * 60)
    print("Example: SlidingWindow with Welford's Algorithm")
    print("=" * 60)

    np.random.seed(42)
    window = SlidingWindow(size=50, min_periods=5)

    print("\nStreaming 200 values through a window of size 50:\n")
    values = np.random.normal(100, 15, 200)
    checkpoints = [25, 50, 100, 200]

    for i, val in enumerate(values, 1):
        window.add(val)
        if i in checkpoints:
            print(f"  After {i} values:")
            print(f"    Window count: {window.count()}")
            print(f"    Mean: {window.mean():.2f}")
            print(f"    Std:  {window.std():.2f}")
            print(f"    Min:  {window.min():.2f}")
            print(f"    Max:  {window.max():.2f}")

    print(f"\n  Final window has {window.count()} values (max size = 50)")


if __name__ == "__main__":
    example_streaming_engine()
    example_incremental_aggregator()
    example_sliding_window()

    print("\n" + "=" * 60)
    print("All streaming examples completed!")
    print("=" * 60)
