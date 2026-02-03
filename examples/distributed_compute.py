#!/usr/bin/env python
"""Distributed compute backends example for Forge.

This example demonstrates DaskBackend, RayBackend, and the
get_distributed_backend dispatcher. Since distributed libraries
may not be installed, examples check availability gracefully.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forge.distributed import DaskBackend, RayBackend, get_distributed_backend


def create_sample_data(n: int = 10000) -> pd.DataFrame:
    """Create a larger dataset suitable for distributed processing."""
    np.random.seed(42)
    return pd.DataFrame({
        "user_id": np.random.choice([f"U{i:04d}" for i in range(100)], n),
        "amount": np.random.lognormal(4, 1, n),
        "quantity": np.random.randint(1, 20, n),
        "category": np.random.choice(["electronics", "clothing", "food", "books"], n),
        "score": np.random.normal(0, 1, n),
    })


def example_backend_availability():
    """Example: Check which distributed backends are available."""
    print("=" * 60)
    print("Example: Distributed Backend Availability")
    print("=" * 60)

    backends = [
        ("Dask", DaskBackend()),
        ("Ray", RayBackend()),
    ]

    print("\nChecking installed distributed backends:\n")
    for name, backend in backends:
        available = backend.is_available()
        status = "AVAILABLE" if available else "not installed"
        print(f"  {name:12s}: {status}")

    # Try auto-detection
    print("\nAttempting auto-detection...")
    try:
        backend = get_distributed_backend("auto")
        print(f"  Auto-detected: {backend.name}")
    except Exception as e:
        print(f"  No distributed backend available: {type(e).__name__}")
        print("  (Install dask, ray, or pyspark for distributed execution)")


def example_dask_backend():
    """Example: Use Dask for distributed feature computation."""
    print("\n" + "=" * 60)
    print("Example: DaskBackend")
    print("=" * 60)

    backend = DaskBackend(n_workers=2)
    if not backend.is_available():
        print("\n  Dask is not installed. Skipping.")
        print("  Install with: pip install 'dask[distributed]'")
        return

    df = create_sample_data(5000)

    try:
        backend.connect()
        print(f"\n  Connected to Dask cluster")

        # map_partitions: apply a function to each chunk
        def compute_features(chunk: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({
                "revenue": chunk["amount"] * chunk["quantity"],
                "log_amount": np.log1p(chunk["amount"]),
                "score_squared": chunk["score"] ** 2,
            })

        result = backend.map_partitions(df, compute_features, n_partitions=4)
        print(f"  map_partitions result: {result.shape}")
        print(f"  Columns: {list(result.columns)}")

        # groupby_apply: distributed group operations
        def group_stats(group: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({
                "category": [group["category"].iloc[0]],
                "mean_amount": [group["amount"].mean()],
                "total_quantity": [group["quantity"].sum()],
            })

        grouped = backend.groupby_apply(df, "category", group_stats)
        print(f"\n  groupby_apply result: {grouped.shape}")
        print(grouped.to_string(index=False))

    finally:
        backend.disconnect()
        print("\n  Disconnected from Dask cluster")


def example_ray_backend():
    """Example: Use Ray for distributed feature computation."""
    print("\n" + "=" * 60)
    print("Example: RayBackend")
    print("=" * 60)

    backend = RayBackend(num_cpus=2)
    if not backend.is_available():
        print("\n  Ray is not installed. Skipping.")
        print("  Install with: pip install ray")
        return

    df = create_sample_data(5000)

    try:
        backend.connect()
        print(f"\n  Connected to Ray cluster")

        # map_partitions with Ray
        def compute_features(chunk: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({
                "revenue": chunk["amount"] * chunk["quantity"],
                "log_amount": np.log1p(chunk["amount"]),
            })

        result = backend.map_partitions(df, compute_features, n_partitions=4)
        print(f"  map_partitions result: {result.shape}")
        print(f"  Revenue mean: {result['revenue'].mean():.2f}")

    finally:
        backend.disconnect()
        print("\n  Disconnected from Ray cluster")


if __name__ == "__main__":
    example_backend_availability()
    example_dask_backend()
    example_ray_backend()

    print("\n" + "=" * 60)
    print("All distributed compute examples completed!")
    print("=" * 60)
