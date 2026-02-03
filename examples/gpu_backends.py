#!/usr/bin/env python
"""GPU compute backend example for Forge.

This example demonstrates how to use CPUBackend, get_backend,
and the GPU-to-CPU fallback pattern for transparent hardware
acceleration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forge.backends import CPUBackend, ComputeDevice, GPUBackend, get_backend


def create_sample_data() -> pd.DataFrame:
    """Create sample numeric data for backend operations."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "price": np.random.lognormal(4, 0.5, n),
        "quantity": np.random.randint(1, 100, n).astype(float),
        "discount": np.random.uniform(0, 0.3, n),
        "category": np.random.choice(["A", "B", "C", "D"], n),
    })


def example_cpu_backend():
    """Example: Use CPUBackend for numeric and encoding operations."""
    print("=" * 60)
    print("Example: CPUBackend Operations")
    print("=" * 60)

    df = create_sample_data()
    backend = CPUBackend()

    print(f"\nBackend device: {backend.device.value}")
    print(f"Available: {backend.is_available()}")

    # Arithmetic operations
    revenue = backend.multiply(df["price"], df["quantity"])
    net_price = backend.divide(df["price"], df["discount"] + 1)
    log_price = backend.log(df["price"])
    sqrt_qty = backend.sqrt(df["quantity"])
    price_sq = backend.power(df["price"], 2.0)

    print(f"\nArithmetic operations on {len(df)} rows:")
    print(f"  Revenue (price * quantity):  mean={revenue.mean():.2f}")
    print(f"  Net price (price / (1+disc)): mean={net_price.mean():.2f}")
    print(f"  Log(price):                  mean={log_price.mean():.2f}")
    print(f"  Sqrt(quantity):              mean={sqrt_qty.mean():.2f}")
    print(f"  Price^2:                     mean={price_sq.mean():.2f}")

    # Aggregation operations
    rolling_mean = backend.rolling_agg(df["price"], window=10, agg_func="mean")
    group_mean = backend.group_agg(df, "category", "price", "mean")
    print(f"\n  Rolling mean (w=10):         mean={rolling_mean.mean():.2f}")
    print(f"  Group mean by category:      {group_mean.unique()[:4]}")

    # Encoding operations
    ohe = backend.one_hot_encode(df["category"])
    ordinal = backend.ordinal_encode(df["category"])
    print(f"\n  One-hot encoded columns: {list(ohe.columns)}")
    print(f"  Ordinal encoded unique:  {sorted(ordinal.dropna().unique())}")

    # Polynomial and interaction features
    numeric_df = df[["price", "quantity"]].head(5)
    poly = backend.polynomial_features(numeric_df, degree=2)
    print(f"\n  Polynomial features (degree=2): {list(poly.columns)}")

    # Quantile binning
    bins = backend.quantile_bin(df["price"], n_bins=5)
    print(f"  Quantile bins (5): {sorted(bins.dropna().unique())}")


def example_auto_backend_dispatch():
    """Example: Automatic backend selection with GPU fallback."""
    print("\n" + "=" * 60)
    print("Example: Auto Backend Dispatch (GPU Fallback)")
    print("=" * 60)

    # Auto-detect: picks GPU if available, falls back to CPU
    backend = get_backend("auto")
    print(f"\nAuto-detected backend: {backend.device.value}")
    print(f"Backend available: {backend.is_available()}")

    # Check GPU availability explicitly
    gpu = GPUBackend()
    print(f"\nGPU backend available: {gpu.is_available()}")
    if not gpu.is_available():
        print("  (RAPIDS cuDF not installed; using CPU backend)")

    # All operations work identically regardless of backend
    df = create_sample_data()
    result = backend.multiply(df["price"], df["quantity"])
    log_result = backend.log(df["price"])
    interactions = backend.interaction_features(df, ["price", "quantity", "discount"])

    print(f"\nOperations with {backend.device.value} backend:")
    print(f"  Multiply result mean: {result.mean():.2f}")
    print(f"  Log result mean:      {log_result.mean():.2f}")
    print(f"  Interaction columns:  {list(interactions.columns)}")


def example_backend_comparison():
    """Example: Compare CPU backend operations side by side."""
    print("\n" + "=" * 60)
    print("Example: Backend Feature Generation Pattern")
    print("=" * 60)

    df = create_sample_data()
    backend = get_backend("cpu")

    print("\nGenerating features through the backend API:\n")

    features = {}
    features["revenue"] = backend.multiply(df["price"], df["quantity"])
    features["log_price"] = backend.log(df["price"])
    features["sqrt_quantity"] = backend.sqrt(df["quantity"])
    features["price_squared"] = backend.power(df["price"], 2.0)
    features["rolling_price_mean"] = backend.rolling_agg(df["price"], 20, "mean")
    features["group_price_mean"] = backend.group_agg(df, "category", "price", "mean")

    result = pd.DataFrame(features, index=df.index)
    print(f"  Generated {result.shape[1]} features for {result.shape[0]} rows")
    print(f"\n  Feature summary:")
    for col in result.columns:
        print(f"    {col:25s} mean={result[col].mean():.2f}")


if __name__ == "__main__":
    example_cpu_backend()
    example_auto_backend_dispatch()
    example_backend_comparison()

    print("\n" + "=" * 60)
    print("All GPU backend examples completed!")
    print("=" * 60)
