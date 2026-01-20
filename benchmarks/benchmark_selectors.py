"""Benchmarks for feature selectors."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable

# Try to import forge components
try:
    from forge.selectors import (
        StatisticalSelector,
        ImportanceSelector,
        CorrelationSelector,
        VarianceSelector,
    )

    FORGE_AVAILABLE = True
except ImportError:
    FORGE_AVAILABLE = False


def generate_feature_data(
    n_rows: int, n_features: int, n_informative: int = 10
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic feature data with known informative features."""
    np.random.seed(42)

    # Generate random features
    X = pd.DataFrame(
        np.random.randn(n_rows, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    # Make some features informative (correlated with target)
    y = np.random.randint(0, 2, size=n_rows)
    for i in range(min(n_informative, n_features)):
        X[f"feature_{i}"] = X[f"feature_{i}"] + y * np.random.uniform(1, 3)

    return X, pd.Series(y, name="target")


def benchmark_function(
    func: Callable, *args, n_iterations: int = 5, **kwargs
) -> dict:
    """Benchmark a function over multiple iterations."""
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "n_iterations": n_iterations,
    }


def benchmark_statistical_selector(
    n_rows: int, n_features: int, k: int = 10
) -> dict:
    """Benchmark statistical feature selection."""
    if not FORGE_AVAILABLE:
        return {"error": "Forge not installed"}

    X, y = generate_feature_data(n_rows, n_features)
    selector = StatisticalSelector(k=k, method="mutual_info_classif")

    def run():
        selector.fit(X, y)
        return selector.transform(X)

    result = benchmark_function(run)
    result["n_rows"] = n_rows
    result["n_features"] = n_features
    result["k"] = k
    result["selector"] = "StatisticalSelector"
    return result


def benchmark_importance_selector(
    n_rows: int, n_features: int, k: int = 10
) -> dict:
    """Benchmark importance-based feature selection."""
    if not FORGE_AVAILABLE:
        return {"error": "Forge not installed"}

    X, y = generate_feature_data(n_rows, n_features)
    selector = ImportanceSelector(k=k)

    def run():
        selector.fit(X, y)
        return selector.transform(X)

    result = benchmark_function(run)
    result["n_rows"] = n_rows
    result["n_features"] = n_features
    result["k"] = k
    result["selector"] = "ImportanceSelector"
    return result


def benchmark_correlation_selector(
    n_rows: int, n_features: int, threshold: float = 0.9
) -> dict:
    """Benchmark correlation-based feature selection."""
    if not FORGE_AVAILABLE:
        return {"error": "Forge not installed"}

    X, y = generate_feature_data(n_rows, n_features)
    selector = CorrelationSelector(threshold=threshold)

    def run():
        selector.fit(X, y)
        return selector.transform(X)

    result = benchmark_function(run)
    result["n_rows"] = n_rows
    result["n_features"] = n_features
    result["threshold"] = threshold
    result["selector"] = "CorrelationSelector"
    return result


def benchmark_variance_selector(
    n_rows: int, n_features: int, threshold: float = 0.01
) -> dict:
    """Benchmark variance-based feature selection."""
    if not FORGE_AVAILABLE:
        return {"error": "Forge not installed"}

    X, y = generate_feature_data(n_rows, n_features)
    selector = VarianceSelector(threshold=threshold)

    def run():
        selector.fit(X, y)
        return selector.transform(X)

    result = benchmark_function(run)
    result["n_rows"] = n_rows
    result["n_features"] = n_features
    result["threshold"] = threshold
    result["selector"] = "VarianceSelector"
    return result


def run_selector_benchmarks() -> list[dict]:
    """Run all selector benchmarks."""
    results = []
    sizes = [(1000, 50), (10000, 100), (100000, 200)]

    for n_rows, n_features in sizes:
        print(f"Benchmarking selectors with {n_rows} rows, {n_features} features...")

        results.append(benchmark_statistical_selector(n_rows, n_features))
        results.append(benchmark_importance_selector(n_rows, n_features))
        results.append(benchmark_correlation_selector(n_rows, n_features))
        results.append(benchmark_variance_selector(n_rows, n_features))

    return results


if __name__ == "__main__":
    results = run_selector_benchmarks()
    for r in results:
        if "error" not in r:
            print(
                f"{r['selector']}: {r['mean_time']:.4f}s "
                f"(+/- {r['std_time']:.4f}s) "
                f"[{r['n_rows']} rows, {r['n_features']} features]"
            )
