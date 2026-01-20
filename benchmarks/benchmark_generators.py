"""Benchmarks for feature generators."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable

# Try to import forge components
try:
    from forge.generators.numeric import InteractionGenerator, PolynomialGenerator
    from forge.generators.categorical import TargetEncoder, OneHotEncoder
    from forge.generators.temporal import DateTimeComponents

    FORGE_AVAILABLE = True
except ImportError:
    FORGE_AVAILABLE = False


def generate_numeric_data(n_rows: int, n_cols: int) -> pd.DataFrame:
    """Generate synthetic numeric data for benchmarking."""
    np.random.seed(42)
    data = {
        f"num_{i}": np.random.randn(n_rows) * 100 + np.random.randint(-50, 50)
        for i in range(n_cols)
    }
    return pd.DataFrame(data)


def generate_categorical_data(
    n_rows: int, n_cols: int, cardinality: int = 10
) -> pd.DataFrame:
    """Generate synthetic categorical data for benchmarking."""
    np.random.seed(42)
    categories = [f"cat_{i}" for i in range(cardinality)]
    data = {
        f"cat_{i}": np.random.choice(categories, size=n_rows) for i in range(n_cols)
    }
    return pd.DataFrame(data)


def generate_datetime_data(n_rows: int) -> pd.DataFrame:
    """Generate synthetic datetime data for benchmarking."""
    np.random.seed(42)
    start_date = pd.Timestamp("2020-01-01")
    dates = pd.date_range(start=start_date, periods=n_rows, freq="H")
    return pd.DataFrame({"timestamp": dates})


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


def benchmark_interaction_generator(n_rows: int, n_cols: int) -> dict:
    """Benchmark interaction feature generation."""
    if not FORGE_AVAILABLE:
        return {"error": "Forge not installed"}

    X = generate_numeric_data(n_rows, n_cols)
    y = np.random.randint(0, 2, size=n_rows)

    generator = InteractionGenerator()

    def run():
        generator.fit(X, y)
        return generator.transform(X)

    result = benchmark_function(run)
    result["n_rows"] = n_rows
    result["n_cols"] = n_cols
    result["generator"] = "InteractionGenerator"
    return result


def benchmark_polynomial_generator(n_rows: int, n_cols: int, degree: int = 2) -> dict:
    """Benchmark polynomial feature generation."""
    if not FORGE_AVAILABLE:
        return {"error": "Forge not installed"}

    X = generate_numeric_data(n_rows, n_cols)
    y = np.random.randint(0, 2, size=n_rows)

    generator = PolynomialGenerator(degree=degree)

    def run():
        generator.fit(X, y)
        return generator.transform(X)

    result = benchmark_function(run)
    result["n_rows"] = n_rows
    result["n_cols"] = n_cols
    result["degree"] = degree
    result["generator"] = "PolynomialGenerator"
    return result


def benchmark_target_encoder(
    n_rows: int, n_cols: int, cardinality: int = 10
) -> dict:
    """Benchmark target encoding."""
    if not FORGE_AVAILABLE:
        return {"error": "Forge not installed"}

    X = generate_categorical_data(n_rows, n_cols, cardinality)
    y = np.random.randint(0, 2, size=n_rows)

    encoder = TargetEncoder()

    def run():
        encoder.fit(X, y)
        return encoder.transform(X)

    result = benchmark_function(run)
    result["n_rows"] = n_rows
    result["n_cols"] = n_cols
    result["cardinality"] = cardinality
    result["generator"] = "TargetEncoder"
    return result


def benchmark_datetime_generator(n_rows: int) -> dict:
    """Benchmark datetime feature extraction."""
    if not FORGE_AVAILABLE:
        return {"error": "Forge not installed"}

    X = generate_datetime_data(n_rows)
    y = np.random.randint(0, 2, size=n_rows)

    generator = DateTimeComponents(columns=["timestamp"])

    def run():
        generator.fit(X, y)
        return generator.transform(X)

    result = benchmark_function(run)
    result["n_rows"] = n_rows
    result["generator"] = "DateTimeComponents"
    return result


def run_generator_benchmarks() -> list[dict]:
    """Run all generator benchmarks."""
    results = []
    sizes = [(1000, 5), (10000, 10), (100000, 20)]

    for n_rows, n_cols in sizes:
        print(f"Benchmarking generators with {n_rows} rows, {n_cols} columns...")

        results.append(benchmark_interaction_generator(n_rows, n_cols))
        results.append(benchmark_polynomial_generator(n_rows, n_cols))
        results.append(benchmark_target_encoder(n_rows, n_cols))
        results.append(benchmark_datetime_generator(n_rows))

    return results


if __name__ == "__main__":
    results = run_generator_benchmarks()
    for r in results:
        if "error" not in r:
            print(
                f"{r['generator']}: {r['mean_time']:.4f}s "
                f"(+/- {r['std_time']:.4f}s) "
                f"[{r['n_rows']} rows]"
            )
