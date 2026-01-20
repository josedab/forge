"""Benchmarks for end-to-end pipeline performance."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable

# Try to import forge components
try:
    from forge import AutoFeatureTransformer

    FORGE_AVAILABLE = True
except ImportError:
    FORGE_AVAILABLE = False

# Try to import sklearn for comparison
try:
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def generate_mixed_data(
    n_rows: int, n_numeric: int = 10, n_categorical: int = 5
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate mixed-type data for benchmarking."""
    np.random.seed(42)

    data = {}

    # Numeric columns
    for i in range(n_numeric):
        data[f"num_{i}"] = np.random.randn(n_rows) * 100

    # Categorical columns
    for i in range(n_categorical):
        cardinality = np.random.randint(5, 20)
        categories = [f"cat_{i}_{j}" for j in range(cardinality)]
        data[f"cat_{i}"] = np.random.choice(categories, size=n_rows)

    # Add some missing values
    df = pd.DataFrame(data)
    mask = np.random.random(df.shape) < 0.05
    df = df.mask(mask)

    # Target variable
    y = pd.Series(np.random.randint(0, 2, size=n_rows), name="target")

    return df, y


def benchmark_function(
    func: Callable, *args, n_iterations: int = 3, **kwargs
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


def benchmark_forge_pipeline(
    n_rows: int, n_numeric: int, n_categorical: int, max_features: int = 50
) -> dict:
    """Benchmark Forge AutoFeatureTransformer."""
    if not FORGE_AVAILABLE:
        return {"error": "Forge not installed"}

    X, y = generate_mixed_data(n_rows, n_numeric, n_categorical)
    transformer = AutoFeatureTransformer(max_features=max_features)

    def run():
        return transformer.fit_transform(X, y)

    result = benchmark_function(run)
    result["n_rows"] = n_rows
    result["n_numeric"] = n_numeric
    result["n_categorical"] = n_categorical
    result["max_features"] = max_features
    result["pipeline"] = "Forge"
    return result


def benchmark_sklearn_pipeline(
    n_rows: int, n_numeric: int, n_categorical: int
) -> dict:
    """Benchmark equivalent sklearn pipeline for comparison."""
    if not SKLEARN_AVAILABLE:
        return {"error": "sklearn not installed"}

    X, y = generate_mixed_data(n_rows, n_numeric, n_categorical)

    numeric_cols = [c for c in X.columns if c.startswith("num_")]
    categorical_cols = [c for c in X.columns if c.startswith("cat_")]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    def run():
        return preprocessor.fit_transform(X, y)

    result = benchmark_function(run)
    result["n_rows"] = n_rows
    result["n_numeric"] = n_numeric
    result["n_categorical"] = n_categorical
    result["pipeline"] = "sklearn"
    return result


def run_pipeline_benchmarks() -> list[dict]:
    """Run all pipeline benchmarks."""
    results = []
    sizes = [
        (1000, 10, 5),
        (10000, 20, 10),
        (100000, 30, 15),
    ]

    for n_rows, n_numeric, n_categorical in sizes:
        print(
            f"Benchmarking pipelines with {n_rows} rows, "
            f"{n_numeric} numeric, {n_categorical} categorical..."
        )

        results.append(benchmark_forge_pipeline(n_rows, n_numeric, n_categorical))
        results.append(benchmark_sklearn_pipeline(n_rows, n_numeric, n_categorical))

    return results


def compare_results(results: list[dict]) -> None:
    """Compare Forge vs sklearn performance."""
    forge_results = [r for r in results if r.get("pipeline") == "Forge"]
    sklearn_results = [r for r in results if r.get("pipeline") == "sklearn"]

    print("\n" + "=" * 60)
    print("Performance Comparison: Forge vs sklearn")
    print("=" * 60)

    for forge, sklearn in zip(forge_results, sklearn_results):
        if "error" in forge or "error" in sklearn:
            continue

        speedup = sklearn["mean_time"] / forge["mean_time"]
        print(f"\nDataset: {forge['n_rows']} rows")
        print(f"  Forge:   {forge['mean_time']:.4f}s")
        print(f"  sklearn: {sklearn['mean_time']:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    results = run_pipeline_benchmarks()
    compare_results(results)
