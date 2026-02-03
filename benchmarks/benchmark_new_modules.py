"""Benchmarks for new forge modules: streaming, search, backends, packs, cache, graph, catalog."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable

# Try to import forge components
try:
    from forge import (
        StreamingEngine,
        CPUBackend,
        FinanceFeaturePack,
        FeatureCache,
        CacheConfig,
        AutoFeatureSearch,
        RandomSearch,
    )

    FORGE_AVAILABLE = True
except ImportError:
    FORGE_AVAILABLE = False

try:
    from forge.streaming import IncrementalAggregator

    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False

try:
    from forge.search.search import EvolutionarySearch

    SEARCH_AVAILABLE = True
except ImportError:
    SEARCH_AVAILABLE = False

try:
    from forge import EcommerceFeaturePack

    ECOMMERCE_AVAILABLE = True
except ImportError:
    ECOMMERCE_AVAILABLE = False

try:
    from forge.graph import EntityGraph, GraphBuilder, GraphFeatureGenerator

    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False

try:
    from forge.catalog import CatalogStore, CatalogEntry

    CATALOG_AVAILABLE = True
except ImportError:
    CATALOG_AVAILABLE = False


# ── Data Generators ──────────────────────────────────────────────


def generate_numeric_data(n_rows: int, n_cols: int) -> pd.DataFrame:
    """Generate synthetic numeric data for benchmarking."""
    np.random.seed(42)
    data = {
        f"num_{i}": np.random.randn(n_rows) * 100 + np.random.randint(-50, 50)
        for i in range(n_cols)
    }
    return pd.DataFrame(data)


def generate_finance_data(n_rows: int) -> pd.DataFrame:
    """Generate synthetic financial price/volume data."""
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)
    volumes = np.random.randint(1000, 100000, size=n_rows).astype(float)
    return pd.DataFrame({"close": prices, "volume": volumes})


def generate_ecommerce_data(n_rows: int) -> pd.DataFrame:
    """Generate synthetic e-commerce transaction data."""
    np.random.seed(42)
    n_customers = max(10, n_rows // 10)
    n_products = max(5, n_rows // 20)
    customer_ids = np.random.randint(1, n_customers + 1, size=n_rows)
    product_ids = np.random.randint(1, n_products + 1, size=n_rows)
    amounts = np.random.uniform(5.0, 500.0, size=n_rows)
    timestamps = pd.date_range("2020-01-01", periods=n_rows, freq="h")
    return pd.DataFrame({
        "customer_id": customer_ids,
        "product_id": product_ids,
        "amount": amounts,
        "timestamp": timestamps,
    })


def generate_graph_edge_data(n_nodes: int) -> pd.DataFrame:
    """Generate synthetic edge data for graph benchmarks."""
    np.random.seed(42)
    avg_edges_per_node = 5
    n_edges = n_nodes * avg_edges_per_node
    sources = np.random.randint(0, n_nodes, size=n_edges)
    targets = np.random.randint(0, n_nodes, size=n_edges)
    # Remove self-loops
    mask = sources != targets
    sources = sources[mask]
    targets = targets[mask]
    return pd.DataFrame({
        "source": [f"node_{s}" for s in sources],
        "target": [f"node_{t}" for t in targets],
    })


# ── Benchmark Helper ─────────────────────────────────────────────


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


# ── 1. Streaming Throughput ──────────────────────────────────────


def benchmark_streaming_throughput(n_rows: int) -> dict:
    """Benchmark StreamingEngine.process() throughput."""
    if not FORGE_AVAILABLE or not STREAMING_AVAILABLE:
        return {"error": "Forge streaming not installed"}

    X = generate_numeric_data(n_rows, n_cols=5)
    engine = StreamingEngine(generators=[
        IncrementalAggregator(stats=["mean", "std"]),
    ])

    def run():
        engine.reset()
        return engine.process(X)

    result = benchmark_function(run)
    result["n_rows"] = n_rows
    result["benchmark"] = "StreamingEngine.process"
    return result


# ── 2. AutoFeature Search ────────────────────────────────────────


def benchmark_random_search(n_candidates: int) -> dict:
    """Benchmark RandomSearch with different candidate counts."""
    if not FORGE_AVAILABLE:
        return {"error": "Forge not installed"}

    np.random.seed(42)
    X = generate_numeric_data(200, n_cols=5)
    y = pd.Series(np.random.randint(0, 2, size=200))

    search = RandomSearch(
        n_candidates=n_candidates,
        scoring="accuracy",
        cv=2,
        random_state=42,
    )

    def run():
        search.results_ = []
        search.best_features_ = []
        search._is_fitted = False
        return search.fit(X, y)

    result = benchmark_function(run, n_iterations=3)
    result["n_candidates"] = n_candidates
    result["n_rows"] = 200
    result["benchmark"] = "RandomSearch.fit"
    return result


def benchmark_evolutionary_search(population_size: int, n_generations: int) -> dict:
    """Benchmark EvolutionarySearch with different population/generation sizes."""
    if not SEARCH_AVAILABLE:
        return {"error": "Forge search not installed"}

    np.random.seed(42)
    X = generate_numeric_data(200, n_cols=5)
    y = pd.Series(np.random.randint(0, 2, size=200))

    search = EvolutionarySearch(
        population_size=population_size,
        n_generations=n_generations,
        scoring="accuracy",
        cv=2,
        random_state=42,
    )

    def run():
        search.results_ = []
        search.best_features_ = []
        search.history_ = []
        search._is_fitted = False
        return search.fit(X, y)

    result = benchmark_function(run, n_iterations=3)
    result["population_size"] = population_size
    result["n_generations"] = n_generations
    result["n_rows"] = 200
    result["benchmark"] = "EvolutionarySearch.fit"
    return result


# ── 3. CPU Backend Operations ────────────────────────────────────


def benchmark_cpu_backend_add(n_rows: int) -> dict:
    """Benchmark CPUBackend.add() at different data sizes."""
    if not FORGE_AVAILABLE:
        return {"error": "Forge not installed"}

    np.random.seed(42)
    a = pd.Series(np.random.randn(n_rows))
    b = pd.Series(np.random.randn(n_rows))
    backend = CPUBackend()

    def run():
        return backend.add(a, b)

    result = benchmark_function(run)
    result["n_rows"] = n_rows
    result["benchmark"] = "CPUBackend.add"
    return result


def benchmark_cpu_backend_multiply(n_rows: int) -> dict:
    """Benchmark CPUBackend.multiply() at different data sizes."""
    if not FORGE_AVAILABLE:
        return {"error": "Forge not installed"}

    np.random.seed(42)
    a = pd.Series(np.random.randn(n_rows))
    b = pd.Series(np.random.randn(n_rows))
    backend = CPUBackend()

    def run():
        return backend.multiply(a, b)

    result = benchmark_function(run)
    result["n_rows"] = n_rows
    result["benchmark"] = "CPUBackend.multiply"
    return result


def benchmark_cpu_backend_polynomial(n_rows: int, n_cols: int) -> dict:
    """Benchmark CPUBackend.polynomial_features() at different data sizes."""
    if not FORGE_AVAILABLE:
        return {"error": "Forge not installed"}

    X = generate_numeric_data(n_rows, n_cols)
    backend = CPUBackend()

    def run():
        return backend.polynomial_features(X, degree=2)

    result = benchmark_function(run)
    result["n_rows"] = n_rows
    result["n_cols"] = n_cols
    result["benchmark"] = "CPUBackend.polynomial_features"
    return result


def benchmark_cpu_backend_one_hot(n_rows: int) -> dict:
    """Benchmark CPUBackend.one_hot_encode() at different data sizes."""
    if not FORGE_AVAILABLE:
        return {"error": "Forge not installed"}

    np.random.seed(42)
    categories = [f"cat_{i}" for i in range(20)]
    series = pd.Series(np.random.choice(categories, size=n_rows), name="feature")
    backend = CPUBackend()

    def run():
        return backend.one_hot_encode(series)

    result = benchmark_function(run)
    result["n_rows"] = n_rows
    result["benchmark"] = "CPUBackend.one_hot_encode"
    return result


# ── 4. Domain Packs ──────────────────────────────────────────────


def benchmark_finance_pack(n_rows: int) -> dict:
    """Benchmark FinanceFeaturePack.fit_transform() at different sizes."""
    if not FORGE_AVAILABLE:
        return {"error": "Forge not installed"}

    X = generate_finance_data(n_rows)
    pack = FinanceFeaturePack(windows=[5, 10, 20])

    def run():
        return pack.fit_transform(X)

    result = benchmark_function(run)
    result["n_rows"] = n_rows
    result["benchmark"] = "FinanceFeaturePack.fit_transform"
    return result


def benchmark_ecommerce_pack(n_rows: int) -> dict:
    """Benchmark EcommerceFeaturePack.fit_transform() at different sizes."""
    if not ECOMMERCE_AVAILABLE:
        return {"error": "Forge ecommerce pack not installed"}

    X = generate_ecommerce_data(n_rows)
    pack = EcommerceFeaturePack(
        customer_col="customer_id",
        amount_col="amount",
        timestamp_col="timestamp",
        product_col="product_id",
    )

    def run():
        return pack.fit_transform(X)

    result = benchmark_function(run)
    result["n_rows"] = n_rows
    result["benchmark"] = "EcommerceFeaturePack.fit_transform"
    return result


# ── 5. Feature Caching ───────────────────────────────────────────


def benchmark_cache_put_get(n_rows: int) -> dict:
    """Benchmark FeatureCache put/get/hit overhead at different DataFrame sizes."""
    if not FORGE_AVAILABLE:
        return {"error": "Forge not installed"}

    X = generate_numeric_data(n_rows, n_cols=10)
    config = CacheConfig(max_memory_items=200)
    cache = FeatureCache(config=config)

    # Pre-compute a key
    key = cache.compute_key(X, params={"test": "benchmark"})

    def run():
        cache.clear()
        # Put
        cache.put(key, X, transformer_name="BenchmarkTransformer")
        # Get (cache hit)
        result = cache.get(key)
        # Second get (another hit)
        result = cache.get(key)
        return result

    result = benchmark_function(run)
    result["n_rows"] = n_rows
    result["benchmark"] = "FeatureCache.put_get"
    return result


# ── 6. Graph Features ────────────────────────────────────────────


def benchmark_graph_features(n_nodes: int) -> dict:
    """Benchmark GraphFeatureGenerator with different graph sizes."""
    if not GRAPH_AVAILABLE:
        return {"error": "Forge graph module not installed"}

    edge_df = generate_graph_edge_data(n_nodes)

    gen = GraphFeatureGenerator(
        source_col="source",
        target_col="target",
        features=["degree", "weighted_degree", "community"],
    )

    def run():
        gen._is_fitted = False
        return gen.fit_transform(edge_df)

    result = benchmark_function(run, n_iterations=3)
    result["n_nodes"] = n_nodes
    result["n_edges"] = len(edge_df)
    result["benchmark"] = "GraphFeatureGenerator.fit_transform"
    return result


# ── 7. Catalog Operations ────────────────────────────────────────


def benchmark_catalog_register(n_entries: int) -> dict:
    """Benchmark CatalogStore register operations at different scales."""
    if not CATALOG_AVAILABLE:
        return {"error": "Forge catalog module not installed"}

    entries = [
        CatalogEntry(
            name=f"feature_{i}",
            description=f"Description for feature number {i}",
            dtype="float64",
            source_columns=[f"col_{i % 10}"],
            transformation=f"transform_{i % 5}",
            owner=f"team_{i % 3}",
            tags={"domain": f"domain_{i % 4}", "type": f"type_{i % 6}"},
        )
        for i in range(n_entries)
    ]

    def run():
        store = CatalogStore()
        for entry in entries:
            store.add(entry)
        return store

    result = benchmark_function(run)
    result["n_entries"] = n_entries
    result["benchmark"] = "CatalogStore.register"
    return result


def benchmark_catalog_search(n_entries: int) -> dict:
    """Benchmark CatalogStore search operations at different scales."""
    if not CATALOG_AVAILABLE:
        return {"error": "Forge catalog module not installed"}

    store = CatalogStore()
    for i in range(n_entries):
        store.add(CatalogEntry(
            name=f"feature_{i}",
            description=f"Description for feature number {i}",
            dtype="float64",
            source_columns=[f"col_{i % 10}"],
            transformation=f"transform_{i % 5}",
            owner=f"team_{i % 3}",
            tags={"domain": f"domain_{i % 4}", "type": f"type_{i % 6}"},
        ))

    def run():
        # Text search
        results_text = store.search("feature number")
        # Tag search
        results_tags = store.search(tags={"domain": "domain_2"})
        # Owner search
        results_owner = store.search(owner="team_1")
        # Combined search
        results_combined = store.search("feature", tags={"domain": "domain_0"})
        return results_combined

    result = benchmark_function(run)
    result["n_entries"] = n_entries
    result["benchmark"] = "CatalogStore.search"
    return result


# ── Main Runner ──────────────────────────────────────────────────


def run_all_benchmarks() -> list[dict]:
    """Run all benchmarks for new modules."""
    results = []

    # 1. Streaming throughput
    print("=" * 60)
    print("1. Streaming Throughput Benchmarks")
    print("=" * 60)
    for n_rows in [10_000, 100_000, 1_000_000]:
        print(f"  Benchmarking StreamingEngine with {n_rows:,} rows...")
        results.append(benchmark_streaming_throughput(n_rows))

    # 2. AutoFeature search
    print("=" * 60)
    print("2. AutoFeature Search Benchmarks")
    print("=" * 60)
    for n_candidates in [50, 100, 200]:
        print(f"  Benchmarking RandomSearch with {n_candidates} candidates...")
        results.append(benchmark_random_search(n_candidates))

    for pop_size, n_gens in [(20, 5), (50, 10), (100, 20)]:
        print(
            f"  Benchmarking EvolutionarySearch with "
            f"population={pop_size}, generations={n_gens}..."
        )
        results.append(benchmark_evolutionary_search(pop_size, n_gens))

    # 3. CPU Backend operations
    print("=" * 60)
    print("3. CPU Backend Benchmarks")
    print("=" * 60)
    for n_rows in [10_000, 100_000, 1_000_000]:
        print(f"  Benchmarking CPUBackend with {n_rows:,} rows...")
        results.append(benchmark_cpu_backend_add(n_rows))
        results.append(benchmark_cpu_backend_multiply(n_rows))
        results.append(benchmark_cpu_backend_one_hot(n_rows))

    for n_rows, n_cols in [(10_000, 5), (100_000, 10), (1_000_000, 5)]:
        print(
            f"  Benchmarking CPUBackend.polynomial_features "
            f"with {n_rows:,} rows, {n_cols} cols..."
        )
        results.append(benchmark_cpu_backend_polynomial(n_rows, n_cols))

    # 4. Domain packs
    print("=" * 60)
    print("4. Domain Pack Benchmarks")
    print("=" * 60)
    for n_rows in [10_000, 100_000, 1_000_000]:
        print(f"  Benchmarking FinanceFeaturePack with {n_rows:,} rows...")
        results.append(benchmark_finance_pack(n_rows))

    for n_rows in [10_000, 100_000, 1_000_000]:
        print(f"  Benchmarking EcommerceFeaturePack with {n_rows:,} rows...")
        results.append(benchmark_ecommerce_pack(n_rows))

    # 5. Feature caching
    print("=" * 60)
    print("5. Feature Cache Benchmarks")
    print("=" * 60)
    for n_rows in [1_000, 10_000, 100_000]:
        print(f"  Benchmarking FeatureCache with {n_rows:,} rows...")
        results.append(benchmark_cache_put_get(n_rows))

    # 6. Graph features
    print("=" * 60)
    print("6. Graph Feature Benchmarks")
    print("=" * 60)
    for n_nodes in [100, 1_000, 10_000]:
        print(f"  Benchmarking GraphFeatureGenerator with {n_nodes:,} nodes...")
        results.append(benchmark_graph_features(n_nodes))

    # 7. Catalog operations
    print("=" * 60)
    print("7. Catalog Benchmarks")
    print("=" * 60)
    for n_entries in [100, 1_000]:
        print(f"  Benchmarking CatalogStore register with {n_entries:,} entries...")
        results.append(benchmark_catalog_register(n_entries))

    for n_entries in [100, 1_000]:
        print(f"  Benchmarking CatalogStore search with {n_entries:,} entries...")
        results.append(benchmark_catalog_search(n_entries))

    return results


def print_results(results: list[dict]) -> None:
    """Print benchmark results in a formatted table."""
    print()
    print("=" * 80)
    print(f"{'Benchmark':<45} {'Mean (s)':>10} {'Std (s)':>10} "
          f"{'Min (s)':>10} {'Max (s)':>10}")
    print("-" * 80)

    for r in results:
        if "error" in r:
            print(f"  SKIPPED: {r.get('benchmark', 'unknown')} - {r['error']}")
            continue

        label = r.get("benchmark", "unknown")
        # Build a size suffix
        parts = []
        if "n_rows" in r:
            parts.append(f"{r['n_rows']:,}r")
        if "n_nodes" in r:
            parts.append(f"{r['n_nodes']:,}n")
        if "n_entries" in r:
            parts.append(f"{r['n_entries']:,}e")
        if "n_candidates" in r:
            parts.append(f"{r['n_candidates']}cand")
        if "population_size" in r:
            parts.append(f"pop{r['population_size']}")
        if "n_generations" in r:
            parts.append(f"gen{r['n_generations']}")
        if "n_cols" in r:
            parts.append(f"{r['n_cols']}c")

        size_str = " " + ",".join(parts) if parts else ""
        display = f"{label}{size_str}"

        print(
            f"  {display:<43} {r['mean_time']:>10.4f} {r['std_time']:>10.4f} "
            f"{r['min_time']:>10.4f} {r['max_time']:>10.4f}"
        )

    print("=" * 80)


if __name__ == "__main__":
    results = run_all_benchmarks()
    print_results(results)
