"""Integration tests for multi-module workflows across new forge modules.

Tests verify that streaming, caching, domain packs, catalog, graph features,
search, and backends work correctly together in realistic workflows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from forge import (
    AutoFeatureSearch,
    CacheConfig,
    CatalogEntry,
    CatalogStore,
    CPUBackend,
    DataAnalyzer,
    EntityGraph,
    FeatureCache,
    FeatureCatalogStore,
    FinanceFeaturePack,
    GraphBuilder,
    GraphFeatureGenerator,
    StreamingEngine,
    get_backend,
)
from forge.streaming import IncrementalAggregator, StreamingWindowAggregator


class TestStreamingWithCache:
    """Tests for streaming feature generation combined with caching."""

    def test_streaming_results_cached_and_retrieved(
        self, sample_numeric_df: pd.DataFrame
    ):
        """Process data through StreamingEngine, cache results, verify cache hit."""
        engine = StreamingEngine(
            generators=[
                IncrementalAggregator(stats=["mean", "std"]),
            ],
            passthrough=False,
        )

        result_first = engine.process(sample_numeric_df)
        assert isinstance(result_first, pd.DataFrame)
        assert len(result_first) == len(sample_numeric_df)

        cache = FeatureCache(config=CacheConfig(max_memory_items=50))
        key = cache.compute_key(sample_numeric_df, params={"engine": "streaming_v1"})
        cache.put(key, result_first, transformer_name="StreamingEngine")

        assert cache.has(key)
        cached_result = cache.get(key)
        assert cached_result is not None
        pd.testing.assert_frame_equal(cached_result, result_first)
        assert cache.stats["hits"] == 1

    def test_cache_hit_returns_same_results_on_second_run(
        self, sample_numeric_df: pd.DataFrame
    ):
        """Run streaming engine twice and verify cache hit on second run."""
        engine = StreamingEngine(
            generators=[
                IncrementalAggregator(stats=["mean", "min", "max"]),
                StreamingWindowAggregator(window_size=50, stats=["mean"]),
            ],
            passthrough=False,
        )

        cache = FeatureCache(config=CacheConfig(max_memory_items=100))

        # First run: cache miss
        key = cache.compute_key(sample_numeric_df, params={"pipeline": "stream_v1"})
        result_from_cache = cache.get(key)
        assert result_from_cache is None
        assert cache.stats["misses"] == 1

        result_first = engine.process(sample_numeric_df)
        cache.put(key, result_first)

        # Second run: cache hit
        result_second = cache.get(key)
        assert result_second is not None
        pd.testing.assert_frame_equal(result_second, result_first)
        assert cache.stats["hits"] == 1

    def test_streaming_engine_stats_after_batches(
        self, sample_numeric_df: pd.DataFrame
    ):
        """Verify streaming engine statistics update after multiple batches."""
        engine = StreamingEngine(
            generators=[IncrementalAggregator(stats=["mean"])],
            passthrough=True,
        )

        batch1 = sample_numeric_df.iloc[:50]
        batch2 = sample_numeric_df.iloc[50:]
        engine.process(batch1)
        engine.process(batch2)

        stats = engine.stats
        assert stats["n_batches_processed"] == 2
        assert stats["n_events_processed"] == len(sample_numeric_df)
        assert stats["n_generators"] == 1
        assert stats["n_features"] > 0

    def test_cache_invalidation_after_streaming(
        self, sample_numeric_df: pd.DataFrame
    ):
        """Verify that invalidated cache entries are no longer retrievable."""
        engine = StreamingEngine(
            generators=[IncrementalAggregator(stats=["mean"])],
            passthrough=False,
        )

        cache = FeatureCache(config=CacheConfig(max_memory_items=50))
        result = engine.process(sample_numeric_df)
        key = cache.compute_key(sample_numeric_df, params={"version": "1"})
        cache.put(key, result)

        assert cache.has(key)
        removed = cache.invalidate(key)
        assert removed is True
        assert not cache.has(key)
        assert cache.get(key) is None


class TestPacksWithCatalog:
    """Tests for domain feature packs integrated with the catalog store."""

    @pytest.fixture
    def finance_df(self) -> pd.DataFrame:
        """DataFrame with finance-like columns for FinanceFeaturePack."""
        np.random.seed(42)
        n = 200
        prices = 100 + np.cumsum(np.random.normal(0, 1, n))
        return pd.DataFrame(
            {
                "close": prices,
                "volume": np.random.randint(1000, 10000, n),
            }
        )

    def test_finance_pack_features_registered_in_catalog(
        self, finance_df: pd.DataFrame
    ):
        """Generate finance features and register them in CatalogStore."""
        pack = FinanceFeaturePack(
            features=["returns", "sma", "rsi"],
            windows=[5, 10],
        )
        features = pack.fit_transform(finance_df)

        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(finance_df)
        assert len(features.columns) > 0

        store = CatalogStore()
        for col_name in features.columns:
            entry = CatalogEntry(
                name=col_name,
                description=f"Finance feature: {col_name}",
                dtype=str(features[col_name].dtype),
                source_columns=["close", "volume"],
                transformation="FinanceFeaturePack",
                owner="quant-team",
                tags={"domain": "finance", "pack": "finance"},
            )
            store.add(entry)

        assert store.size == len(features.columns)

    def test_catalog_search_returns_registered_features(
        self, finance_df: pd.DataFrame
    ):
        """Search and retrieve features from catalog after registration."""
        pack = FinanceFeaturePack(
            features=["returns", "volatility", "sma"],
            windows=[5],
        )
        features = pack.fit_transform(finance_df)

        store = CatalogStore()
        for col_name in features.columns:
            entry = CatalogEntry(
                name=col_name,
                description=f"Finance indicator: {col_name}",
                tags={"domain": "finance"},
                owner="quant-team",
            )
            store.add(entry)

        results = store.search("volatility")
        assert len(results) > 0
        assert any("volatility" in r.name for r in results)

        results_by_tag = store.search(tags={"domain": "finance"})
        assert len(results_by_tag) == len(features.columns)

    def test_catalog_store_deprecation_workflow(
        self, finance_df: pd.DataFrame
    ):
        """Deprecate features in catalog and verify they are excluded from search."""
        pack = FinanceFeaturePack(
            features=["returns", "sma"],
            windows=[5],
        )
        features = pack.fit_transform(finance_df)

        store = CatalogStore()
        for col_name in features.columns:
            store.add(CatalogEntry(name=col_name, owner="team-a"))

        # Deprecate one feature
        sma_cols = [c for c in features.columns if "sma" in c]
        assert len(sma_cols) > 0
        store.deprecate(sma_cols[0], reason="Replaced by EMA")

        # Search excluding deprecated
        results = store.search(include_deprecated=False)
        result_names = {r.name for r in results}
        assert sma_cols[0] not in result_names

        # Search including deprecated
        results_all = store.search(include_deprecated=True)
        result_all_names = {r.name for r in results_all}
        assert sma_cols[0] in result_all_names

    def test_feature_catalog_store_from_transformer(
        self, finance_df: pd.DataFrame
    ):
        """Use FeatureCatalogStore.add_from_transformer to populate catalog."""
        pack = FinanceFeaturePack(
            features=["returns", "rsi", "macd"],
        )
        pack.fit(finance_df)

        catalog = FeatureCatalogStore()
        count = catalog.add_from_transformer(
            pack, owner="ml-team", tags={"source": "pack"}
        )

        assert count > 0
        assert catalog.size == count

        all_entries = catalog.list_all()
        assert all(e.owner == "ml-team" for e in all_entries)
        assert all(e.tags.get("source") == "pack" for e in all_entries)


class TestGraphWithDistributed:
    """Tests for graph feature generation from DataFrames."""

    @pytest.fixture
    def edge_df(self) -> pd.DataFrame:
        """DataFrame representing edges in a simple graph."""
        np.random.seed(42)
        sources = [f"user_{i}" for i in range(20) for _ in range(3)]
        targets = [f"item_{np.random.randint(0, 10)}" for _ in range(60)]
        weights = np.random.uniform(0.5, 5.0, 60)
        return pd.DataFrame(
            {
                "source": sources,
                "target": targets,
                "weight": weights,
            }
        )

    def test_build_entity_graph_from_dataframe(self, edge_df: pd.DataFrame):
        """Build an EntityGraph from a DataFrame and verify structure."""
        builder = GraphBuilder(
            source_col="source",
            target_col="target",
            weight_col="weight",
            directed=False,
        )
        graph = builder.build(edge_df)

        assert isinstance(graph, EntityGraph)
        assert graph.n_nodes > 0
        assert graph.n_edges > 0

        # All source and target values should be nodes
        all_nodes = set(graph.nodes)
        for src in edge_df["source"].unique():
            assert str(src) in all_nodes
        for tgt in edge_df["target"].unique():
            assert str(tgt) in all_nodes

    def test_graph_feature_generator_produces_numeric_columns(
        self, edge_df: pd.DataFrame
    ):
        """Generate graph features and verify they are valid numeric columns."""
        gen = GraphFeatureGenerator(
            source_col="source",
            target_col="target",
            weight_col="weight",
            features=["degree", "weighted_degree", "pagerank"],
            directed=False,
        )
        gen.fit(edge_df)
        features = gen.transform(edge_df)

        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(edge_df)
        assert len(features.columns) > 0

        # All columns should be numeric
        for col in features.columns:
            assert pd.api.types.is_numeric_dtype(features[col]), (
                f"Column {col} is not numeric"
            )

        # Feature names should follow the graph_ prefix convention
        feature_names = gen.get_feature_names_out()
        assert all(name.startswith("graph_") for name in feature_names)

    def test_graph_features_with_community_detection(
        self, edge_df: pd.DataFrame
    ):
        """Generate community features and verify valid labels."""
        gen = GraphFeatureGenerator(
            source_col="source",
            target_col="target",
            features=["degree", "community", "clustering_coefficient"],
            directed=False,
        )
        features = gen.fit_transform(edge_df)

        assert "graph_community" in features.columns
        assert "graph_degree" in features.columns
        assert "graph_clustering" in features.columns

        # Community labels should be non-negative
        assert (features["graph_community"] >= 0).all()
        # Degree should be non-negative
        assert (features["graph_degree"] >= 0).all()
        # Clustering coefficient should be between 0 and 1
        assert (features["graph_clustering"] >= 0).all()
        assert (features["graph_clustering"] <= 1).all()

    def test_graph_adjacency_matrix_consistency(self, edge_df: pd.DataFrame):
        """Verify adjacency matrix is consistent with the entity graph."""
        builder = GraphBuilder(
            source_col="source",
            target_col="target",
            directed=True,
        )
        graph = builder.build(edge_df)
        matrix, node_list = graph.to_adjacency_matrix()

        assert matrix.shape == (graph.n_nodes, graph.n_nodes)
        assert len(node_list) == graph.n_nodes

        # At least some edges should produce nonzero entries
        assert matrix.sum() > 0


class TestSearchWithBackends:
    """Tests for AutoFeatureSearch with CPU backend."""

    def test_auto_feature_search_random_method(
        self, sample_numeric_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Run AutoFeatureSearch with random method and verify output."""
        search = AutoFeatureSearch(
            method="random",
            n_features=5,
            n_candidates=20,
            cv=2,
            random_state=42,
        )
        result = search.fit_transform(sample_numeric_df, sample_target_binary)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_numeric_df)
        assert len(result.columns) <= 5
        assert len(result.columns) > 0

    def test_auto_feature_search_feature_names_out(
        self, sample_numeric_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Verify get_feature_names_out matches transform output."""
        search = AutoFeatureSearch(
            method="random",
            n_features=5,
            n_candidates=15,
            cv=2,
            random_state=42,
        )
        search.fit(sample_numeric_df, sample_target_binary)

        names = search.get_feature_names_out()
        result = search.transform(sample_numeric_df)

        assert list(result.columns) == names

    def test_cpu_backend_operations(self, sample_numeric_df: pd.DataFrame):
        """Verify CPUBackend can perform core operations on the data."""
        backend = CPUBackend()

        assert backend.is_available()

        series_a = sample_numeric_df["age"].astype(float)
        series_b = sample_numeric_df["income"].astype(float)

        added = backend.add(series_a, series_b)
        assert len(added) == len(sample_numeric_df)
        assert isinstance(added, pd.Series)

        multiplied = backend.multiply(series_a, series_b)
        assert len(multiplied) == len(sample_numeric_df)

        log_result = backend.log(series_a)
        assert len(log_result) == len(sample_numeric_df)
        assert not log_result.isna().all()

    def test_get_backend_returns_cpu_by_default(self):
        """Verify get_backend with 'cpu' returns a CPUBackend."""
        backend = get_backend("cpu")
        assert isinstance(backend, CPUBackend)
        assert backend.is_available()

    def test_search_results_are_retrievable(
        self, sample_numeric_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Verify search results can be inspected after fit."""
        search = AutoFeatureSearch(
            method="random",
            n_features=10,
            n_candidates=30,
            cv=2,
            random_state=42,
        )
        search.fit(sample_numeric_df, sample_target_binary)

        results = search.get_search_results()
        assert isinstance(results, list)
        assert len(results) > 0

        # Each result should have a name and a score
        for r in results:
            assert isinstance(r.name, str)
            assert len(r.name) > 0
            assert isinstance(r.score, float)


class TestEndToEndPipeline:
    """Full end-to-end pipeline: analyze, generate, select, cache, catalog."""

    @pytest.fixture
    def finance_data(self) -> pd.DataFrame:
        """DataFrame with finance-like data for the full pipeline."""
        np.random.seed(42)
        n = 150
        prices = 100 + np.cumsum(np.random.normal(0, 1, n))
        return pd.DataFrame(
            {
                "close": prices,
                "volume": np.random.randint(500, 5000, n),
                "open": prices + np.random.normal(0, 0.5, n),
                "high": prices + np.abs(np.random.normal(0, 1, n)),
                "low": prices - np.abs(np.random.normal(0, 1, n)),
            }
        )

    @pytest.fixture
    def finance_target(self) -> pd.Series:
        """Binary target for the finance pipeline."""
        np.random.seed(42)
        return pd.Series(np.random.choice([0, 1], 150), name="direction")

    def test_analyze_then_generate_with_pack(
        self, finance_data: pd.DataFrame
    ):
        """Analyze data and generate features using a domain pack."""
        analyzer = DataAnalyzer()
        report = analyzer.analyze(finance_data)

        assert report is not None
        assert len(report.column_types) == len(finance_data.columns)

        pack = FinanceFeaturePack(
            features=["returns", "sma", "volatility", "rsi"],
            windows=[5, 10],
        )
        features = pack.fit_transform(finance_data)

        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(finance_data)
        assert len(features.columns) > 0

    def test_generate_select_with_search(
        self, finance_data: pd.DataFrame, finance_target: pd.Series
    ):
        """Generate features with a pack and select with AutoFeatureSearch."""
        pack = FinanceFeaturePack(
            features=["returns", "sma", "volatility"],
            windows=[5],
        )
        generated = pack.fit_transform(finance_data)
        generated = generated.fillna(0)

        search = AutoFeatureSearch(
            method="random",
            n_features=5,
            n_candidates=15,
            cv=2,
            random_state=42,
        )
        selected = search.fit_transform(generated, finance_target)

        assert isinstance(selected, pd.DataFrame)
        assert len(selected) == len(finance_data)
        assert len(selected.columns) <= 5

    def test_cache_and_catalog_full_workflow(
        self, finance_data: pd.DataFrame
    ):
        """Generate features, cache them, register in catalog."""
        pack = FinanceFeaturePack(
            features=["returns", "sma", "rsi"],
            windows=[5],
        )
        features = pack.fit_transform(finance_data)

        # Cache the features
        cache = FeatureCache(config=CacheConfig(max_memory_items=50))
        key = cache.compute_key(finance_data, transformer=pack)
        cache.put(key, features, transformer_name="FinanceFeaturePack")

        assert cache.has(key)
        cached = cache.get(key)
        pd.testing.assert_frame_equal(cached, features)

        # Register in catalog
        catalog = FeatureCatalogStore()
        count = catalog.add_from_transformer(
            pack, owner="pipeline", tags={"stage": "generated"}
        )
        assert count > 0

        catalog_stats = catalog.get_statistics()
        assert catalog_stats["total_features"] == count
        assert catalog_stats["active_features"] == count

    def test_pipeline_produces_usable_features_for_sklearn(
        self, finance_data: pd.DataFrame, finance_target: pd.Series
    ):
        """Full pipeline output can be used to train an sklearn classifier."""
        # Step 1: Generate features
        pack = FinanceFeaturePack(
            features=["returns", "sma", "volatility", "momentum"],
            windows=[5, 10],
        )
        features = pack.fit_transform(finance_data)
        features = features.fillna(0).replace([np.inf, -np.inf], 0)

        assert len(features.columns) > 0
        assert features.shape[0] == len(finance_target)

        # Step 2: Cache results
        cache = FeatureCache(config=CacheConfig(max_memory_items=50))
        key = cache.compute_key(finance_data, transformer=pack)
        cache.put(key, features)

        # Step 3: Register in catalog
        store = CatalogStore()
        for col in features.columns:
            store.add(CatalogEntry(
                name=col,
                description="Feature from finance pack",
                tags={"domain": "finance", "stage": "production"},
                owner="ml-pipeline",
            ))

        assert store.size == len(features.columns)

        # Step 4: Train sklearn model on produced features
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        scores = cross_val_score(clf, features, finance_target, cv=3)

        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)

    def test_streaming_to_cache_to_catalog_workflow(
        self, sample_numeric_df: pd.DataFrame, sample_target_binary: pd.Series
    ):
        """Stream data, cache incremental results, register final features."""
        engine = StreamingEngine(
            generators=[
                IncrementalAggregator(stats=["mean", "std"]),
            ],
            passthrough=False,
        )

        # Process in two batches
        batch1 = sample_numeric_df.iloc[:50]
        batch2 = sample_numeric_df.iloc[50:]
        engine.process(batch1)
        final_features = engine.process(batch2)

        assert len(final_features) == len(batch2)
        assert len(final_features.columns) > 0

        # Cache
        cache = FeatureCache(config=CacheConfig(max_memory_items=100))
        key = cache.compute_key(
            sample_numeric_df,
            params={"pipeline": "streaming", "stats": ["mean", "std"]},
        )
        cache.put(key, final_features, transformer_name="StreamingEngine")
        assert cache.has(key)

        # Catalog
        store = CatalogStore()
        for col in final_features.columns:
            store.add(CatalogEntry(
                name=col,
                description="Streaming aggregation feature",
                tags={"source": "streaming"},
            ))

        results = store.search(tags={"source": "streaming"})
        assert len(results) == len(final_features.columns)
