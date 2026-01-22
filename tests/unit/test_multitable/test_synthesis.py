"""Tests for deep feature synthesis module."""

import numpy as np
import pandas as pd
import pytest

from forge.multitable.relationships import RelationshipGraph
from forge.multitable.synthesis import (
    DeepFeatureSynthesis,
    AggregationPrimitive,
    TransformPrimitive,
    AGGREGATION_PRIMITIVES,
    create_dfs,
)


@pytest.fixture
def customers_df():
    """Create a sample customers table."""
    return pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "age": [25, 30, 35, 40, 45],
    })


@pytest.fixture
def orders_df():
    """Create a sample orders table."""
    return pd.DataFrame({
        "order_id": [101, 102, 103, 104, 105, 106, 107],
        "customer_id": [1, 1, 2, 3, 3, 3, 5],
        "amount": [100.0, 150.0, 200.0, 50.0, 75.0, 125.0, 300.0],
    })


@pytest.fixture
def relationship_graph(customers_df, orders_df):
    """Create a relationship graph for testing."""
    graph = RelationshipGraph()
    graph.add_table("customers", customers_df, primary_key="customer_id")
    graph.add_table("orders", orders_df, primary_key="order_id")
    graph.add_relationship(
        parent="customers",
        parent_col="customer_id",
        child="orders",
        child_col="customer_id",
    )
    return graph


class TestAggregationPrimitives:
    """Tests for aggregation primitives."""

    def test_sum_primitive(self):
        """Test sum aggregation."""
        primitive = AGGREGATION_PRIMITIVES["sum"]
        series = pd.Series([1, 2, 3, 4, 5])
        result = primitive.apply(series)
        assert result == 15.0

    def test_mean_primitive(self):
        """Test mean aggregation."""
        primitive = AGGREGATION_PRIMITIVES["mean"]
        series = pd.Series([1, 2, 3, 4, 5])
        result = primitive.apply(series)
        assert result == 3.0

    def test_count_primitive(self):
        """Test count aggregation."""
        primitive = AGGREGATION_PRIMITIVES["count"]
        series = pd.Series([1, 2, 3, 4, 5])
        result = primitive.apply(series)
        assert result == 5

    def test_nunique_primitive(self):
        """Test nunique aggregation."""
        primitive = AGGREGATION_PRIMITIVES["nunique"]
        series = pd.Series(["a", "b", "a", "c"])
        result = primitive.apply(series)
        assert result == 3

    def test_std_primitive(self):
        """Test std aggregation."""
        primitive = AGGREGATION_PRIMITIVES["std"]
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = primitive.apply(series)
        assert result > 0


class TestDeepFeatureSynthesis:
    """Tests for DeepFeatureSynthesis class."""

    def test_synthesis_basic(self, relationship_graph):
        """Test basic feature synthesis."""
        dfs = DeepFeatureSynthesis(
            graph=relationship_graph,
            target_table="customers",
            max_depth=1,
        )

        result = dfs.synthesize()

        assert len(result.features) == 5  # 5 customers
        assert len(result.feature_definitions) > 0

    def test_synthesis_aggregations(self, relationship_graph):
        """Test that aggregation features are generated."""
        dfs = DeepFeatureSynthesis(
            graph=relationship_graph,
            target_table="customers",
            agg_primitives=["mean", "sum", "count"],
            max_depth=1,
        )

        result = dfs.synthesize()

        # Should have features for order amounts
        feature_names = list(result.features.columns)
        assert any("amount" in name for name in feature_names)
        assert any("mean" in name or "sum" in name or "count" in name
                   for name in feature_names)

    def test_synthesis_depth_control(self, relationship_graph):
        """Test that max_depth is respected."""
        dfs = DeepFeatureSynthesis(
            graph=relationship_graph,
            target_table="customers",
            max_depth=1,
        )

        result = dfs.synthesize()

        # All features should be at depth 1
        for defn in result.feature_definitions:
            assert defn.depth <= 1

    def test_synthesis_max_features(self, relationship_graph):
        """Test that max_features is respected."""
        dfs = DeepFeatureSynthesis(
            graph=relationship_graph,
            target_table="customers",
            max_features=5,
            max_depth=1,
        )

        result = dfs.synthesize()
        assert len(result.features.columns) <= 5

    def test_feature_definitions(self, relationship_graph):
        """Test that feature definitions are recorded."""
        dfs = DeepFeatureSynthesis(
            graph=relationship_graph,
            target_table="customers",
            max_depth=1,
        )

        result = dfs.synthesize()

        for defn in result.feature_definitions:
            assert defn.name in result.features.columns
            assert defn.source_table == "orders"
            assert len(defn.path) > 0

    def test_ignore_columns(self, relationship_graph):
        """Test that ignored columns are excluded."""
        dfs = DeepFeatureSynthesis(
            graph=relationship_graph,
            target_table="customers",
            ignore_columns=["order_id"],
            max_depth=1,
        )

        result = dfs.synthesize()
        feature_names = list(result.features.columns)

        # Should not have features based on order_id
        assert not any("order_id" in name for name in feature_names)


class TestCreateDFS:
    """Tests for create_dfs convenience function."""

    def test_create_dfs_auto_detect(self, customers_df, orders_df):
        """Test DFS creation with auto-detected relationships."""
        tables = {
            "customers": customers_df,
            "orders": orders_df,
        }

        dfs = create_dfs(tables, target_table="customers", max_depth=1)
        result = dfs.synthesize()

        assert len(result.features) == 5
        assert len(result.feature_definitions) > 0
