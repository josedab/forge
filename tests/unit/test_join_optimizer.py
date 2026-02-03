"""Tests for cross-table join-path optimization and aggregation."""

from __future__ import annotations

import pandas as pd
import pytest

from forge.multitable.join_optimizer import (
    CrossTableAggregator,
    JoinPath,
    JoinPathFinder,
)


@pytest.fixture
def relational_tables() -> dict[str, pd.DataFrame]:
    customers = pd.DataFrame({
        "customer_id": [1, 2, 3, 4],
        "name": ["Alice", "Bob", "Carol", "Dave"],
    })
    orders = pd.DataFrame({
        "order_id": [10, 11, 12, 13, 14],
        "customer_id": [1, 1, 2, 3, 3],
        "amount": [100.0, 150.0, 200.0, 50.0, 300.0],
        "product_id": [1, 2, 1, 3, 2],
    })
    products = pd.DataFrame({
        "product_id": [1, 2, 3],
        "price": [10.0, 20.0, 30.0],
        "category": ["A", "B", "A"],
    })
    return {"customers": customers, "orders": orders, "products": products}


@pytest.fixture
def relationships() -> dict[tuple[str, str], tuple[str, str]]:
    return {
        ("customers", "orders"): ("customer_id", "customer_id"),
        ("orders", "products"): ("product_id", "product_id"),
    }


class TestJoinPathFinder:
    def test_direct_path(self, relationships: dict) -> None:
        finder = JoinPathFinder()
        paths = finder.find_paths(relationships, "customers", "orders")
        assert len(paths) >= 1
        assert paths[0].depth == 1
        assert paths[0].tables == ["customers", "orders"]

    def test_two_hop_path(self, relationships: dict) -> None:
        finder = JoinPathFinder(max_depth=3)
        paths = finder.find_paths(relationships, "customers", "products")
        assert len(paths) >= 1
        assert paths[0].depth == 2
        assert paths[0].tables == ["customers", "orders", "products"]

    def test_no_path(self, relationships: dict) -> None:
        extra = {**relationships}
        finder = JoinPathFinder()
        paths = finder.find_paths(extra, "customers", "nonexistent")
        assert len(paths) == 0

    def test_max_depth_limit(self, relationships: dict) -> None:
        finder = JoinPathFinder(max_depth=1)
        paths = finder.find_paths(relationships, "customers", "products")
        assert len(paths) == 0  # Requires 2 hops

    def test_find_all_reachable(self, relationships: dict) -> None:
        finder = JoinPathFinder(max_depth=3)
        reachable = finder.find_all_reachable(relationships, "customers")
        assert "orders" in reachable
        assert "products" in reachable
        assert reachable["orders"].depth == 1
        assert reachable["products"].depth == 2

    def test_bidirectional(self, relationships: dict) -> None:
        finder = JoinPathFinder()
        paths = finder.find_paths(relationships, "orders", "customers")
        assert len(paths) >= 1


class TestJoinPath:
    def test_str_representation(self) -> None:
        path = JoinPath(
            tables=["customers", "orders"],
            join_keys=[("customer_id", "customer_id")],
            depth=1,
        )
        s = str(path)
        assert "customers" in s
        assert "orders" in s

    def test_estimated_cost(self) -> None:
        p1 = JoinPath(tables=["a", "b"], join_keys=[("x", "y")], depth=1)
        p2 = JoinPath(tables=["a", "b", "c"], join_keys=[("x", "y"), ("y", "z")], depth=2)
        assert p2.estimated_cost > p1.estimated_cost


class TestCrossTableAggregator:
    def test_basic_aggregation(
        self, relational_tables: dict[str, pd.DataFrame], relationships: dict
    ) -> None:
        finder = JoinPathFinder()
        paths = finder.find_paths(relationships, "customers", "orders")
        path = paths[0]

        agg = CrossTableAggregator(agg_functions=["count", "mean", "sum"])
        result = agg.aggregate(
            relational_tables, path,
            target_key="customer_id",
            agg_columns=["amount"],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert any("amount" in c for c in result.columns)

    def test_two_hop_aggregation(
        self, relational_tables: dict[str, pd.DataFrame], relationships: dict
    ) -> None:
        finder = JoinPathFinder(max_depth=3)
        paths = finder.find_paths(relationships, "customers", "products")
        path = paths[0]

        agg = CrossTableAggregator(agg_functions=["count", "mean"])
        result = agg.aggregate(
            relational_tables, path,
            target_key="customer_id",
            agg_columns=["price"],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_auto_detect_columns(
        self, relational_tables: dict[str, pd.DataFrame], relationships: dict
    ) -> None:
        finder = JoinPathFinder()
        paths = finder.find_paths(relationships, "customers", "orders")

        agg = CrossTableAggregator()
        result = agg.aggregate(
            relational_tables, paths[0],
            target_key="customer_id",
        )
        assert result.shape[1] > 0

    def test_empty_join(self) -> None:
        path = JoinPath(tables=["a"], join_keys=[], depth=0)
        agg = CrossTableAggregator()
        result = agg.aggregate({}, path, target_key="id")
        assert result.empty
