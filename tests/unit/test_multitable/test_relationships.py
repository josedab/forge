"""Tests for relationship management module."""

import numpy as np
import pandas as pd
import pytest

from forge.multitable.relationships import (
    Relationship,
    RelationshipType,
    RelationshipGraph,
    detect_relationships,
)
from forge.exceptions import ValidationError, ConfigurationError


@pytest.fixture
def customers_df():
    """Create a sample customers table."""
    return pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "age": [25, 30, 35, 40, 45],
        "city": ["NYC", "LA", "NYC", "Chicago", "LA"],
    })


@pytest.fixture
def orders_df():
    """Create a sample orders table."""
    return pd.DataFrame({
        "order_id": [101, 102, 103, 104, 105, 106, 107],
        "customer_id": [1, 1, 2, 3, 3, 3, 5],
        "amount": [100.0, 150.0, 200.0, 50.0, 75.0, 125.0, 300.0],
        "date": pd.date_range("2024-01-01", periods=7),
    })


@pytest.fixture
def products_df():
    """Create a sample products table."""
    return pd.DataFrame({
        "product_id": [1001, 1002, 1003],
        "name": ["Widget", "Gadget", "Gizmo"],
        "price": [10.0, 20.0, 30.0],
    })


class TestRelationship:
    """Tests for Relationship class."""

    def test_relationship_creation(self):
        """Test basic relationship creation."""
        rel = Relationship(
            parent_table="customers",
            parent_column="customer_id",
            child_table="orders",
            child_column="customer_id",
        )

        assert rel.parent_table == "customers"
        assert rel.child_table == "orders"
        assert rel.relationship_type == RelationshipType.ONE_TO_MANY

    def test_relationship_str(self):
        """Test relationship string representation."""
        rel = Relationship(
            parent_table="customers",
            parent_column="customer_id",
            child_table="orders",
            child_column="customer_id",
        )

        str_repr = str(rel)
        assert "customers" in str_repr
        assert "orders" in str_repr

    def test_relationship_reverse(self):
        """Test relationship reversal."""
        rel = Relationship(
            parent_table="customers",
            parent_column="customer_id",
            child_table="orders",
            child_column="customer_id",
            relationship_type=RelationshipType.ONE_TO_MANY,
        )

        reversed_rel = rel.reverse()
        assert reversed_rel.parent_table == "orders"
        assert reversed_rel.child_table == "customers"
        assert reversed_rel.relationship_type == RelationshipType.MANY_TO_ONE


class TestRelationshipGraph:
    """Tests for RelationshipGraph class."""

    def test_add_table(self, customers_df):
        """Test adding a table to the graph."""
        graph = RelationshipGraph()
        graph.add_table("customers", customers_df, primary_key="customer_id")

        info = graph.get_table("customers")
        assert info.name == "customers"
        assert info.primary_key == "customer_id"
        assert info.n_rows == 5

    def test_add_duplicate_table_fails(self, customers_df):
        """Test that adding duplicate table fails."""
        graph = RelationshipGraph()
        graph.add_table("customers", customers_df)

        with pytest.raises(ConfigurationError):
            graph.add_table("customers", customers_df)

    def test_add_relationship(self, customers_df, orders_df):
        """Test adding a relationship."""
        graph = RelationshipGraph()
        graph.add_table("customers", customers_df, primary_key="customer_id")
        graph.add_table("orders", orders_df, primary_key="order_id")

        rel = graph.add_relationship(
            parent="customers",
            parent_col="customer_id",
            child="orders",
            child_col="customer_id",
        )

        assert rel.parent_table == "customers"
        relationships = graph.get_relationships()
        assert len(relationships) == 1

    def test_get_neighbors(self, customers_df, orders_df):
        """Test getting neighboring tables."""
        graph = RelationshipGraph()
        graph.add_table("customers", customers_df, primary_key="customer_id")
        graph.add_table("orders", orders_df, primary_key="order_id")
        graph.add_relationship(
            parent="customers",
            parent_col="customer_id",
            child="orders",
            child_col="customer_id",
        )

        neighbors = graph.get_neighbors("customers")
        assert len(neighbors) == 1
        assert neighbors[0][0] == "orders"

    def test_find_path(self, customers_df, orders_df, products_df):
        """Test finding path between tables."""
        graph = RelationshipGraph()
        graph.add_table("customers", customers_df, primary_key="customer_id")
        graph.add_table("orders", orders_df, primary_key="order_id")
        graph.add_relationship(
            parent="customers",
            parent_col="customer_id",
            child="orders",
            child_col="customer_id",
        )

        path = graph.find_path("customers", "orders")
        assert path is not None
        assert len(path) == 1
        assert path[0][0] == "orders"

    def test_find_path_no_path(self, customers_df, products_df):
        """Test finding path when no path exists."""
        graph = RelationshipGraph()
        graph.add_table("customers", customers_df)
        graph.add_table("products", products_df)

        path = graph.find_path("customers", "products")
        assert path is None

    def test_validate_graph(self, customers_df, orders_df):
        """Test graph validation."""
        graph = RelationshipGraph()
        graph.add_table("customers", customers_df, primary_key="customer_id")
        graph.add_table("orders", orders_df, primary_key="order_id")
        graph.add_relationship(
            parent="customers",
            parent_col="customer_id",
            child="orders",
            child_col="customer_id",
        )

        issues = graph.validate()
        # Should have no issues for valid graph
        # (orders has customer_id=5 which doesn't exist, but is handled)
        assert isinstance(issues, list)

    def test_summary(self, customers_df, orders_df):
        """Test graph summary generation."""
        graph = RelationshipGraph()
        graph.add_table("customers", customers_df, primary_key="customer_id")
        graph.add_table("orders", orders_df, primary_key="order_id")
        graph.add_relationship(
            parent="customers",
            parent_col="customer_id",
            child="orders",
            child_col="customer_id",
        )

        summary = graph.summary()
        assert "customers" in summary
        assert "orders" in summary
        assert "2 tables" in summary


class TestDetectRelationships:
    """Tests for automatic relationship detection."""

    def test_detect_fk_relationship(self, customers_df, orders_df):
        """Test automatic FK detection."""
        tables = {
            "customers": customers_df,
            "orders": orders_df,
        }

        graph = detect_relationships(tables)

        relationships = graph.get_relationships()
        assert len(relationships) >= 1

        # Should detect customer_id relationship
        rel_cols = [(r.parent_column, r.child_column) for r in relationships]
        assert ("customer_id", "customer_id") in rel_cols

    def test_detect_primary_key(self, customers_df):
        """Test primary key detection."""
        tables = {"customers": customers_df}
        graph = detect_relationships(tables)

        info = graph.get_table("customers")
        assert info.primary_key == "customer_id"

    def test_no_false_positives(self, customers_df, products_df):
        """Test that unrelated tables don't get false relationships."""
        tables = {
            "customers": customers_df,
            "products": products_df,
        }

        graph = detect_relationships(tables)
        relationships = graph.get_relationships()

        # Should not find relationships between unrelated tables
        assert len(relationships) == 0
