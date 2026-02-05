"""Integration tests for multi-table feature synthesis."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forge import (
    DeepFeatureSynthesis,
    MultiTableTransformer,
    RelationshipGraph,
    detect_relationships,
    multi_table_features,
)
from forge.multitable.relationships import RelationshipType


@pytest.fixture
def customers_df():
    """Create a sample customers table."""
    np.random.seed(42)
    return pd.DataFrame({
        "customer_id": range(1, 101),
        "name": [f"Customer_{i}" for i in range(1, 101)],
        "age": np.random.randint(18, 70, 100),
        "income": np.random.uniform(20000, 150000, 100),
        "region": np.random.choice(["North", "South", "East", "West"], 100),
        "signup_date": pd.date_range("2020-01-01", periods=100, freq="D"),
    })


@pytest.fixture
def orders_df():
    """Create a sample orders table."""
    np.random.seed(42)
    n_orders = 500
    return pd.DataFrame({
        "order_id": range(1, n_orders + 1),
        "customer_id": np.random.randint(1, 101, n_orders),
        "product_id": np.random.randint(1, 51, n_orders),
        "order_date": pd.date_range("2020-01-01", periods=n_orders, freq="2H"),
        "amount": np.random.uniform(10, 500, n_orders),
        "quantity": np.random.randint(1, 10, n_orders),
    })


@pytest.fixture
def products_df():
    """Create a sample products table."""
    np.random.seed(42)
    return pd.DataFrame({
        "product_id": range(1, 51),
        "product_name": [f"Product_{i}" for i in range(1, 51)],
        "category": np.random.choice(["Electronics", "Clothing", "Food", "Home"], 50),
        "price": np.random.uniform(5, 200, 50),
    })


@pytest.fixture
def sample_tables(customers_df, orders_df, products_df):
    """Create a dict of sample tables."""
    return {
        "customers": customers_df,
        "orders": orders_df,
        "products": products_df,
    }


class TestRelationshipGraph:
    """Tests for RelationshipGraph class."""

    def test_add_tables_and_relationships(self, customers_df, orders_df):
        """Test adding tables and relationships manually."""
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
        assert rel.child_table == "orders"
        assert len(graph.get_relationships()) == 1

    def test_get_neighbors(self, customers_df, orders_df):
        """Test getting neighboring tables."""
        graph = RelationshipGraph()
        graph.add_table("customers", customers_df, primary_key="customer_id")
        graph.add_table("orders", orders_df, primary_key="order_id")
        graph.add_relationship(
            parent="customers", parent_col="customer_id",
            child="orders", child_col="customer_id",
        )

        neighbors = graph.get_neighbors("customers")
        assert len(neighbors) == 1
        assert neighbors[0][0] == "orders"

    def test_find_path(self, sample_tables):
        """Test finding path between tables."""
        graph = RelationshipGraph()
        for name, df in sample_tables.items():
            graph.add_table(name, df)

        graph.add_relationship(
            parent="customers", parent_col="customer_id",
            child="orders", child_col="customer_id",
        )
        graph.add_relationship(
            parent="products", parent_col="product_id",
            child="orders", child_col="product_id",
        )

        path = graph.find_path("customers", "products")
        assert path is not None
        assert len(path) == 2  # customers -> orders -> products

    def test_validate_graph(self, customers_df, orders_df):
        """Test graph validation."""
        graph = RelationshipGraph()
        graph.add_table("customers", customers_df, primary_key="customer_id")
        graph.add_table("orders", orders_df, primary_key="order_id")
        graph.add_relationship(
            parent="customers", parent_col="customer_id",
            child="orders", child_col="customer_id",
        )

        issues = graph.validate()
        # Should have no major issues with properly formed data
        assert isinstance(issues, list)

    def test_summary(self, customers_df, orders_df):
        """Test graph summary generation."""
        graph = RelationshipGraph()
        graph.add_table("customers", customers_df, primary_key="customer_id")
        graph.add_table("orders", orders_df, primary_key="order_id")
        graph.add_relationship(
            parent="customers", parent_col="customer_id",
            child="orders", child_col="customer_id",
        )

        summary = graph.summary()
        assert "customers" in summary
        assert "orders" in summary
        assert "2 tables" in summary


class TestDetectRelationships:
    """Tests for automatic relationship detection."""

    def test_detect_simple_relationships(self, sample_tables):
        """Test detecting relationships in simple schema."""
        graph = detect_relationships(sample_tables)

        # Should detect customer_id relationship
        relationships = graph.get_relationships()
        assert len(relationships) >= 1

        # Check that customers -> orders relationship exists
        rel_pairs = [(r.parent_table, r.child_table) for r in relationships]
        assert ("customers", "orders") in rel_pairs or ("orders", "customers") in rel_pairs

    def test_detect_primary_keys(self, sample_tables):
        """Test that primary keys are detected."""
        graph = detect_relationships(sample_tables)

        customers_info = graph.get_table("customers")
        # Should detect customer_id as primary key
        assert customers_info.primary_key is not None


class TestMultiTableTransformer:
    """Tests for MultiTableTransformer class."""

    def test_basic_fit_transform(self, sample_tables):
        """Test basic fit and transform."""
        transformer = MultiTableTransformer(
            target_table="customers",
            max_depth=1,
            max_features=20,
            verbose=0,
        )

        result = transformer.fit_transform(sample_tables)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_tables["customers"])
        # Should have more columns than original customers table
        assert len(result.columns) >= len(sample_tables["customers"].columns)

    def test_with_manual_relationships(self, sample_tables):
        """Test with manually specified relationships."""
        transformer = MultiTableTransformer(
            target_table="customers",
            relationships=[
                {
                    "parent": "customers",
                    "parent_col": "customer_id",
                    "child": "orders",
                    "child_col": "customer_id",
                }
            ],
            auto_detect_relationships=False,
            max_depth=1,
            verbose=0,
        )

        result = transformer.fit_transform(sample_tables)
        assert isinstance(result, pd.DataFrame)

    def test_transform_after_fit(self, sample_tables):
        """Test that transform works after fit."""
        transformer = MultiTableTransformer(
            target_table="customers",
            max_depth=1,
            max_features=10,
            verbose=0,
        )

        transformer.fit(sample_tables)

        # Transform the same data
        result1 = transformer.transform(sample_tables)

        # Transform new data (same schema)
        new_tables = {
            "customers": sample_tables["customers"].head(50),
            "orders": sample_tables["orders"][sample_tables["orders"]["customer_id"] <= 50],
            "products": sample_tables["products"],
        }
        result2 = transformer.transform(new_tables)

        # Same columns
        assert list(result1.columns) == list(result2.columns)

    def test_get_feature_names_out(self, sample_tables):
        """Test get_feature_names_out method."""
        transformer = MultiTableTransformer(
            target_table="customers",
            max_depth=1,
            verbose=0,
        )

        transformer.fit(sample_tables)
        names = transformer.get_feature_names_out()

        assert isinstance(names, list)
        assert len(names) > 0

    def test_get_feature_definitions(self, sample_tables):
        """Test get_feature_definitions method."""
        transformer = MultiTableTransformer(
            target_table="customers",
            max_depth=1,
            verbose=0,
        )

        transformer.fit(sample_tables)
        definitions = transformer.get_feature_definitions()

        assert isinstance(definitions, list)

    def test_include_target_columns_false(self, sample_tables):
        """Test with include_target_columns=False."""
        transformer = MultiTableTransformer(
            target_table="customers",
            max_depth=1,
            include_target_columns=False,
            verbose=0,
        )

        result = transformer.fit_transform(sample_tables)

        # Original customer columns should not be present (unless they're also generated)
        original_cols = set(sample_tables["customers"].columns)
        result_cols = set(result.columns)
        # At least some original columns should be absent
        assert original_cols != result_cols

    def test_save_and_load(self, sample_tables):
        """Test saving and loading transformer."""
        transformer = MultiTableTransformer(
            target_table="customers",
            max_depth=1,
            verbose=0,
        )
        transformer.fit(sample_tables)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            transformer.save(path)
            loaded = MultiTableTransformer.load(path)

            assert loaded._is_fitted
            assert loaded.target_table == "customers"
            assert loaded.max_depth == 1
        finally:
            Path(path).unlink()


class TestMultiTableFeaturesConvenience:
    """Tests for multi_table_features convenience function."""

    def test_basic_usage(self, sample_tables):
        """Test basic usage of convenience function."""
        result = multi_table_features(
            tables=sample_tables,
            target_table="customers",
            max_depth=1,
            max_features=10,
            verbose=0,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_tables["customers"])


class TestDeepFeatureSynthesis:
    """Tests for DeepFeatureSynthesis class."""

    def test_basic_synthesis(self, sample_tables):
        """Test basic feature synthesis."""
        graph = detect_relationships(sample_tables)

        dfs = DeepFeatureSynthesis(
            graph=graph,
            target_table="customers",
            max_depth=1,
            max_features=20,
            verbose=0,
        )

        result = dfs.synthesize()

        assert result.features is not None
        assert len(result.features) == len(sample_tables["customers"])

    def test_agg_primitives(self, sample_tables):
        """Test with specific aggregation primitives."""
        graph = detect_relationships(sample_tables)

        dfs = DeepFeatureSynthesis(
            graph=graph,
            target_table="customers",
            agg_primitives=["mean", "count", "sum"],
            max_depth=1,
            verbose=0,
        )

        result = dfs.synthesize()
        assert result.features is not None


class TestIntegrationWorkflows:
    """End-to-end integration tests."""

    def test_ecommerce_workflow(self, sample_tables):
        """Test full e-commerce feature engineering workflow."""
        # Step 1: Detect relationships
        graph = detect_relationships(sample_tables)

        # Step 2: Generate features
        transformer = MultiTableTransformer(
            target_table="customers",
            max_depth=2,
            max_features=50,
            verbose=0,
        )

        features = transformer.fit_transform(sample_tables)

        # Step 3: Verify features
        assert isinstance(features, pd.DataFrame)
        assert len(features) == 100  # 100 customers

        # Step 4: Get feature info
        names = transformer.get_feature_names_out()
        assert len(names) > len(sample_tables["customers"].columns)

    def test_with_sklearn_model(self, sample_tables):
        """Test multi-table features with sklearn model."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        # Generate features
        transformer = MultiTableTransformer(
            target_table="customers",
            max_depth=1,
            max_features=20,
            verbose=0,
        )

        features = transformer.fit_transform(sample_tables)

        # Create binary target
        y = (sample_tables["customers"]["income"] > 75000).astype(int)

        # Select only numeric features
        numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        X = features[numeric_cols].fillna(0)

        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_relationship_reversal(self, customers_df, orders_df):
        """Test relationship reversal for bidirectional traversal."""
        graph = RelationshipGraph()
        graph.add_table("customers", customers_df, primary_key="customer_id")
        graph.add_table("orders", orders_df, primary_key="order_id")

        rel = graph.add_relationship(
            parent="customers",
            parent_col="customer_id",
            child="orders",
            child_col="customer_id",
        )

        # Test reverse relationship
        reversed_rel = rel.reverse()
        assert reversed_rel.parent_table == "orders"
        assert reversed_rel.child_table == "customers"
        assert reversed_rel.relationship_type == RelationshipType.MANY_TO_ONE
