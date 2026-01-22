#!/usr/bin/env python
"""Multi-table feature engineering workflow example.

This example demonstrates how to use Forge's multi-table feature synthesis
capabilities to automatically generate features from related tables,
similar to what you would find in a typical relational database.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forge import (
    MultiTableTransformer,
    RelationshipGraph,
    DeepFeatureSynthesis,
    detect_relationships,
    multi_table_features,
)


def create_ecommerce_data() -> dict[str, pd.DataFrame]:
    """Create sample e-commerce data with multiple related tables."""
    np.random.seed(42)

    # Customers table
    n_customers = 100
    customers = pd.DataFrame({
        "customer_id": range(1, n_customers + 1),
        "name": [f"Customer_{i}" for i in range(1, n_customers + 1)],
        "age": np.random.randint(18, 70, n_customers),
        "income": np.random.uniform(25000, 150000, n_customers),
        "region": np.random.choice(["North", "South", "East", "West"], n_customers),
        "signup_date": pd.date_range("2020-01-01", periods=n_customers, freq="D"),
        "is_premium": np.random.choice([True, False], n_customers, p=[0.3, 0.7]),
    })

    # Products table
    n_products = 50
    products = pd.DataFrame({
        "product_id": range(1, n_products + 1),
        "product_name": [f"Product_{i}" for i in range(1, n_products + 1)],
        "category": np.random.choice(
            ["Electronics", "Clothing", "Food", "Home", "Books"], n_products
        ),
        "price": np.random.uniform(10, 500, n_products),
        "rating": np.random.uniform(1, 5, n_products),
    })

    # Orders table (many-to-many: customers -> orders -> products)
    n_orders = 500
    orders = pd.DataFrame({
        "order_id": range(1, n_orders + 1),
        "customer_id": np.random.randint(1, n_customers + 1, n_orders),
        "product_id": np.random.randint(1, n_products + 1, n_orders),
        "order_date": pd.date_range("2020-01-01", periods=n_orders, freq="2H"),
        "quantity": np.random.randint(1, 5, n_orders),
        "discount": np.random.uniform(0, 0.3, n_orders),
    })

    # Calculate order amount
    orders = orders.merge(products[["product_id", "price"]], on="product_id")
    orders["amount"] = orders["quantity"] * orders["price"] * (1 - orders["discount"])
    orders = orders.drop("price", axis=1)

    # Reviews table (one-to-many: orders -> reviews)
    n_reviews = 200
    reviewed_orders = np.random.choice(orders["order_id"].values, n_reviews, replace=False)
    reviews = pd.DataFrame({
        "review_id": range(1, n_reviews + 1),
        "order_id": reviewed_orders,
        "rating": np.random.randint(1, 6, n_reviews),
        "review_text": [f"Review text {i}" for i in range(1, n_reviews + 1)],
    })

    return {
        "customers": customers,
        "products": products,
        "orders": orders,
        "reviews": reviews,
    }


def example_relationship_detection():
    """Example: Automatic relationship detection between tables."""
    print("=" * 60)
    print("Example: Automatic Relationship Detection")
    print("=" * 60)

    tables = create_ecommerce_data()

    print(f"\nTables in dataset:")
    for name, df in tables.items():
        print(f"  - {name}: {len(df)} rows, {len(df.columns)} columns")

    # Auto-detect relationships
    graph = detect_relationships(tables)

    print(f"\nDetected relationships:")
    for rel in graph.get_relationships():
        print(f"  - {rel.parent_table}.{rel.parent_column} -> "
              f"{rel.child_table}.{rel.child_column}")

    # Display graph summary
    print(f"\n{graph.summary()}")


def example_manual_relationships():
    """Example: Manually define relationships between tables."""
    print("\n" + "=" * 60)
    print("Example: Manual Relationship Definition")
    print("=" * 60)

    tables = create_ecommerce_data()

    # Create graph and add tables
    graph = RelationshipGraph()

    for name, df in tables.items():
        primary_key = f"{name.rstrip('s')}_id"  # e.g., customer_id, product_id
        if primary_key in df.columns:
            graph.add_table(name, df, primary_key=primary_key)
        else:
            graph.add_table(name, df)

    # Define relationships manually
    graph.add_relationship(
        parent="customers",
        parent_col="customer_id",
        child="orders",
        child_col="customer_id",
    )
    graph.add_relationship(
        parent="products",
        parent_col="product_id",
        child="orders",
        child_col="product_id",
    )
    graph.add_relationship(
        parent="orders",
        parent_col="order_id",
        child="reviews",
        child_col="order_id",
    )

    print(f"Manually defined {len(graph.get_relationships())} relationships")

    # Find path between tables
    path = graph.find_path("customers", "reviews")
    if path:
        print(f"\nPath from customers to reviews: {' -> '.join(r.child_table for r in path)}")


def example_multi_table_transformer():
    """Example: Generate features from multiple tables."""
    print("\n" + "=" * 60)
    print("Example: Multi-Table Feature Generation")
    print("=" * 60)

    tables = create_ecommerce_data()

    # Create transformer targeting the customers table
    transformer = MultiTableTransformer(
        target_table="customers",
        max_depth=2,           # How deep to traverse relationships
        max_features=30,       # Maximum features to generate
        agg_primitives=["mean", "sum", "count", "min", "max"],  # Aggregation functions
        verbose=1,
    )

    # Generate features
    customer_features = transformer.fit_transform(tables)

    print(f"\nOriginal customer columns: {len(tables['customers'].columns)}")
    print(f"Generated feature columns: {len(customer_features.columns)}")

    print(f"\nSample generated features:")
    feature_names = transformer.get_feature_names_out()
    for name in feature_names[:10]:
        print(f"  - {name}")

    # Show sample data
    print(f"\nSample feature values (first 3 customers):")
    numeric_cols = customer_features.select_dtypes(include=[np.number]).columns[:5]
    print(customer_features[numeric_cols].head(3).to_string())


def example_convenience_function():
    """Example: Quick feature generation with convenience function."""
    print("\n" + "=" * 60)
    print("Example: Convenience Function")
    print("=" * 60)

    tables = create_ecommerce_data()

    # One-line feature generation
    features = multi_table_features(
        tables=tables,
        target_table="customers",
        max_depth=1,
        max_features=20,
        verbose=0,
    )

    print(f"Generated {len(features.columns)} features for {len(features)} customers")


def example_deep_feature_synthesis():
    """Example: Advanced deep feature synthesis."""
    print("\n" + "=" * 60)
    print("Example: Deep Feature Synthesis")
    print("=" * 60)

    tables = create_ecommerce_data()

    # First detect relationships
    graph = detect_relationships(tables)

    # Create DFS instance for fine-grained control
    dfs = DeepFeatureSynthesis(
        graph=graph,
        target_table="customers",
        max_depth=2,
        max_features=50,
        agg_primitives=["mean", "sum", "count", "std", "min", "max"],
        trans_primitives=["day", "month", "year"],  # Date transformations
        verbose=1,
    )

    # Synthesize features
    result = dfs.synthesize()

    print(f"\nSynthesized features: {len(result.feature_names)}")
    print(f"\nFeature definitions:")
    for defn in result.definitions[:5]:
        print(f"  - {defn}")


def example_sklearn_integration():
    """Example: Integration with sklearn pipeline."""
    print("\n" + "=" * 60)
    print("Example: sklearn Integration")
    print("=" * 60)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    tables = create_ecommerce_data()

    # Generate features for customers
    transformer = MultiTableTransformer(
        target_table="customers",
        max_depth=1,
        max_features=25,
        include_target_columns=True,
        verbose=0,
    )

    features = transformer.fit_transform(tables)

    # Create target: predict premium customers
    y = tables["customers"]["is_premium"].astype(int)

    # Select only numeric features
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    X = features[numeric_cols].fillna(0)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nFeatures used: {len(X.columns)}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"\nTop 5 important features:")
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    print(importance.head(5).to_string(index=False))


def example_transform_new_data():
    """Example: Transform new data after fitting."""
    print("\n" + "=" * 60)
    print("Example: Transform New Data")
    print("=" * 60)

    tables = create_ecommerce_data()

    # Fit transformer
    transformer = MultiTableTransformer(
        target_table="customers",
        max_depth=1,
        max_features=15,
        verbose=0,
    )
    transformer.fit(tables)

    # Simulate new data (subset of existing for demo)
    new_tables = {
        "customers": tables["customers"].iloc[:10].copy(),
        "orders": tables["orders"][tables["orders"]["customer_id"] <= 10].copy(),
        "products": tables["products"].copy(),
        "reviews": tables["reviews"].copy(),
    }

    # Transform new data
    new_features = transformer.transform(new_tables)

    print(f"Original training customers: {len(tables['customers'])}")
    print(f"New customers to transform: {len(new_tables['customers'])}")
    print(f"Generated features: {len(new_features.columns)}")


if __name__ == "__main__":
    example_relationship_detection()
    example_manual_relationships()
    example_multi_table_transformer()
    example_convenience_function()
    example_deep_feature_synthesis()
    example_sklearn_integration()
    example_transform_new_data()

    print("\n" + "=" * 60)
    print("All multi-table examples completed!")
    print("=" * 60)
