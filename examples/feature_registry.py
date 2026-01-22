#!/usr/bin/env python
"""Feature registry usage example.

This example demonstrates how to use Forge's feature registry capabilities
to manage, version, and serve feature definitions in a production-like
environment.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from forge.registry import (
    LocalRegistry,
    FeatureDefinition,
    FeatureSet,
    RegistryConfig,
)
from forge.registry.base import FeatureType
from forge.registry.local_registry import create_local_registry, create_registry


def create_sample_feature_data() -> pd.DataFrame:
    """Create sample feature data for materialization."""
    np.random.seed(42)
    n_customers = 100

    return pd.DataFrame({
        "customer_id": range(1, n_customers + 1),
        "total_spend": np.random.uniform(100, 5000, n_customers),
        "order_count": np.random.randint(1, 50, n_customers),
        "avg_order_value": np.random.uniform(50, 200, n_customers),
        "days_since_last_order": np.random.randint(1, 365, n_customers),
        "event_timestamp": pd.date_range("2024-01-01", periods=n_customers, freq="D"),
    })


def example_basic_registry_setup():
    """Example: Setting up a local feature registry."""
    print("=" * 60)
    print("Example: Basic Registry Setup")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = Path(tmpdir) / "my_feature_registry"

        # Method 1: Using convenience function (recommended)
        registry = create_local_registry(
            path=str(registry_path),
            project_name="ecommerce_features",
        )

        print(f"Registry created at: {registry_path}")
        print(f"Project name: {registry.config.project_name}")
        print(f"Connected: {registry._connected}")

        registry.disconnect()


def example_define_features():
    """Example: Defining feature metadata."""
    print("\n" + "=" * 60)
    print("Example: Feature Definitions")
    print("=" * 60)

    # Define individual features with metadata
    total_spend = FeatureDefinition(
        name="total_spend",
        dtype=FeatureType.FLOAT64,
        description="Total customer spending in the last 12 months",
        source_columns=["order_amount"],
        transformation="sum",
        entity="customer",
        tags=["financial", "aggregation", "ml-ready"],
    )

    order_count = FeatureDefinition(
        name="order_count",
        dtype=FeatureType.INT64,
        description="Number of orders placed by customer",
        source_columns=["order_id"],
        transformation="count",
        entity="customer",
        tags=["behavioral", "aggregation"],
    )

    avg_order_value = FeatureDefinition(
        name="avg_order_value",
        dtype=FeatureType.FLOAT64,
        description="Average order value for customer",
        source_columns=["order_amount"],
        transformation="mean",
        entity="customer",
        tags=["financial", "aggregation"],
    )

    days_since_last_order = FeatureDefinition(
        name="days_since_last_order",
        dtype=FeatureType.INT64,
        description="Days since customer's last order",
        source_columns=["order_date"],
        transformation="recency",
        entity="customer",
        tags=["temporal", "engagement"],
    )

    print("Defined features:")
    for feature in [total_spend, order_count, avg_order_value, days_since_last_order]:
        print(f"  - {feature.name} ({feature.dtype.value}): {feature.description[:50]}...")

    return [total_spend, order_count, avg_order_value, days_since_last_order]


def example_feature_sets():
    """Example: Grouping features into feature sets."""
    print("\n" + "=" * 60)
    print("Example: Feature Sets")
    print("=" * 60)

    features = example_define_features()

    # Create a feature set
    customer_features = FeatureSet(
        name="customer_behavioral_features",
        features=features,
        entity="customer",
        description="Customer behavioral and financial features for ML models",
        tags=["production", "ml-ready"],
    )

    print(f"\nFeature Set: {customer_features.name}")
    print(f"  Entity: {customer_features.entity}")
    print(f"  Features: {len(customer_features.features)}")
    print(f"  Description: {customer_features.description}")

    return customer_features


def example_register_features():
    """Example: Registering features in the registry."""
    print("\n" + "=" * 60)
    print("Example: Feature Registration")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        registry = create_local_registry(
            path=str(Path(tmpdir) / "registry"),
            project_name="demo",
        )

        # Define a feature
        feature = FeatureDefinition(
            name="customer_ltv",
            dtype=FeatureType.FLOAT64,
            description="Customer lifetime value prediction",
            entity="customer",
        )

        # Register single feature
        version = registry.register_feature(feature)
        print(f"Registered feature: {feature.name}")
        print(f"  Version: {version.version}")

        # Register a feature set
        feature_set = FeatureSet(
            name="ltv_features",
            features=[
                feature,
                FeatureDefinition(
                    name="purchase_frequency",
                    dtype=FeatureType.FLOAT64,
                    entity="customer",
                ),
            ],
            entity="customer",
        )

        fs_version = registry.register_feature_set(feature_set)
        print(f"\nRegistered feature set: {feature_set.name}")
        print(f"  Version: {fs_version}")

        registry.disconnect()


def example_list_and_retrieve():
    """Example: Listing and retrieving features."""
    print("\n" + "=" * 60)
    print("Example: List and Retrieve Features")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        registry = create_local_registry(
            path=str(Path(tmpdir) / "registry"),
            project_name="demo",
        )

        # Register some features
        feature_set = example_feature_sets()
        registry.register_feature_set(feature_set)

        # List all features
        all_features = registry.list_features()
        print(f"All registered features: {all_features}")

        # List features by entity
        customer_features = registry.list_features(entity="customer")
        print(f"Customer features: {customer_features}")

        # List feature sets
        all_sets = registry.list_feature_sets()
        print(f"Feature sets: {all_sets}")

        # Get specific feature
        feature = registry.get_feature("total_spend")
        if feature:
            print(f"\nRetrieved feature: {feature.name}")
            print(f"  Type: {feature.dtype.value}")
            print(f"  Entity: {feature.entity}")

        # Get feature set
        fs = registry.get_feature_set("customer_behavioral_features")
        if fs:
            print(f"\nRetrieved feature set: {fs.name}")
            print(f"  Contains {len(fs.features)} features")

        registry.disconnect()


def example_materialize_data():
    """Example: Materializing feature data to the registry."""
    print("\n" + "=" * 60)
    print("Example: Data Materialization")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        registry = create_local_registry(
            path=str(Path(tmpdir) / "registry"),
            project_name="demo",
        )

        # Register feature set
        feature_set = example_feature_sets()
        registry.register_feature_set(feature_set)

        # Create feature data
        data = create_sample_feature_data()
        print(f"Feature data shape: {data.shape}")

        # Materialize to registry
        rows = registry.materialize("customer_behavioral_features", data)
        print(f"Materialized {rows} rows to registry")

        registry.disconnect()


def example_historical_features():
    """Example: Retrieving historical features for training."""
    print("\n" + "=" * 60)
    print("Example: Historical Feature Retrieval")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        registry = create_local_registry(
            path=str(Path(tmpdir) / "registry"),
            project_name="demo",
        )

        # Register and materialize
        feature_set = example_feature_sets()
        registry.register_feature_set(feature_set)

        data = create_sample_feature_data()
        registry.materialize("customer_behavioral_features", data)

        # Create entity DataFrame for point-in-time lookup
        entity_df = pd.DataFrame({
            "customer_id": [1, 5, 10, 20, 50],
        })

        # Get historical features
        features = registry.get_historical_features(
            entity_df=entity_df,
            features=["total_spend", "order_count", "avg_order_value"],
            entity_column="customer_id",
        )

        print(f"Retrieved features for {len(entity_df)} entities")
        print(f"\nFeature data:")
        print(features.to_string(index=False))

        registry.disconnect()


def example_export_import():
    """Example: Export and import feature definitions."""
    print("\n" + "=" * 60)
    print("Example: Export/Import Definitions")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        registry1_path = Path(tmpdir) / "registry1"
        registry2_path = Path(tmpdir) / "registry2"
        export_path = Path(tmpdir) / "feature_definitions.json"

        # Create first registry and register features
        registry1 = create_local_registry(
            path=str(registry1_path),
            project_name="project1",
        )

        feature_set = example_feature_sets()
        registry1.register_feature_set(feature_set)

        # Export definitions
        registry1.export_definitions(str(export_path))
        print(f"Exported definitions to: {export_path.name}")
        registry1.disconnect()

        # Create second registry and import
        registry2 = create_local_registry(
            path=str(registry2_path),
            project_name="project2",
        )

        registry2.import_definitions(str(export_path))
        print(f"Imported definitions to new registry")

        # Verify import
        features = registry2.list_features()
        print(f"Features in new registry: {features}")

        registry2.disconnect()


def example_versioning():
    """Example: Feature versioning workflow."""
    print("\n" + "=" * 60)
    print("Example: Feature Versioning")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        registry = create_local_registry(
            path=str(Path(tmpdir) / "registry"),
            project_name="demo",
        )

        # Version 1
        feature_v1 = FeatureDefinition(
            name="churn_score",
            dtype=FeatureType.FLOAT64,
            description="Customer churn probability - initial model",
            entity="customer",
        )
        v1 = registry.register_feature(feature_v1)
        print(f"Registered v1: {v1.version}")

        # Version 2 (updated definition)
        feature_v2 = FeatureDefinition(
            name="churn_score",
            dtype=FeatureType.FLOAT64,
            description="Customer churn probability - improved model with more features",
            entity="customer",
        )
        v2 = registry.register_feature(feature_v2)
        print(f"Registered v2: {v2.version}")

        # Version 3 (another update)
        feature_v3 = FeatureDefinition(
            name="churn_score",
            dtype=FeatureType.FLOAT64,
            description="Customer churn probability - production-ready",
            entity="customer",
        )
        v3 = registry.register_feature(feature_v3)
        print(f"Registered v3: {v3.version}")

        # Check internal version history
        versions = registry._versions.get("churn_score", [])
        print(f"\nVersion history: {[v.version for v in versions]}")

        registry.disconnect()


def example_context_manager():
    """Example: Using registry as context manager."""
    print("\n" + "=" * 60)
    print("Example: Context Manager Pattern")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = RegistryConfig(
            registry_type="local",
            project_name="context_demo",
            connection_params={"path": str(Path(tmpdir) / "registry")},
        )

        # Using context manager (auto-connects and disconnects)
        with LocalRegistry(config) as registry:
            feature = FeatureDefinition(
                name="session_duration",
                dtype=FeatureType.FLOAT64,
                entity="session",
            )
            registry.register_feature(feature)
            print(f"Registered feature within context")
            print(f"Registry connected: {registry._connected}")

        print(f"Registry connected after context: {registry._connected}")


def example_full_workflow():
    """Example: Complete feature engineering workflow with registry."""
    print("\n" + "=" * 60)
    print("Example: Complete Workflow")
    print("=" * 60)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Create registry
        registry = create_local_registry(
            path=str(Path(tmpdir) / "feature_store"),
            project_name="ecommerce_ml",
        )
        print("Step 1: Created feature registry")

        # Step 2: Define features
        features = [
            FeatureDefinition(
                name="total_spend",
                dtype=FeatureType.FLOAT64,
                description="Total customer spending",
                entity="customer",
            ),
            FeatureDefinition(
                name="order_count",
                dtype=FeatureType.INT64,
                description="Number of orders",
                entity="customer",
            ),
            FeatureDefinition(
                name="avg_order_value",
                dtype=FeatureType.FLOAT64,
                description="Average order value",
                entity="customer",
            ),
        ]

        feature_set = FeatureSet(
            name="customer_ml_features",
            features=features,
            entity="customer",
        )
        registry.register_feature_set(feature_set)
        print("Step 2: Registered feature definitions")

        # Step 3: Materialize feature data
        data = create_sample_feature_data()
        registry.materialize("customer_ml_features", data)
        print("Step 3: Materialized feature data")

        # Step 4: Retrieve features for training
        entity_df = pd.DataFrame({
            "customer_id": data["customer_id"].values,
        })

        training_features = registry.get_historical_features(
            entity_df=entity_df,
            features=["total_spend", "order_count", "avg_order_value"],
            entity_column="customer_id",
        )
        print("Step 4: Retrieved training features")

        # Step 5: Train model (simplified example)
        # Create synthetic target
        y = (data["total_spend"] > data["total_spend"].median()).astype(int)

        X = training_features[["total_spend", "order_count", "avg_order_value"]].fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        print(f"Step 5: Trained model - Accuracy: {accuracy:.3f}")

        # Step 6: Export for production
        export_path = Path(tmpdir) / "production_features.json"
        registry.export_definitions(str(export_path))
        print(f"Step 6: Exported definitions for production")

        registry.disconnect()
        print("\nWorkflow complete!")


if __name__ == "__main__":
    example_basic_registry_setup()
    example_define_features()
    example_feature_sets()
    example_register_features()
    example_list_and_retrieve()
    example_materialize_data()
    example_historical_features()
    example_export_import()
    example_versioning()
    example_context_manager()
    example_full_workflow()

    print("\n" + "=" * 60)
    print("All feature registry examples completed!")
    print("=" * 60)
