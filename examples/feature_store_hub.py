#!/usr/bin/env python
"""Feature store hub example for Forge.

This example demonstrates the create_feature_store factory,
LocalRegistry, and the registry pattern for managing feature
definitions, versions, and materialized data.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from forge.registry import (
    FeatureDefinition, FeatureSet, LocalRegistry, RegistryConfig,
)
from forge.registry.base import FeatureType, create_feature_definitions_from_dataframe
from forge.registry.hub import create_feature_store


def example_local_registry():
    """Example: Use LocalRegistry for development feature management."""
    print("=" * 60)
    print("Example: LocalRegistry for Feature Management")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = RegistryConfig(
            registry_type="local", project_name="demo_project",
            connection_params={"path": str(Path(tmpdir) / "registry")},
            default_entity="customer",
        )
        registry = LocalRegistry(config)
        registry.connect()

        features = [
            FeatureDefinition(name="log_income", dtype=FeatureType.FLOAT64,
                              description="Log-transformed income",
                              source_columns=["income"], entity="customer", owner="ml-team"),
            FeatureDefinition(name="age_group", dtype=FeatureType.STRING,
                              description="Binned age category",
                              source_columns=["age"], entity="customer", owner="ml-team"),
            FeatureDefinition(name="purchase_count", dtype=FeatureType.INT64,
                              description="Total purchases",
                              source_columns=["orders"], entity="customer", owner="data-eng"),
        ]

        print(f"\nRegistering {len(features)} features...")
        for feat in features:
            version = registry.register_feature(feat)
            print(f"  {feat.name}: version={version.version}, hash={version.feature_hash}")

        print(f"\nRegistered features: {registry.list_features()}")
        feat = registry.get_feature("log_income")
        print(f"\nRetrieved 'log_income': {feat.description} (owner={feat.owner})")
        registry.disconnect()


def example_feature_sets():
    """Example: Group features into FeatureSets and materialize data."""
    print("\n" + "=" * 60)
    print("Example: FeatureSets and Data Materialization")
    print("=" * 60)

    np.random.seed(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        config = RegistryConfig(
            registry_type="local", project_name="ecommerce",
            connection_params={"path": str(Path(tmpdir) / "registry")},
        )
        registry = LocalRegistry(config)
        registry.connect()

        feature_set = FeatureSet(
            name="customer_features", entity="customer",
            description="Core customer features for churn model",
            features=[
                FeatureDefinition(name="total_spend", dtype=FeatureType.FLOAT64,
                                  entity="customer"),
                FeatureDefinition(name="avg_order_value", dtype=FeatureType.FLOAT64,
                                  entity="customer"),
            ],
        )

        version = registry.register_feature_set(feature_set)
        print(f"\nRegistered '{feature_set.name}' (v={version}): {feature_set.feature_names}")

        n = 100
        data = pd.DataFrame({
            "customer_id": [f"C{i:04d}" for i in range(n)],
            "total_spend": np.random.lognormal(6, 1, n),
            "avg_order_value": np.random.lognormal(3, 0.5, n),
        })
        rows = registry.materialize("customer_features", data)
        print(f"  Materialized {rows} rows")

        entity_df = pd.DataFrame({"customer_id": ["C0001", "C0010", "C0050"]})
        result = registry.get_historical_features(
            entity_df=entity_df, features=["total_spend", "avg_order_value"],
            entity_column="customer_id",
        )
        print(f"\nHistorical features:\n{result.to_string(index=False)}")
        registry.disconnect()


def example_factory_pattern():
    """Example: Use create_feature_store factory for different backends."""
    print("\n" + "=" * 60)
    print("Example: create_feature_store Factory Pattern")
    print("=" * 60)

    np.random.seed(42)
    df = pd.DataFrame({
        "user_id": range(50),
        "score": np.random.normal(0, 1, 50),
        "amount": np.random.lognormal(3, 1, 50),
        "is_active": np.random.choice([True, False], 50),
    })

    definitions = create_feature_definitions_from_dataframe(
        df, entity="user", exclude_columns=["user_id"]
    )
    print(f"\nAuto-detected {len(definitions)} features from DataFrame:")
    for fd in definitions:
        print(f"  {fd.name}: dtype={fd.dtype.value}")

    # Factory pattern: each backend requires its SDK
    for name in ["tecton", "hopsworks", "databricks", "sagemaker"]:
        try:
            store = create_feature_store(name)
            print(f"  Created {name}: {type(store).__name__}")
        except Exception as e:
            print(f"  {name}: {type(e).__name__} (SDK not installed, expected)")


if __name__ == "__main__":
    example_local_registry()
    example_feature_sets()
    example_factory_pattern()

    print("\n" + "=" * 60)
    print("All feature store hub examples completed!")
    print("=" * 60)
