#!/usr/bin/env python
"""Domain-specific feature packs example for Forge.

This example demonstrates FinanceFeaturePack, HealthcareFeaturePack,
EcommerceFeaturePack, and GeospatialFeaturePack for generating
industry-specific features from raw data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forge.packs import (
    EcommerceFeaturePack,
    FeaturePackRegistry,
    FinanceFeaturePack,
    GeospatialFeaturePack,
    HealthcareFeaturePack,
)


def example_finance_pack():
    """Example: Generate financial technical indicators."""
    print("=" * 60)
    print("Example: FinanceFeaturePack")
    print("=" * 60)

    np.random.seed(42)
    n = 200
    prices = 100 + np.cumsum(np.random.normal(0, 1, n))
    stock_data = pd.DataFrame({
        "close": prices,
        "volume": np.random.randint(1000, 50000, n).astype(float),
    })

    pack = FinanceFeaturePack(windows=[5, 10, 20])
    features = pack.fit_transform(stock_data)

    print(f"\nInput: {stock_data.shape[0]} trading days, {stock_data.shape[1]} columns")
    print(f"Generated: {features.shape[1]} financial features\n")
    print(f"Available features: {pack.available_features()[:8]}...")
    print(f"\nSample feature values (last row):")
    for col in list(features.columns)[:8]:
        val = features[col].iloc[-1]
        print(f"  {col:30s} = {val:.4f}" if not np.isnan(val) else f"  {col:30s} = NaN")


def example_healthcare_pack():
    """Example: Generate healthcare clinical features."""
    print("\n" + "=" * 60)
    print("Example: HealthcareFeaturePack")
    print("=" * 60)

    np.random.seed(42)
    n = 300
    patient_data = pd.DataFrame({
        "age": np.random.randint(18, 90, n).astype(float),
        "weight": np.random.normal(75, 15, n),
        "height": np.random.normal(170, 10, n),
        "systolic": np.random.normal(130, 20, n),
        "diastolic": np.random.normal(80, 10, n),
        "lab_glucose": np.random.normal(100, 25, n),
        "lab_cholesterol": np.random.normal(200, 40, n),
    })

    pack = HealthcareFeaturePack()
    features = pack.fit_transform(patient_data)

    print(f"\nInput: {n} patients, {patient_data.shape[1]} raw columns")
    print(f"Generated: {features.shape[1]} clinical features\n")
    print(f"Feature names:")
    for col in features.columns:
        print(f"  - {col}")


def example_ecommerce_pack():
    """Example: Generate e-commerce RFM and behavioral features."""
    print("\n" + "=" * 60)
    print("Example: EcommerceFeaturePack")
    print("=" * 60)

    np.random.seed(42)
    n = 500
    transactions = pd.DataFrame({
        "customer_id": np.random.choice([f"C{i:03d}" for i in range(50)], n),
        "amount": np.random.lognormal(3, 1, n),
        "product_id": np.random.choice([f"P{i:03d}" for i in range(20)], n),
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="2h"),
    })

    pack = EcommerceFeaturePack(customer_col="customer_id", amount_col="amount")
    features = pack.fit_transform(transactions)

    print(f"\nInput: {n} transactions from {transactions['customer_id'].nunique()} customers")
    print(f"Generated: {features.shape[1]} e-commerce features\n")
    print(f"Feature columns:")
    for col in features.columns:
        print(f"  - {col}")

    print(f"\nSample values (first row):")
    for col in list(features.columns)[:5]:
        val = features[col].iloc[0]
        print(f"  {col:35s} = {val:.2f}")


def example_geospatial_pack():
    """Example: Generate geospatial features from coordinates."""
    print("\n" + "=" * 60)
    print("Example: GeospatialFeaturePack")
    print("=" * 60)

    np.random.seed(42)
    n = 200
    location_data = pd.DataFrame({
        "latitude": np.random.uniform(40.5, 41.0, n),
        "longitude": np.random.uniform(-74.2, -73.7, n),
    })

    pack = GeospatialFeaturePack(
        reference_points={
            "times_square": (40.7580, -73.9855),
            "jfk_airport": (40.6413, -73.7781),
        },
        n_clusters=4,
    )
    features = pack.fit_transform(location_data)

    print(f"\nInput: {n} locations around NYC")
    print(f"Generated: {features.shape[1]} geospatial features\n")
    print(f"Feature columns:")
    for col in features.columns:
        val = features[col].iloc[0]
        print(f"  {col:40s} = {val:.4f}")


if __name__ == "__main__":
    example_finance_pack()
    example_healthcare_pack()
    example_ecommerce_pack()
    example_geospatial_pack()

    print("\n" + "=" * 60)
    print("All domain pack examples completed!")
    print("=" * 60)
