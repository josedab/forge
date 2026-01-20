#!/usr/bin/env python
"""Basic usage example for Forge.

This example demonstrates the core functionality of Forge for
automated feature engineering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forge import AutoFeatureTransformer, DataAnalyzer


def create_sample_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create sample dataset for demonstration."""
    np.random.seed(42)
    n_samples = 1000

    # Create features
    data = {
        # Numeric features
        "age": np.random.randint(18, 80, n_samples),
        "income": np.random.lognormal(10, 1, n_samples),
        "credit_score": np.random.normal(700, 50, n_samples),
        "years_employed": np.random.exponential(5, n_samples),
        # Categorical features
        "education": np.random.choice(
            ["high_school", "bachelors", "masters", "phd"], n_samples
        ),
        "employment_type": np.random.choice(
            ["full_time", "part_time", "self_employed", "unemployed"], n_samples
        ),
        "region": np.random.choice(
            ["north", "south", "east", "west"], n_samples
        ),
    }

    X = pd.DataFrame(data)

    # Create target (loan approval based on features)
    prob = (
        0.3
        + 0.2 * (X["income"] > X["income"].median()).astype(float)
        + 0.2 * (X["credit_score"] > 700).astype(float)
        + 0.1 * (X["education"].isin(["masters", "phd"])).astype(float)
        + 0.1 * (X["years_employed"] > 3).astype(float)
    )
    y = pd.Series((np.random.random(n_samples) < prob).astype(int), name="approved")

    return X, y


def example_data_analysis():
    """Example: Analyze data before feature engineering."""
    print("=" * 60)
    print("Example: Data Analysis")
    print("=" * 60)

    X, y = create_sample_data()

    # Create analyzer
    analyzer = DataAnalyzer()

    # Analyze data
    report = analyzer.analyze(X, y)

    print(f"\nDataset shape: {X.shape}")
    print(f"\nColumn types detected:")
    for col_info in report.columns.values():
        print(f"  - {col_info.name}: {col_info.type.value}")

    print(f"\nTarget distribution:")
    print(y.value_counts(normalize=True))

    # Get quality assessment
    quality = analyzer.assess_quality(X)
    print(f"\nData quality issues: {len(quality.get('issues', []))}")


def example_basic_transformation():
    """Example: Basic feature transformation."""
    print("\n" + "=" * 60)
    print("Example: Basic Feature Transformation")
    print("=" * 60)

    X, y = create_sample_data()

    # Create transformer
    transformer = AutoFeatureTransformer(
        max_features=30,
        verbose=1,
    )

    # Fit and transform
    X_transformed = transformer.fit_transform(X, y)

    print(f"\nOriginal features: {X.shape[1]}")
    print(f"Transformed features: {X_transformed.shape[1]}")

    # Show feature names
    print("\nGenerated features (first 10):")
    for name in transformer.get_feature_names_out()[:10]:
        print(f"  - {name}")


def example_custom_transformations():
    """Example: Custom transformation settings."""
    print("\n" + "=" * 60)
    print("Example: Custom Transformation Settings")
    print("=" * 60)

    X, y = create_sample_data()

    # Configure specific transformations
    transformer = AutoFeatureTransformer(
        # Numeric transformations
        numeric_transformations=["log", "sqrt", "square"],
        # Categorical encoding
        categorical_encoding="target",
        # Missing value handling
        missing_strategy="auto",
        # Feature selection
        selection_method="importance",
        max_features=20,
        # Reproducibility
        random_state=42,
        verbose=1,
    )

    X_transformed = transformer.fit_transform(X, y)

    print(f"\nFeatures after transformation: {X_transformed.shape[1]}")


def example_feature_importance():
    """Example: Analyze feature importance."""
    print("\n" + "=" * 60)
    print("Example: Feature Importance Analysis")
    print("=" * 60)

    X, y = create_sample_data()

    transformer = AutoFeatureTransformer(
        max_features=20,
        verbose=0,
    )
    transformer.fit(X, y)

    # Get feature importance
    importance = transformer.get_feature_importance()

    print("\nTop 10 most important features:")
    print(importance.head(10).to_string(index=False))


def example_transform_new_data():
    """Example: Transform new data after fitting."""
    print("\n" + "=" * 60)
    print("Example: Transform New Data")
    print("=" * 60)

    X, y = create_sample_data()

    # Split into train/test
    train_size = int(0.8 * len(X))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train = y.iloc[:train_size]

    # Fit on training data only
    transformer = AutoFeatureTransformer(max_features=15, verbose=0)
    X_train_transformed = transformer.fit_transform(X_train, y_train)

    # Transform test data (uses learned parameters)
    X_test_transformed = transformer.transform(X_test)

    print(f"Training features: {X_train_transformed.shape}")
    print(f"Test features: {X_test_transformed.shape}")
    print(f"\nFeature columns match: {list(X_train_transformed.columns) == list(X_test_transformed.columns)}")


if __name__ == "__main__":
    example_data_analysis()
    example_basic_transformation()
    example_custom_transformations()
    example_feature_importance()
    example_transform_new_data()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
