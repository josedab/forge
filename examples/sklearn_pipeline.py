#!/usr/bin/env python
"""Example of using Forge with scikit-learn pipelines.

This example demonstrates how to integrate Forge's AutoFeatureTransformer
with scikit-learn's Pipeline and model selection tools.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from forge import AutoFeatureTransformer


def create_sample_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create sample dataset for demonstration."""
    np.random.seed(42)
    n_samples = 2000

    data = {
        "age": np.random.randint(18, 80, n_samples),
        "income": np.random.lognormal(10, 1, n_samples),
        "credit_score": np.random.normal(700, 50, n_samples),
        "years_employed": np.random.exponential(5, n_samples),
        "num_accounts": np.random.poisson(3, n_samples),
        "education": np.random.choice(
            ["high_school", "bachelors", "masters", "phd"], n_samples
        ),
        "employment_type": np.random.choice(
            ["full_time", "part_time", "self_employed"], n_samples
        ),
        "region": np.random.choice(["north", "south", "east", "west"], n_samples),
    }

    X = pd.DataFrame(data)

    # Create target
    prob = (
        0.25
        + 0.2 * (X["income"] > X["income"].median()).astype(float)
        + 0.2 * (X["credit_score"] > 700).astype(float)
        + 0.15 * (X["education"].isin(["masters", "phd"])).astype(float)
        + 0.1 * (X["years_employed"] > 3).astype(float)
    )
    y = pd.Series((np.random.random(n_samples) < prob).astype(int), name="approved")

    return X, y


def example_basic_pipeline():
    """Example: Basic pipeline with Forge and classifier."""
    print("=" * 60)
    print("Example: Basic Pipeline")
    print("=" * 60)

    X, y = create_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create pipeline
    pipeline = Pipeline(
        [
            ("features", AutoFeatureTransformer(max_features=30, verbose=0)),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # Fit and predict
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")


def example_cross_validation():
    """Example: Cross-validation with pipeline."""
    print("\n" + "=" * 60)
    print("Example: Cross-Validation")
    print("=" * 60)

    X, y = create_sample_data()

    pipeline = Pipeline(
        [
            ("features", AutoFeatureTransformer(max_features=20, verbose=0)),
            ("classifier", RandomForestClassifier(n_estimators=50, random_state=42)),
        ]
    )

    # 5-fold cross-validation
    scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")

    print(f"\nCross-validation ROC-AUC scores: {scores}")
    print(f"Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")


def example_multiple_classifiers():
    """Example: Compare different classifiers."""
    print("\n" + "=" * 60)
    print("Example: Comparing Classifiers")
    print("=" * 60)

    X, y = create_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create feature transformer once
    feature_transformer = AutoFeatureTransformer(max_features=25, verbose=0)
    X_train_features = feature_transformer.fit_transform(X_train, y_train)
    X_test_features = feature_transformer.transform(X_test)

    # Scale for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_test_scaled = scaler.transform(X_test_features)

    classifiers = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
    }

    print(f"\nFeatures generated: {X_train_features.shape[1]}")
    print("\nModel comparison:")
    print("-" * 50)

    for name, clf in classifiers.items():
        # Use scaled data for logistic regression
        if name == "Logistic Regression":
            clf.fit(X_train_scaled, y_train)
            y_proba = clf.predict_proba(X_test_scaled)[:, 1]
        else:
            clf.fit(X_train_features, y_train)
            y_proba = clf.predict_proba(X_test_features)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        print(f"{name:25s}: ROC-AUC = {auc:.4f}")


def example_grid_search():
    """Example: Grid search with pipeline."""
    print("\n" + "=" * 60)
    print("Example: Grid Search")
    print("=" * 60)

    X, y = create_sample_data()

    # Create pipeline with named steps for grid search
    pipeline = Pipeline(
        [
            ("features", AutoFeatureTransformer(verbose=0)),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    # Define parameter grid
    param_grid = {
        "features__max_features": [10, 20, 30],
        "classifier__n_estimators": [50, 100],
        "classifier__max_depth": [5, 10, None],
    }

    # Grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X, y)

    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best ROC-AUC: {grid_search.best_score_:.4f}")


def example_pipeline_with_scaling():
    """Example: Pipeline with feature scaling."""
    print("\n" + "=" * 60)
    print("Example: Pipeline with Scaling")
    print("=" * 60)

    X, y = create_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pipeline: Feature engineering -> Scaling -> Classifier
    pipeline = Pipeline(
        [
            ("features", AutoFeatureTransformer(max_features=25, verbose=0)),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print(f"\nLogistic Regression ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    # Show feature coefficients
    feature_names = pipeline.named_steps["features"].get_feature_names_out()
    coefficients = pipeline.named_steps["classifier"].coef_[0]

    # Get top features by absolute coefficient
    feature_importance = pd.DataFrame(
        {"feature": feature_names, "coefficient": coefficients}
    )
    feature_importance["abs_coef"] = feature_importance["coefficient"].abs()
    feature_importance = feature_importance.sort_values("abs_coef", ascending=False)

    print("\nTop 10 features by coefficient magnitude:")
    print(feature_importance.head(10)[["feature", "coefficient"]].to_string(index=False))


def example_feature_analysis():
    """Example: Analyze generated features."""
    print("\n" + "=" * 60)
    print("Example: Feature Analysis")
    print("=" * 60)

    X, y = create_sample_data()

    # Create transformer with specific settings
    transformer = AutoFeatureTransformer(
        max_features=20,
        numeric_transformations=["log", "sqrt"],
        categorical_encoding="target",
        selection_method="importance",
        verbose=0,
    )

    X_transformed = transformer.fit_transform(X, y)

    print(f"\nOriginal features: {X.shape[1]}")
    print(f"Generated features: {X_transformed.shape[1]}")

    # Get importance
    importance = transformer.get_feature_importance()
    print("\nFeature importance ranking:")
    print(importance.to_string(index=False))


if __name__ == "__main__":
    example_basic_pipeline()
    example_cross_validation()
    example_multiple_classifiers()
    example_grid_search()
    example_pipeline_with_scaling()
    example_feature_analysis()

    print("\n" + "=" * 60)
    print("All sklearn pipeline examples completed!")
    print("=" * 60)
