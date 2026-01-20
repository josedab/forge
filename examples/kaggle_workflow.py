#!/usr/bin/env python
"""Example Kaggle competition workflow using Forge.

This example demonstrates a typical Kaggle competition workflow:
1. Data exploration and analysis
2. Feature engineering with Forge
3. Model training and validation
4. Ensemble and submission preparation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from forge import AutoFeatureTransformer, DataAnalyzer
from forge.selectors import ImportanceSelector
from forge.transformers import ForgePipeline


def create_kaggle_like_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Create data that simulates a Kaggle competition dataset."""
    np.random.seed(42)

    n_train = 5000
    n_test = 2000

    def generate_features(n: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                # Numeric features
                "feature_1": np.random.normal(0, 1, n),
                "feature_2": np.random.exponential(2, n),
                "feature_3": np.random.lognormal(0, 0.5, n),
                "feature_4": np.random.uniform(0, 100, n),
                "feature_5": np.random.poisson(5, n),
                "feature_6": np.random.normal(50, 10, n),
                "feature_7": np.random.gamma(2, 2, n),
                "feature_8": np.random.beta(2, 5, n) * 100,
                # Categorical features
                "cat_1": np.random.choice(["A", "B", "C", "D"], n),
                "cat_2": np.random.choice(["type_1", "type_2", "type_3"], n),
                "cat_3": np.random.choice(["low", "medium", "high"], n),
                # Features with missing values
                "feature_missing_1": np.where(
                    np.random.random(n) > 0.1, np.random.normal(0, 1, n), np.nan
                ),
                "feature_missing_2": np.where(
                    np.random.random(n) > 0.2, np.random.normal(0, 1, n), np.nan
                ),
            }
        )

    X_train = generate_features(n_train)
    X_test = generate_features(n_test)

    # Create target based on features (with some noise)
    def compute_target(X: pd.DataFrame) -> pd.Series:
        signal = (
            0.3 * X["feature_1"]
            + 0.2 * np.log1p(X["feature_2"])
            + 0.15 * (X["cat_1"] == "A").astype(float)
            + 0.1 * (X["feature_4"] > 50).astype(float)
            + 0.1 * X["feature_6"] / 100
            + np.random.normal(0, 0.2, len(X))
        )
        return pd.Series((signal > signal.median()).astype(int), name="target")

    y_train = compute_target(X_train)

    return X_train, X_test, y_train


def step1_explore_data(X_train: pd.DataFrame, y_train: pd.Series):
    """Step 1: Explore and understand the data."""
    print("=" * 60)
    print("Step 1: Data Exploration")
    print("=" * 60)

    # Basic info
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Target distribution:\n{y_train.value_counts(normalize=True)}")

    # Use Forge analyzer
    analyzer = DataAnalyzer()
    report = analyzer.analyze(X_train, y_train)

    print("\nColumn types detected:")
    for col, info in report.columns.items():
        print(f"  {col}: {info.type.value}")

    # Quality assessment
    quality = analyzer.assess_quality(X_train)

    if "missing" in quality:
        print("\nMissing values:")
        for col, info in quality["missing"].items():
            print(f"  {col}: {info['percentage']:.1f}%")


def step2_feature_engineering(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, AutoFeatureTransformer]:
    """Step 2: Automated feature engineering."""
    print("\n" + "=" * 60)
    print("Step 2: Feature Engineering")
    print("=" * 60)

    # Create transformer with competition-tuned settings
    transformer = AutoFeatureTransformer(
        # Generate many features initially
        max_features=50,
        # Use multiple numeric transformations
        numeric_transformations=["log", "sqrt", "square"],
        # Target encoding for categories (common in Kaggle)
        categorical_encoding="target",
        # Handle missing values
        missing_strategy="auto",
        # Use importance-based selection
        selection_method="importance",
        # Reproducibility
        random_state=42,
        verbose=1,
    )

    # Fit on training data
    X_train_fe = transformer.fit_transform(X_train, y_train)

    # Transform test data
    X_test_fe = transformer.transform(X_test)

    print(f"\nOriginal features: {X_train.shape[1]}")
    print(f"Engineered features: {X_train_fe.shape[1]}")

    # Show top features
    importance = transformer.get_feature_importance()
    print("\nTop 10 features:")
    print(importance.head(10).to_string(index=False))

    return X_train_fe, X_test_fe, transformer


def step3_model_validation(
    X_train: pd.DataFrame, y_train: pd.Series
) -> dict[str, float]:
    """Step 3: Cross-validation with multiple models."""
    print("\n" + "=" * 60)
    print("Step 3: Model Validation")
    print("=" * 60)

    # Scale for linear models
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print("\nCross-validation results:")
    print("-" * 50)

    for name, model in models.items():
        # Use scaled data for linear models
        if name == "Logistic Regression":
            X_for_model = X_scaled
        else:
            X_for_model = X_train.values

        # Get cross-validated predictions
        y_pred_proba = cross_val_predict(
            model, X_for_model, y_train, cv=cv, method="predict_proba"
        )[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        auc = roc_auc_score(y_train, y_pred_proba)
        acc = accuracy_score(y_train, y_pred)
        f1 = f1_score(y_train, y_pred)

        results[name] = {"auc": auc, "accuracy": acc, "f1": f1}

        print(f"{name:25s} | AUC: {auc:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

    return results


def step4_ensemble_and_predict(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series
) -> np.ndarray:
    """Step 4: Create ensemble predictions."""
    print("\n" + "=" * 60)
    print("Step 4: Ensemble Prediction")
    print("=" * 60)

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    models = [
        ("lr", LogisticRegression(random_state=42, max_iter=1000)),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
        ("gb", GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
    ]

    predictions = []
    weights = [0.2, 0.4, 0.4]  # Weight based on CV performance

    for name, model in models:
        if name == "lr":
            model.fit(X_train_scaled, y_train)
            pred = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train.values, y_train)
            pred = model.predict_proba(X_test.values)[:, 1]

        predictions.append(pred)
        print(f"Trained {name}, prediction range: [{pred.min():.3f}, {pred.max():.3f}]")

    # Weighted ensemble
    ensemble_pred = np.zeros(len(X_test))
    for pred, weight in zip(predictions, weights):
        ensemble_pred += weight * pred

    print(f"\nEnsemble prediction range: [{ensemble_pred.min():.3f}, {ensemble_pred.max():.3f}]")

    return ensemble_pred


def step5_create_submission(predictions: np.ndarray, threshold: float = 0.5):
    """Step 5: Create submission file."""
    print("\n" + "=" * 60)
    print("Step 5: Create Submission")
    print("=" * 60)

    # Create submission DataFrame
    submission = pd.DataFrame(
        {
            "id": range(len(predictions)),
            "target_probability": predictions,
            "target": (predictions > threshold).astype(int),
        }
    )

    print(f"\nSubmission shape: {submission.shape}")
    print(f"Prediction distribution:")
    print(submission["target"].value_counts(normalize=True))

    # In a real competition, you would save:
    # submission[["id", "target"]].to_csv("submission.csv", index=False)
    print("\nSubmission file would be saved as: submission.csv")

    return submission


def main():
    """Run the complete Kaggle workflow."""
    print("Kaggle Competition Workflow with Forge")
    print("=" * 60)

    # Create data
    X_train, X_test, y_train = create_kaggle_like_data()

    # Step 1: Explore
    step1_explore_data(X_train, y_train)

    # Step 2: Feature Engineering
    X_train_fe, X_test_fe, transformer = step2_feature_engineering(
        X_train, X_test, y_train
    )

    # Step 3: Validate
    cv_results = step3_model_validation(X_train_fe, y_train)

    # Step 4: Ensemble
    predictions = step4_ensemble_and_predict(X_train_fe, X_test_fe, y_train)

    # Step 5: Submit
    submission = step5_create_submission(predictions)

    print("\n" + "=" * 60)
    print("Workflow Complete!")
    print("=" * 60)
    print("\nSummary:")
    print(f"  - Original features: {X_train.shape[1]}")
    print(f"  - Engineered features: {X_train_fe.shape[1]}")
    print(f"  - Best CV AUC: {max(r['auc'] for r in cv_results.values()):.4f}")
    print(f"  - Test predictions generated: {len(predictions)}")


if __name__ == "__main__":
    main()
