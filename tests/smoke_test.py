"""Smoke tests for quick installation verification.

Run with: pytest tests/smoke_test.py -v
Expected to complete in under 10 seconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_import_forge() -> None:
    """Verify core forge package imports successfully."""
    import forge

    assert hasattr(forge, "__version__")
    assert hasattr(forge, "AutoFeatureTransformer")
    assert hasattr(forge, "DataAnalyzer")


def test_version_format() -> None:
    """Verify version string is well-formed."""
    from forge import __version__

    parts = __version__.split(".")
    assert len(parts) >= 2, f"Expected semver, got {__version__}"


def test_basic_fit_transform() -> None:
    """Verify AutoFeatureTransformer can fit and transform a small dataset."""
    from forge import AutoFeatureTransformer

    rng = np.random.RandomState(42)
    X = pd.DataFrame({
        "num_a": rng.normal(0, 1, 50),
        "num_b": rng.uniform(0, 10, 50),
        "cat": rng.choice(["x", "y", "z"], 50),
    })
    y = pd.Series(rng.choice([0, 1], 50), name="target")

    transformer = AutoFeatureTransformer(max_features=10)
    result = transformer.fit_transform(X, y)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 50
    assert result.shape[1] >= X.shape[1]


def test_data_analyzer() -> None:
    """Verify DataAnalyzer produces a report."""
    from forge import DataAnalyzer

    df = pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "salary": [50000, 60000, 70000, 80000, 90000],
        "dept": ["A", "B", "A", "B", "A"],
    })

    analyzer = DataAnalyzer()
    report = analyzer.analyze(df)
    assert report is not None


def test_sklearn_pipeline_compatibility() -> None:
    """Verify forge works inside an sklearn Pipeline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from forge import AutoFeatureTransformer

    rng = np.random.RandomState(42)
    X = pd.DataFrame({
        "a": rng.normal(0, 1, 100),
        "b": rng.normal(5, 2, 100),
    })
    y = pd.Series((X["a"] + X["b"] > 5).astype(int), name="target")

    pipe = Pipeline([
        ("features", AutoFeatureTransformer(max_features=5)),
        ("clf", LogisticRegression(max_iter=200)),
    ])

    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == 100
