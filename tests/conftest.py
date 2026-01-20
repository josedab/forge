"""Pytest fixtures for Forge tests."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_numeric_df() -> pd.DataFrame:
    """DataFrame with numeric columns."""
    np.random.seed(42)
    n = 100

    return pd.DataFrame(
        {
            "age": np.random.randint(18, 80, n),
            "income": np.random.normal(50000, 15000, n),
            "score": np.random.uniform(0, 100, n),
            "count": np.random.poisson(5, n),
        }
    )


@pytest.fixture
def sample_categorical_df() -> pd.DataFrame:
    """DataFrame with categorical columns."""
    np.random.seed(42)
    n = 100

    return pd.DataFrame(
        {
            "color": np.random.choice(["red", "blue", "green"], n),
            "size": np.random.choice(["S", "M", "L", "XL"], n),
            "category": pd.Categorical(
                np.random.choice(["A", "B", "C"], n), categories=["A", "B", "C"]
            ),
            "flag": np.random.choice([True, False], n),
        }
    )


@pytest.fixture
def sample_temporal_df() -> pd.DataFrame:
    """DataFrame with temporal columns."""
    np.random.seed(42)
    n = 100

    base_date = datetime(2023, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(n)]

    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "timestamp": pd.to_datetime(dates) + pd.to_timedelta(
                np.random.randint(0, 86400, n), unit="s"
            ),
            "value": np.random.normal(100, 20, n),
        }
    )


@pytest.fixture
def sample_text_df() -> pd.DataFrame:
    """DataFrame with text columns."""
    np.random.seed(42)

    texts = [
        "This is a sample text.",
        "Another example sentence here.",
        "Short text",
        "This is a much longer text that contains multiple words and sentences. It should be interesting to analyze.",
        "Quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating!",
        "Data science and analytics.",
        "Python programming language.",
        "Natural language processing.",
        "Feature engineering techniques.",
    ]

    return pd.DataFrame(
        {
            "text": np.random.choice(texts, 100),
            "title": np.random.choice(["Title A", "Title B", "Title C"], 100),
        }
    )


@pytest.fixture
def sample_mixed_df() -> pd.DataFrame:
    """DataFrame with mixed column types."""
    np.random.seed(42)
    n = 100

    base_date = datetime(2023, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(n)]

    return pd.DataFrame(
        {
            # Numeric
            "age": np.random.randint(18, 80, n),
            "income": np.random.normal(50000, 15000, n),
            "score": np.random.uniform(0, 100, n),
            # Categorical
            "category": np.random.choice(["A", "B", "C"], n),
            "color": np.random.choice(["red", "blue", "green"], n),
            # Temporal
            "date": pd.to_datetime(dates),
            # Text
            "description": np.random.choice(
                ["Short desc", "Medium description here", "A much longer description text"], n
            ),
        }
    )


@pytest.fixture
def sample_target_binary() -> pd.Series:
    """Binary classification target."""
    np.random.seed(42)
    return pd.Series(np.random.choice([0, 1], 100), name="target")


@pytest.fixture
def sample_target_multiclass() -> pd.Series:
    """Multiclass classification target."""
    np.random.seed(42)
    return pd.Series(np.random.choice([0, 1, 2], 100), name="target")


@pytest.fixture
def sample_target_regression() -> pd.Series:
    """Regression target."""
    np.random.seed(42)
    return pd.Series(np.random.normal(100, 20, 100), name="target")


@pytest.fixture
def sample_df_with_missing() -> pd.DataFrame:
    """DataFrame with missing values."""
    np.random.seed(42)
    n = 100

    df = pd.DataFrame(
        {
            "complete": np.random.normal(0, 1, n),
            "few_missing": np.random.normal(0, 1, n),
            "many_missing": np.random.normal(0, 1, n),
            "categorical": np.random.choice(["A", "B", "C"], n),
        }
    )

    # Add missing values
    df.loc[np.random.choice(n, 5, replace=False), "few_missing"] = np.nan
    df.loc[np.random.choice(n, 30, replace=False), "many_missing"] = np.nan
    df.loc[np.random.choice(n, 10, replace=False), "categorical"] = None

    return df


@pytest.fixture
def sample_df_with_outliers() -> pd.DataFrame:
    """DataFrame with outliers."""
    np.random.seed(42)
    n = 100

    df = pd.DataFrame(
        {
            "normal": np.random.normal(0, 1, n),
            "with_outliers": np.random.normal(0, 1, n),
        }
    )

    # Add outliers
    df.loc[0, "with_outliers"] = 100
    df.loc[1, "with_outliers"] = -100

    return df


@pytest.fixture
def small_df() -> pd.DataFrame:
    """Small DataFrame for quick tests."""
    return pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "c": ["x", "y", "x", "y", "x"],
        }
    )


@pytest.fixture
def small_target() -> pd.Series:
    """Small target for quick tests."""
    return pd.Series([0, 1, 0, 1, 0], name="target")


@pytest.fixture
def high_cardinality_df() -> pd.DataFrame:
    """DataFrame with high cardinality categorical column."""
    np.random.seed(42)
    n = 1000

    return pd.DataFrame(
        {
            "id": [f"id_{i}" for i in range(n)],
            "group": np.random.choice([f"g_{i}" for i in range(100)], n),
            "value": np.random.normal(0, 1, n),
        }
    )


@pytest.fixture
def time_series_df() -> pd.DataFrame:
    """Time series DataFrame for lag/rolling tests."""
    np.random.seed(42)
    n = 365

    dates = pd.date_range("2023-01-01", periods=n, freq="D")

    return pd.DataFrame(
        {
            "date": dates,
            "value": np.cumsum(np.random.normal(0, 1, n)) + 100,
            "group": np.repeat(["A", "B"], [n // 2 + n % 2, n // 2])[:n],
        }
    )


@pytest.fixture
def correlated_df() -> pd.DataFrame:
    """DataFrame with correlated features."""
    np.random.seed(42)
    n = 100

    x = np.random.normal(0, 1, n)

    return pd.DataFrame(
        {
            "x": x,
            "y": x + np.random.normal(0, 0.1, n),  # Highly correlated with x
            "z": x * 2 + np.random.normal(0, 0.1, n),  # Highly correlated with x
            "independent": np.random.normal(0, 1, n),  # Independent
        }
    )


@pytest.fixture
def feature_importance_df() -> pd.DataFrame:
    """DataFrame for testing feature importance."""
    np.random.seed(42)
    n = 500

    # Create features with known importance
    important1 = np.random.normal(0, 1, n)
    important2 = np.random.normal(0, 1, n)
    noise1 = np.random.normal(0, 1, n)
    noise2 = np.random.normal(0, 1, n)

    return pd.DataFrame(
        {
            "important1": important1,
            "important2": important2,
            "noise1": noise1,
            "noise2": noise2,
        }
    )


@pytest.fixture
def feature_importance_target(feature_importance_df: pd.DataFrame) -> pd.Series:
    """Target variable correlated with important features."""
    np.random.seed(42)
    y = (
        2 * feature_importance_df["important1"]
        + feature_importance_df["important2"]
        + np.random.normal(0, 0.1, len(feature_importance_df))
    )
    return pd.Series((y > y.median()).astype(int), name="target")


# Markers for conditional tests
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "requires_shap: marks tests requiring shap")
    config.addinivalue_line("markers", "requires_viz: marks tests requiring matplotlib")


def _has_shap() -> bool:
    """Check if shap is available."""
    try:
        import shap
        return True
    except ImportError:
        return False


def _has_matplotlib() -> bool:
    """Check if matplotlib is available."""
    try:
        import matplotlib
        return True
    except ImportError:
        return False


# Skip markers
requires_shap = pytest.mark.skipif(
    not _has_shap(), reason="shap not installed"
)

requires_viz = pytest.mark.skipif(
    not _has_matplotlib(), reason="matplotlib not installed"
)
