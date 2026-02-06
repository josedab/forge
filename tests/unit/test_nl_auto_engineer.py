"""Tests for NaturalLanguageFeatureEngineer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.nlgen.auto_engineer import (
    NaturalLanguageFeatureEngineer,
    detect_domain,
)

# ---- fixtures --------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "age": rng.randint(18, 65, 100),
        "income": rng.uniform(20_000, 150_000, 100),
        "debt": rng.uniform(0, 80_000, 100),
        "tenure_months": rng.randint(1, 60, 100),
        "plan": rng.choice(["basic", "pro", "enterprise"], 100),
        "signup_date": pd.date_range("2020-01-01", periods=100, freq="D"),
    })


@pytest.fixture
def target(sample_df: pd.DataFrame) -> pd.Series:
    rng = np.random.RandomState(0)
    return pd.Series(rng.randint(0, 2, len(sample_df)), name="churn")


# ---- domain detection tests ------------------------------------------


class TestDetectDomain:
    def test_ecommerce_detected(self) -> None:
        result = detect_domain("Predicting customer churn for a SaaS subscription product")
        assert result.domain == "ecommerce"
        assert result.confidence > 0

    def test_finance_detected(self) -> None:
        result = detect_domain("Credit risk scoring for loan default prediction")
        assert result.domain == "finance"
        assert result.confidence > 0

    def test_healthcare_detected(self) -> None:
        result = detect_domain("Predicting patient readmission in a hospital")
        assert result.domain == "healthcare"
        assert result.confidence > 0

    def test_general_fallback(self) -> None:
        result = detect_domain("Some generic prediction problem")
        assert result.domain == "general"

    def test_empty_string(self) -> None:
        result = detect_domain("")
        assert result.domain == "general"
        assert result.confidence == 0.0


# ---- NaturalLanguageFeatureEngineer tests ----------------------------


class TestNaturalLanguageFeatureEngineer:
    def test_fit_transform_basic(self, sample_df: pd.DataFrame, target: pd.Series) -> None:
        eng = NaturalLanguageFeatureEngineer(
            problem="Predicting customer churn for a SaaS product",
        )
        result = eng.fit_transform(sample_df, target)
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) > len(sample_df.columns)

    def test_get_feature_names_out(self, sample_df: pd.DataFrame, target: pd.Series) -> None:
        eng = NaturalLanguageFeatureEngineer(problem="churn prediction")
        eng.fit_transform(sample_df, target)
        names = eng.get_feature_names_out()
        assert len(names) > 0

    def test_domain_stored(self, sample_df: pd.DataFrame) -> None:
        eng = NaturalLanguageFeatureEngineer(problem="credit risk scoring for loan default")
        eng.fit(sample_df)
        assert eng._domain is not None
        assert eng._domain.domain == "finance"

    def test_include_originals_false(self, sample_df: pd.DataFrame) -> None:
        eng = NaturalLanguageFeatureEngineer(
            problem="churn", include_originals=False,
        )
        result = eng.fit_transform(sample_df)
        # Original columns should NOT be present
        for col in sample_df.columns:
            assert col not in result.columns or col.endswith("_log1p")

    def test_max_features_respected(self, sample_df: pd.DataFrame) -> None:
        eng = NaturalLanguageFeatureEngineer(problem="churn", max_features=3)
        result = eng.fit_transform(sample_df)
        new_cols = set(result.columns) - set(sample_df.columns)
        assert len(new_cols) <= 3

    def test_not_fitted_raises(self) -> None:
        eng = NaturalLanguageFeatureEngineer(problem="test")
        with pytest.raises(Exception):
            eng.transform(pd.DataFrame({"a": [1]}))

    def test_invalid_input_raises(self, sample_df: pd.DataFrame) -> None:
        eng = NaturalLanguageFeatureEngineer(problem="test")
        with pytest.raises(Exception):
            eng.fit("not a dataframe")  # type: ignore[arg-type]

    def test_empty_df_raises(self) -> None:
        eng = NaturalLanguageFeatureEngineer(problem="test")
        with pytest.raises(Exception):
            eng.fit(pd.DataFrame())

    def test_get_result(self, sample_df: pd.DataFrame, target: pd.Series) -> None:
        eng = NaturalLanguageFeatureEngineer(problem="customer churn SaaS")
        eng.fit_transform(sample_df, target)
        result = eng.get_result()
        assert result is not None
        assert result.domain.domain == "ecommerce"
        assert result.n_generated_features > 0

    def test_no_target(self, sample_df: pd.DataFrame) -> None:
        eng = NaturalLanguageFeatureEngineer(problem="churn prediction")
        result = eng.fit_transform(sample_df)
        assert isinstance(result, pd.DataFrame)
        # Without target, no target encoding columns
        target_enc_cols = [c for c in result.columns if c.endswith("_target_enc")]
        assert len(target_enc_cols) == 0

    def test_sklearn_clone(self, sample_df: pd.DataFrame) -> None:
        from sklearn.base import clone
        eng = NaturalLanguageFeatureEngineer(problem="churn", max_features=10)
        eng2 = clone(eng)
        assert eng2.problem == "churn"
        assert eng2.max_features == 10
