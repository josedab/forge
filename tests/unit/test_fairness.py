"""Tests for compliance and fairness features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.fairness import (
    BiasDetector,
    BiasReport,
    FairFeatureGenerator,
    FairnessMetric,
)


@pytest.fixture
def biased_df():
    np.random.seed(42)
    n = 500
    gender = np.random.choice(["M", "F"], n)
    # income is correlated with gender
    income = np.where(gender == "M", 60000, 40000) + np.random.randn(n) * 5000
    age = np.random.randint(20, 60, n).astype(float)
    score = income * 0.001 + np.random.randn(n) * 0.5
    return pd.DataFrame({
        "gender": gender,
        "income": income,
        "age": age,
        "score": score,
    })


@pytest.fixture
def unbiased_df():
    np.random.seed(42)
    n = 500
    gender = np.random.choice(["M", "F"], n)
    # Features independent of gender
    feature1 = np.random.randn(n) * 10
    feature2 = np.random.randn(n) * 5
    return pd.DataFrame({
        "gender": gender,
        "feature1": feature1,
        "feature2": feature2,
    })


@pytest.fixture
def binary_target():
    np.random.seed(42)
    return np.random.randint(0, 2, 500)


class TestBiasDetector:
    """Tests for BiasDetector."""

    def test_fit_basic(self, biased_df):
        detector = BiasDetector(protected_columns=["gender"])
        detector.fit(biased_df)
        assert detector._is_fitted
        assert len(detector.bias_reports_) > 0

    def test_detects_biased_feature(self, biased_df):
        detector = BiasDetector(
            protected_columns=["gender"],
            metrics=[FairnessMetric.CORRELATION],
            threshold=0.2,
        )
        detector.fit(biased_df)
        assert "income" in detector.biased_features_

    def test_unbiased_data_clean(self, unbiased_df):
        detector = BiasDetector(
            protected_columns=["gender"],
            metrics=[FairnessMetric.CORRELATION],
            threshold=0.3,
        )
        detector.fit(unbiased_df)
        assert len(detector.biased_features_) == 0

    def test_demographic_parity(self, biased_df):
        detector = BiasDetector(
            protected_columns=["gender"],
            metrics=[FairnessMetric.DEMOGRAPHIC_PARITY],
            threshold=0.1,
        )
        detector.fit(biased_df)
        parity_reports = [
            r for r in detector.bias_reports_
            if r.metric == FairnessMetric.DEMOGRAPHIC_PARITY
        ]
        assert len(parity_reports) > 0

    def test_disparate_impact(self, biased_df):
        detector = BiasDetector(
            protected_columns=["gender"],
            metrics=[FairnessMetric.DISPARATE_IMPACT],
        )
        detector.fit(biased_df)
        di_reports = [
            r for r in detector.bias_reports_
            if r.metric == FairnessMetric.DISPARATE_IMPACT
        ]
        assert len(di_reports) > 0

    def test_equalized_odds(self, biased_df):
        # Use a target correlated with the features
        target = (biased_df["income"] > biased_df["income"].median()).astype(int).values
        detector = BiasDetector(
            protected_columns=["gender"],
            metrics=[FairnessMetric.EQUALIZED_ODDS],
        )
        detector.fit(biased_df, target)
        eo_reports = [
            r for r in detector.bias_reports_
            if r.metric == FairnessMetric.EQUALIZED_ODDS
        ]
        assert len(eo_reports) > 0

    def test_no_protected_columns_warning(self, biased_df):
        detector = BiasDetector(protected_columns=["nonexistent"])
        detector.fit(biased_df)
        assert len(detector.bias_reports_) == 0

    def test_get_summary(self, biased_df):
        detector = BiasDetector(
            protected_columns=["gender"],
            threshold=0.2,
        )
        detector.fit(biased_df)
        summary = detector.get_summary()
        assert "total_checks" in summary
        assert "biased_features" in summary
        assert summary["total_checks"] > 0

    def test_multiple_metrics(self, biased_df):
        detector = BiasDetector(
            protected_columns=["gender"],
            metrics=[FairnessMetric.CORRELATION, FairnessMetric.DEMOGRAPHIC_PARITY],
        )
        detector.fit(biased_df)
        metrics_found = {r.metric for r in detector.bias_reports_}
        assert FairnessMetric.CORRELATION in metrics_found
        assert FairnessMetric.DEMOGRAPHIC_PARITY in metrics_found


class TestFairFeatureGenerator:
    """Tests for FairFeatureGenerator."""

    def test_fit_transform_basic(self, biased_df):
        gen = FairFeatureGenerator(protected_columns=["gender"])
        result = gen.fit_transform(biased_df)
        assert result.shape[0] == len(biased_df)
        fair_cols = [c for c in result.columns if c.endswith("_fair")]
        assert len(fair_cols) > 0

    def test_fair_features_less_correlated(self, biased_df):
        gen = FairFeatureGenerator(protected_columns=["gender"])
        result = gen.fit_transform(biased_df)

        # Encode gender for correlation check
        gender_num = np.where(biased_df["gender"] == "M", 1, 0).astype(float)

        # Original correlation
        orig_corr = abs(float(np.corrcoef(biased_df["income"], gender_num)[0, 1]))

        # Fair feature correlation
        fair_corr = abs(float(np.corrcoef(result["income_fair"], gender_num)[0, 1]))

        assert fair_corr < orig_corr

    def test_preserves_original_columns(self, biased_df):
        gen = FairFeatureGenerator(protected_columns=["gender"])
        result = gen.fit_transform(biased_df)
        for col in biased_df.columns:
            assert col in result.columns

    def test_get_feature_names_out(self, biased_df):
        gen = FairFeatureGenerator(protected_columns=["gender"])
        gen.fit(biased_df)
        names = gen.get_feature_names_out()
        assert all("_fair" in n for n in names)

    def test_transform_before_fit_raises(self, biased_df):
        gen = FairFeatureGenerator()
        with pytest.raises(RuntimeError, match="must be fitted"):
            gen.transform(biased_df)

    def test_no_protected_returns_copy(self, biased_df):
        gen = FairFeatureGenerator(protected_columns=["nonexistent"])
        result = gen.fit_transform(biased_df)
        assert result.shape == biased_df.shape


class TestBiasReport:
    """Tests for BiasReport dataclass."""

    def test_str_biased(self):
        report = BiasReport(
            feature="income",
            protected_attribute="gender",
            metric=FairnessMetric.CORRELATION,
            score=0.8,
            is_biased=True,
            threshold=0.3,
        )
        s = str(report)
        assert "BIASED" in s
        assert "income" in s

    def test_str_clean(self):
        report = BiasReport(
            feature="age",
            protected_attribute="gender",
            metric=FairnessMetric.CORRELATION,
            score=0.05,
            is_biased=False,
            threshold=0.3,
        )
        s = str(report)
        assert "OK" in s
