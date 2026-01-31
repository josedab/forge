"""Tests for natural language feature synthesis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.nlgen import NLFeatureSynthesizer


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "income": [50000, 60000, 70000, 80000, 90000],
        "debt": [10000, 20000, 5000, 15000, 25000],
        "age": [25, 30, 35, 40, 45],
        "score": [0.8, 0.6, 0.9, 0.7, 0.5],
    })


@pytest.fixture
def synth():
    return NLFeatureSynthesizer(schema={
        "income": "numeric", "debt": "numeric",
        "age": "numeric", "score": "numeric",
    })


class TestNLFeatureSynthesizer:
    def test_ratio(self, synth, sample_df):
        result = synth.synthesize("ratio of income to debt")
        assert result.success
        assert "income_div_debt" in result.feature_names
        df_new = result.execute(sample_df)
        assert "income_div_debt" in df_new.columns
        assert df_new["income_div_debt"].iloc[0] == pytest.approx(5.0)

    def test_product(self, synth, sample_df):
        result = synth.synthesize("product of age and score")
        assert result.success
        df_new = result.execute(sample_df)
        assert "age_mul_score" in df_new.columns

    def test_difference(self, synth, sample_df):
        result = synth.synthesize("difference between income and debt")
        assert result.success
        df_new = result.execute(sample_df)
        assert "income_minus_debt" in df_new.columns

    def test_sum(self, synth, sample_df):
        result = synth.synthesize("sum of income and debt")
        assert result.success
        df_new = result.execute(sample_df)
        assert "income_plus_debt" in df_new.columns

    def test_log_transform(self, synth, sample_df):
        result = synth.synthesize("log transform of income")
        assert result.success
        df_new = result.execute(sample_df)
        assert "income_log1p" in df_new.columns

    def test_square(self, synth, sample_df):
        result = synth.synthesize("age squared")
        assert result.success
        df_new = result.execute(sample_df)
        assert "age_squared" in df_new.columns
        assert df_new["age_squared"].iloc[0] == 625

    def test_sqrt(self, synth, sample_df):
        result = synth.synthesize("square root of income")
        assert result.success
        df_new = result.execute(sample_df)
        assert "income_sqrt" in df_new.columns

    def test_binning(self, synth, sample_df):
        result = synth.synthesize("bin age into 3 quantiles")
        assert result.success
        df_new = result.execute(sample_df)
        assert "age_binned_3" in df_new.columns

    def test_null_flag(self, synth):
        result = synth.synthesize("flag if age is null")
        assert result.success
        df = pd.DataFrame({"age": [1, np.nan, 3]})
        df_new = result.execute(df)
        assert "age_is_null" in df_new.columns
        assert df_new["age_is_null"].sum() == 1

    def test_gt_flag(self, synth, sample_df):
        result = synth.synthesize("flag if income > 70000")
        assert result.success
        df_new = result.execute(sample_df)
        assert any("income_gt_" in c for c in df_new.columns)

    def test_lt_flag(self, synth, sample_df):
        result = synth.synthesize("flag if score < 0.7")
        assert result.success
        df_new = result.execute(sample_df)
        assert any("score_lt_" in c for c in df_new.columns)

    def test_unparseable(self, synth):
        result = synth.synthesize("do something magical")
        assert not result.success
        assert "Could not parse" in result.error

    def test_unparseable_execute_fails(self, synth):
        result = synth.synthesize("do something magical")
        with pytest.raises(RuntimeError):
            result.execute(pd.DataFrame())

    def test_schema_validation_warning(self):
        synth = NLFeatureSynthesizer(schema={"x": "numeric"})
        result = synth.synthesize("ratio of x to missing_col")
        assert any("missing_col" in w for w in result.warnings)

    def test_from_dataframe(self, sample_df):
        synth = NLFeatureSynthesizer.from_dataframe(sample_df)
        result = synth.synthesize("ratio of income to debt")
        assert result.success

    def test_batch(self, synth, sample_df):
        results = synth.synthesize_batch([
            "ratio of income to debt",
            "age squared",
        ])
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_operator_syntax(self, synth, sample_df):
        result = synth.synthesize("income / debt")
        assert result.success
        df_new = result.execute(sample_df)
        assert "income_div_debt" in df_new.columns

    def test_division_by_zero_safe(self, synth):
        result = synth.synthesize("ratio of income to debt")
        df = pd.DataFrame({"income": [100], "debt": [0]})
        df_new = result.execute(df)
        assert df_new["income_div_debt"].iloc[0] == 0
