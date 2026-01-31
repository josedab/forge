"""Tests for natural language feature synthesis (rule-based and LLM)."""

from __future__ import annotations

import pandas as pd
import pytest

from forge.nlgen import NLFeatureSynthesizer, SynthesisResult
from forge.nlgen.llm_synthesizer import LLMFeatureSynthesizer


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "income": [50000, 60000, 70000, 80000],
        "debt": [10000, 20000, 5000, 30000],
        "age": [25, 35, 45, 55],
        "city": ["NYC", "LA", "SF", "NYC"],
    })


@pytest.fixture
def synth() -> NLFeatureSynthesizer:
    return NLFeatureSynthesizer(
        schema={"income": "numeric", "debt": "numeric", "age": "numeric", "city": "categorical"}
    )


class TestNLFeatureSynthesizer:
    """Tests for rule-based NL synthesis."""

    def test_ratio(self, synth: NLFeatureSynthesizer, sample_df: pd.DataFrame) -> None:
        result = synth.synthesize("ratio of income to debt")
        assert result.success
        assert "income_div_debt" in result.feature_names
        df_out = result.execute(sample_df)
        assert "income_div_debt" in df_out.columns

    def test_product(self, synth: NLFeatureSynthesizer, sample_df: pd.DataFrame) -> None:
        result = synth.synthesize("product of income and age")
        assert result.success
        df_out = result.execute(sample_df)
        assert "income_mul_age" in df_out.columns

    def test_difference(self, synth: NLFeatureSynthesizer, sample_df: pd.DataFrame) -> None:
        result = synth.synthesize("difference between income and debt")
        assert result.success
        df_out = result.execute(sample_df)
        assert "income_minus_debt" in df_out.columns
        assert df_out["income_minus_debt"].iloc[0] == 40000

    def test_log_transform(self, synth: NLFeatureSynthesizer, sample_df: pd.DataFrame) -> None:
        result = synth.synthesize("log transform of income")
        assert result.success
        df_out = result.execute(sample_df)
        assert "income_log1p" in df_out.columns

    def test_sqrt(self, synth: NLFeatureSynthesizer, sample_df: pd.DataFrame) -> None:
        result = synth.synthesize("square root of age")
        assert result.success
        df_out = result.execute(sample_df)
        assert "age_sqrt" in df_out.columns

    def test_square(self, synth: NLFeatureSynthesizer, sample_df: pd.DataFrame) -> None:
        result = synth.synthesize("age squared")
        assert result.success
        df_out = result.execute(sample_df)
        assert "age_squared" in df_out.columns
        assert df_out["age_squared"].iloc[0] == 625

    def test_binning(self, synth: NLFeatureSynthesizer, sample_df: pd.DataFrame) -> None:
        result = synth.synthesize("bin income into 3 quantiles")
        assert result.success
        df_out = result.execute(sample_df)
        assert "income_binned_3" in df_out.columns

    def test_null_flag(self, synth: NLFeatureSynthesizer) -> None:
        df = pd.DataFrame({"income": [1, None, 3, None]})
        result = synth.synthesize("flag if income is null")
        assert result.success
        df_out = result.execute(df)
        assert "income_is_null" in df_out.columns
        assert df_out["income_is_null"].tolist() == [0, 1, 0, 1]

    def test_gt_flag(self, synth: NLFeatureSynthesizer, sample_df: pd.DataFrame) -> None:
        result = synth.synthesize("flag if age > 40")
        assert result.success
        df_out = result.execute(sample_df)
        assert "age_gt_40" in df_out.columns

    def test_unparseable_description(self, synth: NLFeatureSynthesizer) -> None:
        result = synth.synthesize("something completely random and unparseable 12345")
        assert not result.success
        assert "Could not parse" in result.error

    def test_schema_validation(self) -> None:
        synth = NLFeatureSynthesizer(schema={"a": "numeric"})
        result = synth.synthesize("ratio of a to b")
        assert result.success
        assert any("not found" in w for w in result.warnings)

    def test_from_dataframe(self, sample_df: pd.DataFrame) -> None:
        synth = NLFeatureSynthesizer.from_dataframe(sample_df)
        result = synth.synthesize("ratio of income to debt")
        assert result.success

    def test_synthesize_batch(self, synth: NLFeatureSynthesizer) -> None:
        results = synth.synthesize_batch([
            "ratio of income to debt",
            "log transform of age",
        ])
        assert len(results) == 2
        assert all(r.success for r in results)


class TestLLMFeatureSynthesizer:
    """Tests for LLM-enhanced synthesis (no API key = rule fallback)."""

    def test_fallback_to_rules(self, sample_df: pd.DataFrame) -> None:
        synth = LLMFeatureSynthesizer(
            schema={"income": "numeric", "debt": "numeric"},
            fallback_to_rules=True,
        )
        result = synth.synthesize("ratio of income to debt")
        assert result.success
        assert "income_div_debt" in result.feature_names

    def test_no_api_key_no_fallback(self) -> None:
        synth = LLMFeatureSynthesizer(
            schema={"a": "numeric"},
            fallback_to_rules=False,
        )
        result = synth.synthesize("something complex")
        # Without API key and without fallback, complex descriptions fail
        assert not result.success

    def test_from_dataframe(self, sample_df: pd.DataFrame) -> None:
        synth = LLMFeatureSynthesizer.from_dataframe(sample_df)
        result = synth.synthesize("product of income and debt")
        assert result.success

    def test_batch(self) -> None:
        synth = LLMFeatureSynthesizer(
            schema={"x": "numeric", "y": "numeric"},
        )
        results = synth.synthesize_batch(["ratio of x to y", "log transform of x"])
        assert len(results) == 2
        assert all(r.success for r in results)


class TestSynthesisResultExecution:
    """Tests for code execution safety."""

    def test_execute_on_failed_raises(self) -> None:
        result = SynthesisResult(
            description="test", code="", feature_names=[], success=False, error="fail"
        )
        with pytest.raises(RuntimeError, match="Cannot execute"):
            result.execute(pd.DataFrame())

    def test_execute_invalid_code_raises(self) -> None:
        result = SynthesisResult(
            description="test",
            code="result = df.copy()\nresult['x'] = df['nonexistent']",
            feature_names=["x"],
            success=True,
        )
        with pytest.raises(RuntimeError, match="Execution failed"):
            result.execute(pd.DataFrame({"a": [1]}))
