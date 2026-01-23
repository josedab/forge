"""Tests for data contracts module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.contracts import (
    ContractGenerator,
    ContractValidator,
    DataContract,
    Violation,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "income": [30000.0, 45000.0, 60000.0, 75000.0, 90000.0],
        "city": ["NY", "LA", "NY", "SF", "LA"],
    })


class TestContractGenerator:
    def test_from_dataframe(self, sample_df):
        gen = ContractGenerator()
        contract = gen.from_dataframe(sample_df)
        assert "age" in contract.columns
        assert "income" in contract.columns
        assert "city" in contract.columns
        assert contract.min_rows > 0

    def test_numeric_constraints(self, sample_df):
        gen = ContractGenerator()
        contract = gen.from_dataframe(sample_df)
        age_c = contract.columns["age"]
        assert age_c.min_value is not None
        assert age_c.max_value is not None
        assert age_c.mean_range is not None
        assert age_c.min_value < 25
        assert age_c.max_value > 45

    def test_categorical_constraints(self, sample_df):
        gen = ContractGenerator()
        contract = gen.from_dataframe(sample_df)
        city_c = contract.columns["city"]
        assert city_c.allowed_values is not None
        assert "NY" in city_c.allowed_values

    def test_tolerance(self, sample_df):
        gen_tight = ContractGenerator(tolerance=1.0)
        gen_wide = ContractGenerator(tolerance=3.0)
        c_tight = gen_tight.from_dataframe(sample_df).columns["age"]
        c_wide = gen_wide.from_dataframe(sample_df).columns["age"]
        # Wider tolerance = wider range
        assert c_wide.max_value > c_tight.max_value


class TestContractValidator:
    def test_valid_data(self, sample_df):
        gen = ContractGenerator()
        contract = gen.from_dataframe(sample_df)
        validator = ContractValidator(contract)
        violations = validator.validate(sample_df)
        assert len(violations) == 0

    def test_missing_column(self, sample_df):
        gen = ContractGenerator()
        contract = gen.from_dataframe(sample_df)
        validator = ContractValidator(contract)
        violations = validator.validate(sample_df.drop(columns=["age"]))
        assert any(v.column == "age" for v in violations)

    def test_value_out_of_range(self, sample_df):
        gen = ContractGenerator(tolerance=1.01)
        contract = gen.from_dataframe(sample_df)
        validator = ContractValidator(contract)

        bad_df = sample_df.copy()
        bad_df.loc[0, "age"] = -1000
        violations = validator.validate(bad_df)
        assert any("min_value" in v.rule for v in violations)

    def test_unexpected_category(self, sample_df):
        gen = ContractGenerator()
        contract = gen.from_dataframe(sample_df)
        validator = ContractValidator(contract)

        bad_df = sample_df.copy()
        bad_df.loc[0, "city"] = "MARS"
        violations = validator.validate(bad_df)
        assert any("allowed_values" in v.rule for v in violations)

    def test_excess_nulls(self, sample_df):
        gen = ContractGenerator(null_tolerance=0.0)
        contract = gen.from_dataframe(sample_df)
        # contract expects 0 nulls + 0 tolerance = 0
        contract.columns["age"].max_null_pct = 0.0
        contract.columns["age"].nullable = False

        bad_df = sample_df.copy()
        bad_df.loc[0, "age"] = np.nan
        validator = ContractValidator(contract)
        violations = validator.validate(bad_df)
        assert any("nullable" in v.rule for v in violations)

    def test_too_few_rows(self, sample_df):
        gen = ContractGenerator()
        contract = gen.from_dataframe(sample_df)
        contract.min_rows = 100

        validator = ContractValidator(contract)
        violations = validator.validate(sample_df)
        assert any("min_rows" in v.rule for v in violations)

    def test_error_strictness(self, sample_df):
        gen = ContractGenerator()
        contract = gen.from_dataframe(sample_df)
        contract.min_rows = 100
        validator = ContractValidator(contract, strictness="error")
        with pytest.raises(ValueError, match="Contract violations"):
            validator.validate(sample_df)

    def test_silent_strictness(self, sample_df):
        gen = ContractGenerator()
        contract = gen.from_dataframe(sample_df)
        contract.min_rows = 100
        validator = ContractValidator(contract, strictness="silent")
        violations = validator.validate(sample_df)
        assert len(violations) > 0  # still returns violations, just doesn't raise


class TestDataContract:
    def test_serialization(self, sample_df):
        gen = ContractGenerator()
        contract = gen.from_dataframe(sample_df, name="test")
        d = contract.to_dict()
        restored = DataContract.from_dict(d)
        assert restored.name == "test"
        assert "age" in restored.columns
        assert restored.columns["age"].min_value is not None

    def test_save_load(self, sample_df, tmp_path):
        gen = ContractGenerator()
        contract = gen.from_dataframe(sample_df, name="persist_test")
        path = tmp_path / "contract.json"
        contract.save(path)

        loaded = DataContract.load(path)
        assert loaded.name == "persist_test"
        assert len(loaded.columns) == 3


class TestViolation:
    def test_str(self):
        v = Violation(column="age", rule="min_value", message="too low")
        assert "[ERROR]" in str(v)
        assert "age" in str(v)

    def test_warning_severity(self):
        v = Violation(column="x", rule="mean", message="drift", severity="warning")
        assert "[WARNING]" in str(v)
