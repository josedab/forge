"""Tests for data contracts module — generation, validation, CI/CD."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forge.contracts import (
    ContractGenerator,
    ContractValidator,
    DataContract,
)
from forge.contracts.validation import (
    ValidationReport,
    ci_validate,
    export_contract_yaml,
    generate_contract,
    validate_data_file,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "age": np.random.uniform(20, 60, 100),
        "income": np.random.uniform(30000, 100000, 100),
        "category": np.random.choice(["A", "B", "C"], 100),
    })


@pytest.fixture
def contract(sample_df: pd.DataFrame) -> DataContract:
    gen = ContractGenerator(tolerance=1.5)
    return gen.from_dataframe(sample_df)


class TestContractGenerator:
    def test_generates_contract(self, sample_df: pd.DataFrame) -> None:
        gen = ContractGenerator()
        contract = gen.from_dataframe(sample_df)
        assert len(contract.columns) == 3
        assert "age" in contract.columns
        assert "income" in contract.columns
        assert "category" in contract.columns

    def test_numeric_ranges(self, sample_df: pd.DataFrame) -> None:
        gen = ContractGenerator(tolerance=1.5)
        contract = gen.from_dataframe(sample_df)
        age_c = contract.columns["age"]
        assert age_c.min_value is not None
        assert age_c.max_value is not None
        assert age_c.min_value < sample_df["age"].min()
        assert age_c.max_value > sample_df["age"].max()

    def test_categorical_allowed_values(self, sample_df: pd.DataFrame) -> None:
        gen = ContractGenerator()
        contract = gen.from_dataframe(sample_df)
        cat_c = contract.columns["category"]
        assert cat_c.allowed_values is not None
        assert set(cat_c.allowed_values) == {"A", "B", "C"}

    def test_expected_columns(self, sample_df: pd.DataFrame) -> None:
        gen = ContractGenerator()
        contract = gen.from_dataframe(sample_df)
        assert contract.expected_columns == ["age", "income", "category"]


class TestContractValidator:
    def test_valid_data(self, sample_df: pd.DataFrame, contract: DataContract) -> None:
        validator = ContractValidator(contract, strictness="silent")
        violations = validator.validate(sample_df)
        # Contract was generated from this data, should pass
        assert len([v for v in violations if v.severity == "error"]) == 0

    def test_missing_column(self, contract: DataContract) -> None:
        df = pd.DataFrame({"age": [30], "income": [50000]})  # missing category
        validator = ContractValidator(contract, strictness="silent")
        violations = validator.validate(df)
        assert any(v.rule == "column_exists" for v in violations)

    def test_null_violation(self, contract: DataContract) -> None:
        df = pd.DataFrame({
            "age": [None] * 100,
            "income": [50000] * 100,
            "category": ["A"] * 100,
        })
        validator = ContractValidator(contract, strictness="silent")
        violations = validator.validate(df)
        assert any(v.rule in ("nullable", "max_null_pct") for v in violations)

    def test_out_of_range(self, contract: DataContract) -> None:
        df = pd.DataFrame({
            "age": [999999],  # way out of range
            "income": [50000],
            "category": ["A"],
        })
        validator = ContractValidator(contract, strictness="silent")
        violations = validator.validate(df)
        assert any(v.rule == "max_value" for v in violations)

    def test_unexpected_category(self, contract: DataContract) -> None:
        df = pd.DataFrame({
            "age": [30] * 20,
            "income": [50000] * 20,
            "category": ["Z"] * 20,  # not in allowed values
        })
        validator = ContractValidator(contract, strictness="silent")
        violations = validator.validate(df)
        assert any(v.rule == "allowed_values" for v in violations)

    def test_strictness_error_raises(self, contract: DataContract) -> None:
        df = pd.DataFrame({"age": [999999], "income": [50000], "category": ["A"]})
        validator = ContractValidator(contract, strictness="error")
        with pytest.raises(ValueError, match="Contract violations"):
            validator.validate(df)


class TestDataContractSerialization:
    def test_json_round_trip(self, contract: DataContract, tmp_path: Path) -> None:
        path = tmp_path / "contract.json"
        contract.save(path)
        loaded = DataContract.load(path)
        assert loaded.name == contract.name
        assert set(loaded.columns.keys()) == set(contract.columns.keys())

    def test_to_dict_from_dict(self, contract: DataContract) -> None:
        d = contract.to_dict()
        loaded = DataContract.from_dict(d)
        assert loaded.version == contract.version
        assert len(loaded.columns) == len(contract.columns)


class TestContractYAMLExport:
    def test_yaml_export(self, contract: DataContract) -> None:
        yaml_str = export_contract_yaml(contract)
        assert "name:" in yaml_str
        assert "columns:" in yaml_str
        assert "age:" in yaml_str

    def test_yaml_contains_expected_columns(self, contract: DataContract) -> None:
        yaml_str = export_contract_yaml(contract)
        assert "expected_columns:" in yaml_str
        for col in contract.expected_columns:
            assert col in yaml_str


class TestCIValidation:
    def test_validate_data_file(
        self, sample_df: pd.DataFrame, contract: DataContract, tmp_path: Path
    ) -> None:
        data_path = tmp_path / "data.csv"
        contract_path = tmp_path / "contract.json"
        sample_df.to_csv(data_path, index=False)
        contract.save(contract_path)

        report = validate_data_file(data_path, contract_path)
        assert isinstance(report, ValidationReport)
        assert report.n_rows == 100

    def test_valid_data_passes(
        self, sample_df: pd.DataFrame, contract: DataContract, tmp_path: Path
    ) -> None:
        data_path = tmp_path / "data.csv"
        contract_path = tmp_path / "contract.json"
        sample_df.to_csv(data_path, index=False)
        contract.save(contract_path)

        report = validate_data_file(data_path, contract_path)
        assert report.passed

    def test_bad_data_fails(
        self, contract: DataContract, tmp_path: Path
    ) -> None:
        bad_df = pd.DataFrame({
            "age": [999999], "income": [50000], "category": ["Z"]
        })
        data_path = tmp_path / "bad.csv"
        contract_path = tmp_path / "contract.json"
        bad_df.to_csv(data_path, index=False)
        contract.save(contract_path)

        report = validate_data_file(data_path, contract_path)
        assert not report.passed
        assert report.n_errors > 0

    def test_report_summary(
        self, sample_df: pd.DataFrame, contract: DataContract, tmp_path: Path
    ) -> None:
        data_path = tmp_path / "data.csv"
        contract_path = tmp_path / "contract.json"
        sample_df.to_csv(data_path, index=False)
        contract.save(contract_path)

        report = validate_data_file(data_path, contract_path)
        summary = report.summary()
        assert "Contract Validation" in summary

    def test_generate_contract_function(
        self, sample_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        data_path = tmp_path / "data.csv"
        out_path = tmp_path / "generated.json"
        sample_df.to_csv(data_path, index=False)

        contract = generate_contract(data_path, out_path)
        assert out_path.exists()
        assert len(contract.columns) == 3

    def test_ci_validate_exit_code(
        self, sample_df: pd.DataFrame, contract: DataContract, tmp_path: Path
    ) -> None:
        data_path = tmp_path / "data.csv"
        contract_path = tmp_path / "contract.json"
        sample_df.to_csv(data_path, index=False)
        contract.save(contract_path)

        exit_code = ci_validate(str(data_path), str(contract_path))
        assert exit_code == 0
