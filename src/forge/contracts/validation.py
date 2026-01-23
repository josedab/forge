"""CI/CD integration and multi-format export for data contracts.

Provides functions for validating data contracts in CI/CD pipelines,
exporting contracts as YAML, and generating validation reports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from forge.contracts import (
    ContractGenerator,
    ContractValidator,
    DataContract,
    Violation,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Report from a CI/CD contract validation run."""

    contract_name: str
    data_path: str
    n_rows: int
    n_violations: int
    n_errors: int
    n_warnings: int
    violations: list[Violation]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "contract_name": self.contract_name,
            "data_path": self.data_path,
            "n_rows": self.n_rows,
            "n_violations": self.n_violations,
            "n_errors": self.n_errors,
            "n_warnings": self.n_warnings,
            "passed": self.passed,
            "violations": [
                {"column": v.column, "rule": v.rule, "message": v.message, "severity": v.severity}
                for v in self.violations
            ],
        }

    def summary(self) -> str:
        """Human-readable summary."""
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        lines = [
            f"Contract Validation: {status}",
            f"  Contract: {self.contract_name}",
            f"  Data: {self.data_path} ({self.n_rows} rows)",
            f"  Violations: {self.n_violations} ({self.n_errors} errors, {self.n_warnings} warnings)",
        ]
        for v in self.violations[:10]:
            lines.append(f"    [{v.severity.upper()}] {v.column}: {v.message}")
        if len(self.violations) > 10:
            lines.append(f"    ... and {len(self.violations) - 10} more")
        return "\n".join(lines)


def validate_data_file(
    data_path: str | Path,
    contract_path: str | Path,
    fail_on_error: bool = True,
    fail_on_warning: bool = False,
) -> ValidationReport:
    """Validate a data file against a contract — designed for CI/CD.

    Args:
        data_path: Path to CSV or Parquet data file.
        contract_path: Path to contract JSON file.
        fail_on_error: Return non-passing if any errors found.
        fail_on_warning: Return non-passing if any warnings found.

    Returns:
        ValidationReport with results.
    """
    data_path = Path(data_path)
    contract = DataContract.load(contract_path)

    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    validator = ContractValidator(contract, strictness="silent")
    violations = validator.validate(df)

    n_errors = sum(1 for v in violations if v.severity == "error")
    n_warnings = sum(1 for v in violations if v.severity != "error")

    passed = True
    if fail_on_error and n_errors > 0:
        passed = False
    if fail_on_warning and n_warnings > 0:
        passed = False

    return ValidationReport(
        contract_name=contract.name,
        data_path=str(data_path),
        n_rows=len(df),
        n_violations=len(violations),
        n_errors=n_errors,
        n_warnings=n_warnings,
        violations=violations,
        passed=passed,
    )


def generate_contract(
    data_path: str | Path,
    output_path: str | Path,
    name: str = "auto_contract",
    tolerance: float = 1.5,
    format: str = "json",
) -> DataContract:
    """Generate a contract from a data file and save it.

    Args:
        data_path: Path to training data CSV/Parquet.
        output_path: Where to save the contract.
        name: Contract name.
        tolerance: Range tolerance multiplier.
        format: Output format ('json' or 'yaml').

    Returns:
        Generated DataContract.
    """
    data_path = Path(data_path)

    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    gen = ContractGenerator(tolerance=tolerance)
    contract = gen.from_dataframe(df, name=name)

    output_path = Path(output_path)
    if format == "yaml":
        output_path.write_text(export_contract_yaml(contract))
    else:
        contract.save(output_path)

    return contract


def export_contract_yaml(contract: DataContract) -> str:
    """Export a data contract as YAML.

    Args:
        contract: DataContract to export.

    Returns:
        YAML string representation.
    """
    lines = [
        f"name: {contract.name}",
        f"version: {contract.version}",
        f"min_rows: {contract.min_rows}",
        "expected_columns:",
    ]
    for col in contract.expected_columns:
        lines.append(f"  - {col}")

    lines.append("columns:")
    for col_name, cc in contract.columns.items():
        lines.append(f"  {col_name}:")
        lines.append(f"    dtype: {cc.dtype}")
        lines.append(f"    nullable: {str(cc.nullable).lower()}")
        lines.append(f"    max_null_pct: {cc.max_null_pct}")
        if cc.min_value is not None:
            lines.append(f"    min_value: {cc.min_value}")
        if cc.max_value is not None:
            lines.append(f"    max_value: {cc.max_value}")
        if cc.mean_range is not None:
            lines.append(f"    mean_range: [{cc.mean_range[0]}, {cc.mean_range[1]}]")
        if cc.std_range is not None:
            lines.append(f"    std_range: [{cc.std_range[0]}, {cc.std_range[1]}]")
        if cc.allowed_values is not None:
            lines.append("    allowed_values:")
            for v in cc.allowed_values:
                lines.append(f"      - {v}")
        if cc.n_unique_range is not None:
            lines.append(f"    n_unique_range: [{cc.n_unique_range[0]}, {cc.n_unique_range[1]}]")

    return "\n".join(lines) + "\n"


def ci_validate(
    data_path: str,
    contract_path: str,
    fail_on_warning: bool = False,
) -> int:
    """Entry point for CI/CD pipeline validation.

    Returns 0 for pass, 1 for fail. Prints a summary to stdout.

    Args:
        data_path: Path to data file.
        contract_path: Path to contract file.
        fail_on_warning: Fail on warnings too.

    Returns:
        Exit code (0 = pass, 1 = fail).
    """
    report = validate_data_file(
        data_path, contract_path,
        fail_on_error=True, fail_on_warning=fail_on_warning,
    )
    print(report.summary())
    return 0 if report.passed else 1
