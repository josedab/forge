"""Auto-profiling data contracts for feature pipelines.

Generates data contracts from training data (expected schema, value
ranges, distribution shapes) and enforces them at transform-time
with configurable strictness.

Example:
    >>> from forge.contracts import ContractGenerator, ContractValidator
    >>> gen = ContractGenerator()
    >>> contract = gen.from_dataframe(X_train)
    >>> validator = ContractValidator(contract)
    >>> violations = validator.validate(X_new)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

Strictness = Literal["warn", "error", "silent"]


@dataclass
class ColumnContract:
    """Contract for a single column."""

    name: str
    dtype: str
    nullable: bool = True
    max_null_pct: float = 1.0
    n_unique_range: tuple[int, int] | None = None
    # Numeric constraints
    min_value: float | None = None
    max_value: float | None = None
    mean_range: tuple[float, float] | None = None
    std_range: tuple[float, float] | None = None
    # Categorical constraints
    allowed_values: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "nullable": self.nullable,
            "max_null_pct": self.max_null_pct,
            "n_unique_range": list(self.n_unique_range) if self.n_unique_range else None,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mean_range": list(self.mean_range) if self.mean_range else None,
            "std_range": list(self.std_range) if self.std_range else None,
            "allowed_values": self.allowed_values,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ColumnContract:
        return cls(
            name=data["name"],
            dtype=data["dtype"],
            nullable=data.get("nullable", True),
            max_null_pct=data.get("max_null_pct", 1.0),
            n_unique_range=tuple(data["n_unique_range"]) if data.get("n_unique_range") else None,
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            mean_range=tuple(data["mean_range"]) if data.get("mean_range") else None,
            std_range=tuple(data["std_range"]) if data.get("std_range") else None,
            allowed_values=data.get("allowed_values"),
        )


@dataclass
class DataContract:
    """Full data contract for a DataFrame."""

    name: str = "default"
    columns: dict[str, ColumnContract] = field(default_factory=dict)
    min_rows: int = 0
    expected_columns: list[str] = field(default_factory=list)
    version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "min_rows": self.min_rows,
            "expected_columns": self.expected_columns,
            "columns": {k: v.to_dict() for k, v in self.columns.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataContract:
        columns = {
            k: ColumnContract.from_dict(v)
            for k, v in data.get("columns", {}).items()
        }
        return cls(
            name=data.get("name", "default"),
            columns=columns,
            min_rows=data.get("min_rows", 0),
            expected_columns=data.get("expected_columns", []),
            version=data.get("version", "1.0.0"),
        )

    def save(self, path: str | Path) -> None:
        """Save contract to JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> DataContract:
        """Load contract from JSON file."""
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)


@dataclass
class Violation:
    """A single contract violation."""

    column: str
    rule: str
    message: str
    severity: str = "error"  # error, warning

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.column}: {self.message}"


class ContractGenerator:
    """Generates data contracts from training DataFrames.

    Parameters:
        tolerance: Multiplier for range tolerance (e.g., 1.5 = 50% wider).
        null_tolerance: Additional null pct tolerance above observed.
    """

    def __init__(
        self,
        tolerance: float = 1.5,
        null_tolerance: float = 0.1,
    ) -> None:
        self.tolerance = tolerance
        self.null_tolerance = null_tolerance

    def from_dataframe(
        self,
        df: pd.DataFrame,
        name: str = "auto_contract",
    ) -> DataContract:
        """Generate a contract from a DataFrame.

        Args:
            df: Training DataFrame to profile.
            name: Contract name.

        Returns:
            DataContract with column constraints.
        """
        columns: dict[str, ColumnContract] = {}

        for col_name in df.columns:
            col = df[col_name]
            columns[col_name] = self._profile_column(col)

        return DataContract(
            name=name,
            columns=columns,
            min_rows=max(1, len(df) // 10),
            expected_columns=list(df.columns),
        )

    def _profile_column(self, col: pd.Series) -> ColumnContract:
        """Profile a single column into a contract."""
        null_pct = float(col.isna().mean())
        n_unique = int(col.nunique())

        contract = ColumnContract(
            name=str(col.name),
            dtype=str(col.dtype),
            nullable=null_pct > 0,
            max_null_pct=min(1.0, null_pct + self.null_tolerance),
        )

        if pd.api.types.is_numeric_dtype(col):
            desc = col.dropna().describe()
            if len(desc) > 0:
                col_min = float(desc.get("min", 0))
                col_max = float(desc.get("max", 0))
                col_mean = float(desc.get("mean", 0))
                col_std = float(desc.get("std", 0))
                spread = max(abs(col_max - col_min) * (self.tolerance - 1), 1e-6)

                contract.min_value = col_min - spread
                contract.max_value = col_max + spread
                mean_spread = max(col_std * self.tolerance, 1e-6)
                contract.mean_range = (col_mean - mean_spread, col_mean + mean_spread)
                contract.std_range = (0, col_std * self.tolerance * 2)

        elif pd.api.types.is_object_dtype(col) or pd.api.types.is_categorical_dtype(col):
            if n_unique <= 100:
                contract.allowed_values = sorted(col.dropna().unique().astype(str).tolist())
            contract.n_unique_range = (1, max(n_unique * 2, 10))

        return contract


class ContractValidator:
    """Validates DataFrames against data contracts.

    Parameters:
        contract: DataContract to enforce.
        strictness: How to handle violations — 'error', 'warn', or 'silent'.
    """

    def __init__(
        self,
        contract: DataContract,
        strictness: Strictness = "warn",
    ) -> None:
        self.contract = contract
        self.strictness = strictness

    def validate(self, df: pd.DataFrame) -> list[Violation]:
        """Validate a DataFrame against the contract.

        Args:
            df: DataFrame to validate.

        Returns:
            List of Violation objects.

        Raises:
            ValueError: If strictness='error' and violations found.
        """
        violations: list[Violation] = []

        # Check row count
        if len(df) < self.contract.min_rows:
            violations.append(Violation(
                column="__dataframe__",
                rule="min_rows",
                message=f"Expected ≥{self.contract.min_rows} rows, got {len(df)}",
            ))

        # Check expected columns present
        for col_name in self.contract.expected_columns:
            if col_name not in df.columns:
                violations.append(Violation(
                    column=col_name,
                    rule="column_exists",
                    message=f"Expected column '{col_name}' not found",
                ))

        # Check column contracts
        for col_name, col_contract in self.contract.columns.items():
            if col_name not in df.columns:
                continue
            col_violations = self._validate_column(df[col_name], col_contract)
            violations.extend(col_violations)

        if violations and self.strictness == "error":
            msg = "\n".join(str(v) for v in violations)
            raise ValueError(f"Contract violations:\n{msg}")

        if violations and self.strictness == "warn":
            for v in violations:
                logger.warning(str(v))

        return violations

    def _validate_column(
        self, col: pd.Series, contract: ColumnContract
    ) -> list[Violation]:
        """Validate a single column."""
        violations: list[Violation] = []
        col_name = str(col.name)

        # Null check
        null_pct = float(col.isna().mean())
        if not contract.nullable and null_pct > 0:
            violations.append(Violation(
                column=col_name, rule="nullable",
                message=f"Column is not nullable but has {null_pct:.1%} nulls",
            ))
        elif null_pct > contract.max_null_pct:
            violations.append(Violation(
                column=col_name, rule="max_null_pct",
                message=f"Null rate {null_pct:.1%} exceeds max {contract.max_null_pct:.1%}",
            ))

        if pd.api.types.is_numeric_dtype(col):
            non_null = col.dropna()
            if len(non_null) > 0:
                # Value range
                if contract.min_value is not None:
                    actual_min = float(non_null.min())
                    if actual_min < contract.min_value:
                        violations.append(Violation(
                            column=col_name, rule="min_value",
                            message=f"Min value {actual_min:.4f} below contract min {contract.min_value:.4f}",
                        ))

                if contract.max_value is not None:
                    actual_max = float(non_null.max())
                    if actual_max > contract.max_value:
                        violations.append(Violation(
                            column=col_name, rule="max_value",
                            message=f"Max value {actual_max:.4f} above contract max {contract.max_value:.4f}",
                        ))

                # Mean drift
                if contract.mean_range is not None:
                    actual_mean = float(non_null.mean())
                    lo, hi = contract.mean_range
                    if actual_mean < lo or actual_mean > hi:
                        violations.append(Violation(
                            column=col_name, rule="mean_range",
                            message=f"Mean {actual_mean:.4f} outside range [{lo:.4f}, {hi:.4f}]",
                            severity="warning",
                        ))

        elif contract.allowed_values is not None:
            actual_values = set(col.dropna().astype(str).unique())
            unexpected = actual_values - set(contract.allowed_values)
            if unexpected:
                violations.append(Violation(
                    column=col_name, rule="allowed_values",
                    message=f"Unexpected values: {sorted(unexpected)[:5]}",
                    severity="warning",
                ))

        return violations
