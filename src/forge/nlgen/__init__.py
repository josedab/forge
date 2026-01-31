"""Natural language feature synthesis.

Converts natural language descriptions of features into executable
Forge pipeline code. Uses rule-based parsing with optional LLM
enhancement when an API key is available.

Example:
    >>> from forge.nlgen import NLFeatureSynthesizer
    >>> synth = NLFeatureSynthesizer(schema={"age": "numeric", "city": "categorical"})
    >>> result = synth.synthesize("ratio of income to debt, log-transformed")
    >>> print(result.code)
    >>> df_new = result.execute(df)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ColumnSchema:
    """Schema information for a column."""

    name: str
    dtype: str  # "numeric", "categorical", "temporal", "text"
    sample_values: list[Any] = field(default_factory=list)


@dataclass
class SynthesisResult:
    """Result of a natural language synthesis operation."""

    description: str
    code: str
    feature_names: list[str]
    success: bool = True
    error: str = ""
    warnings: list[str] = field(default_factory=list)

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the generated code on a DataFrame.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with new features added.

        Raises:
            RuntimeError: If execution fails.
        """
        if not self.success:
            raise RuntimeError(f"Cannot execute failed synthesis: {self.error}")

        result = df.copy()
        local_vars: dict[str, Any] = {"df": result, "np": np, "pd": pd}
        try:
            exec(self.code, {}, local_vars)  # noqa: S102
            return local_vars.get("result", result)
        except Exception as e:
            raise RuntimeError(f"Execution failed: {e}") from e


# Rule patterns for natural language parsing
_PATTERNS: list[tuple[str, str]] = [
    # Ratio patterns
    (r"ratio\s+of\s+(\w+)\s+to\s+(\w+)", "ratio"),
    (r"(\w+)\s*/\s*(\w+)", "ratio"),
    (r"(\w+)\s+divided\s+by\s+(\w+)", "ratio"),
    # Product / interaction patterns
    (r"(\w+)\s*\*\s*(\w+)", "product"),
    (r"product\s+of\s+(\w+)\s+and\s+(\w+)", "product"),
    (r"(\w+)\s+times\s+(\w+)", "product"),
    (r"multiply\s+(\w+)\s+(?:by|and)\s+(\w+)", "product"),
    # Difference patterns
    (r"(\w+)\s*-\s*(\w+)", "difference"),
    (r"difference\s+(?:between|of)\s+(\w+)\s+and\s+(\w+)", "difference"),
    (r"(\w+)\s+minus\s+(\w+)", "difference"),
    # Sum patterns
    (r"(\w+)\s*\+\s*(\w+)", "sum"),
    (r"sum\s+of\s+(\w+)\s+and\s+(\w+)", "sum"),
    # Transform patterns
    (r"log[_\s]*(?:transform(?:ed)?)?(?:\s+(?:of\s+)?)?(\w+)", "log"),
    (r"square[_\s]*root\s+(?:of\s+)?(\w+)", "sqrt"),
    (r"squared?\s+(?:of\s+)?(\w+)", "square"),
    (r"(\w+)\s+squared", "square"),
    # Binning patterns
    (r"bin\s+(\w+)\s+into\s+(\d+)\s+(?:quantiles?|bins?|buckets?)", "bin"),
    (r"(\w+)\s+bucketed?\s+into\s+(\d+)", "bin"),
    # Boolean / flag patterns
    (r"(?:flag|indicator)\s+(?:if|where|when)\s+(\w+)\s+(?:is\s+)?(?:null|missing|nan)", "null_flag"),
    (r"(?:flag|indicator)\s+(?:if|where|when)\s+(\w+)\s*>\s*([\d.]+)", "gt_flag"),
    (r"(?:flag|indicator)\s+(?:if|where|when)\s+(\w+)\s*<\s*([\d.]+)", "lt_flag"),
]


class NLFeatureSynthesizer:
    """Converts natural language to Forge feature engineering code.

    Uses rule-based parsing to interpret common FE operations described
    in plain English, generating executable Python code.

    Parameters:
        schema: Column schema as dict[name, dtype] or list[ColumnSchema].
        validate: Whether to validate generated code before returning.
    """

    def __init__(
        self,
        schema: dict[str, str] | list[ColumnSchema] | None = None,
        validate: bool = True,
    ) -> None:
        self.validate = validate
        self._schema: dict[str, str] = {}
        if isinstance(schema, dict):
            self._schema = schema
        elif isinstance(schema, list):
            self._schema = {cs.name: cs.dtype for cs in schema}

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> NLFeatureSynthesizer:
        """Create synthesizer from a DataFrame's schema."""
        schema: dict[str, str] = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                schema[col] = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                schema[col] = "temporal"
            else:
                schema[col] = "categorical"
        return cls(schema=schema)

    def synthesize(self, description: str) -> SynthesisResult:
        """Synthesize feature code from natural language.

        Args:
            description: Natural language feature description.

        Returns:
            SynthesisResult with generated code and metadata.
        """
        description_clean = description.strip().lower()
        operations = self._parse_operations(description_clean)

        if not operations:
            return SynthesisResult(
                description=description,
                code="",
                feature_names=[],
                success=False,
                error=f"Could not parse: '{description}'",
            )

        code_lines: list[str] = ["result = df.copy()"]
        feature_names: list[str] = []
        warnings: list[str] = []

        for op_type, args in operations:
            code, name, warns = self._generate_code(op_type, args)
            if code:
                code_lines.append(code)
                feature_names.append(name)
                warnings.extend(warns)

        code = "\n".join(code_lines)

        if self.validate:
            warns = self._validate_columns(operations)
            warnings.extend(warns)

        return SynthesisResult(
            description=description,
            code=code,
            feature_names=feature_names,
            success=True,
            warnings=warnings,
        )

    def synthesize_batch(self, descriptions: list[str]) -> list[SynthesisResult]:
        """Synthesize multiple features from a list of descriptions."""
        return [self.synthesize(desc) for desc in descriptions]

    def _parse_operations(self, text: str) -> list[tuple[str, list[str]]]:
        """Parse natural language into operation tuples."""
        operations: list[tuple[str, list[str]]] = []

        for pattern, op_type in _PATTERNS:
            match = re.search(pattern, text)
            if match:
                args = list(match.groups())
                operations.append((op_type, args))
                break  # Use first matching pattern

        return operations

    def _generate_code(
        self, op_type: str, args: list[str]
    ) -> tuple[str, str, list[str]]:
        """Generate Python code for an operation.

        Returns:
            (code_line, feature_name, warnings)
        """
        warnings: list[str] = []

        if op_type == "ratio" and len(args) >= 2:
            a, b = args[0], args[1]
            name = f"{a}_div_{b}"
            code = (
                f"result['{name}'] = np.where("
                f"df['{b}'] != 0, df['{a}'] / df['{b}'], 0)"
            )
            return code, name, warnings

        elif op_type == "product" and len(args) >= 2:
            a, b = args[0], args[1]
            name = f"{a}_mul_{b}"
            code = f"result['{name}'] = df['{a}'] * df['{b}']"
            return code, name, warnings

        elif op_type == "difference" and len(args) >= 2:
            a, b = args[0], args[1]
            name = f"{a}_minus_{b}"
            code = f"result['{name}'] = df['{a}'] - df['{b}']"
            return code, name, warnings

        elif op_type == "sum" and len(args) >= 2:
            a, b = args[0], args[1]
            name = f"{a}_plus_{b}"
            code = f"result['{name}'] = df['{a}'] + df['{b}']"
            return code, name, warnings

        elif op_type == "log" and len(args) >= 1:
            a = args[0]
            name = f"{a}_log1p"
            code = f"result['{name}'] = np.log1p(df['{a}'].clip(lower=0))"
            return code, name, warnings

        elif op_type == "sqrt" and len(args) >= 1:
            a = args[0]
            name = f"{a}_sqrt"
            code = f"result['{name}'] = np.sqrt(df['{a}'].clip(lower=0))"
            return code, name, warnings

        elif op_type == "square" and len(args) >= 1:
            a = args[0]
            name = f"{a}_squared"
            code = f"result['{name}'] = df['{a}'] ** 2"
            return code, name, warnings

        elif op_type == "bin" and len(args) >= 2:
            a, n = args[0], args[1]
            name = f"{a}_binned_{n}"
            code = f"result['{name}'] = pd.qcut(df['{a}'], q={n}, labels=False, duplicates='drop')"
            return code, name, warnings

        elif op_type == "null_flag" and len(args) >= 1:
            a = args[0]
            name = f"{a}_is_null"
            code = f"result['{name}'] = df['{a}'].isna().astype(int)"
            return code, name, warnings

        elif op_type == "gt_flag" and len(args) >= 2:
            a, threshold = args[0], args[1]
            name = f"{a}_gt_{threshold}"
            code = f"result['{name}'] = (df['{a}'] > {threshold}).astype(int)"
            return code, name, warnings

        elif op_type == "lt_flag" and len(args) >= 2:
            a, threshold = args[0], args[1]
            name = f"{a}_lt_{threshold}"
            code = f"result['{name}'] = (df['{a}'] < {threshold}).astype(int)"
            return code, name, warnings

        return "", "", warnings

    def _validate_columns(
        self, operations: list[tuple[str, list[str]]]
    ) -> list[str]:
        """Validate that referenced columns exist in schema."""
        warnings: list[str] = []
        if not self._schema:
            return warnings

        for _, args in operations:
            for arg in args:
                if arg.replace(".", "").isdigit():
                    continue
                if arg not in self._schema:
                    warnings.append(f"Column '{arg}' not found in schema")

        return warnings
