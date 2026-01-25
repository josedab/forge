"""Transformation grammar defining the search space for feature discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class TransformOp(str, Enum):
    """Available transformation operations."""

    # Unary operations
    LOG = "log"
    LOG1P = "log1p"
    SQRT = "sqrt"
    SQUARE = "square"
    CUBE = "cube"
    ABS = "abs"
    RECIPROCAL = "reciprocal"
    SIGMOID = "sigmoid"

    # Binary operations
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"

    # Aggregation operations
    ZSCORE = "zscore"
    MINMAX = "minmax"
    RANK = "rank"
    BINNING = "binning"

    # Identity
    IDENTITY = "identity"


# Classification of operations
UNARY_OPS = {
    TransformOp.LOG, TransformOp.LOG1P, TransformOp.SQRT,
    TransformOp.SQUARE, TransformOp.CUBE, TransformOp.ABS,
    TransformOp.RECIPROCAL, TransformOp.SIGMOID, TransformOp.ZSCORE,
    TransformOp.MINMAX, TransformOp.RANK, TransformOp.BINNING,
    TransformOp.IDENTITY,
}

BINARY_OPS = {
    TransformOp.ADD, TransformOp.SUBTRACT,
    TransformOp.MULTIPLY, TransformOp.DIVIDE,
}


@dataclass
class TransformNode:
    """A single transformation node in a feature program.

    Represents one transformation step. Can be a unary operation
    (applied to one column) or binary (applied to two columns).

    Args:
        op: The transformation operation.
        columns: Source column name(s).
        params: Optional parameters for the operation.
    """

    op: TransformOp
    columns: list[str]
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Generate a descriptive feature name."""
        cols = "_".join(self.columns)
        if self.op == TransformOp.IDENTITY:
            return cols
        return f"{self.op.value}_{cols}"

    @property
    def is_unary(self) -> bool:
        return self.op in UNARY_OPS

    @property
    def is_binary(self) -> bool:
        return self.op in BINARY_OPS

    def apply(self, X: pd.DataFrame) -> pd.Series:
        """Apply this transformation to the data.

        Args:
            X: Input DataFrame.

        Returns:
            Transformed Series.
        """
        if self.op == TransformOp.IDENTITY:
            return X[self.columns[0]].copy()

        if self.is_unary:
            col = X[self.columns[0]]
            return _apply_unary(col, self.op, self.params)

        # Binary
        left = X[self.columns[0]]
        right = X[self.columns[1]]
        return _apply_binary(left, right, self.op)


def _apply_unary(
    series: pd.Series, op: TransformOp, params: dict[str, Any]
) -> pd.Series:
    """Apply a unary transformation."""
    if op == TransformOp.LOG:
        return np.log(series.clip(lower=1e-10))  # type: ignore[no-any-return]
    if op == TransformOp.LOG1P:
        return np.log1p(series.clip(lower=0))  # type: ignore[no-any-return]
    if op == TransformOp.SQRT:
        return np.sqrt(series.clip(lower=0))  # type: ignore[no-any-return]
    if op == TransformOp.SQUARE:
        return series ** 2
    if op == TransformOp.CUBE:
        return series ** 3
    if op == TransformOp.ABS:
        return series.abs()
    if op == TransformOp.RECIPROCAL:
        return 1.0 / series.replace(0, np.nan)
    if op == TransformOp.SIGMOID:
        return 1.0 / (1.0 + np.exp(-series.clip(-500, 500)))  # type: ignore[no-any-return]
    if op == TransformOp.ZSCORE:
        std = series.std()
        if std == 0:
            return pd.Series(0.0, index=series.index)
        return (series - series.mean()) / std
    if op == TransformOp.MINMAX:
        rng = series.max() - series.min()
        if rng == 0:
            return pd.Series(0.0, index=series.index)
        return (series - series.min()) / rng  # type: ignore[no-any-return]
    if op == TransformOp.RANK:
        return series.rank(pct=True)
    if op == TransformOp.BINNING:
        n_bins = params.get("n_bins", 10)
        return pd.cut(series, bins=n_bins, labels=False, duplicates="drop")  # type: ignore[no-any-return]
    return series.copy()


def _apply_binary(
    left: pd.Series, right: pd.Series, op: TransformOp
) -> pd.Series:
    """Apply a binary transformation."""
    if op == TransformOp.ADD:
        return left + right  # type: ignore[no-any-return]
    if op == TransformOp.SUBTRACT:
        return left - right  # type: ignore[no-any-return]
    if op == TransformOp.MULTIPLY:
        return left * right  # type: ignore[no-any-return]
    if op == TransformOp.DIVIDE:
        return left / right.replace(0, np.nan)  # type: ignore[no-any-return]
    return left.copy()


class TransformGrammar:
    """Defines the search space of feature transformations.

    Controls which operations are allowed, maximum depth of
    composed transformations, and which columns can be used.

    Args:
        unary_ops: Allowed unary operations. None for all.
        binary_ops: Allowed binary operations. None for all.
        max_depth: Maximum depth of transformation chains.
        allow_self_interaction: Whether a column can interact with itself.

    Example:
        >>> grammar = TransformGrammar(max_depth=2)
        >>> programs = grammar.sample_programs(X, n=100, rng=rng)
    """

    def __init__(
        self,
        unary_ops: list[TransformOp] | None = None,
        binary_ops: list[TransformOp] | None = None,
        max_depth: int = 2,
        allow_self_interaction: bool = False,
    ) -> None:
        self.unary_ops = unary_ops or list(UNARY_OPS - {TransformOp.IDENTITY})
        self.binary_ops = binary_ops or list(BINARY_OPS)
        self.max_depth = max_depth
        self.allow_self_interaction = allow_self_interaction

    def sample_programs(
        self,
        X: pd.DataFrame,
        n: int,
        rng: np.random.RandomState | None = None,
    ) -> list[list[TransformNode]]:
        """Sample random transformation programs from the grammar.

        A "program" is a list of TransformNode steps.

        Args:
            X: Input DataFrame (used for column names).
            n: Number of programs to generate.
            rng: Random state for reproducibility.

        Returns:
            List of transformation programs.
        """
        if rng is None:
            rng = np.random.RandomState()

        numeric_cols: list[str] = list(X.select_dtypes(include=[np.number]).columns)
        if not numeric_cols:
            return []

        programs: list[list[TransformNode]] = []
        for _ in range(n):
            depth = rng.randint(1, self.max_depth + 1)
            program = self._sample_one(numeric_cols, depth, rng)
            programs.append(program)
        return programs

    def _sample_one(
        self,
        columns: list[str],
        depth: int,
        rng: np.random.RandomState,
    ) -> list[TransformNode]:
        """Sample a single transformation program."""
        program: list[TransformNode] = []

        # First step: choose unary or binary on original columns
        if rng.random() < 0.5 and len(columns) >= 2:
            # Binary operation
            op = self.binary_ops[rng.randint(len(self.binary_ops))]
            if self.allow_self_interaction:
                cols = list(rng.choice(columns, size=2, replace=True))
            else:
                cols = list(rng.choice(columns, size=2, replace=False))
            program.append(TransformNode(op=op, columns=cols))
        else:
            # Unary operation
            op = self.unary_ops[rng.randint(len(self.unary_ops))]
            col = str(rng.choice(columns))
            program.append(TransformNode(op=op, columns=[col]))

        # Additional depth: chain unary operations
        for _ in range(1, depth):
            op = self.unary_ops[rng.randint(len(self.unary_ops))]
            # The column reference here is symbolic — will be applied in order
            program.append(TransformNode(op=op, columns=["_prev"]))

        return program

    def evaluate_program(
        self, program: list[TransformNode], X: pd.DataFrame
    ) -> pd.Series:
        """Execute a transformation program on data.

        Args:
            program: List of transformation nodes.
            X: Input DataFrame.

        Returns:
            Resulting Series from the transformation chain.
        """
        result = program[0].apply(X)
        for node in program[1:]:
            # Chain: apply unary op on previous result
            temp = pd.DataFrame({"_prev": result})
            result = _apply_unary(temp["_prev"], node.op, node.params)
        return result

    def program_name(self, program: list[TransformNode]) -> str:
        """Generate a human-readable name for a program."""
        parts: list[str] = []
        for node in program:
            if node.columns == ["_prev"]:
                parts.append(node.op.value)
            else:
                parts.append(node.name)
        return "->".join(parts)
