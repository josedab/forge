"""Type definitions for Forge."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Type aliases
ArrayLike = Union[np.ndarray, pd.Series, list[Any]]
DataFrameLike = Union[pd.DataFrame, dict[str, ArrayLike]]
FeatureMatrix = Union[pd.DataFrame, "NDArray[np.floating[Any]]"]
Target = Union[pd.Series, "NDArray[Any]"]


class ColumnType(str, Enum):
    """Enumeration of column types."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


class SelectionMethod(str, Enum):
    """Feature selection methods."""

    IMPORTANCE = "importance"
    STATISTICAL = "statistical"
    CORRELATION = "correlation"
    VARIANCE = "variance"
    SHAP = "shap"


class ImputationStrategy(str, Enum):
    """Missing value imputation strategies."""

    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"
    KNN = "knn"
    ITERATIVE = "iterative"
    AUTO = "auto"


class EncodingStrategy(str, Enum):
    """Categorical encoding strategies."""

    ONEHOT = "onehot"
    TARGET = "target"
    FREQUENCY = "frequency"
    ORDINAL = "ordinal"
    BINARY = "binary"
    AUTO = "auto"


@dataclass
class ColumnInfo:
    """Information about a DataFrame column."""

    name: str
    dtype: str
    inferred_type: ColumnType
    null_count: int
    null_ratio: float
    unique_count: int
    unique_ratio: float
    sample_values: list[Any] = field(default_factory=list)

    @property
    def is_numeric(self) -> bool:
        return self.inferred_type == ColumnType.NUMERIC

    @property
    def is_categorical(self) -> bool:
        return self.inferred_type == ColumnType.CATEGORICAL

    @property
    def is_datetime(self) -> bool:
        return self.inferred_type == ColumnType.DATETIME

    @property
    def is_text(self) -> bool:
        return self.inferred_type == ColumnType.TEXT

    @property
    def has_missing(self) -> bool:
        return self.null_count > 0


@dataclass
class FeatureInfo:
    """Information about a generated feature."""

    name: str
    source_columns: list[str]
    generator: str
    importance: float | None = None
    selected: bool = True


@dataclass
class QualityIssue:
    """Represents a data quality issue."""

    column: str
    issue_type: str
    severity: str  # "low", "medium", "high"
    description: str
    suggestion: str | None = None


@dataclass
class AnalysisReport:
    """Report from data analysis."""

    n_rows: int
    n_columns: int
    column_info: dict[str, ColumnInfo]
    column_types: dict[str, ColumnType]
    quality_issues: list[QualityIssue]
    statistics: dict[str, dict[str, Any]]
    target_info: dict[str, Any] | None = None

    def summary(self) -> str:
        """Return a summary of the analysis."""
        lines = [
            f"Dataset: {self.n_rows} rows x {self.n_columns} columns",
            "",
            "Column Types:",
        ]

        type_counts: dict[ColumnType, int] = {}
        for col_type in self.column_types.values():
            type_counts[col_type] = type_counts.get(col_type, 0) + 1

        for col_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  - {col_type.value}: {count}")

        if self.quality_issues:
            lines.append("")
            lines.append(f"Quality Issues: {len(self.quality_issues)}")
            for issue in self.quality_issues[:5]:
                lines.append(f"  - [{issue.severity}] {issue.column}: {issue.description}")
            if len(self.quality_issues) > 5:
                lines.append(f"  ... and {len(self.quality_issues) - 5} more")

        return "\n".join(lines)


# Protocol for sklearn-compatible transformers
T = TypeVar("T", bound="TransformerProtocol")


class TransformerProtocol(Protocol):
    """Protocol for sklearn-compatible transformers."""

    def fit(self: T, X: pd.DataFrame, y: pd.Series | None = None) -> T: ...

    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame: ...

    def get_feature_names_out(self) -> list[str]: ...


class SelectorProtocol(Protocol):
    """Protocol for feature selectors."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> SelectorProtocol: ...

    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...

    def get_support(self, indices: bool = False) -> NDArray[Any] | list[str]: ...
