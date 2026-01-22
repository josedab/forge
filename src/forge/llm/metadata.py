"""Metadata extraction for LLM-powered feature discovery."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from forge.analyzer.type_inference import TypeInferrer
from forge.types import ColumnType


@dataclass
class ColumnMetadata:
    """Rich metadata about a single column."""

    name: str
    dtype: str
    inferred_type: ColumnType
    null_count: int
    null_ratio: float
    unique_count: int
    unique_ratio: float
    sample_values: list[Any]
    statistics: dict[str, Any] = field(default_factory=dict)
    patterns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "inferred_type": self.inferred_type.value,
            "null_count": self.null_count,
            "null_ratio": round(self.null_ratio, 4),
            "unique_count": self.unique_count,
            "unique_ratio": round(self.unique_ratio, 4),
            "sample_values": [str(v) for v in self.sample_values[:5]],
            "statistics": self.statistics,
            "patterns": self.patterns,
        }

    def to_prompt_string(self) -> str:
        """Convert to a concise string for LLM prompts."""
        lines = [
            f"Column: {self.name}",
            f"  Type: {self.inferred_type.value} (dtype: {self.dtype})",
            f"  Missing: {self.null_ratio:.1%} ({self.null_count} nulls)",
            f"  Unique: {self.unique_count} values ({self.unique_ratio:.1%})",
        ]

        if self.sample_values:
            samples = ", ".join(str(v) for v in self.sample_values[:3])
            lines.append(f"  Samples: {samples}")

        if self.statistics:
            stats_str = ", ".join(
                f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in list(self.statistics.items())[:5]
            )
            lines.append(f"  Stats: {stats_str}")

        if self.patterns:
            lines.append(f"  Patterns: {', '.join(self.patterns[:3])}")

        return "\n".join(lines)


@dataclass
class DatasetMetadata:
    """Comprehensive metadata about a dataset."""

    n_rows: int
    n_columns: int
    columns: dict[str, ColumnMetadata]
    column_types_summary: dict[str, int]
    correlations: dict[tuple[str, str], float] = field(default_factory=dict)
    target_info: dict[str, Any] | None = None
    domain_hints: list[str] = field(default_factory=list)
    metadata_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "columns": {k: v.to_dict() for k, v in self.columns.items()},
            "column_types_summary": self.column_types_summary,
            "correlations": {f"{k[0]}|{k[1]}": v for k, v in self.correlations.items()},
            "target_info": self.target_info,
            "domain_hints": self.domain_hints,
        }

    def to_prompt_string(self, max_columns: int = 20) -> str:
        """Convert to a prompt-friendly string representation."""
        lines = [
            f"Dataset Overview: {self.n_rows:,} rows × {self.n_columns} columns",
            "",
            "Column Type Distribution:",
        ]

        for col_type, count in sorted(
            self.column_types_summary.items(), key=lambda x: -x[1]
        ):
            lines.append(f"  - {col_type}: {count}")

        if self.domain_hints:
            lines.append("")
            lines.append(f"Domain Hints: {', '.join(self.domain_hints)}")

        lines.append("")
        lines.append("Column Details:")

        for i, (_, col_meta) in enumerate(self.columns.items()):
            if i >= max_columns:
                lines.append(f"  ... and {len(self.columns) - max_columns} more columns")
                break
            lines.append(col_meta.to_prompt_string())
            lines.append("")

        if self.target_info:
            lines.append("Target Variable:")
            for k, v in self.target_info.items():
                lines.append(f"  {k}: {v}")

        if self.correlations:
            lines.append("")
            lines.append("High Correlations:")
            sorted_corrs = sorted(
                self.correlations.items(), key=lambda x: abs(x[1]), reverse=True
            )[:10]
            for (col1, col2), corr in sorted_corrs:
                lines.append(f"  {col1} <-> {col2}: {corr:.3f}")

        return "\n".join(lines)

    def compute_hash(self) -> str:
        """Compute a hash for caching purposes."""
        content = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.md5(content.encode()).hexdigest()[:16]


class MetadataExtractor:
    """Extracts rich metadata from DataFrames for LLM analysis.

    This class analyzes a DataFrame and produces structured metadata
    that can be efficiently passed to an LLM for feature suggestions.

    Example:
        >>> extractor = MetadataExtractor()
        >>> metadata = extractor.extract(X, y)
        >>> print(metadata.to_prompt_string())
    """

    def __init__(
        self,
        max_sample_values: int = 5,
        correlation_threshold: float = 0.5,
        categorical_threshold: int = 50,
    ) -> None:
        """Initialize the metadata extractor.

        Args:
            max_sample_values: Maximum sample values to store per column.
            correlation_threshold: Minimum correlation to report.
            categorical_threshold: Max unique values for categorical detection.
        """
        self.max_sample_values = max_sample_values
        self.correlation_threshold = correlation_threshold
        self.type_inferrer = TypeInferrer(categorical_threshold=categorical_threshold)

    def extract(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> DatasetMetadata:
        """Extract comprehensive metadata from a DataFrame.

        Args:
            X: Input DataFrame.
            y: Optional target variable.

        Returns:
            DatasetMetadata with rich column information.
        """
        column_types = self.type_inferrer.infer_types(X)
        columns: dict[str, ColumnMetadata] = {}

        for col in X.columns:
            columns[col] = self._extract_column_metadata(
                X[col], col, column_types.get(col, ColumnType.UNKNOWN)
            )

        type_counts = {}
        for col_type in column_types.values():
            type_counts[col_type.value] = type_counts.get(col_type.value, 0) + 1

        correlations = self._compute_correlations(X, column_types)
        target_info = self._extract_target_info(y) if y is not None else None
        domain_hints = self._infer_domain_hints(X, column_types)

        metadata = DatasetMetadata(
            n_rows=len(X),
            n_columns=len(X.columns),
            columns=columns,
            column_types_summary=type_counts,
            correlations=correlations,
            target_info=target_info,
            domain_hints=domain_hints,
        )
        metadata.metadata_hash = metadata.compute_hash()

        return metadata

    def _extract_column_metadata(
        self, series: pd.Series, name: str, col_type: ColumnType
    ) -> ColumnMetadata:
        """Extract metadata for a single column."""
        non_null = series.dropna()
        samples = (
            non_null.sample(min(self.max_sample_values, len(non_null))).tolist()
            if len(non_null) > 0
            else []
        )

        statistics = self._compute_column_statistics(series, col_type)
        patterns = self._detect_patterns(series, name, col_type)

        return ColumnMetadata(
            name=name,
            dtype=str(series.dtype),
            inferred_type=col_type,
            null_count=int(series.isna().sum()),
            null_ratio=float(series.isna().mean()),
            unique_count=int(series.nunique()),
            unique_ratio=float(series.nunique() / len(series)) if len(series) > 0 else 0,
            sample_values=samples,
            statistics=statistics,
            patterns=patterns,
        )

    def _compute_column_statistics(
        self, series: pd.Series, col_type: ColumnType
    ) -> dict[str, Any]:
        """Compute statistics based on column type."""
        stats: dict[str, Any] = {}

        if col_type == ColumnType.NUMERIC:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if len(numeric) > 0:
                stats["mean"] = float(numeric.mean())
                stats["std"] = float(numeric.std())
                stats["min"] = float(numeric.min())
                stats["max"] = float(numeric.max())
                stats["median"] = float(numeric.median())
                stats["skewness"] = float(numeric.skew())
                q1, q3 = numeric.quantile([0.25, 0.75])
                stats["q1"] = float(q1)
                stats["q3"] = float(q3)
                stats["iqr"] = float(q3 - q1)
                if stats["std"] > 0:
                    stats["cv"] = float(stats["std"] / abs(stats["mean"])) if stats["mean"] != 0 else None

        elif col_type == ColumnType.CATEGORICAL:
            value_counts = series.value_counts()
            if len(value_counts) > 0:
                stats["mode"] = str(value_counts.index[0])
                stats["mode_frequency"] = float(value_counts.iloc[0] / len(series))
                stats["top_5"] = list(value_counts.head(5).index.astype(str))

        elif col_type == ColumnType.DATETIME:
            try:
                dt = pd.to_datetime(series, errors="coerce").dropna()
                if len(dt) > 0:
                    stats["min_date"] = str(dt.min())
                    stats["max_date"] = str(dt.max())
                    stats["date_range_days"] = (dt.max() - dt.min()).days
            except Exception:
                pass

        elif col_type == ColumnType.TEXT:
            text = series.dropna().astype(str)
            if len(text) > 0:
                stats["avg_length"] = float(text.str.len().mean())
                stats["max_length"] = int(text.str.len().max())
                stats["avg_word_count"] = float(text.str.split().str.len().mean())

        return stats

    def _detect_patterns(
        self, series: pd.Series, name: str, col_type: ColumnType
    ) -> list[str]:
        """Detect patterns in column name and values."""
        patterns = []
        name_lower = name.lower()

        # Name-based patterns
        if any(kw in name_lower for kw in ["id", "key", "code"]):
            patterns.append("identifier")
        if any(kw in name_lower for kw in ["date", "time", "timestamp", "created", "updated"]):
            patterns.append("temporal")
        if any(kw in name_lower for kw in ["price", "cost", "amount", "revenue", "income"]):
            patterns.append("monetary")
        if any(kw in name_lower for kw in ["count", "qty", "quantity", "num", "number"]):
            patterns.append("count")
        if any(kw in name_lower for kw in ["ratio", "rate", "percent", "pct"]):
            patterns.append("ratio")
        if any(kw in name_lower for kw in ["lat", "lon", "latitude", "longitude", "geo"]):
            patterns.append("geospatial")
        if any(kw in name_lower for kw in ["email", "phone", "address", "url"]):
            patterns.append("contact")
        if any(kw in name_lower for kw in ["name", "title", "description"]):
            patterns.append("text_field")
        if any(kw in name_lower for kw in ["flag", "is_", "has_", "bool"]):
            patterns.append("boolean_indicator")
        if any(kw in name_lower for kw in ["category", "type", "class", "status", "state"]):
            patterns.append("categorical_indicator")
        if "_" in name or name != name_lower:
            if name.endswith("_at") or name.endswith("_date"):
                patterns.append("timestamp_suffix")
            if name.endswith("_id"):
                patterns.append("foreign_key")

        # Value-based patterns for numeric columns
        if col_type == ColumnType.NUMERIC:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if len(numeric) > 0:
                if (numeric >= 0).all() and (numeric <= 1).all():
                    patterns.append("probability_range")
                if (numeric == numeric.astype(int)).all():
                    patterns.append("integer_values")
                if (numeric > 0).all():
                    patterns.append("positive_only")

        return patterns

    def _compute_correlations(
        self, X: pd.DataFrame, column_types: dict[str, ColumnType]
    ) -> dict[tuple[str, str], float]:
        """Compute significant correlations between numeric columns."""
        numeric_cols = [
            col for col, ctype in column_types.items()
            if ctype == ColumnType.NUMERIC
        ]

        if len(numeric_cols) < 2:
            return {}

        correlations = {}
        try:
            numeric_df = X[numeric_cols].apply(pd.to_numeric, errors="coerce")
            corr_matrix = numeric_df.corr()

            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i + 1:]:
                    corr_val = corr_matrix.loc[col1, col2]
                    if pd.notna(corr_val) and abs(corr_val) >= self.correlation_threshold:
                        correlations[(col1, col2)] = float(corr_val)
        except Exception:
            pass

        return correlations

    def _extract_target_info(self, y: pd.Series) -> dict[str, Any]:
        """Extract information about the target variable."""
        info: dict[str, Any] = {
            "name": y.name if y.name else "target",
            "dtype": str(y.dtype),
            "null_count": int(y.isna().sum()),
            "unique_count": int(y.nunique()),
        }

        if y.dtype in ["int64", "float64"]:
            unique_ratio = y.nunique() / len(y)
            if unique_ratio < 0.05 or y.nunique() <= 10:
                info["task_type"] = "classification"
                info["classes"] = list(y.unique()[:10])
                info["class_distribution"] = y.value_counts().head(10).to_dict()
            else:
                info["task_type"] = "regression"
                info["mean"] = float(y.mean())
                info["std"] = float(y.std())
                info["min"] = float(y.min())
                info["max"] = float(y.max())
        else:
            info["task_type"] = "classification"
            info["classes"] = list(y.unique()[:10])
            info["class_distribution"] = y.value_counts().head(10).to_dict()

        return info

    def _infer_domain_hints(
        self, X: pd.DataFrame, column_types: dict[str, ColumnType]
    ) -> list[str]:
        """Infer domain hints from column names and patterns."""
        hints = set()
        col_names_lower = [c.lower() for c in X.columns]

        # Finance/E-commerce indicators
        finance_keywords = ["price", "revenue", "cost", "profit", "transaction", "payment"]
        if any(kw in " ".join(col_names_lower) for kw in finance_keywords):
            hints.add("finance_or_ecommerce")

        # Healthcare indicators
        health_keywords = ["patient", "diagnosis", "medical", "health", "drug", "treatment"]
        if any(kw in " ".join(col_names_lower) for kw in health_keywords):
            hints.add("healthcare")

        # Marketing indicators
        marketing_keywords = ["campaign", "click", "impression", "conversion", "customer"]
        if any(kw in " ".join(col_names_lower) for kw in marketing_keywords):
            hints.add("marketing")

        # Time series indicators
        if any(ct == ColumnType.DATETIME for ct in column_types.values()):
            hints.add("temporal_data")

        # Geospatial indicators
        geo_keywords = ["lat", "lon", "latitude", "longitude", "city", "country", "region"]
        if any(kw in " ".join(col_names_lower) for kw in geo_keywords):
            hints.add("geospatial")

        # Text-heavy dataset
        text_count = sum(1 for ct in column_types.values() if ct == ColumnType.TEXT)
        if text_count > len(column_types) * 0.3:
            hints.add("text_heavy")

        return list(hints)
