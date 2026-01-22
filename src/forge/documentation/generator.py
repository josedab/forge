"""Feature documentation generator."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from forge.types import ColumnType

if TYPE_CHECKING:
    pass


@dataclass
class FeatureDoc:
    """Documentation for a single feature."""

    name: str
    dtype: str
    description: str = ""
    statistics: dict[str, Any] = field(default_factory=dict)
    source_columns: list[str] = field(default_factory=list)
    transformation: str = ""
    category: str = "unknown"
    tags: list[str] = field(default_factory=list)
    quality_notes: list[str] = field(default_factory=list)
    example_values: list[Any] = field(default_factory=list)
    created_at: str = ""
    version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "dtype": self.dtype,
            "statistics": self.statistics,
            "source_columns": self.source_columns,
            "transformation": self.transformation,
            "category": self.category,
            "tags": self.tags,
            "quality_notes": self.quality_notes,
            "example_values": self.example_values,
            "created_at": self.created_at,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureDoc:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            dtype=data.get("dtype", "unknown"),
            statistics=data.get("statistics", {}),
            source_columns=data.get("source_columns", []),
            transformation=data.get("transformation", ""),
            category=data.get("category", "unknown"),
            tags=data.get("tags", []),
            quality_notes=data.get("quality_notes", []),
            example_values=data.get("example_values", []),
            created_at=data.get("created_at", ""),
            version=data.get("version", "1.0.0"),
        )

    def compute_hash(self) -> str:
        """Compute a hash for this feature doc."""
        content = f"{self.name}:{self.dtype}:{self.transformation}:{sorted(self.source_columns)}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class DatasetDoc:
    """Documentation for an entire dataset."""

    name: str
    features: list[FeatureDoc] = field(default_factory=list)
    description: str = ""
    n_samples: int = 0
    n_features: int = 0
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    quality_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "features": [f.to_dict() for f in self.features],
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "quality_summary": self.quality_summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetDoc:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            features=[FeatureDoc.from_dict(f) for f in data.get("features", [])],
            n_samples=data.get("n_samples", 0),
            n_features=data.get("n_features", 0),
            created_at=data.get("created_at", ""),
            metadata=data.get("metadata", {}),
            quality_summary=data.get("quality_summary", {}),
        )

    def get_feature(self, name: str) -> FeatureDoc | None:
        """Get feature documentation by name."""
        for f in self.features:
            if f.name == name:
                return f
        return None

    def filter_by_category(self, category: str) -> list[FeatureDoc]:
        """Filter features by category."""
        return [f for f in self.features if f.category == category]

    def filter_by_tag(self, tag: str) -> list[FeatureDoc]:
        """Filter features by tag."""
        return [f for f in self.features if tag in f.tags]


class FeatureDocumentationGenerator:
    """Generates documentation for features automatically.

    Analyzes features and creates comprehensive documentation
    including descriptions, statistics, quality notes, and more.

    Example:
        >>> generator = FeatureDocumentationGenerator()
        >>> docs = generator.generate(X, feature_descriptions={"price": "Product price"})
        >>> generator.export_markdown(docs, "features.md")
    """

    # Category inference rules
    CATEGORY_PATTERNS = {
        "temporal": ["date", "time", "year", "month", "day", "hour", "minute", "second", "timestamp"],
        "identifier": ["id", "key", "code", "uuid", "guid"],
        "geographic": ["lat", "lon", "latitude", "longitude", "geo", "location", "address", "zip", "postal", "country", "city", "state"],
        "financial": ["price", "cost", "amount", "revenue", "profit", "fee", "tax", "salary", "income"],
        "demographic": ["age", "gender", "sex", "birth", "occupation", "education", "marital"],
        "metric": ["count", "total", "sum", "avg", "mean", "rate", "ratio", "percent", "score"],
        "boolean": ["is_", "has_", "flag", "active", "enabled", "valid"],
    }

    def __init__(
        self,
        include_statistics: bool = True,
        include_quality_notes: bool = True,
        include_examples: bool = True,
        n_examples: int = 5,
        auto_describe: bool = True,
    ) -> None:
        """Initialize the documentation generator.

        Args:
            include_statistics: Include statistical summaries.
            include_quality_notes: Include data quality notes.
            include_examples: Include example values.
            n_examples: Number of example values to include.
            auto_describe: Auto-generate descriptions from names.
        """
        self.include_statistics = include_statistics
        self.include_quality_notes = include_quality_notes
        self.include_examples = include_examples
        self.n_examples = n_examples
        self.auto_describe = auto_describe

    def generate(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        dataset_name: str = "Dataset",
        dataset_description: str = "",
        feature_descriptions: dict[str, str] | None = None,
        feature_tags: dict[str, list[str]] | None = None,
    ) -> DatasetDoc:
        """Generate documentation for a dataset.

        Args:
            X: Input DataFrame.
            y: Optional target variable.
            dataset_name: Name for the dataset.
            dataset_description: Description of the dataset.
            feature_descriptions: Manual descriptions for features.
            feature_tags: Manual tags for features.

        Returns:
            DatasetDoc with comprehensive documentation.
        """
        feature_descriptions = feature_descriptions or {}
        feature_tags = feature_tags or {}

        feature_docs = []
        for col in X.columns:
            doc = self._document_feature(
                X[col],
                name=col,
                description=feature_descriptions.get(col),
                tags=feature_tags.get(col, []),
            )
            feature_docs.append(doc)

        # Generate quality summary
        quality_summary = self._generate_quality_summary(X, feature_docs)

        return DatasetDoc(
            name=dataset_name,
            description=dataset_description,
            features=feature_docs,
            n_samples=len(X),
            n_features=len(X.columns),
            created_at=datetime.now().isoformat(),
            metadata={
                "target_column": y.name if y is not None else None,
                "memory_usage_mb": X.memory_usage(deep=True).sum() / 1024 / 1024,
            },
            quality_summary=quality_summary,
        )

    def _document_feature(
        self,
        series: pd.Series,
        name: str,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> FeatureDoc:
        """Document a single feature."""
        # Infer dtype
        dtype = self._infer_dtype(series)

        # Infer category
        category = self._infer_category(name, dtype, series)

        # Generate description
        if description:
            desc = description
        elif self.auto_describe:
            desc = self._auto_describe(name, dtype, category)
        else:
            desc = ""

        # Compute statistics
        statistics = {}
        if self.include_statistics:
            statistics = self._compute_statistics(series, dtype)

        # Quality notes
        quality_notes = []
        if self.include_quality_notes:
            quality_notes = self._check_quality(series, dtype)

        # Example values
        example_values = []
        if self.include_examples:
            example_values = self._get_examples(series)

        # Auto-generate tags
        auto_tags = self._infer_tags(name, dtype, series)
        all_tags = list(set((tags or []) + auto_tags))

        return FeatureDoc(
            name=name,
            description=desc,
            dtype=dtype,
            statistics=statistics,
            source_columns=[name],
            transformation="original",
            category=category,
            tags=all_tags,
            quality_notes=quality_notes,
            example_values=example_values,
            created_at=datetime.now().isoformat(),
        )

    def _infer_dtype(self, series: pd.Series) -> str:
        """Infer the data type of a series."""
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        elif pd.api.types.is_integer_dtype(series):
            return "integer"
        elif pd.api.types.is_float_dtype(series):
            return "float"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        elif isinstance(series.dtype, pd.CategoricalDtype):
            return "categorical"
        elif pd.api.types.is_object_dtype(series):
            # Check if it looks like categorical
            nunique = series.nunique()
            if nunique < len(series) * 0.5 and nunique < 100:
                return "categorical"
            return "string"
        else:
            return "unknown"

    def _infer_category(self, name: str, dtype: str, series: pd.Series) -> str:
        """Infer the category of a feature."""
        name_lower = name.lower()

        for category, patterns in self.CATEGORY_PATTERNS.items():
            for pattern in patterns:
                if pattern in name_lower:
                    return category

        # Infer from dtype
        if dtype == "datetime":
            return "temporal"
        elif dtype == "boolean":
            return "boolean"
        elif dtype in ["integer", "float"]:
            return "numeric"
        elif dtype in ["categorical", "string"]:
            return "categorical"

        return "other"

    def _auto_describe(self, name: str, dtype: str, category: str) -> str:
        """Auto-generate a description from the feature name."""
        # Convert name to readable format
        readable = name.replace("_", " ").replace("-", " ")
        readable = " ".join(word.capitalize() for word in readable.split())

        type_desc = {
            "integer": "integer value",
            "float": "numeric value",
            "boolean": "boolean flag",
            "datetime": "date/time",
            "categorical": "categorical value",
            "string": "text value",
        }.get(dtype, "value")

        category_desc = {
            "temporal": "representing a time-related measurement",
            "identifier": "used as an identifier",
            "geographic": "representing geographic information",
            "financial": "representing financial data",
            "demographic": "representing demographic information",
            "metric": "representing a calculated metric",
            "boolean": "indicating a true/false condition",
        }.get(category, "")

        desc = f"{readable} ({type_desc})"
        if category_desc:
            desc += f" {category_desc}"

        return desc

    def _compute_statistics(self, series: pd.Series, dtype: str) -> dict[str, Any]:
        """Compute statistics for a feature."""
        stats: dict[str, Any] = {
            "count": int(series.count()),
            "missing": int(series.isna().sum()),
            "missing_pct": float(series.isna().mean() * 100),
            "unique": int(series.nunique()),
        }

        if dtype in ["integer", "float"]:
            numeric_series = pd.to_numeric(series, errors="coerce")
            stats.update({
                "mean": float(numeric_series.mean()) if not numeric_series.isna().all() else None,
                "std": float(numeric_series.std()) if not numeric_series.isna().all() else None,
                "min": float(numeric_series.min()) if not numeric_series.isna().all() else None,
                "max": float(numeric_series.max()) if not numeric_series.isna().all() else None,
                "median": float(numeric_series.median()) if not numeric_series.isna().all() else None,
                "q25": float(numeric_series.quantile(0.25)) if not numeric_series.isna().all() else None,
                "q75": float(numeric_series.quantile(0.75)) if not numeric_series.isna().all() else None,
            })
            # Skewness and kurtosis
            if len(numeric_series.dropna()) > 2:
                stats["skewness"] = float(numeric_series.skew())
                stats["kurtosis"] = float(numeric_series.kurtosis())

        elif dtype == "categorical":
            value_counts = series.value_counts()
            stats["top_values"] = value_counts.head(5).to_dict()
            stats["cardinality"] = int(series.nunique())

        elif dtype == "datetime":
            dt_series = pd.to_datetime(series, errors="coerce")
            stats["min_date"] = str(dt_series.min()) if not dt_series.isna().all() else None
            stats["max_date"] = str(dt_series.max()) if not dt_series.isna().all() else None

        return stats

    def _check_quality(self, series: pd.Series, dtype: str) -> list[str]:
        """Check data quality and generate notes."""
        notes = []

        # Missing values
        missing_pct = series.isna().mean() * 100
        if missing_pct > 0:
            if missing_pct > 50:
                notes.append(f"High missing rate ({missing_pct:.1f}%) - consider dropping or imputing")
            elif missing_pct > 10:
                notes.append(f"Moderate missing rate ({missing_pct:.1f}%) - imputation recommended")
            else:
                notes.append(f"Low missing rate ({missing_pct:.1f}%)")

        # Cardinality for categorical
        if dtype in ["categorical", "string"]:
            nunique = series.nunique()
            if nunique == 1:
                notes.append("Constant value - no predictive power")
            elif nunique == len(series):
                notes.append("High cardinality (all unique) - likely an identifier")
            elif nunique > len(series) * 0.9:
                notes.append("Very high cardinality - consider encoding or dropping")

        # Numeric checks
        if dtype in ["integer", "float"]:
            numeric_series = pd.to_numeric(series, errors="coerce")

            # Zeros
            zero_pct = (numeric_series == 0).mean() * 100
            if zero_pct > 50:
                notes.append(f"High percentage of zeros ({zero_pct:.1f}%)")

            # Outliers (using IQR)
            q1, q3 = numeric_series.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                outliers = ((numeric_series < q1 - 1.5 * iqr) | (numeric_series > q3 + 1.5 * iqr)).mean() * 100
                if outliers > 5:
                    notes.append(f"Contains outliers ({outliers:.1f}% of values)")

            # Skewness
            if len(numeric_series.dropna()) > 2:
                skew = abs(numeric_series.skew())
                if skew > 2:
                    notes.append(f"Highly skewed distribution (skewness: {skew:.2f}) - consider transformation")

        return notes

    def _get_examples(self, series: pd.Series) -> list[Any]:
        """Get example values from a series."""
        non_null = series.dropna()
        if len(non_null) == 0:
            return []

        # Get diverse examples
        unique_vals = non_null.unique()
        if len(unique_vals) <= self.n_examples:
            examples = unique_vals.tolist()
        else:
            # Sample from unique values
            indices = np.linspace(0, len(unique_vals) - 1, self.n_examples, dtype=int)
            examples = [unique_vals[i] for i in indices]

        # Convert to JSON-serializable types
        serializable = []
        for val in examples:
            if isinstance(val, (np.integer, np.floating)):
                serializable.append(float(val) if isinstance(val, np.floating) else int(val))
            elif isinstance(val, (pd.Timestamp, datetime)):
                serializable.append(str(val))
            elif isinstance(val, np.ndarray):
                serializable.append(val.tolist())
            else:
                serializable.append(val)

        return serializable

    def _infer_tags(self, name: str, dtype: str, series: pd.Series) -> list[str]:
        """Infer tags for a feature."""
        tags = []
        name_lower = name.lower()

        # Type tags
        tags.append(f"type:{dtype}")

        # Pattern-based tags
        if "id" in name_lower or name_lower.endswith("_id"):
            tags.append("identifier")
        if any(p in name_lower for p in ["price", "cost", "amount", "revenue"]):
            tags.append("financial")
        if any(p in name_lower for p in ["date", "time", "timestamp"]):
            tags.append("temporal")

        # Quality-based tags
        if series.isna().any():
            tags.append("has_missing")
        if series.nunique() == 1:
            tags.append("constant")
        if series.nunique() == len(series):
            tags.append("unique")

        return tags

    def _generate_quality_summary(
        self, X: pd.DataFrame, feature_docs: list[FeatureDoc]
    ) -> dict[str, Any]:
        """Generate overall quality summary."""
        total_missing = X.isna().sum().sum()
        total_cells = X.shape[0] * X.shape[1]

        features_with_issues = sum(1 for f in feature_docs if f.quality_notes)
        constant_features = sum(1 for f in feature_docs if "constant" in f.tags)

        return {
            "total_missing_values": int(total_missing),
            "missing_percentage": float(total_missing / total_cells * 100) if total_cells > 0 else 0,
            "features_with_quality_issues": features_with_issues,
            "constant_features": constant_features,
            "categories": self._count_by_category(feature_docs),
            "dtypes": self._count_by_dtype(feature_docs),
        }

    def _count_by_category(self, feature_docs: list[FeatureDoc]) -> dict[str, int]:
        """Count features by category."""
        counts: dict[str, int] = {}
        for f in feature_docs:
            counts[f.category] = counts.get(f.category, 0) + 1
        return counts

    def _count_by_dtype(self, feature_docs: list[FeatureDoc]) -> dict[str, int]:
        """Count features by dtype."""
        counts: dict[str, int] = {}
        for f in feature_docs:
            counts[f.dtype] = counts.get(f.dtype, 0) + 1
        return counts


def generate_feature_docs(
    X: pd.DataFrame,
    y: pd.Series | None = None,
    dataset_name: str = "Dataset",
    **kwargs: Any,
) -> DatasetDoc:
    """Convenience function to generate feature documentation.

    Args:
        X: Input DataFrame.
        y: Optional target variable.
        dataset_name: Name for the dataset.
        **kwargs: Additional arguments for generator.

    Returns:
        DatasetDoc with documentation.
    """
    generator = FeatureDocumentationGenerator(**kwargs)
    return generator.generate(X, y, dataset_name=dataset_name)


# Alias for consistency
generate_documentation = generate_feature_docs
