"""Enhanced documentation generator with multi-format export.

Extends the base FeatureDocumentationGenerator with richer descriptions,
Markdown/HTML export, and cross-feature relationship summaries.

Example:
    >>> from forge.documentation.enhanced import EnhancedDocGenerator
    >>> gen = EnhancedDocGenerator()
    >>> doc = gen.generate(df, pipeline)
    >>> print(gen.to_markdown(doc))
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from forge.documentation.generator import DatasetDoc, FeatureDoc

logger = logging.getLogger(__name__)


@dataclass
class RelationshipInfo:
    """Describes a relationship between two features.

    Attributes:
        feature_a: First feature name.
        feature_b: Second feature name.
        correlation: Pearson correlation coefficient.
        relationship_type: Type of relationship.
    """

    feature_a: str
    feature_b: str
    correlation: float
    relationship_type: str = "linear"


@dataclass
class EnhancedDatasetDoc:
    """Extended dataset documentation with relationships and formatting.

    Attributes:
        base: The base DatasetDoc.
        relationships: Cross-feature relationships.
        summary_text: Auto-generated summary text.
    """

    base: DatasetDoc
    relationships: list[RelationshipInfo] = field(default_factory=list)
    summary_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "base": self.base.to_dict(),
            "relationships": [
                {
                    "feature_a": r.feature_a,
                    "feature_b": r.feature_b,
                    "correlation": r.correlation,
                    "relationship_type": r.relationship_type,
                }
                for r in self.relationships
            ],
            "summary_text": self.summary_text,
        }


class EnhancedDocGenerator:
    """Enhanced documentation generator with multi-format export.

    Parameters
    ----------
    correlation_threshold : float
        Minimum absolute correlation to report as a relationship.
    max_example_values : int
        Maximum example values to include per feature.
    """

    def __init__(
        self,
        correlation_threshold: float = 0.5,
        max_example_values: int = 5,
    ) -> None:
        self.correlation_threshold = correlation_threshold
        self.max_example_values = max_example_values

    def generate(
        self,
        df: pd.DataFrame,
        name: str = "dataset",
        description: str = "",
    ) -> EnhancedDatasetDoc:
        """Generate enhanced documentation for a DataFrame.

        Args:
            df: Input DataFrame.
            name: Dataset name.
            description: Dataset description.

        Returns:
            EnhancedDatasetDoc with features, relationships, and summary.
        """
        features: list[FeatureDoc] = []
        for col in df.columns:
            features.append(self._document_feature(df, col))

        base = DatasetDoc(
            name=name,
            features=features,
            description=description,
            n_samples=len(df),
            n_features=len(df.columns),
        )

        relationships = self._find_relationships(df)
        summary = self._generate_summary(base, relationships)

        return EnhancedDatasetDoc(
            base=base,
            relationships=relationships,
            summary_text=summary,
        )

    def to_markdown(self, doc: EnhancedDatasetDoc) -> str:
        """Export documentation as Markdown.

        Args:
            doc: Enhanced dataset documentation.

        Returns:
            Markdown string.
        """
        lines: list[str] = []
        base = doc.base
        lines.append(f"# {base.name}")
        lines.append("")
        if base.description:
            lines.append(base.description)
            lines.append("")
        lines.append(f"**Samples:** {base.n_samples}  ")
        lines.append(f"**Features:** {base.n_features}")
        lines.append("")

        # Summary
        if doc.summary_text:
            lines.append("## Summary")
            lines.append("")
            lines.append(doc.summary_text)
            lines.append("")

        # Feature table
        lines.append("## Features")
        lines.append("")
        lines.append("| Name | Type | Description | Nulls |")
        lines.append("|------|------|-------------|-------|")
        for feat in base.features:
            null_pct = ""
            if feat.statistics and "null_percentage" in feat.statistics:
                null_pct = f"{feat.statistics['null_percentage']:.1f}%"
            lines.append(
                f"| {feat.name} | {feat.dtype} "
                f"| {feat.description} | {null_pct} |"
            )
        lines.append("")

        # Relationships
        if doc.relationships:
            lines.append("## Feature Relationships")
            lines.append("")
            for r in doc.relationships:
                lines.append(
                    f"- **{r.feature_a}** ↔ **{r.feature_b}**: "
                    f"r={r.correlation:.3f} ({r.relationship_type})"
                )
            lines.append("")

        return "\n".join(lines)

    def to_html(self, doc: EnhancedDatasetDoc) -> str:
        """Export documentation as HTML.

        Args:
            doc: Enhanced dataset documentation.

        Returns:
            HTML string.
        """
        base = doc.base
        parts = [
            "<html><head><style>"
            "body{font-family:sans-serif;margin:2em;}"
            "table{border-collapse:collapse;width:100%;}"
            "th,td{border:1px solid #ccc;padding:8px;text-align:left;}"
            "th{background:#f5f5f5;}"
            "</style></head><body>"
        ]
        parts.append(f"<h1>{_escape(base.name)}</h1>")
        if base.description:
            parts.append(f"<p>{_escape(base.description)}</p>")
        parts.append(
            f"<p><strong>Samples:</strong> {base.n_samples} | "
            f"<strong>Features:</strong> {base.n_features}</p>"
        )

        if doc.summary_text:
            parts.append(f"<h2>Summary</h2><p>{_escape(doc.summary_text)}</p>")

        # Feature table
        parts.append("<h2>Features</h2><table>")
        parts.append("<tr><th>Name</th><th>Type</th><th>Description</th></tr>")
        for feat in base.features:
            parts.append(
                f"<tr><td>{_escape(feat.name)}</td>"
                f"<td>{_escape(feat.dtype)}</td>"
                f"<td>{_escape(feat.description)}</td></tr>"
            )
        parts.append("</table>")

        parts.append("</body></html>")
        return "\n".join(parts)

    def to_json(self, doc: EnhancedDatasetDoc) -> str:
        """Export documentation as JSON.

        Args:
            doc: Enhanced dataset documentation.

        Returns:
            JSON string.
        """
        return json.dumps(doc.to_dict(), indent=2, default=str)

    def _document_feature(
        self, df: pd.DataFrame, col: str,
    ) -> FeatureDoc:
        series = df[col]
        stats: dict[str, Any] = {
            "null_count": int(series.isna().sum()),
            "null_percentage": float(series.isna().mean() * 100),
        }
        dtype = str(series.dtype)
        description = self._auto_describe(col, series)
        category = self._infer_category(col, series)

        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if len(clean) > 0:
                stats["mean"] = float(clean.mean())
                stats["std"] = float(clean.std())
                stats["min"] = float(clean.min())
                stats["max"] = float(clean.max())

        examples = series.dropna().head(self.max_example_values).tolist()

        return FeatureDoc(
            name=col,
            dtype=dtype,
            description=description,
            statistics=stats,
            category=category,
            example_values=examples,
        )

    @staticmethod
    def _auto_describe(col: str, series: pd.Series) -> str:  # type: ignore[type-arg]
        parts = col.replace("_", " ").replace("-", " ").strip()
        dtype = series.dtype
        if pd.api.types.is_numeric_dtype(dtype):
            return f"Numeric feature: {parts}"
        if pd.api.types.is_bool_dtype(dtype):
            return f"Boolean flag: {parts}"
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return f"Timestamp: {parts}"
        return f"Categorical/text feature: {parts}"

    @staticmethod
    def _infer_category(col: str, series: pd.Series) -> str:  # type: ignore[type-arg]
        if pd.api.types.is_datetime64_any_dtype(series.dtype):
            return "temporal"
        if pd.api.types.is_numeric_dtype(series.dtype):
            return "numeric"
        if pd.api.types.is_bool_dtype(series.dtype):
            return "boolean"
        return "categorical"

    def _find_relationships(
        self, df: pd.DataFrame,
    ) -> list[RelationshipInfo]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return []

        corr = df[numeric_cols].corr()
        relationships: list[RelationshipInfo] = []
        seen: set[tuple[str, str]] = set()

        for i, col_a in enumerate(numeric_cols):
            for col_b in numeric_cols[i + 1 :]:
                key = (col_a, col_b)
                if key in seen:
                    continue
                seen.add(key)
                r = corr.loc[col_a, col_b]
                if abs(r) >= self.correlation_threshold:
                    rtype = "positive" if r > 0 else "negative"
                    relationships.append(
                        RelationshipInfo(
                            feature_a=col_a,
                            feature_b=col_b,
                            correlation=float(r),
                            relationship_type=rtype,
                        )
                    )
        relationships.sort(key=lambda x: abs(x.correlation), reverse=True)
        return relationships

    @staticmethod
    def _generate_summary(
        base: DatasetDoc, relationships: list[RelationshipInfo],
    ) -> str:
        n_numeric = sum(
            1 for f in base.features
            if f.category == "numeric"
        )
        n_cat = sum(
            1 for f in base.features
            if f.category == "categorical"
        )
        parts = [
            f"Dataset '{base.name}' contains {base.n_samples} samples "
            f"and {base.n_features} features "
            f"({n_numeric} numeric, {n_cat} categorical)."
        ]
        if relationships:
            parts.append(
                f" Found {len(relationships)} notable correlations "
                f"(|r| >= threshold)."
            )
        return "".join(parts)


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
