"""Export feature documentation to various formats."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from forge.documentation.generator import DatasetDoc, FeatureDoc
from forge.documentation.catalog import FeatureCatalog

if TYPE_CHECKING:
    pass


class BaseExporter(ABC):
    """Abstract base class for documentation exporters."""

    @abstractmethod
    def export_dataset(self, dataset: DatasetDoc, path: str | Path) -> None:
        """Export dataset documentation."""
        pass

    @abstractmethod
    def export_catalog(self, catalog: FeatureCatalog, path: str | Path) -> None:
        """Export catalog documentation."""
        pass

    @abstractmethod
    def export_feature(self, feature: FeatureDoc) -> str:
        """Export a single feature to string."""
        pass


class MarkdownExporter(BaseExporter):
    """Export documentation to Markdown format.

    Example:
        >>> exporter = MarkdownExporter()
        >>> exporter.export_dataset(dataset_doc, "features.md")
    """

    def __init__(
        self,
        include_statistics: bool = True,
        include_examples: bool = True,
        include_quality_notes: bool = True,
        include_toc: bool = True,
    ) -> None:
        """Initialize Markdown exporter.

        Args:
            include_statistics: Include statistics tables.
            include_examples: Include example values.
            include_quality_notes: Include quality notes.
            include_toc: Include table of contents.
        """
        self.include_statistics = include_statistics
        self.include_examples = include_examples
        self.include_quality_notes = include_quality_notes
        self.include_toc = include_toc

    def export_dataset(self, dataset: DatasetDoc, path: str | Path) -> None:
        """Export dataset documentation to Markdown.

        Args:
            dataset: Dataset documentation.
            path: Output file path.
        """
        content = self._render_dataset(dataset)
        Path(path).write_text(content)

    def export_catalog(self, catalog: FeatureCatalog, path: str | Path) -> None:
        """Export catalog to Markdown.

        Args:
            catalog: Feature catalog.
            path: Output file path.
        """
        content = self._render_catalog(catalog)
        Path(path).write_text(content)

    def export_feature(self, feature: FeatureDoc) -> str:
        """Export a single feature to Markdown string.

        Args:
            feature: Feature documentation.

        Returns:
            Markdown string.
        """
        return self._render_feature(feature)

    def _render_dataset(self, dataset: DatasetDoc) -> str:
        """Render dataset documentation."""
        lines = [
            f"# {dataset.name}",
            "",
            dataset.description if dataset.description else "*No description provided*",
            "",
            "## Overview",
            "",
            f"- **Samples**: {dataset.n_samples:,}",
            f"- **Features**: {dataset.n_features}",
            f"- **Created**: {dataset.created_at}",
            "",
        ]

        # Quality summary
        if dataset.quality_summary:
            lines.extend([
                "## Quality Summary",
                "",
                f"- Missing values: {dataset.quality_summary.get('missing_percentage', 0):.1f}%",
                f"- Features with issues: {dataset.quality_summary.get('features_with_quality_issues', 0)}",
                f"- Constant features: {dataset.quality_summary.get('constant_features', 0)}",
                "",
            ])

            # Category breakdown
            categories = dataset.quality_summary.get("categories", {})
            if categories:
                lines.extend(["### Categories", ""])
                for cat, count in sorted(categories.items()):
                    lines.append(f"- {cat}: {count}")
                lines.append("")

        # Table of contents
        if self.include_toc and dataset.features:
            lines.extend(["## Table of Contents", ""])
            for feature in dataset.features:
                anchor = feature.name.lower().replace("_", "-").replace(" ", "-")
                lines.append(f"- [{feature.name}](#{anchor})")
            lines.append("")

        # Features
        lines.extend(["## Features", ""])
        for feature in dataset.features:
            lines.append(self._render_feature(feature))
            lines.append("")

        return "\n".join(lines)

    def _render_feature(self, feature: FeatureDoc) -> str:
        """Render a single feature."""
        lines = [
            f"### {feature.name}",
            "",
            feature.description if feature.description else "*No description*",
            "",
            f"- **Type**: {feature.dtype}",
            f"- **Category**: {feature.category}",
        ]

        if feature.tags:
            lines.append(f"- **Tags**: {', '.join(feature.tags)}")

        if feature.transformation and feature.transformation != "original":
            lines.append(f"- **Transformation**: {feature.transformation}")

        if feature.source_columns and feature.source_columns != [feature.name]:
            lines.append(f"- **Source columns**: {', '.join(feature.source_columns)}")

        lines.append("")

        # Statistics
        if self.include_statistics and feature.statistics:
            lines.extend(self._render_statistics(feature.statistics))

        # Examples
        if self.include_examples and feature.example_values:
            lines.extend([
                "**Example values**:",
                f"`{feature.example_values}`",
                "",
            ])

        # Quality notes
        if self.include_quality_notes and feature.quality_notes:
            lines.extend([
                "**Quality notes**:",
                "",
            ])
            for note in feature.quality_notes:
                lines.append(f"- {note}")
            lines.append("")

        return "\n".join(lines)

    def _render_statistics(self, stats: dict[str, Any]) -> list[str]:
        """Render statistics table."""
        lines = ["**Statistics**:", ""]

        # Numeric stats table
        numeric_stats = ["count", "missing", "mean", "std", "min", "max", "median"]
        has_numeric = any(k in stats for k in numeric_stats[2:])

        if has_numeric:
            lines.append("| Statistic | Value |")
            lines.append("|-----------|-------|")
            for stat in numeric_stats:
                if stat in stats and stats[stat] is not None:
                    value = stats[stat]
                    if isinstance(value, float):
                        lines.append(f"| {stat} | {value:.4f} |")
                    else:
                        lines.append(f"| {stat} | {value} |")
            lines.append("")
        else:
            # Basic stats
            if "count" in stats:
                lines.append(f"- Count: {stats['count']}")
            if "missing" in stats:
                lines.append(f"- Missing: {stats['missing']} ({stats.get('missing_pct', 0):.1f}%)")
            if "unique" in stats:
                lines.append(f"- Unique values: {stats['unique']}")
            lines.append("")

        # Top values for categorical
        if "top_values" in stats:
            lines.extend(["**Top values**:", ""])
            for val, count in list(stats["top_values"].items())[:5]:
                lines.append(f"- `{val}`: {count}")
            lines.append("")

        return lines

    def _render_catalog(self, catalog: FeatureCatalog) -> str:
        """Render catalog documentation."""
        stats = catalog.get_statistics()

        lines = [
            "# Feature Catalog",
            "",
            "## Overview",
            "",
            f"- **Total features**: {stats['total_features']}",
            f"- **Datasets**: {stats['total_datasets']}",
            "",
        ]

        # Categories
        if stats.get("categories"):
            lines.extend(["### Categories", ""])
            for cat, count in sorted(stats["categories"].items()):
                lines.append(f"- {cat}: {count}")
            lines.append("")

        # Features by dataset
        for dataset_name in catalog.list_datasets():
            entries = catalog.search(dataset=dataset_name)
            lines.extend([
                f"## {dataset_name}",
                "",
                f"*{len(entries)} features*",
                "",
            ])

            for entry in entries:
                lines.append(f"### {entry.feature_doc.name}")
                lines.append("")
                lines.append(entry.feature_doc.description or "*No description*")
                lines.append("")
                lines.append(f"- **Type**: {entry.feature_doc.dtype}")
                lines.append(f"- **Status**: {entry.status}")
                if entry.owner:
                    lines.append(f"- **Owner**: {entry.owner}")
                lines.append("")

        return "\n".join(lines)


class HTMLExporter(BaseExporter):
    """Export documentation to HTML format.

    Example:
        >>> exporter = HTMLExporter()
        >>> exporter.export_dataset(dataset_doc, "features.html")
    """

    def __init__(
        self,
        include_statistics: bool = True,
        include_search: bool = True,
        theme: Literal["light", "dark"] = "light",
    ) -> None:
        """Initialize HTML exporter.

        Args:
            include_statistics: Include statistics.
            include_search: Include JavaScript search.
            theme: Color theme.
        """
        self.include_statistics = include_statistics
        self.include_search = include_search
        self.theme = theme

    def export_dataset(self, dataset: DatasetDoc, path: str | Path) -> None:
        """Export dataset to HTML."""
        content = self._render_html(dataset)
        Path(path).write_text(content)

    def export_catalog(self, catalog: FeatureCatalog, path: str | Path) -> None:
        """Export catalog to HTML."""
        # Convert catalog to a combined dataset for rendering
        all_features = [entry.feature_doc for entry in catalog]
        combined = DatasetDoc(
            name="Feature Catalog",
            description=f"Combined catalog with {len(all_features)} features",
            features=all_features,
            n_features=len(all_features),
        )
        self.export_dataset(combined, path)

    def export_feature(self, feature: FeatureDoc) -> str:
        """Export feature to HTML string."""
        return self._render_feature_html(feature)

    def _render_html(self, dataset: DatasetDoc) -> str:
        """Render full HTML document."""
        bg_color = "#ffffff" if self.theme == "light" else "#1a1a1a"
        text_color = "#333333" if self.theme == "light" else "#e0e0e0"
        card_bg = "#f8f9fa" if self.theme == "light" else "#2d2d2d"

        features_html = "\n".join(
            self._render_feature_html(f) for f in dataset.features
        )

        search_script = """
        <script>
        function searchFeatures() {
            const query = document.getElementById('search').value.toLowerCase();
            const cards = document.querySelectorAll('.feature-card');
            cards.forEach(card => {
                const text = card.textContent.toLowerCase();
                card.style.display = text.includes(query) ? 'block' : 'none';
            });
        }
        </script>
        """ if self.include_search else ""

        search_input = """
        <div style="margin-bottom: 20px;">
            <input type="text" id="search" placeholder="Search features..."
                   onkeyup="searchFeatures()"
                   style="width: 100%; padding: 10px; font-size: 16px; border: 1px solid #ddd; border-radius: 4px;">
        </div>
        """ if self.include_search else ""

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{dataset.name} - Feature Documentation</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: {text_color};
            background-color: {bg_color};
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{ color: {text_color}; }}
        .feature-card {{
            background: {card_bg};
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .feature-name {{
            font-size: 1.2em;
            font-weight: bold;
            color: #2563eb;
            margin-bottom: 10px;
        }}
        .tag {{
            display: inline-block;
            background: #e0e7ff;
            color: #3730a3;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            margin-right: 5px;
        }}
        .stat-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }}
        .stat-table th, .stat-table td {{
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .quality-note {{
            background: #fef3c7;
            border-left: 4px solid #f59e0b;
            padding: 10px;
            margin: 10px 0;
        }}
        .overview {{
            background: {card_bg};
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
    </style>
    {search_script}
</head>
<body>
    <h1>{dataset.name}</h1>
    <div class="overview">
        <p>{dataset.description or 'Feature documentation'}</p>
        <p><strong>Samples:</strong> {dataset.n_samples:,} |
           <strong>Features:</strong> {dataset.n_features}</p>
    </div>
    {search_input}
    <div id="features">
        {features_html}
    </div>
</body>
</html>"""

    def _render_feature_html(self, feature: FeatureDoc) -> str:
        """Render a feature card."""
        tags_html = " ".join(f'<span class="tag">{tag}</span>' for tag in feature.tags[:5])

        stats_html = ""
        if self.include_statistics and feature.statistics:
            rows = []
            for key in ["count", "missing", "mean", "std", "min", "max"]:
                if key in feature.statistics and feature.statistics[key] is not None:
                    val = feature.statistics[key]
                    if isinstance(val, float):
                        rows.append(f"<tr><td>{key}</td><td>{val:.4f}</td></tr>")
                    else:
                        rows.append(f"<tr><td>{key}</td><td>{val}</td></tr>")
            if rows:
                stats_html = f"""
                <table class="stat-table">
                    <tr><th>Statistic</th><th>Value</th></tr>
                    {"".join(rows)}
                </table>
                """

        quality_html = ""
        if feature.quality_notes:
            notes = "<br>".join(feature.quality_notes)
            quality_html = f'<div class="quality-note">{notes}</div>'

        return f"""
        <div class="feature-card" data-name="{feature.name}">
            <div class="feature-name">{feature.name}</div>
            <p>{feature.description}</p>
            <p><strong>Type:</strong> {feature.dtype} |
               <strong>Category:</strong> {feature.category}</p>
            <div>{tags_html}</div>
            {stats_html}
            {quality_html}
        </div>
        """


class JSONExporter(BaseExporter):
    """Export documentation to JSON format.

    Example:
        >>> exporter = JSONExporter()
        >>> exporter.export_dataset(dataset_doc, "features.json")
    """

    def __init__(self, indent: int = 2) -> None:
        """Initialize JSON exporter.

        Args:
            indent: JSON indentation level.
        """
        self.indent = indent

    def export_dataset(self, dataset: DatasetDoc, path: str | Path) -> None:
        """Export dataset to JSON."""
        content = json.dumps(dataset.to_dict(), indent=self.indent, default=str)
        Path(path).write_text(content)

    def export_catalog(self, catalog: FeatureCatalog, path: str | Path) -> None:
        """Export catalog to JSON."""
        catalog.save(path)

    def export_feature(self, feature: FeatureDoc) -> str:
        """Export feature to JSON string."""
        return json.dumps(feature.to_dict(), indent=self.indent, default=str)


def export_documentation(
    source: DatasetDoc | FeatureCatalog,
    path: str | Path,
    format: Literal["markdown", "html", "json"] = "markdown",
    **kwargs: Any,
) -> None:
    """Export documentation to a file.

    Args:
        source: Dataset documentation or catalog.
        path: Output file path.
        format: Output format.
        **kwargs: Additional arguments for exporter.
    """
    exporters = {
        "markdown": MarkdownExporter,
        "html": HTMLExporter,
        "json": JSONExporter,
    }

    if format not in exporters:
        raise ValueError(f"Unknown format: {format}. Supported: {list(exporters.keys())}")

    exporter = exporters[format](**kwargs)

    if isinstance(source, DatasetDoc):
        exporter.export_dataset(source, path)
    else:
        exporter.export_catalog(source, path)
