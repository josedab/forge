"""Tests for documentation exporters."""

import json
import tempfile
from pathlib import Path

import pytest

from forge.documentation.generator import FeatureDoc, DatasetDoc
from forge.documentation.catalog import FeatureCatalog
from forge.documentation.export import (
    MarkdownExporter,
    HTMLExporter,
    JSONExporter,
    export_documentation,
)


@pytest.fixture
def sample_feature():
    """Create sample feature doc."""
    return FeatureDoc(
        name="user_age",
        dtype="int64",
        description="User age in years",
        category="numeric",
        tags=["demographic", "user"],
        statistics={
            "count": 1000,
            "missing": 5,
            "mean": 35.5,
            "std": 12.3,
            "min": 18,
            "max": 85,
        },
        example_values=[25, 30, 45, 62],
        quality_notes=["5 missing values (0.5%)"],
    )


@pytest.fixture
def sample_dataset(sample_feature):
    """Create sample dataset doc."""
    features = [
        sample_feature,
        FeatureDoc(
            name="income",
            dtype="float64",
            description="Annual income",
            category="numeric",
            tags=["financial"],
            statistics={"count": 1000, "mean": 55000.0},
        ),
        FeatureDoc(
            name="category",
            dtype="object",
            description="Product category",
            category="categorical",
            tags=["product"],
            statistics={
                "count": 1000,
                "unique": 5,
                "top_values": {"Electronics": 300, "Clothing": 250},
            },
        ),
    ]

    return DatasetDoc(
        name="Customer Data",
        description="Customer transaction dataset",
        features=features,
        n_samples=1000,
        n_features=3,
        quality_summary={
            "missing_percentage": 0.5,
            "features_with_quality_issues": 1,
            "constant_features": 0,
            "categories": {"numeric": 2, "categorical": 1},
        },
    )


@pytest.fixture
def sample_catalog(sample_dataset):
    """Create sample catalog."""
    catalog = FeatureCatalog()
    catalog.add_dataset(sample_dataset, owner="data_team")
    return catalog


class TestMarkdownExporter:
    """Tests for MarkdownExporter."""

    def test_export_feature(self, sample_feature):
        """Test single feature export."""
        exporter = MarkdownExporter()
        markdown = exporter.export_feature(sample_feature)

        assert "### user_age" in markdown
        assert "User age in years" in markdown
        assert "int64" in markdown
        assert "numeric" in markdown

    def test_export_feature_with_statistics(self, sample_feature):
        """Test feature export with statistics."""
        exporter = MarkdownExporter(include_statistics=True)
        markdown = exporter.export_feature(sample_feature)

        assert "Statistics" in markdown or "mean" in markdown
        assert "35.5" in markdown or "count" in markdown

    def test_export_feature_with_examples(self, sample_feature):
        """Test feature export with examples."""
        exporter = MarkdownExporter(include_examples=True)
        markdown = exporter.export_feature(sample_feature)

        assert "Example values" in markdown

    def test_export_feature_with_quality_notes(self, sample_feature):
        """Test feature export with quality notes."""
        exporter = MarkdownExporter(include_quality_notes=True)
        markdown = exporter.export_feature(sample_feature)

        assert "Quality notes" in markdown
        assert "missing" in markdown.lower()

    def test_export_dataset(self, sample_dataset):
        """Test dataset export."""
        exporter = MarkdownExporter()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            path = Path(f.name)

        try:
            exporter.export_dataset(sample_dataset, path)
            content = path.read_text()

            assert "# Customer Data" in content
            assert "Customer transaction dataset" in content
            assert "1,000" in content or "1000" in content
            assert "user_age" in content
        finally:
            path.unlink()

    def test_export_dataset_with_toc(self, sample_dataset):
        """Test dataset export with table of contents."""
        exporter = MarkdownExporter(include_toc=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            path = Path(f.name)

        try:
            exporter.export_dataset(sample_dataset, path)
            content = path.read_text()

            assert "Table of Contents" in content
        finally:
            path.unlink()

    def test_export_catalog(self, sample_catalog):
        """Test catalog export."""
        exporter = MarkdownExporter()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            path = Path(f.name)

        try:
            exporter.export_catalog(sample_catalog, path)
            content = path.read_text()

            assert "Feature Catalog" in content
            assert "Customer Data" in content
        finally:
            path.unlink()

    def test_statistics_table_format(self, sample_feature):
        """Test statistics table formatting."""
        exporter = MarkdownExporter(include_statistics=True)
        markdown = exporter.export_feature(sample_feature)

        # Should have table markers
        assert "|" in markdown

    def test_top_values_for_categorical(self, sample_dataset):
        """Test top values display for categorical features."""
        exporter = MarkdownExporter(include_statistics=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            path = Path(f.name)

        try:
            exporter.export_dataset(sample_dataset, path)
            content = path.read_text()

            assert "Top values" in content or "Electronics" in content
        finally:
            path.unlink()


class TestHTMLExporter:
    """Tests for HTMLExporter."""

    def test_export_feature(self, sample_feature):
        """Test single feature export."""
        exporter = HTMLExporter()
        html = exporter.export_feature(sample_feature)

        assert "feature-card" in html
        assert "user_age" in html
        assert "User age in years" in html

    def test_export_dataset(self, sample_dataset):
        """Test dataset export."""
        exporter = HTMLExporter()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            path = Path(f.name)

        try:
            exporter.export_dataset(sample_dataset, path)
            content = path.read_text()

            assert "<!DOCTYPE html>" in content
            assert "Customer Data" in content
            assert "<style>" in content
        finally:
            path.unlink()

    def test_export_with_search(self, sample_dataset):
        """Test export with search functionality."""
        exporter = HTMLExporter(include_search=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            path = Path(f.name)

        try:
            exporter.export_dataset(sample_dataset, path)
            content = path.read_text()

            assert "searchFeatures" in content
            assert "<input" in content
        finally:
            path.unlink()

    def test_export_without_search(self, sample_dataset):
        """Test export without search."""
        exporter = HTMLExporter(include_search=False)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            path = Path(f.name)

        try:
            exporter.export_dataset(sample_dataset, path)
            content = path.read_text()

            assert "searchFeatures" not in content
        finally:
            path.unlink()

    def test_light_theme(self, sample_dataset):
        """Test light theme."""
        exporter = HTMLExporter(theme="light")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            path = Path(f.name)

        try:
            exporter.export_dataset(sample_dataset, path)
            content = path.read_text()

            assert "#ffffff" in content or "white" in content.lower()
        finally:
            path.unlink()

    def test_dark_theme(self, sample_dataset):
        """Test dark theme."""
        exporter = HTMLExporter(theme="dark")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            path = Path(f.name)

        try:
            exporter.export_dataset(sample_dataset, path)
            content = path.read_text()

            assert "#1a1a1a" in content or "dark" in content.lower()
        finally:
            path.unlink()

    def test_export_catalog(self, sample_catalog):
        """Test catalog export."""
        exporter = HTMLExporter()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            path = Path(f.name)

        try:
            exporter.export_catalog(sample_catalog, path)
            content = path.read_text()

            assert "<!DOCTYPE html>" in content
            assert "Feature Catalog" in content
        finally:
            path.unlink()

    def test_statistics_table(self, sample_dataset):
        """Test statistics in HTML table."""
        exporter = HTMLExporter(include_statistics=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            path = Path(f.name)

        try:
            exporter.export_dataset(sample_dataset, path)
            content = path.read_text()

            assert "stat-table" in content
            assert "<table" in content
        finally:
            path.unlink()

    def test_tags_display(self, sample_feature):
        """Test tags displayed as badges."""
        exporter = HTMLExporter()
        html = exporter.export_feature(sample_feature)

        assert "tag" in html
        assert "demographic" in html


class TestJSONExporter:
    """Tests for JSONExporter."""

    def test_export_feature(self, sample_feature):
        """Test single feature export."""
        exporter = JSONExporter()
        json_str = exporter.export_feature(sample_feature)

        data = json.loads(json_str)

        assert data["name"] == "user_age"
        assert data["dtype"] == "int64"

    def test_export_dataset(self, sample_dataset):
        """Test dataset export."""
        exporter = JSONExporter()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            exporter.export_dataset(sample_dataset, path)
            content = path.read_text()
            data = json.loads(content)

            assert data["name"] == "Customer Data"
            assert data["n_samples"] == 1000
            assert len(data["features"]) == 3
        finally:
            path.unlink()

    def test_export_catalog(self, sample_catalog):
        """Test catalog export."""
        exporter = JSONExporter()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            exporter.export_catalog(sample_catalog, path)
            content = path.read_text()
            data = json.loads(content)

            assert "entries" in data or "features" in data
        finally:
            path.unlink()

    def test_custom_indent(self, sample_feature):
        """Test custom indentation."""
        exporter = JSONExporter(indent=4)
        json_str = exporter.export_feature(sample_feature)

        # Should have 4-space indentation
        lines = json_str.split("\n")
        indented_lines = [l for l in lines if l.startswith("    ")]
        assert len(indented_lines) > 0


class TestExportDocumentationFunction:
    """Tests for export_documentation convenience function."""

    def test_export_markdown(self, sample_dataset):
        """Test markdown export."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            path = Path(f.name)

        try:
            export_documentation(sample_dataset, path, format="markdown")
            content = path.read_text()

            assert "# Customer Data" in content
        finally:
            path.unlink()

    def test_export_html(self, sample_dataset):
        """Test HTML export."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            path = Path(f.name)

        try:
            export_documentation(sample_dataset, path, format="html")
            content = path.read_text()

            assert "<!DOCTYPE html>" in content
        finally:
            path.unlink()

    def test_export_json(self, sample_dataset):
        """Test JSON export."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            export_documentation(sample_dataset, path, format="json")
            content = path.read_text()
            data = json.loads(content)

            assert data["name"] == "Customer Data"
        finally:
            path.unlink()

    def test_export_catalog_markdown(self, sample_catalog):
        """Test catalog markdown export."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            path = Path(f.name)

        try:
            export_documentation(sample_catalog, path, format="markdown")
            content = path.read_text()

            assert "Feature Catalog" in content
        finally:
            path.unlink()

    def test_invalid_format(self, sample_dataset):
        """Test invalid format raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            path = Path(f.name)

        try:
            with pytest.raises(ValueError):
                export_documentation(sample_dataset, path, format="invalid")
        finally:
            path.unlink()

    def test_with_kwargs(self, sample_dataset):
        """Test passing kwargs to exporter."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            path = Path(f.name)

        try:
            export_documentation(
                sample_dataset,
                path,
                format="markdown",
                include_statistics=False,
                include_toc=False,
            )
            content = path.read_text()

            # Should not have TOC
            assert "Table of Contents" not in content
        finally:
            path.unlink()
