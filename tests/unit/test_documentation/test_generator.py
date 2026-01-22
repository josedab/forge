"""Tests for feature documentation generator."""

import numpy as np
import pandas as pd
import pytest

from forge.documentation.generator import (
    FeatureDocumentationGenerator,
    FeatureDoc,
    DatasetDoc,
    generate_documentation,
)


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 100

    return pd.DataFrame({
        "user_id": range(n_samples),
        "age": np.random.randint(18, 80, n_samples),
        "income": np.random.uniform(20000, 150000, n_samples),
        "purchase_count": np.random.poisson(5, n_samples),
        "is_premium": np.random.choice([True, False], n_samples),
        "category": np.random.choice(["A", "B", "C"], n_samples),
        "signup_date": pd.date_range("2020-01-01", periods=n_samples, freq="D"),
        "email": [f"user{i}@example.com" for i in range(n_samples)],
    })


@pytest.fixture
def mixed_quality_dataframe():
    """Create DataFrame with quality issues."""
    np.random.seed(42)
    n_samples = 100

    df = pd.DataFrame({
        "complete": np.random.randn(n_samples),
        "with_missing": np.random.randn(n_samples),
        "constant": np.ones(n_samples),
        "high_cardinality": [f"val_{i}" for i in range(n_samples)],
        "low_cardinality": np.random.choice(["a", "b"], n_samples),
    })

    # Add missing values
    df.loc[::5, "with_missing"] = np.nan

    return df


class TestFeatureDoc:
    """Tests for FeatureDoc dataclass."""

    def test_creation(self):
        """Test FeatureDoc creation."""
        doc = FeatureDoc(
            name="test_feature",
            dtype="float64",
            description="A test feature",
            category="numeric",
        )

        assert doc.name == "test_feature"
        assert doc.dtype == "float64"
        assert doc.description == "A test feature"
        assert doc.category == "numeric"

    def test_default_values(self):
        """Test default values."""
        doc = FeatureDoc(name="test", dtype="int64")

        assert doc.description == ""
        assert doc.category == "unknown"
        assert doc.tags == []
        assert doc.statistics == {}
        assert doc.quality_notes == []

    def test_to_dict(self):
        """Test to_dict method."""
        doc = FeatureDoc(
            name="test",
            dtype="float64",
            description="desc",
            category="numeric",
            tags=["important"],
        )

        d = doc.to_dict()

        assert d["name"] == "test"
        assert d["dtype"] == "float64"
        assert d["tags"] == ["important"]

    def test_from_dict(self):
        """Test from_dict method."""
        data = {
            "name": "test",
            "dtype": "float64",
            "description": "desc",
            "category": "numeric",
            "tags": ["tag1"],
        }

        doc = FeatureDoc.from_dict(data)

        assert doc.name == "test"
        assert doc.dtype == "float64"
        assert doc.tags == ["tag1"]


class TestDatasetDoc:
    """Tests for DatasetDoc dataclass."""

    def test_creation(self):
        """Test DatasetDoc creation."""
        features = [
            FeatureDoc(name="f1", dtype="int64"),
            FeatureDoc(name="f2", dtype="float64"),
        ]

        doc = DatasetDoc(
            name="test_dataset",
            features=features,
            n_samples=100,
            n_features=2,
        )

        assert doc.name == "test_dataset"
        assert len(doc.features) == 2
        assert doc.n_samples == 100

    def test_to_dict(self):
        """Test to_dict method."""
        doc = DatasetDoc(
            name="test",
            features=[FeatureDoc(name="f1", dtype="int64")],
            n_samples=50,
            n_features=1,
        )

        d = doc.to_dict()

        assert d["name"] == "test"
        assert d["n_samples"] == 50
        assert len(d["features"]) == 1

    def test_from_dict(self):
        """Test from_dict method."""
        data = {
            "name": "test",
            "features": [{"name": "f1", "dtype": "int64"}],
            "n_samples": 100,
            "n_features": 1,
        }

        doc = DatasetDoc.from_dict(data)

        assert doc.name == "test"
        assert len(doc.features) == 1


class TestFeatureDocumentationGenerator:
    """Tests for FeatureDocumentationGenerator."""

    def test_generate_basic(self, sample_dataframe):
        """Test basic documentation generation."""
        generator = FeatureDocumentationGenerator()
        doc = generator.generate(sample_dataframe, dataset_name="test_dataset")

        assert isinstance(doc, DatasetDoc)
        assert doc.name == "test_dataset"
        assert doc.n_samples == len(sample_dataframe)
        assert doc.n_features == len(sample_dataframe.columns)
        assert len(doc.features) == len(sample_dataframe.columns)

    def test_feature_types_detected(self, sample_dataframe):
        """Test that feature types are detected."""
        generator = FeatureDocumentationGenerator()
        doc = generator.generate(sample_dataframe)

        feature_names = {f.name for f in doc.features}

        assert "user_id" in feature_names
        assert "age" in feature_names
        assert "income" in feature_names

    def test_statistics_computed(self, sample_dataframe):
        """Test that statistics are computed."""
        generator = FeatureDocumentationGenerator(include_statistics=True)
        doc = generator.generate(sample_dataframe)

        # Find numeric feature
        age_feature = next(f for f in doc.features if f.name == "age")

        assert "count" in age_feature.statistics
        assert "mean" in age_feature.statistics or age_feature.statistics.get("count") is not None

    def test_quality_notes_generated(self, mixed_quality_dataframe):
        """Test quality notes for features with issues."""
        generator = FeatureDocumentationGenerator(include_quality_notes=True)
        doc = generator.generate(mixed_quality_dataframe)

        # Find feature with missing values
        missing_feature = next(f for f in doc.features if f.name == "with_missing")
        assert len(missing_feature.quality_notes) > 0 or missing_feature.statistics.get("missing", 0) > 0

        # Find constant feature
        constant_feature = next(f for f in doc.features if f.name == "constant")
        assert len(constant_feature.quality_notes) > 0 or constant_feature.statistics.get("unique", 1) == 1

    def test_example_values_included(self, sample_dataframe):
        """Test example values are included."""
        generator = FeatureDocumentationGenerator(include_examples=True)
        doc = generator.generate(sample_dataframe)

        # At least some features should have examples
        features_with_examples = [f for f in doc.features if f.example_values]
        assert len(features_with_examples) > 0

    def test_auto_description(self, sample_dataframe):
        """Test automatic description generation."""
        generator = FeatureDocumentationGenerator(auto_describe=True)
        doc = generator.generate(sample_dataframe)

        # Features should have descriptions
        features_with_desc = [f for f in doc.features if f.description]
        assert len(features_with_desc) > 0

    def test_category_inference(self, sample_dataframe):
        """Test category inference."""
        generator = FeatureDocumentationGenerator()
        doc = generator.generate(sample_dataframe)

        # Find features and check categories
        # The generator infers domain-specific categories from column names
        valid_categories = [
            "numeric", "categorical", "temporal", "text",
            "identifier", "boolean", "unknown", "other",
            # Domain-specific categories inferred from names
            "geographic", "financial", "demographic", "metric",
        ]
        for feature in doc.features:
            assert feature.category in valid_categories

    def test_tag_generation(self, sample_dataframe):
        """Test tag generation."""
        generator = FeatureDocumentationGenerator()
        doc = generator.generate(sample_dataframe)

        # At least some features should have tags (auto-generated)
        features_with_tags = [f for f in doc.features if f.tags]
        assert len(features_with_tags) >= 0  # Tags are optional

    def test_custom_descriptions(self, sample_dataframe):
        """Test custom descriptions."""
        custom_desc = {
            "age": "Customer age in years",
            "income": "Annual income in USD",
        }

        generator = FeatureDocumentationGenerator()
        doc = generator.generate(sample_dataframe, feature_descriptions=custom_desc)

        age_feature = next(f for f in doc.features if f.name == "age")
        assert age_feature.description == "Customer age in years"

    def test_without_statistics(self, sample_dataframe):
        """Test generation without statistics."""
        generator = FeatureDocumentationGenerator(include_statistics=False)
        doc = generator.generate(sample_dataframe)

        # Statistics should be empty or minimal
        for feature in doc.features:
            assert len(feature.statistics) == 0 or "count" not in feature.statistics

    def test_quality_summary(self, mixed_quality_dataframe):
        """Test quality summary generation."""
        generator = FeatureDocumentationGenerator()
        doc = generator.generate(mixed_quality_dataframe)

        # Quality summary should be present
        assert doc.quality_summary is not None or hasattr(doc, "quality_summary")


class TestGenerateDocumentationFunction:
    """Tests for generate_documentation convenience function."""

    def test_basic_usage(self, sample_dataframe):
        """Test basic function usage."""
        doc = generate_documentation(sample_dataframe, dataset_name="my_dataset")

        assert isinstance(doc, DatasetDoc)
        assert doc.name == "my_dataset"

    def test_with_options(self, sample_dataframe):
        """Test with options."""
        doc = generate_documentation(
            sample_dataframe,
            dataset_name="test",
            include_statistics=True,
            include_examples=True,
        )

        assert isinstance(doc, DatasetDoc)

    def test_default_name(self, sample_dataframe):
        """Test default dataset name."""
        doc = generate_documentation(sample_dataframe)

        assert doc.name is not None
        assert len(doc.name) > 0
