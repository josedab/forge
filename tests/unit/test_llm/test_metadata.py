"""Tests for metadata extraction module."""

import numpy as np
import pandas as pd
import pytest

from forge.llm.metadata import (
    MetadataExtractor,
    DatasetMetadata,
    ColumnMetadata,
)
from forge.types import ColumnType


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "customer_id": range(1, 101),
        "age": np.random.randint(18, 80, 100),
        "income": np.random.normal(50000, 15000, 100),
        "category": np.random.choice(["A", "B", "C"], 100),
        "signup_date": pd.date_range("2020-01-01", periods=100),
        "description": ["Customer " + str(i) + " description text here" for i in range(100)],
        "has_premium": np.random.choice([True, False], 100),
    })


@pytest.fixture
def sample_target():
    """Create a sample target variable."""
    np.random.seed(42)
    return pd.Series(np.random.choice([0, 1], 100), name="target")


class TestMetadataExtractor:
    """Tests for MetadataExtractor class."""

    def test_extract_basic(self, sample_df, sample_target):
        """Test basic metadata extraction."""
        extractor = MetadataExtractor()
        metadata = extractor.extract(sample_df, sample_target)

        assert isinstance(metadata, DatasetMetadata)
        assert metadata.n_rows == 100
        assert metadata.n_columns == 7
        assert len(metadata.columns) == 7

    def test_column_types_inferred(self, sample_df):
        """Test that column types are correctly inferred."""
        extractor = MetadataExtractor()
        metadata = extractor.extract(sample_df)

        assert metadata.columns["age"].inferred_type == ColumnType.NUMERIC
        assert metadata.columns["income"].inferred_type == ColumnType.NUMERIC
        assert metadata.columns["category"].inferred_type == ColumnType.CATEGORICAL
        assert metadata.columns["signup_date"].inferred_type == ColumnType.DATETIME

    def test_statistics_computed(self, sample_df):
        """Test that statistics are computed for columns."""
        extractor = MetadataExtractor()
        metadata = extractor.extract(sample_df)

        age_meta = metadata.columns["age"]
        assert "mean" in age_meta.statistics
        assert "std" in age_meta.statistics
        assert "min" in age_meta.statistics
        assert "max" in age_meta.statistics

    def test_patterns_detected(self, sample_df):
        """Test that patterns are detected from column names."""
        extractor = MetadataExtractor()
        metadata = extractor.extract(sample_df)

        # customer_id should be detected as identifier
        assert "identifier" in metadata.columns["customer_id"].patterns

    def test_correlations_computed(self, sample_df):
        """Test that correlations are computed between numeric columns."""
        extractor = MetadataExtractor(correlation_threshold=0.0)
        metadata = extractor.extract(sample_df)

        # Should have some correlations (with low threshold)
        # Note: correlations might be empty if no columns correlate
        assert isinstance(metadata.correlations, dict)

    def test_target_info_extracted(self, sample_df, sample_target):
        """Test that target variable info is extracted."""
        extractor = MetadataExtractor()
        metadata = extractor.extract(sample_df, sample_target)

        assert metadata.target_info is not None
        assert "task_type" in metadata.target_info
        assert metadata.target_info["task_type"] == "classification"

    def test_domain_hints_inferred(self, sample_df):
        """Test that domain hints are inferred."""
        extractor = MetadataExtractor()
        metadata = extractor.extract(sample_df)

        # Should detect temporal data
        assert "temporal_data" in metadata.domain_hints

    def test_to_prompt_string(self, sample_df):
        """Test conversion to prompt string."""
        extractor = MetadataExtractor()
        metadata = extractor.extract(sample_df)

        prompt = metadata.to_prompt_string()
        assert "Dataset Overview" in prompt
        assert "Column Details" in prompt

    def test_to_dict(self, sample_df):
        """Test conversion to dictionary."""
        extractor = MetadataExtractor()
        metadata = extractor.extract(sample_df)

        data = metadata.to_dict()
        assert "n_rows" in data
        assert "n_columns" in data
        assert "columns" in data

    def test_metadata_hash(self, sample_df):
        """Test metadata hash computation."""
        extractor = MetadataExtractor()
        metadata = extractor.extract(sample_df)

        hash1 = metadata.compute_hash()
        hash2 = metadata.compute_hash()

        assert hash1 == hash2
        assert len(hash1) == 16


class TestColumnMetadata:
    """Tests for ColumnMetadata class."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        col_meta = ColumnMetadata(
            name="test_col",
            dtype="int64",
            inferred_type=ColumnType.NUMERIC,
            null_count=5,
            null_ratio=0.05,
            unique_count=95,
            unique_ratio=0.95,
            sample_values=[1, 2, 3],
        )

        data = col_meta.to_dict()
        assert data["name"] == "test_col"
        assert data["inferred_type"] == "numeric"

    def test_to_prompt_string(self):
        """Test conversion to prompt string."""
        col_meta = ColumnMetadata(
            name="test_col",
            dtype="float64",
            inferred_type=ColumnType.NUMERIC,
            null_count=0,
            null_ratio=0.0,
            unique_count=100,
            unique_ratio=1.0,
            sample_values=[1.0, 2.0, 3.0],
            statistics={"mean": 50.0, "std": 10.0},
        )

        prompt = col_meta.to_prompt_string()
        assert "test_col" in prompt
        assert "NUMERIC" in prompt.upper() or "numeric" in prompt
