"""Tests for multi-table transformer."""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from forge.multitable.transformer import MultiTableTransformer, multi_table_features
from forge.exceptions import NotFittedError, ValidationError


@pytest.fixture
def sample_tables():
    """Create sample tables for testing."""
    customers = pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "age": [25, 30, 35, 40, 45],
    })

    orders = pd.DataFrame({
        "order_id": [101, 102, 103, 104, 105, 106, 107],
        "customer_id": [1, 1, 2, 3, 3, 3, 5],
        "amount": [100.0, 150.0, 200.0, 50.0, 75.0, 125.0, 300.0],
    })

    return {"customers": customers, "orders": orders}


class TestMultiTableTransformer:
    """Tests for MultiTableTransformer class."""

    def test_fit_basic(self, sample_tables):
        """Test basic fitting."""
        transformer = MultiTableTransformer(
            target_table="customers",
            max_depth=1,
        )

        transformer.fit(sample_tables)
        assert transformer._is_fitted

    def test_transform_basic(self, sample_tables):
        """Test basic transformation."""
        transformer = MultiTableTransformer(
            target_table="customers",
            max_depth=1,
        )

        transformer.fit(sample_tables)
        result = transformer.transform(sample_tables)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5  # 5 customers

    def test_fit_transform(self, sample_tables):
        """Test fit_transform method."""
        transformer = MultiTableTransformer(
            target_table="customers",
            max_depth=1,
        )

        result = transformer.fit_transform(sample_tables)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5

    def test_include_target_columns(self, sample_tables):
        """Test that original columns are included."""
        transformer = MultiTableTransformer(
            target_table="customers",
            include_target_columns=True,
            max_depth=1,
        )

        result = transformer.fit_transform(sample_tables)

        # Should include original customer columns
        assert "customer_id" in result.columns
        assert "name" in result.columns

    def test_exclude_target_columns(self, sample_tables):
        """Test that original columns can be excluded."""
        transformer = MultiTableTransformer(
            target_table="customers",
            include_target_columns=False,
            max_depth=1,
        )

        result = transformer.fit_transform(sample_tables)

        # Should only have generated features
        assert "name" not in result.columns

    def test_get_feature_names_out(self, sample_tables):
        """Test get_feature_names_out method."""
        transformer = MultiTableTransformer(
            target_table="customers",
            max_depth=1,
        )

        transformer.fit(sample_tables)
        names = transformer.get_feature_names_out()

        assert isinstance(names, list)
        assert len(names) > 0

    def test_get_feature_definitions(self, sample_tables):
        """Test get_feature_definitions method."""
        transformer = MultiTableTransformer(
            target_table="customers",
            max_depth=1,
        )

        transformer.fit(sample_tables)
        definitions = transformer.get_feature_definitions()

        assert isinstance(definitions, list)
        assert len(definitions) > 0

    def test_get_feature_lineage(self, sample_tables):
        """Test get_feature_lineage method."""
        transformer = MultiTableTransformer(
            target_table="customers",
            max_depth=1,
        )

        transformer.fit(sample_tables)
        names = transformer.get_feature_names_out()

        # Find a generated feature (not original)
        generated = [n for n in names if "orders" in n]
        if generated:
            lineage = transformer.get_feature_lineage(generated[0])
            assert "name" in lineage
            assert "source_table" in lineage

    def test_not_fitted_error(self, sample_tables):
        """Test that NotFittedError is raised before fitting."""
        transformer = MultiTableTransformer(target_table="customers")

        with pytest.raises(NotFittedError):
            transformer.transform(sample_tables)

    def test_missing_target_table_error(self, sample_tables):
        """Test error when target table is missing."""
        transformer = MultiTableTransformer(target_table="nonexistent")

        with pytest.raises(ValidationError):
            transformer.fit(sample_tables)

    def test_manual_relationships(self, sample_tables):
        """Test with manual relationship specification."""
        transformer = MultiTableTransformer(
            target_table="customers",
            relationships=[{
                "parent": "customers",
                "parent_col": "customer_id",
                "child": "orders",
                "child_col": "customer_id",
            }],
            auto_detect_relationships=False,
            max_depth=1,
        )

        result = transformer.fit_transform(sample_tables)
        assert isinstance(result, pd.DataFrame)

    def test_custom_primitives(self, sample_tables):
        """Test with custom primitive selection."""
        transformer = MultiTableTransformer(
            target_table="customers",
            agg_primitives=["mean", "sum"],
            max_depth=1,
        )

        result = transformer.fit_transform(sample_tables)

        # Should only have mean and sum aggregations
        generated_cols = [c for c in result.columns if "orders" in c]
        for col in generated_cols:
            assert "mean" in col or "sum" in col

    def test_max_features_limit(self, sample_tables):
        """Test that max_features is respected."""
        transformer = MultiTableTransformer(
            target_table="customers",
            max_features=3,
            include_target_columns=False,
            max_depth=1,
        )

        result = transformer.fit_transform(sample_tables)
        assert len(result.columns) <= 3

    def test_save_load(self, sample_tables):
        """Test save and load functionality."""
        transformer = MultiTableTransformer(
            target_table="customers",
            max_depth=1,
        )
        transformer.fit(sample_tables)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "transformer.pkl"
            transformer.save(path)

            loaded = MultiTableTransformer.load(path)
            assert loaded._is_fitted
            assert loaded.target_table == "customers"


class TestMultiTableFeaturesFunction:
    """Tests for multi_table_features convenience function."""

    def test_basic_usage(self, sample_tables):
        """Test basic function usage."""
        result = multi_table_features(
            tables=sample_tables,
            target_table="customers",
            max_depth=1,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5

    def test_with_options(self, sample_tables):
        """Test function with additional options."""
        result = multi_table_features(
            tables=sample_tables,
            target_table="customers",
            max_depth=1,
            max_features=5,
            verbose=0,
        )

        assert isinstance(result, pd.DataFrame)
