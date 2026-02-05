"""Tests for feature description and lineage tracking."""

from __future__ import annotations

import pandas as pd

from forge.transformers.descriptions import (
    FeatureDescriber,
    FeatureDescription,
    FeatureLineage,
)


class TestFeatureDescription:
    """Tests for FeatureDescription dataclass."""

    def test_basic_creation(self):
        """Test basic feature description creation."""
        desc = FeatureDescription(
            name="price_qty_multiply",
            description="price multiplied by quantity",
            source_columns=["price", "quantity"],
            generator="InteractionGenerator",
            parameters={"operation": "multiply"},
            feature_type="numeric",
        )

        assert desc.name == "price_qty_multiply"
        assert desc.description == "price multiplied by quantity"
        assert desc.source_columns == ["price", "quantity"]
        assert desc.generator == "InteractionGenerator"

    def test_to_natural_language(self):
        """Test natural language output."""
        desc = FeatureDescription(
            name="age_squared",
            description="age squared (degree 2)",
            source_columns=["age"],
            generator="PolynomialGenerator",
        )

        assert desc.to_natural_language() == "age squared (degree 2)"

    def test_to_dict(self):
        """Test dictionary conversion."""
        desc = FeatureDescription(
            name="color_woe",
            description="Weight of Evidence encoding of color",
            source_columns=["color"],
            generator="WoEEncoder",
            parameters={"regularization": 0.5},
            feature_type="categorical",
        )

        result = desc.to_dict()

        assert result["name"] == "color_woe"
        assert result["description"] == "Weight of Evidence encoding of color"
        assert result["source_columns"] == ["color"]
        assert result["generator"] == "WoEEncoder"
        assert result["parameters"] == {"regularization": 0.5}
        assert result["feature_type"] == "categorical"

    def test_default_feature_type(self):
        """Test default feature type."""
        desc = FeatureDescription(
            name="test",
            description="test feature",
            source_columns=["a"],
            generator="TestGen",
        )

        assert desc.feature_type == "derived"

    def test_default_parameters(self):
        """Test default empty parameters."""
        desc = FeatureDescription(
            name="test",
            description="test feature",
            source_columns=["a"],
            generator="TestGen",
        )

        assert desc.parameters == {}


class TestFeatureDescriber:
    """Tests for FeatureDescriber class."""

    def test_describe_interaction_multiply(self):
        """Test interaction feature description (multiply)."""
        desc = FeatureDescriber.describe_interaction(
            feature_name="price_qty",
            col1="price",
            col2="quantity",
            operation="multiply",
        )

        assert desc.name == "price_qty"
        assert desc.description == "price multiplied by quantity"
        assert "price" in desc.source_columns
        assert "quantity" in desc.source_columns
        assert desc.generator == "InteractionGenerator"

    def test_describe_interaction_divide(self):
        """Test interaction feature description (divide)."""
        desc = FeatureDescriber.describe_interaction(
            feature_name="ratio",
            col1="numerator",
            col2="denominator",
            operation="divide",
        )

        assert desc.description == "numerator divided by denominator"

    def test_describe_interaction_add(self):
        """Test interaction feature description (add)."""
        desc = FeatureDescriber.describe_interaction(
            feature_name="total",
            col1="a",
            col2="b",
            operation="add",
        )

        assert desc.description == "a plus b"

    def test_describe_interaction_subtract(self):
        """Test interaction feature description (subtract)."""
        desc = FeatureDescriber.describe_interaction(
            feature_name="diff",
            col1="a",
            col2="b",
            operation="subtract",
        )

        assert desc.description == "a minus b"

    def test_describe_polynomial_degree_2(self):
        """Test polynomial feature description (squared)."""
        desc = FeatureDescriber.describe_polynomial(
            feature_name="age_squared",
            col="age",
            degree=2,
        )

        assert desc.description == "age squared (degree 2)"
        assert desc.source_columns == ["age"]
        assert desc.generator == "PolynomialGenerator"

    def test_describe_polynomial_degree_3(self):
        """Test polynomial feature description (cubed)."""
        desc = FeatureDescriber.describe_polynomial(
            feature_name="age_cubed",
            col="age",
            degree=3,
        )

        assert desc.description == "age cubed (degree 3)"

    def test_describe_polynomial_degree_n(self):
        """Test polynomial feature description (arbitrary degree)."""
        desc = FeatureDescriber.describe_polynomial(
            feature_name="age_5",
            col="age",
            degree=5,
        )

        assert desc.description == "age raised to power 5"

    def test_describe_encoding_target(self):
        """Test target encoding description."""
        desc = FeatureDescriber.describe_encoding(
            feature_name="color_target",
            col="color",
            encoding_type="target",
        )

        assert "Mean target value" in desc.description
        assert "color" in desc.description
        assert desc.source_columns == ["color"]

    def test_describe_encoding_frequency(self):
        """Test frequency encoding description."""
        desc = FeatureDescriber.describe_encoding(
            feature_name="color_freq",
            col="color",
            encoding_type="frequency",
        )

        assert "Frequency" in desc.description
        assert "color" in desc.description

    def test_describe_encoding_woe(self):
        """Test WoE encoding description."""
        desc = FeatureDescriber.describe_encoding(
            feature_name="color_woe",
            col="color",
            encoding_type="woe",
        )

        assert "Weight of Evidence" in desc.description

    def test_describe_encoding_onehot(self):
        """Test one-hot encoding description."""
        desc = FeatureDescriber.describe_encoding(
            feature_name="color_red",
            col="color",
            encoding_type="onehot",
            category="red",
        )

        # Feature description should include source column and category info
        assert "color" in desc.description
        assert "red" in desc.description or desc.parameters.get("category") == "red"

    def test_describe_temporal_year(self):
        """Test temporal year description."""
        desc = FeatureDescriber.describe_temporal(
            feature_name="date_year",
            col="date",
            component="year",
        )

        assert desc.description == "Year extracted from date"
        assert desc.generator == "DateTimeComponents"

    def test_describe_temporal_month(self):
        """Test temporal month description."""
        desc = FeatureDescriber.describe_temporal(
            feature_name="date_month",
            col="date",
            component="month",
        )

        assert desc.description == "Month (1-12) extracted from date"

    def test_describe_temporal_is_weekend(self):
        """Test temporal is_weekend description."""
        desc = FeatureDescriber.describe_temporal(
            feature_name="date_weekend",
            col="date",
            component="is_weekend",
        )

        assert "weekend" in desc.description.lower()

    def test_describe_lag(self):
        """Test lag feature description."""
        desc = FeatureDescriber.describe_lag(
            feature_name="price_lag_3",
            col="price",
            n=3,
        )

        assert "3 periods ago" in desc.description
        assert desc.generator == "LagGenerator"

    def test_describe_rolling_mean(self):
        """Test rolling mean description."""
        desc = FeatureDescriber.describe_rolling(
            feature_name="price_rolling_7_mean",
            col="price",
            window=7,
            statistic="mean",
        )

        assert "7-period rolling mean" in desc.description
        assert desc.generator == "RollingFeatures"

    def test_describe_rolling_std(self):
        """Test rolling std description."""
        desc = FeatureDescriber.describe_rolling(
            feature_name="price_rolling_7_std",
            col="price",
            window=7,
            statistic="std",
        )

        assert "standard deviation" in desc.description

    def test_describe_generic(self):
        """Test generic describe method."""
        desc = FeatureDescriber.describe(
            feature_name="log_price",
            feature_type="log",
            generator="TransformGenerator",
            col="price",
        )

        assert "logarithm" in desc.description.lower()
        assert desc.source_columns == ["price"]

    def test_describe_unknown_template(self):
        """Test fallback for unknown template."""
        desc = FeatureDescriber.describe(
            feature_name="custom_feature",
            feature_type="unknown_type",
            generator="CustomGenerator",
            col="test",
        )

        # Should not fail, should create some description
        assert desc.name == "custom_feature"
        assert desc.description  # Should have some description

    def test_templates_coverage(self):
        """Test that common templates exist."""
        templates = FeatureDescriber.TEMPLATES

        # Interaction templates
        assert "interaction_multiply" in templates
        assert "interaction_divide" in templates
        assert "interaction_add" in templates
        assert "interaction_subtract" in templates

        # Polynomial templates
        assert "polynomial_2" in templates
        assert "polynomial_3" in templates

        # Transformation templates
        assert "log" in templates
        assert "sqrt" in templates

        # Encoding templates
        assert "target_encoding" in templates
        assert "frequency_encoding" in templates
        assert "woe_encoding" in templates

        # Temporal templates
        assert "datetime_year" in templates
        assert "datetime_month" in templates
        assert "lag" in templates
        assert "rolling_mean" in templates


class TestFeatureLineage:
    """Tests for FeatureLineage class."""

    def test_add_and_get_sources(self):
        """Test adding features and getting sources."""
        lineage = FeatureLineage()
        lineage.add_feature(
            feature_name="price_qty",
            sources=["price", "quantity"],
            operation="multiply",
            generator="InteractionGenerator",
        )

        sources = lineage.get_sources("price_qty")
        assert "price" in sources
        assert "quantity" in sources

    def test_get_sources_unknown_feature(self):
        """Test getting sources for unknown feature."""
        lineage = FeatureLineage()

        # Unknown feature should return itself
        sources = lineage.get_sources("unknown")
        assert sources == ["unknown"]

    def test_add_with_description(self):
        """Test adding feature with description."""
        lineage = FeatureLineage()
        desc = FeatureDescription(
            name="test",
            description="test description",
            source_columns=["a"],
            generator="Test",
        )

        lineage.add_feature(
            feature_name="test",
            sources=["a"],
            operation="transform",
            generator="Test",
            description=desc,
        )

        retrieved = lineage.get_description("test")
        assert retrieved is not None
        assert retrieved.description == "test description"

    def test_get_description_none(self):
        """Test getting description when none exists."""
        lineage = FeatureLineage()
        lineage.add_feature(
            feature_name="test",
            sources=["a"],
            operation="transform",
            generator="Test",
        )

        assert lineage.get_description("test") is None

    def test_get_all_descriptions(self):
        """Test getting all descriptions."""
        lineage = FeatureLineage()
        desc1 = FeatureDescription(
            name="feat1", description="desc1", source_columns=["a"], generator="Gen"
        )
        desc2 = FeatureDescription(
            name="feat2", description="desc2", source_columns=["b"], generator="Gen"
        )

        lineage.add_feature("feat1", ["a"], "op1", "Gen", description=desc1)
        lineage.add_feature("feat2", ["b"], "op2", "Gen", description=desc2)

        all_descs = lineage.get_all_descriptions()
        assert len(all_descs) == 2
        assert "feat1" in all_descs
        assert "feat2" in all_descs

    def test_trace_to_original_simple(self):
        """Test simple lineage tracing."""
        lineage = FeatureLineage()
        lineage.add_feature("feat", ["a", "b"], "multiply", "Gen")

        originals = lineage.trace_to_original("feat")
        assert set(originals) == {"a", "b"}

    def test_trace_to_original_nested(self):
        """Test nested lineage tracing."""
        lineage = FeatureLineage()

        # First level: a, b -> ab
        lineage.add_feature("ab", ["a", "b"], "multiply", "Gen")

        # Second level: ab, c -> abc
        lineage.add_feature("abc", ["ab", "c"], "add", "Gen")

        originals = lineage.trace_to_original("abc")
        assert set(originals) == {"a", "b", "c"}

    def test_trace_to_original_deep_nested(self):
        """Test deeply nested lineage tracing."""
        lineage = FeatureLineage()

        # Build chain: a -> a2 -> a4 -> a8
        lineage.add_feature("a2", ["a"], "square", "Gen")
        lineage.add_feature("a4", ["a2"], "square", "Gen")
        lineage.add_feature("a8", ["a4"], "square", "Gen")

        originals = lineage.trace_to_original("a8")
        assert originals == ["a"]

    def test_trace_unknown_feature(self):
        """Test tracing unknown feature returns itself."""
        lineage = FeatureLineage()

        originals = lineage.trace_to_original("unknown")
        assert originals == ["unknown"]

    def test_to_dataframe(self):
        """Test exporting lineage to DataFrame."""
        lineage = FeatureLineage()
        desc = FeatureDescription(
            name="ab",
            description="a multiplied by b",
            source_columns=["a", "b"],
            generator="InteractionGenerator",
        )

        lineage.add_feature("ab", ["a", "b"], "multiply", "InteractionGenerator", description=desc)
        lineage.add_feature("cd", ["c", "d"], "add", "InteractionGenerator")

        df = lineage.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "feature" in df.columns
        assert "sources" in df.columns
        assert "operation" in df.columns
        assert "generator" in df.columns
        assert "description" in df.columns

        # Check ab row has description
        ab_row = df[df["feature"] == "ab"].iloc[0]
        assert ab_row["description"] == "a multiplied by b"

    def test_add_with_metadata(self):
        """Test adding feature with extra metadata."""
        lineage = FeatureLineage()
        lineage.add_feature(
            feature_name="test",
            sources=["a"],
            operation="transform",
            generator="Test",
            custom_key="custom_value",
            importance=0.85,
        )

        # Metadata should be stored
        assert lineage._lineage["test"]["custom_key"] == "custom_value"
        assert lineage._lineage["test"]["importance"] == 0.85


class TestIntegration:
    """Integration tests for description system."""

    def test_full_pipeline_description(self):
        """Test describing a full feature engineering pipeline."""
        lineage = FeatureLineage()

        # Describe interaction feature
        interaction_desc = FeatureDescriber.describe_interaction(
            "price_qty", "price", "quantity", "multiply"
        )
        lineage.add_feature(
            "price_qty",
            ["price", "quantity"],
            "multiply",
            "InteractionGenerator",
            description=interaction_desc,
        )

        # Describe derived feature
        log_desc = FeatureDescriber.describe(
            "log_price_qty", "log", "TransformGenerator", col="price_qty"
        )
        lineage.add_feature(
            "log_price_qty",
            ["price_qty"],
            "log",
            "TransformGenerator",
            description=log_desc,
        )

        # Trace back to original
        originals = lineage.trace_to_original("log_price_qty")
        assert set(originals) == {"price", "quantity"}

        # Get descriptions
        all_desc = lineage.get_all_descriptions()
        assert len(all_desc) == 2
        assert "price multiplied by quantity" in all_desc["price_qty"].description
        assert "logarithm" in all_desc["log_price_qty"].description.lower()
