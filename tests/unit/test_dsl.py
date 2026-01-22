"""Tests for declarative feature DSL."""

from __future__ import annotations

import pytest

from forge.dsl.compiler import DSLCompiler
from forge.dsl.parser import (
    DSLParser,
    FeatureDSLError,
    FeatureSpec,
    PipelineSpec,
    SelectionSpec,
)

# --- Parser Tests ---

class TestDSLParser:
    """Tests for DSLParser."""

    def test_parse_basic_definition(self):
        definition = {
            "name": "test_pipeline",
            "version": "1.0",
            "features": [
                {"name": "price_log", "source": "price", "transform": "log"},
                {"name": "cat_enc", "source": "category", "transform": "target"},
            ],
        }
        parser = DSLParser()
        spec = parser.parse(definition)
        assert spec.name == "test_pipeline"
        assert len(spec.features) == 2
        assert spec.features[0].name == "price_log"
        assert spec.features[0].transform == "log"

    def test_parse_with_selection(self):
        definition = {
            "name": "with_selection",
            "version": "1.0",
            "features": [
                {"name": "f1", "source": "col1", "transform": "log"},
            ],
            "selection": {"method": "importance", "params": {"k": 10}},
        }
        parser = DSLParser()
        spec = parser.parse(definition)
        assert spec.selection is not None
        assert spec.selection.method == "importance"
        assert spec.selection.params["k"] == 10

    def test_parse_with_params(self):
        definition = {
            "name": "with_params",
            "version": "1.0",
            "features": [
                {
                    "name": "poly",
                    "source": "col1",
                    "transform": "polynomial",
                    "params": {"degree": 3},
                },
            ],
        }
        parser = DSLParser()
        spec = parser.parse(definition)
        assert spec.features[0].params["degree"] == 3

    def test_parse_with_metadata(self):
        definition = {
            "name": "meta_pipeline",
            "version": "1.0",
            "features": [{"name": "f", "source": "c", "transform": "log"}],
            "metadata": {"author": "test", "dataset": "iris"},
        }
        parser = DSLParser()
        spec = parser.parse(definition)
        assert spec.metadata["author"] == "test"

    def test_parse_rejects_non_dict(self):
        parser = DSLParser()
        with pytest.raises(FeatureDSLError, match="Expected dict"):
            parser.parse("not a dict")

    def test_parse_rejects_unsupported_version(self):
        parser = DSLParser()
        with pytest.raises(FeatureDSLError, match="Unsupported DSL version"):
            parser.parse({"version": "99.0", "features": []})

    def test_parse_rejects_invalid_features_type(self):
        parser = DSLParser()
        with pytest.raises(FeatureDSLError, match="must be a list"):
            parser.parse({"name": "test", "features": "not_a_list"})

    def test_parse_strict_rejects_invalid_transform(self):
        definition = {
            "name": "bad",
            "version": "1.0",
            "features": [
                {"name": "f", "source": "c", "transform": "nonexistent"},
            ],
        }
        parser = DSLParser(strict=True)
        with pytest.raises(FeatureDSLError, match="validation failed"):
            parser.parse(definition)

    def test_parse_non_strict_allows_warnings(self):
        definition = {
            "name": "bad",
            "version": "1.0",
            "features": [
                {"name": "f", "source": "c", "transform": "nonexistent"},
            ],
        }
        parser = DSLParser(strict=False)
        spec = parser.parse(definition)
        assert len(spec.features) == 1

    def test_parse_rejects_empty_features(self):
        parser = DSLParser(strict=True)
        with pytest.raises(FeatureDSLError, match="at least one feature"):
            parser.parse({"name": "empty", "version": "1.0", "features": []})

    def test_parse_detects_duplicate_names(self):
        definition = {
            "name": "dupes",
            "version": "1.0",
            "features": [
                {"name": "f1", "source": "c1", "transform": "log"},
                {"name": "f1", "source": "c2", "transform": "sqrt"},
            ],
        }
        parser = DSLParser(strict=True)
        with pytest.raises(FeatureDSLError, match="Duplicate"):
            parser.parse(definition)

    def test_parse_multi_source(self):
        definition = {
            "name": "multi",
            "version": "1.0",
            "features": [
                {"name": "inter", "source": ["col1", "col2"], "transform": "interaction"},
            ],
        }
        parser = DSLParser()
        spec = parser.parse(definition)
        assert spec.features[0].source == ["col1", "col2"]


class TestFeatureSpec:
    """Tests for FeatureSpec validation."""

    def test_valid_spec(self):
        spec = FeatureSpec(name="test", source="col", transform="log")
        assert spec.validate() == []

    def test_missing_name(self):
        spec = FeatureSpec(name="", source="col", transform="log")
        errors = spec.validate()
        assert any("name is required" in e for e in errors)

    def test_missing_source(self):
        spec = FeatureSpec(name="test", source="", transform="log")
        errors = spec.validate()
        assert any("source" in e for e in errors)

    def test_invalid_transform(self):
        spec = FeatureSpec(name="test", source="col", transform="invalid_op")
        errors = spec.validate()
        assert any("unknown transform" in e for e in errors)

    def test_invalid_dtype(self):
        spec = FeatureSpec(name="test", source="col", transform="log", dtype="complex")
        errors = spec.validate()
        assert any("invalid dtype" in e for e in errors)


class TestPipelineSpec:
    """Tests for PipelineSpec validation."""

    def test_valid_pipeline(self):
        spec = PipelineSpec(
            name="test",
            version="1.0",
            description="desc",
            features=[FeatureSpec(name="f", source="c", transform="log")],
        )
        assert spec.validate() == []

    def test_missing_name(self):
        spec = PipelineSpec(
            name="", version="1.0", description="",
            features=[FeatureSpec(name="f", source="c", transform="log")],
        )
        errors = spec.validate()
        assert any("name is required" in e for e in errors)

    def test_invalid_selection_method(self):
        spec = PipelineSpec(
            name="test", version="1.0", description="",
            features=[FeatureSpec(name="f", source="c", transform="log")],
            selection=SelectionSpec(method="unknown"),
        )
        errors = spec.validate()
        assert any("Unknown selection method" in e for e in errors)


# --- Compiler Tests ---

class TestDSLCompiler:
    """Tests for DSLCompiler."""

    def test_to_code_basic(self):
        spec = PipelineSpec(
            name="test_pipeline",
            version="1.0",
            description="A test pipeline",
            features=[
                FeatureSpec(name="price_log", source="price", transform="log"),
            ],
        )
        compiler = DSLCompiler()
        code = compiler.to_code(spec)
        assert "create_pipeline" in code
        assert "price_log" in code
        assert "Pipeline" in code

    def test_to_code_with_selection(self):
        spec = PipelineSpec(
            name="test",
            version="1.0",
            description="",
            features=[
                FeatureSpec(name="f", source="c", transform="log"),
            ],
            selection=SelectionSpec(method="importance", params={"k": 5}),
        )
        compiler = DSLCompiler()
        code = compiler.to_code(spec)
        assert "selection" in code

    def test_to_code_multiple_features(self):
        spec = PipelineSpec(
            name="multi",
            version="1.0",
            description="",
            features=[
                FeatureSpec(name="f1", source="c1", transform="log"),
                FeatureSpec(name="f2", source="c2", transform="sqrt"),
                FeatureSpec(name="f3", source="c3", transform="target"),
            ],
        )
        compiler = DSLCompiler()
        code = compiler.to_code(spec)
        assert "f1" in code
        assert "f2" in code
        assert "f3" in code

    def test_validate_spec(self):
        spec = PipelineSpec(
            name="valid",
            version="1.0",
            description="",
            features=[
                FeatureSpec(name="f", source="c", transform="log"),
            ],
        )
        compiler = DSLCompiler()
        report = compiler.validate_spec(spec)
        assert report["valid"] is True
        assert report["feature_count"] == 1

    def test_validate_spec_with_errors(self):
        spec = PipelineSpec(
            name="",
            version="1.0",
            description="",
            features=[],
        )
        compiler = DSLCompiler()
        report = compiler.validate_spec(spec)
        assert report["valid"] is False
        assert len(report["errors"]) > 0

    def test_validate_spec_with_unmapped_transform(self):
        spec = PipelineSpec(
            name="test",
            version="1.0",
            description="",
            features=[
                FeatureSpec(name="f", source="c", transform="reciprocal"),
            ],
        )
        compiler = DSLCompiler()
        report = compiler.validate_spec(spec)
        assert len(report["warnings"]) > 0


class TestYAMLParsing:
    """Tests for YAML string parsing."""

    def test_parse_yaml_basic(self):
        yaml_str = """
name: yaml_pipeline
version: "1.0"
features:
  - name: price_log
    source: price
    transform: log
  - name: cat_enc
    source: category
    transform: target
"""
        parser = DSLParser()
        try:
            spec = parser.parse_yaml(yaml_str)
            assert spec.name == "yaml_pipeline"
            assert len(spec.features) == 2
        except FeatureDSLError as e:
            if "PyYAML is required" in str(e):
                pytest.skip("PyYAML not installed")
            raise

    def test_parse_yaml_invalid(self):
        parser = DSLParser()
        try:
            with pytest.raises(FeatureDSLError):
                parser.parse_yaml(":::invalid yaml{{{")
        except FeatureDSLError as e:
            if "PyYAML is required" in str(e):
                pytest.skip("PyYAML not installed")
            raise
