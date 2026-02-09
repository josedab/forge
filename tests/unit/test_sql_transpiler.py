"""Tests for SQL transpilation."""

from __future__ import annotations

import pytest

from forge.dsl.parser import DSLParser, FeatureSpec, PipelineSpec
from forge.dsl.sql_transpiler import SQLTranspiler


@pytest.fixture
def simple_spec() -> PipelineSpec:
    parser = DSLParser(strict=False)
    return parser.parse({
        "name": "test",
        "version": "1.0",
        "features": [
            {"name": "price_log", "source": "price", "transform": "log"},
            {"name": "qty_sqrt", "source": "qty", "transform": "sqrt"},
            {"name": "price_sq", "source": "price", "transform": "square"},
        ],
    })


class TestSQLTranspiler:
    def test_postgres_dialect(self, simple_spec: PipelineSpec) -> None:
        t = SQLTranspiler(dialect="postgres")
        result = t.transpile(simple_spec)
        assert "LN" in result.sql
        assert "SQRT" in result.sql
        assert result.dialect == "postgres"

    def test_bigquery_dialect(self, simple_spec: PipelineSpec) -> None:
        t = SQLTranspiler(dialect="bigquery")
        result = t.transpile(simple_spec)
        assert "LN" in result.sql
        assert result.dialect == "bigquery"

    def test_snowflake_dialect(self, simple_spec: PipelineSpec) -> None:
        t = SQLTranspiler(dialect="snowflake")
        result = t.transpile(simple_spec)
        assert "LN" in result.sql

    def test_unsupported_dialect_raises(self) -> None:
        with pytest.raises(ValueError):
            SQLTranspiler(dialect="mysql")  # type: ignore[arg-type]

    def test_interaction(self) -> None:
        spec = PipelineSpec(
            name="test", version="1.0", description="",
            features=[FeatureSpec(name="a_x_b", source=["a", "b"], transform="interaction")],
        )
        t = SQLTranspiler(dialect="postgres")
        result = t.transpile(spec)
        assert "*" in result.sql  # multiplication

    def test_unsupported_transform_warns(self) -> None:
        spec = PipelineSpec(
            name="test", version="1.0", description="",
            features=[FeatureSpec(name="enc", source="cat", transform="onehot")],
        )
        t = SQLTranspiler(dialect="postgres")
        result = t.transpile(spec)
        assert len(result.warnings) > 0

    def test_bin_with_params(self) -> None:
        spec = PipelineSpec(
            name="test", version="1.0", description="",
            features=[FeatureSpec(name="x_bin", source="x", transform="bin", params={"n_bins": 5})],
        )
        t = SQLTranspiler(dialect="postgres")
        result = t.transpile(spec)
        assert "NTILE(5)" in result.sql

    def test_temporal_transforms(self) -> None:
        spec = PipelineSpec(
            name="test", version="1.0", description="",
            features=[
                FeatureSpec(name="m", source="dt", transform="month"),
                FeatureSpec(name="y", source="dt", transform="year"),
            ],
        )
        t = SQLTranspiler(dialect="bigquery")
        result = t.transpile(spec)
        assert "EXTRACT(MONTH" in result.sql
        assert "EXTRACT(YEAR" in result.sql

    def test_transpile_single_feature(self) -> None:
        feat = FeatureSpec(name="x_log", source="x", transform="log")
        t = SQLTranspiler(dialect="postgres")
        expr = t.transpile_feature(feat)
        assert expr is not None
        assert "LN" in expr

    def test_source_table_name(self) -> None:
        spec = PipelineSpec(
            name="test", version="1.0", description="",
            features=[FeatureSpec(name="x_log", source="x", transform="log")],
        )
        t = SQLTranspiler(dialect="postgres", source_table="my_table")
        result = t.transpile(spec)
        assert "my_table" in result.sql
