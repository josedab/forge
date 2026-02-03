"""Tests for notebook experience module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.notebook import (
    ColumnProfiler,
    DatasetExplorer,
    PipelineBuilder,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "income": [30000, 45000, 60000, 75000, 90000],
        "city": ["NY", "LA", "NY", "SF", "LA"],
        "score": [0.8, 0.6, 0.9, 0.7, 0.5],
    })


class TestColumnProfiler:
    def test_profile_numeric(self, sample_df):
        profiler = ColumnProfiler(sample_df)
        p = profiler.profile("age")
        assert p.inferred_type == "numeric"
        assert p.n_rows == 5
        assert p.n_null == 0
        assert "mean" in p.stats

    def test_profile_categorical(self, sample_df):
        profiler = ColumnProfiler(sample_df)
        p = profiler.profile("city")
        assert p.inferred_type == "categorical"
        assert "top_values" in p.stats

    def test_profile_missing(self, sample_df):
        profiler = ColumnProfiler(sample_df)
        with pytest.raises(KeyError):
            profiler.profile("nonexistent")

    def test_profile_all(self, sample_df):
        profiler = ColumnProfiler(sample_df)
        profiles = profiler.profile_all()
        assert len(profiles) == 4

    def test_summary_table(self, sample_df):
        profiler = ColumnProfiler(sample_df)
        summary = profiler.summary_table()
        assert "column" in summary.columns
        assert len(summary) == 4

    def test_to_html(self, sample_df):
        profiler = ColumnProfiler(sample_df)
        p = profiler.profile("age")
        html = p.to_html()
        assert "age" in html
        assert "numeric" in html

    def test_to_dict(self, sample_df):
        profiler = ColumnProfiler(sample_df)
        p = profiler.profile("age")
        d = p.to_dict()
        assert d["name"] == "age"
        assert d["inferred_type"] == "numeric"

    def test_recommendations_skewed(self):
        df = pd.DataFrame({"x": np.exp(np.random.randn(100))})
        profiler = ColumnProfiler(df)
        p = profiler.profile("x")
        assert any("Skewed" in r or "InteractionGenerator" in r for r in p.recommendations)

    def test_recommendations_high_nulls(self):
        df = pd.DataFrame({"x": [1, 2, None, None, None, None, None, None, None, None]})
        profiler = ColumnProfiler(df)
        p = profiler.profile("x")
        assert any("null" in r.lower() or "missing" in r.lower() for r in p.recommendations)

    def test_profile_with_nulls(self):
        df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0]})
        profiler = ColumnProfiler(df)
        p = profiler.profile("x")
        assert p.n_null == 1
        assert p.null_pct > 0


class TestPipelineBuilder:
    def test_add_step(self, sample_df):
        builder = PipelineBuilder(sample_df)
        builder.add_step("numeric", columns=["age", "income"])
        assert len(builder.steps) == 1

    def test_chaining(self, sample_df):
        builder = PipelineBuilder(sample_df)
        result = builder.add_step("numeric").add_step("categorical")
        assert len(result.steps) == 2

    def test_remove_step(self, sample_df):
        builder = PipelineBuilder(sample_df)
        builder.add_step("numeric").add_step("categorical")
        builder.remove_step(0)
        assert len(builder.steps) == 1

    def test_auto_detect_numeric(self, sample_df):
        builder = PipelineBuilder(sample_df)
        builder.add_step("numeric")
        step = builder.steps[0]
        assert "age" in step.columns
        assert "income" in step.columns
        assert "city" not in step.columns

    def test_auto_detect_categorical(self, sample_df):
        builder = PipelineBuilder(sample_df)
        builder.add_step("categorical")
        step = builder.steps[0]
        assert "city" in step.columns

    def test_preview(self, sample_df):
        builder = PipelineBuilder(sample_df)
        builder.add_step("numeric", columns=["age"])
        preview = builder.preview()
        assert "age_squared" in preview.columns

    def test_export_code(self, sample_df):
        builder = PipelineBuilder(sample_df)
        builder.add_step("numeric", columns=["age", "income"])
        builder.add_step("categorical", columns=["city"])
        code = builder.export_code()
        assert "Pipeline" in code
        assert "InteractionGenerator" in code
        assert "TargetEncoder" in code

    def test_export_code_scaler(self, sample_df):
        builder = PipelineBuilder(sample_df)
        builder.add_step("scaler")
        code = builder.export_code()
        assert "StandardScaler" in code


class TestDatasetExplorer:
    def test_overview(self, sample_df):
        explorer = DatasetExplorer(sample_df)
        ov = explorer.overview()
        assert ov["rows"] == 5
        assert ov["columns"] == 4
        assert ov["numeric_columns"] == 3
        assert ov["categorical_columns"] == 1

    def test_suggest_pipeline(self, sample_df):
        explorer = DatasetExplorer(sample_df)
        suggestions = explorer.suggest_pipeline()
        types = [s.generator_type for s in suggestions]
        assert "numeric" in types
        assert "categorical" in types

    def test_overview_memory(self, sample_df):
        explorer = DatasetExplorer(sample_df)
        ov = explorer.overview()
        assert ov["memory_mb"] >= 0
