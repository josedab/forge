"""Tests for the visual feature studio app module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.studio.app import (
    AppConfig,
    ColumnAnalysis,
    PipelineConfig,
    analyze_dataframe,
    compare_dataframes,
    create_streamlit_app_code,
    generate_distribution_summary,
    launch_studio,
)


@pytest.fixture
def sample_df():
    np.random.seed(42)
    return pd.DataFrame({
        "age": np.random.randint(18, 80, 100),
        "income": np.random.normal(50000, 15000, 100),
        "category": np.random.choice(["A", "B", "C"], 100),
        "score": np.random.uniform(0, 1, 100),
    })


class TestAnalyzeDataframe:
    def test_returns_analyses(self, sample_df):
        analyses = analyze_dataframe(sample_df)
        assert len(analyses) == 4
        assert all(isinstance(a, ColumnAnalysis) for a in analyses)

    def test_numeric_stats(self, sample_df):
        analyses = analyze_dataframe(sample_df)
        age_analysis = [a for a in analyses if a.name == "age"][0]
        assert age_analysis.dtype == "numeric"
        assert "mean" in age_analysis.stats
        assert "std" in age_analysis.stats

    def test_categorical_detection(self, sample_df):
        analyses = analyze_dataframe(sample_df)
        cat_analysis = [a for a in analyses if a.name == "category"][0]
        assert cat_analysis.dtype == "categorical"
        assert cat_analysis.unique_count == 3

    def test_null_detection(self):
        df = pd.DataFrame({"a": [1.0, None, 3.0, None, 5.0]})
        analyses = analyze_dataframe(df)
        assert analyses[0].null_count == 2
        assert analyses[0].null_pct == pytest.approx(0.4)

    def test_to_dict(self, sample_df):
        analyses = analyze_dataframe(sample_df)
        d = analyses[0].to_dict()
        assert "name" in d
        assert "dtype" in d


class TestDistributionSummary:
    def test_numeric_histogram(self, sample_df):
        result = generate_distribution_summary(sample_df, "age")
        assert result["type"] == "histogram"
        assert "counts" in result
        assert "edges" in result
        assert "mean" in result

    def test_categorical_bar(self, sample_df):
        result = generate_distribution_summary(sample_df, "category")
        assert result["type"] == "bar"
        assert "labels" in result
        assert "counts" in result


class TestCompareDataframes:
    def test_basic_comparison(self, sample_df):
        after = sample_df.copy()
        after["new_feature"] = after["age"] * after["income"]

        result = compare_dataframes(sample_df, after)
        assert result["columns_before"] == 4
        assert result["columns_after"] == 5
        assert "new_feature" in result["new_columns"]

    def test_removed_columns(self, sample_df):
        after = sample_df.drop(columns=["category"])
        result = compare_dataframes(sample_df, after)
        assert "category" in result["removed_columns"]

    def test_stat_changes(self, sample_df):
        after = sample_df.copy()
        after["income"] = after["income"] * 2
        result = compare_dataframes(sample_df, after)
        assert "income" in result["stat_changes"]


class TestPipelineConfig:
    def test_to_code_default(self):
        config = PipelineConfig(max_features=50)
        code = config.to_code()
        assert "AutoFeatureTransformer" in code
        assert "max_features=50" in code

    def test_to_code_with_steps(self):
        config = PipelineConfig(steps=[
            {"name": "scaler", "code": "StandardScaler()"},
        ])
        code = config.to_code()
        assert "ForgePipeline" in code
        assert "scaler" in code

    def test_to_code_is_valid_python(self):
        config = PipelineConfig(max_features=100)
        code = config.to_code()
        # Should compile without syntax errors
        compile(code, "<string>", "exec")


class TestStreamlitApp:
    def test_create_app_code(self):
        code = create_streamlit_app_code()
        assert "streamlit" in code
        assert "Forge Feature Studio" in code

    def test_custom_config(self):
        config = AppConfig(title="My Studio", port=9000)
        code = create_streamlit_app_code(config)
        assert "My Studio" in code

    def test_launch_studio(self):
        code = launch_studio()
        assert isinstance(code, str)
        assert "streamlit" in code


class TestAppConfig:
    def test_defaults(self):
        config = AppConfig()
        assert config.title == "Forge Feature Studio"
        assert config.port == 8501
        assert config.max_preview_rows == 100
