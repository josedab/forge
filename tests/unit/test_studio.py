"""Tests for Feature Studio."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from forge.studio import (
    ComparisonResult,
    FeatureStudio,
    PipelineStep,
    PreviewResult,
)


@pytest.fixture
def sample_df():
    np.random.seed(42)
    return pd.DataFrame({
        "a": np.random.randn(100),
        "b": np.random.randn(100),
        "c": np.random.uniform(0, 10, 100),
    })


@pytest.fixture
def sample_target():
    np.random.seed(42)
    return np.random.randint(0, 2, 100)


class TestFeatureStudio:
    """Tests for FeatureStudio."""

    def test_load_data(self, sample_df):
        studio = FeatureStudio()
        studio.load_data(sample_df)
        assert studio._data is not None

    def test_load_data_rejects_non_df(self):
        studio = FeatureStudio()
        with pytest.raises(ValueError, match="Expected pd.DataFrame"):
            studio.load_data([[1, 2], [3, 4]])

    def test_add_step(self, sample_df):
        studio = FeatureStudio()
        studio.load_data(sample_df)
        studio.add_step("scale", StandardScaler())
        assert len(studio.get_steps()) == 1

    def test_add_duplicate_step_raises(self, sample_df):
        studio = FeatureStudio()
        studio.load_data(sample_df)
        studio.add_step("scale", StandardScaler())
        with pytest.raises(ValueError, match="already exists"):
            studio.add_step("scale", MinMaxScaler())

    def test_remove_step(self, sample_df):
        studio = FeatureStudio()
        studio.load_data(sample_df)
        studio.add_step("scale", StandardScaler())
        studio.remove_step("scale")
        assert len(studio.get_steps()) == 0

    def test_toggle_step(self, sample_df):
        studio = FeatureStudio()
        studio.load_data(sample_df)
        studio.add_step("scale", StandardScaler())
        studio.toggle_step("scale")
        assert studio.get_steps()[0].enabled is False
        studio.toggle_step("scale")
        assert studio.get_steps()[0].enabled is True

    def test_toggle_nonexistent_raises(self, sample_df):
        studio = FeatureStudio()
        studio.load_data(sample_df)
        with pytest.raises(ValueError, match="not found"):
            studio.toggle_step("nonexistent")

    def test_reorder_steps(self, sample_df):
        studio = FeatureStudio()
        studio.load_data(sample_df)
        studio.add_step("a", StandardScaler())
        studio.add_step("b", MinMaxScaler())
        studio.reorder_steps(["b", "a"])
        steps = studio.get_steps()
        assert steps[0].name == "b"
        assert steps[1].name == "a"

    def test_preview_basic(self, sample_df):
        studio = FeatureStudio()
        studio.load_data(sample_df)
        studio.add_step("scale", StandardScaler())
        preview = studio.preview()
        assert isinstance(preview, PreviewResult)
        assert preview.n_input_features == 3
        assert preview.n_output_features >= 3

    def test_preview_no_data_raises(self):
        studio = FeatureStudio()
        with pytest.raises(RuntimeError, match="No data loaded"):
            studio.preview()

    def test_preview_disabled_step(self, sample_df):
        studio = FeatureStudio()
        studio.load_data(sample_df)
        studio.add_step("scale", StandardScaler())
        studio.toggle_step("scale")
        preview = studio.preview()
        assert preview.n_output_features == 3  # No transformation applied

    def test_compare(self, sample_df, sample_target):
        studio = FeatureStudio()
        studio.load_data(sample_df, sample_target)
        studio.add_step("scale", StandardScaler())

        other = [PipelineStep("minmax", MinMaxScaler())]
        result = studio.compare(other)
        assert isinstance(result, ComparisonResult)
        assert result.a_features > 0
        assert result.b_features > 0

    def test_export_code(self, sample_df):
        studio = FeatureStudio()
        studio.load_data(sample_df)
        studio.add_step("scale", StandardScaler())
        code = studio.export_code()
        assert "Pipeline" in code
        assert "StandardScaler" in code
        assert "create_pipeline" in code

    def test_data_summary(self, sample_df):
        studio = FeatureStudio()
        studio.load_data(sample_df)
        summary = studio.data_summary()
        assert summary["n_rows"] == 100
        assert summary["n_columns"] == 3
        assert "a" in summary["columns"]

    def test_data_summary_no_data(self):
        studio = FeatureStudio()
        summary = studio.data_summary()
        assert "error" in summary

    def test_chaining(self, sample_df):
        studio = (
            FeatureStudio()
            .load_data(sample_df)
            .add_step("scale", StandardScaler())
            .add_step("minmax", MinMaxScaler())
        )
        assert len(studio.get_steps()) == 2

    def test_sample_size(self, sample_df):
        studio = FeatureStudio(sample_size=10)
        studio.load_data(sample_df)
        studio.add_step("scale", StandardScaler())
        preview = studio.preview()
        assert preview.n_output_features >= 3
