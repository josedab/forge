"""Tests for pipeline debugger."""

from __future__ import annotations

import pandas as pd
import pytest

from forge.transformers.debugger import DebugReport, PipelineDebugger, StepResult


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [10.0, 20.0, 30.0, 40.0, 50.0],
    })


class _DoubleTransformer:
    """Simple test transformer that doubles all values."""

    def fit(self, X: pd.DataFrame, y: object = None) -> _DoubleTransformer:
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X * 2


class _AddColTransformer:
    """Test transformer that adds a new column."""

    def fit(self, X: pd.DataFrame, y: object = None) -> _AddColTransformer:
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        out["a_plus_b"] = out["a"] + out["b"]
        return out


class _ErrorTransformer:
    """Test transformer that raises an error."""

    def fit(self, X: pd.DataFrame, y: object = None) -> _ErrorTransformer:
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise ValueError("intentional error")


class TestPipelineDebugger:
    def test_basic_run(self, sample_df: pd.DataFrame) -> None:
        dbg = PipelineDebugger(steps=[
            ("double", _DoubleTransformer()),
        ])
        report = dbg.run(sample_df)
        assert len(report.steps) == 1
        assert report.input_shape == (5, 2)
        assert report.output_shape == (5, 2)

    def test_feature_addition(self, sample_df: pd.DataFrame) -> None:
        dbg = PipelineDebugger(steps=[
            ("add_col", _AddColTransformer()),
        ])
        report = dbg.run(sample_df)
        assert report.steps[0].features_added == ["a_plus_b"]
        assert report.steps[0].net_features == 1

    def test_multi_step(self, sample_df: pd.DataFrame) -> None:
        dbg = PipelineDebugger(steps=[
            ("add_col", _AddColTransformer()),
            ("double", _DoubleTransformer()),
        ])
        report = dbg.run(sample_df)
        assert len(report.steps) == 2
        assert report.output_shape == (5, 3)

    def test_stop_after(self, sample_df: pd.DataFrame) -> None:
        dbg = PipelineDebugger(steps=[
            ("add_col", _AddColTransformer()),
            ("double", _DoubleTransformer()),
        ])
        report = dbg.run(sample_df, stop_after=1)
        assert len(report.steps) == 1

    def test_intermediate_access(self, sample_df: pd.DataFrame) -> None:
        dbg = PipelineDebugger(steps=[
            ("add_col", _AddColTransformer()),
        ])
        dbg.run(sample_df)
        original = dbg.get_intermediate(0)
        after_step = dbg.get_intermediate(1)
        assert original is not None
        assert after_step is not None
        assert "a_plus_b" not in original.columns
        assert "a_plus_b" in after_step.columns

    def test_out_of_range_intermediate(self, sample_df: pd.DataFrame) -> None:
        dbg = PipelineDebugger(steps=[])
        dbg.run(sample_df)
        assert dbg.get_intermediate(999) is None

    def test_error_handling(self, sample_df: pd.DataFrame) -> None:
        dbg = PipelineDebugger(steps=[
            ("err", _ErrorTransformer()),
        ])
        report = dbg.run(sample_df)
        assert len(report.steps[0].errors) == 1
        assert "intentional error" in report.steps[0].errors[0]

    def test_lineage(self, sample_df: pd.DataFrame) -> None:
        dbg = PipelineDebugger(steps=[
            ("add_col", _AddColTransformer()),
        ])
        dbg.run(sample_df)
        lineage = dbg.get_lineage()
        assert "a_plus_b" in lineage
        assert lineage["a_plus_b"].transform_step == "add_col"

    def test_lineage_trace(self, sample_df: pd.DataFrame) -> None:
        dbg = PipelineDebugger(steps=[
            ("add_col", _AddColTransformer()),
        ])
        dbg.run(sample_df)
        sources = dbg.get_lineage_for("a_plus_b")
        assert "a" in sources
        assert "b" in sources

    def test_lineage_trace_unknown(self, sample_df: pd.DataFrame) -> None:
        dbg = PipelineDebugger(steps=[])
        dbg.run(sample_df)
        assert dbg.get_lineage_for("nonexistent") == []

    def test_summary_output(self, sample_df: pd.DataFrame) -> None:
        dbg = PipelineDebugger(steps=[
            ("add_col", _AddColTransformer()),
        ])
        report = dbg.run(sample_df)
        text = report.summary()
        assert "Pipeline Debug Report" in text
        assert "add_col" in text

    def test_slowest_step(self, sample_df: pd.DataFrame) -> None:
        dbg = PipelineDebugger(steps=[
            ("double", _DoubleTransformer()),
            ("add_col", _AddColTransformer()),
        ])
        report = dbg.run(sample_df)
        slowest = report.slowest_step()
        assert slowest is not None

    def test_empty_report_slowest(self) -> None:
        report = DebugReport()
        assert report.slowest_step() is None

    def test_step_result_to_dict(self) -> None:
        sr = StepResult(step_name="x", n_features_in=5, n_features_out=7)
        d = sr.to_dict()
        assert d["step_name"] == "x"
        assert d["n_features_out"] == 7

    def test_capture_stats(self, sample_df: pd.DataFrame) -> None:
        dbg = PipelineDebugger(
            steps=[("add_col", _AddColTransformer())],
            capture_stats=True,
        )
        report = dbg.run(sample_df)
        assert "a_plus_b" in report.steps[0].stats
        assert "mean" in report.steps[0].stats["a_plus_b"]
