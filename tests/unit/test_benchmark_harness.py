"""Tests for the benchmark harness."""

from __future__ import annotations

import pandas as pd

from forge.benchmarks import (
    BenchmarkDataset,
    BenchmarkHarness,
    BenchmarkResult,
    ManualBaseline,
    PassthroughBaseline,
    load_builtin_datasets,
    make_synthetic_dataset,
)


class TestBenchmarkDataset:
    def test_properties(self):
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        ds = BenchmarkDataset(name="test", X=X, y=y, task="classification")
        assert ds.n_rows == 3
        assert ds.n_cols == 2


class TestBaselines:
    def test_manual_baseline(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        y = pd.Series([0, 1, 0])
        bl = ManualBaseline()
        result = bl.fit_transform(X, y)
        assert result.shape == X.shape
        assert bl.name() == "Manual(scale-only)"

    def test_passthrough_baseline(self):
        X = pd.DataFrame({"a": [1, 2, 3], "cat": ["x", "y", "z"]})
        y = pd.Series([0, 1, 0])
        bl = PassthroughBaseline()
        result = bl.fit_transform(X, y)
        assert "a" in result.columns
        assert "cat" not in result.columns


class TestLoadDatasets:
    def test_builtin_datasets(self):
        datasets = load_builtin_datasets()
        assert len(datasets) == 4
        names = {d.name for d in datasets}
        assert "iris" in names
        assert "diabetes" in names

    def test_synthetic_dataset(self):
        ds = make_synthetic_dataset(n_rows=200, n_numeric=5, n_categorical=3)
        assert ds.n_rows == 200
        assert ds.n_cols == 8
        assert "synthetic" in ds.name


class TestBenchmarkResult:
    def test_score_display(self):
        r = BenchmarkResult(
            dataset_name="test", baseline_name="bl",
            fit_time_seconds=0.1, transform_time_seconds=0.2,
            total_time_seconds=0.3, n_features_out=5,
            cv_score_mean=0.95, cv_score_std=0.02,
        )
        assert "0.95" in r.score_display

    def test_error_display(self):
        r = BenchmarkResult(
            dataset_name="test", baseline_name="bl",
            fit_time_seconds=0, transform_time_seconds=0,
            total_time_seconds=0, n_features_out=0,
            cv_score_mean=0, cv_score_std=0, error="boom",
        )
        assert "ERROR" in r.score_display

    def test_to_dict(self):
        r = BenchmarkResult(
            dataset_name="test", baseline_name="bl",
            fit_time_seconds=0.1, transform_time_seconds=0.2,
            total_time_seconds=0.3, n_features_out=5,
            cv_score_mean=0.95, cv_score_std=0.02,
        )
        d = r.to_dict()
        assert d["dataset"] == "test"
        assert d["cv_score_mean"] == 0.95


class TestBenchmarkHarness:
    def test_run_with_simple_baselines(self):
        harness = BenchmarkHarness(cv_folds=3)
        harness.add_baseline("manual", ManualBaseline())
        harness.add_baseline("passthrough", PassthroughBaseline())

        results = harness.run(dataset_names=["iris"])
        assert len(results) == 2
        assert all(r.cv_score_mean > 0 for r in results)

    def test_run_with_synthetic(self):
        ds = make_synthetic_dataset(n_rows=100, n_numeric=3, n_categorical=0)
        harness = BenchmarkHarness(cv_folds=3)
        harness.add_baseline("passthrough", PassthroughBaseline())

        results = harness.run(datasets=[ds])
        assert len(results) == 1

    def test_leaderboard(self):
        harness = BenchmarkHarness(cv_folds=3)
        harness.add_baseline("manual", ManualBaseline())
        harness.add_baseline("passthrough", PassthroughBaseline())
        harness.run(dataset_names=["iris"])

        lb = harness.leaderboard()
        assert not lb.empty
        assert "cv_score_mean" in lb.columns

    def test_leaderboard_markdown(self):
        harness = BenchmarkHarness(cv_folds=3)
        harness.add_baseline("manual", ManualBaseline())
        harness.run(dataset_names=["iris"])

        md = harness.leaderboard_markdown()
        assert "# Forge Benchmark Leaderboard" in md
        assert "iris" in md

    def test_summary(self):
        harness = BenchmarkHarness(cv_folds=3)
        harness.add_baseline("manual", ManualBaseline())
        harness.add_baseline("passthrough", PassthroughBaseline())
        harness.run(dataset_names=["iris"])

        s = harness.summary()
        assert s["total_runs"] == 2
        assert s["datasets"] == 1
        assert s["baselines"] == 2

    def test_empty_results(self):
        harness = BenchmarkHarness()
        assert harness.leaderboard().empty
        assert harness.summary() == {"total_runs": 0}

    def test_dataset_name_filter(self):
        harness = BenchmarkHarness(cv_folds=3)
        harness.add_baseline("passthrough", PassthroughBaseline())
        results = harness.run(dataset_names=["wine"])
        assert len(results) == 1
        assert results[0].dataset_name == "wine"
