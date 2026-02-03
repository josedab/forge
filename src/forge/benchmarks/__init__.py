"""Benchmark harness for comparing Forge against baseline approaches.

Provides a framework for reproducible feature engineering benchmarks
across standard datasets, measuring time-to-transform, feature count,
and downstream model accuracy.

Example:
    >>> from forge.benchmarks import BenchmarkHarness, ForgeBaseline, ManualBaseline
    >>> harness = BenchmarkHarness()
    >>> harness.add_baseline("forge", ForgeBaseline())
    >>> harness.add_baseline("manual", ManualBaseline())
    >>> results = harness.run(datasets=["iris", "titanic"])
    >>> print(harness.leaderboard())
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_wine,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkDataset:
    """A dataset for benchmarking."""

    name: str
    X: pd.DataFrame
    y: pd.Series
    task: str  # "classification" or "regression"
    description: str = ""

    @property
    def n_rows(self) -> int:
        return len(self.X)

    @property
    def n_cols(self) -> int:
        return self.X.shape[1]


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    dataset_name: str
    baseline_name: str
    fit_time_seconds: float
    transform_time_seconds: float
    total_time_seconds: float
    n_features_out: int
    cv_score_mean: float
    cv_score_std: float
    error: str = ""

    @property
    def score_display(self) -> str:
        if self.error:
            return f"ERROR: {self.error}"
        return f"{self.cv_score_mean:.4f} ± {self.cv_score_std:.4f}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "baseline": self.baseline_name,
            "fit_time_s": round(self.fit_time_seconds, 4),
            "transform_time_s": round(self.transform_time_seconds, 4),
            "total_time_s": round(self.total_time_seconds, 4),
            "n_features_out": self.n_features_out,
            "cv_score_mean": round(self.cv_score_mean, 4),
            "cv_score_std": round(self.cv_score_std, 4),
            "error": self.error,
        }


class FeatureEngineeringBaseline(ABC):
    """Abstract baseline for feature engineering comparison."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable name."""

    @abstractmethod
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform data, returning engineered features."""


class ForgeBaseline(FeatureEngineeringBaseline):
    """Forge AutoFeatureTransformer baseline."""

    def __init__(self, max_features: int = 50) -> None:
        self.max_features = max_features
        self._transformer: Any = None

    def name(self) -> str:
        return f"Forge(max_features={self.max_features})"

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        from forge.transformers.auto_transformer import AutoFeatureTransformer

        self._transformer = AutoFeatureTransformer(max_features=self.max_features)
        return self._transformer.fit_transform(X, y)


class ManualBaseline(FeatureEngineeringBaseline):
    """Manual feature engineering baseline (passthrough + scaling)."""

    def __init__(self) -> None:
        self._scaler = StandardScaler()

    def name(self) -> str:
        return "Manual(scale-only)"

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return X.copy()
        result = X.copy()
        result[numeric_cols] = self._scaler.fit_transform(X[numeric_cols])
        return result


class PassthroughBaseline(FeatureEngineeringBaseline):
    """No feature engineering — raw features only."""

    def name(self) -> str:
        return "Passthrough(raw)"

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return X.select_dtypes(include=[np.number]).copy()


def load_builtin_datasets() -> list[BenchmarkDataset]:
    """Load built-in sklearn datasets for benchmarking."""
    datasets: list[BenchmarkDataset] = []

    iris = load_iris(as_frame=True)
    datasets.append(BenchmarkDataset(
        name="iris", X=iris.data, y=iris.target,
        task="classification", description="Iris flower classification (150×4)",
    ))

    wine = load_wine(as_frame=True)
    datasets.append(BenchmarkDataset(
        name="wine", X=wine.data, y=wine.target,
        task="classification", description="Wine recognition (178×13)",
    ))

    cancer = load_breast_cancer(as_frame=True)
    datasets.append(BenchmarkDataset(
        name="breast_cancer", X=cancer.data, y=cancer.target,
        task="classification", description="Breast cancer diagnosis (569×30)",
    ))

    diabetes = load_diabetes(as_frame=True)
    datasets.append(BenchmarkDataset(
        name="diabetes", X=diabetes.data, y=diabetes.target,
        task="regression", description="Diabetes progression (442×10)",
    ))

    return datasets


def make_synthetic_dataset(
    n_rows: int = 1000,
    n_numeric: int = 10,
    n_categorical: int = 5,
    task: str = "classification",
    random_state: int = 42,
) -> BenchmarkDataset:
    """Create a synthetic dataset for benchmarking."""
    rng = np.random.RandomState(random_state)
    data: dict[str, Any] = {}

    for i in range(n_numeric):
        data[f"num_{i}"] = rng.randn(n_rows)

    for i in range(n_categorical):
        n_cats = rng.randint(3, 10)
        data[f"cat_{i}"] = rng.choice([f"val_{j}" for j in range(n_cats)], n_rows)

    X = pd.DataFrame(data)

    if task == "classification":
        y = pd.Series(rng.randint(0, 3, n_rows), name="target")
    else:
        y = pd.Series(rng.randn(n_rows), name="target")

    return BenchmarkDataset(
        name=f"synthetic_{n_rows}x{n_numeric + n_categorical}",
        X=X, y=y, task=task,
        description=f"Synthetic ({n_rows}×{n_numeric + n_categorical})",
    )


class BenchmarkHarness:
    """Orchestrates benchmark comparisons across datasets and baselines.

    Parameters:
        cv_folds: Number of cross-validation folds.
        random_state: Random state for reproducibility.
    """

    def __init__(self, cv_folds: int = 5, random_state: int = 42) -> None:
        self.cv_folds = cv_folds
        self.random_state = random_state
        self._baselines: dict[str, FeatureEngineeringBaseline] = {}
        self._results: list[BenchmarkResult] = []

    def add_baseline(self, key: str, baseline: FeatureEngineeringBaseline) -> None:
        """Register a baseline for comparison."""
        self._baselines[key] = baseline

    def run(
        self,
        datasets: list[BenchmarkDataset] | None = None,
        dataset_names: list[str] | None = None,
    ) -> list[BenchmarkResult]:
        """Run benchmarks across all baselines and datasets.

        Args:
            datasets: Custom datasets. If None, loads built-in datasets.
            dataset_names: Filter built-in datasets by name.

        Returns:
            List of BenchmarkResult objects.
        """
        if datasets is None:
            datasets = load_builtin_datasets()
        if dataset_names:
            datasets = [d for d in datasets if d.name in dataset_names]

        self._results = []
        for dataset in datasets:
            for key, baseline in self._baselines.items():
                result = self._run_single(dataset, key, baseline)
                self._results.append(result)

        return list(self._results)

    def _run_single(
        self, dataset: BenchmarkDataset, key: str,
        baseline: FeatureEngineeringBaseline,
    ) -> BenchmarkResult:
        """Run a single baseline on a single dataset."""
        try:
            t0 = time.perf_counter()
            X_transformed = baseline.fit_transform(dataset.X.copy(), dataset.y.copy())
            t1 = time.perf_counter()

            X_numeric = X_transformed.select_dtypes(include=[np.number])
            if X_numeric.shape[1] == 0:
                return BenchmarkResult(
                    dataset_name=dataset.name, baseline_name=baseline.name(),
                    fit_time_seconds=t1 - t0, transform_time_seconds=0,
                    total_time_seconds=t1 - t0, n_features_out=0,
                    cv_score_mean=0, cv_score_std=0,
                    error="No numeric features produced",
                )

            X_clean = X_numeric.replace([np.inf, -np.inf], np.nan).fillna(0)

            if dataset.task == "classification":
                model = RandomForestClassifier(
                    n_estimators=50, random_state=self.random_state, n_jobs=-1,
                )
                scoring = "accuracy"
            else:
                model = RandomForestRegressor(
                    n_estimators=50, random_state=self.random_state, n_jobs=-1,
                )
                scoring = "r2"

            t2 = time.perf_counter()
            scores = cross_val_score(
                model, X_clean, dataset.y, cv=self.cv_folds, scoring=scoring,
            )
            t3 = time.perf_counter()

            return BenchmarkResult(
                dataset_name=dataset.name, baseline_name=baseline.name(),
                fit_time_seconds=t1 - t0, transform_time_seconds=t3 - t2,
                total_time_seconds=(t1 - t0) + (t3 - t2),
                n_features_out=X_clean.shape[1],
                cv_score_mean=float(scores.mean()),
                cv_score_std=float(scores.std()),
            )
        except Exception as e:
            logger.warning("Benchmark error: %s on %s: %s", baseline.name(), dataset.name, e)
            return BenchmarkResult(
                dataset_name=dataset.name, baseline_name=baseline.name(),
                fit_time_seconds=0, transform_time_seconds=0, total_time_seconds=0,
                n_features_out=0, cv_score_mean=0, cv_score_std=0, error=str(e),
            )

    def leaderboard(self) -> pd.DataFrame:
        """Generate a leaderboard DataFrame from results."""
        if not self._results:
            return pd.DataFrame()
        rows = [r.to_dict() for r in self._results if not r.error]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        return df.sort_values(
            ["dataset", "cv_score_mean"], ascending=[True, False],
        ).reset_index(drop=True)

    def leaderboard_markdown(self) -> str:
        """Generate markdown leaderboard."""
        df = self.leaderboard()
        if df.empty:
            return "No results available."

        lines = ["# Forge Benchmark Leaderboard", ""]
        for dataset_name, group in df.groupby("dataset"):
            lines.append(f"## {dataset_name}")
            lines.append("")
            lines.append("| Rank | Baseline | CV Score | Features | Time (s) |")
            lines.append("|------|----------|----------|----------|----------|")
            for rank, (_, row) in enumerate(group.iterrows(), 1):
                lines.append(
                    f"| {rank} | {row['baseline']} | "
                    f"{row['cv_score_mean']:.4f}±{row['cv_score_std']:.4f} | "
                    f"{row['n_features_out']} | {row['total_time_s']:.2f} |"
                )
            lines.append("")
        return "\n".join(lines)

    @property
    def results(self) -> list[BenchmarkResult]:
        """All benchmark results."""
        return list(self._results)

    def summary(self) -> dict[str, Any]:
        """Summary statistics across all results."""
        valid = [r for r in self._results if not r.error]
        if not valid:
            return {"total_runs": 0}

        baselines = set(r.baseline_name for r in valid)
        datasets = set(r.dataset_name for r in valid)

        wins: dict[str, int] = dict.fromkeys(baselines, 0)
        for ds in datasets:
            ds_results = [r for r in valid if r.dataset_name == ds]
            best = max(ds_results, key=lambda r: r.cv_score_mean)
            wins[best.baseline_name] += 1

        return {
            "total_runs": len(valid),
            "datasets": len(datasets),
            "baselines": len(baselines),
            "wins_per_baseline": wins,
        }
