"""Pipeline debugger and feature explainer.

Provides step-through debugging of feature engineering pipelines,
intermediate-state inspection, and lineage DAG construction.

Example:
    >>> from forge.transformers.debugger import PipelineDebugger
    >>> dbg = PipelineDebugger(pipeline)
    >>> report = dbg.run(X)
    >>> print(report.summary())
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator  # noqa: TC002

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result of executing a single pipeline step.

    Attributes:
        step_name: Name of the step.
        n_features_in: Number of features before this step.
        n_features_out: Number of features after this step.
        features_added: New features created.
        features_removed: Features removed.
        elapsed_ms: Execution time in milliseconds.
        stats: Column-level statistics snapshot.
        errors: Any errors that occurred.
    """

    step_name: str
    n_features_in: int
    n_features_out: int
    features_added: list[str] = field(default_factory=list)
    features_removed: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def net_features(self) -> int:
        return self.n_features_out - self.n_features_in

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_name": self.step_name,
            "n_features_in": self.n_features_in,
            "n_features_out": self.n_features_out,
            "features_added": self.features_added,
            "features_removed": self.features_removed,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "errors": self.errors,
        }


@dataclass
class DebugReport:
    """Full debug report for a pipeline run.

    Attributes:
        steps: Ordered list of step results.
        total_elapsed_ms: Total execution time.
        input_shape: Shape of input data.
        output_shape: Shape of output data.
    """

    steps: list[StepResult] = field(default_factory=list)
    total_elapsed_ms: float = 0.0
    input_shape: tuple[int, int] = (0, 0)
    output_shape: tuple[int, int] = (0, 0)

    def summary(self) -> str:
        """Human-readable summary of the debug run."""
        lines = [
            f"Pipeline Debug Report ({len(self.steps)} steps)",
            f"  Input:  {self.input_shape[0]} rows x {self.input_shape[1]} cols",
            f"  Output: {self.output_shape[0]} rows x {self.output_shape[1]} cols",
            f"  Total time: {self.total_elapsed_ms:.1f} ms",
            "",
        ]
        for i, step in enumerate(self.steps, 1):
            sign = "+" if step.net_features >= 0 else ""
            lines.append(
                f"  [{i}] {step.step_name}: "
                f"{step.n_features_in} → {step.n_features_out} "
                f"({sign}{step.net_features}) "
                f"[{step.elapsed_ms:.1f} ms]"
            )
            if step.errors:
                for err in step.errors:
                    lines.append(f"      ⚠ {err}")
        return "\n".join(lines)

    def slowest_step(self) -> StepResult | None:
        if not self.steps:
            return None
        return max(self.steps, key=lambda s: s.elapsed_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "total_elapsed_ms": round(self.total_elapsed_ms, 3),
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
        }


@dataclass
class LineageNode:
    """Node in a feature lineage DAG."""

    name: str
    source_columns: list[str] = field(default_factory=list)
    transform_step: str = ""
    depth: int = 0


class PipelineDebugger:
    """Step-through debugger for feature engineering pipelines.

    Executes a pipeline one step at a time, recording intermediate
    DataFrames, timing information, and feature lineage.

    Parameters
    ----------
    steps : list of (name, transformer) tuples
        Pipeline steps to debug.
    capture_stats : bool
        Whether to capture per-column statistics at each step.
    """

    def __init__(
        self,
        steps: list[tuple[str, BaseEstimator]],
        capture_stats: bool = True,
    ) -> None:
        self.steps = steps
        self.capture_stats = capture_stats
        self._intermediates: list[pd.DataFrame] = []
        self._lineage: dict[str, LineageNode] = {}

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        *,
        stop_after: int | None = None,
    ) -> DebugReport:
        """Execute the pipeline with debugging.

        Args:
            X: Input DataFrame.
            y: Optional target.
            stop_after: Stop after this many steps (for breakpoints).

        Returns:
            DebugReport with per-step results.
        """
        report = DebugReport()
        report.input_shape = X.shape  # type: ignore[assignment]
        current = X.copy()
        self._intermediates = [current.copy()]

        # Record original columns
        for col in X.columns:
            self._lineage[col] = LineageNode(name=col, depth=0)

        t_start = time.perf_counter()
        n_steps = stop_after if stop_after is not None else len(self.steps)

        for i, (name, transformer) in enumerate(self.steps[:n_steps]):
            step_result = self._run_step(name, transformer, current, y, depth=i + 1)
            report.steps.append(step_result)

            if not step_result.errors:
                current = self._transform(transformer, current, y)

            self._intermediates.append(current.copy())

        report.total_elapsed_ms = (time.perf_counter() - t_start) * 1000
        report.output_shape = current.shape  # type: ignore[assignment]
        return report

    def get_intermediate(self, step_index: int) -> pd.DataFrame | None:
        """Get the intermediate DataFrame after a specific step.

        Args:
            step_index: 0 = input, 1 = after step 1, etc.

        Returns:
            DataFrame or None if index is out of range.
        """
        if 0 <= step_index < len(self._intermediates):
            return self._intermediates[step_index]
        return None

    def get_lineage(self) -> dict[str, LineageNode]:
        """Return the feature lineage DAG."""
        return dict(self._lineage)

    def get_lineage_for(self, feature: str) -> list[str]:
        """Trace the lineage of a single feature back to source columns.

        Args:
            feature: Feature name to trace.

        Returns:
            List of source column names.
        """
        node = self._lineage.get(feature)
        if node is None:
            return []
        if not node.source_columns:
            return [feature]
        sources: list[str] = []
        for src in node.source_columns:
            sources.extend(self.get_lineage_for(src))
        return list(dict.fromkeys(sources))  # deduplicate, preserve order

    def _run_step(
        self,
        name: str,
        transformer: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series | None,
        depth: int,
    ) -> StepResult:
        cols_before = set(X.columns)

        t0 = time.perf_counter()
        errors: list[str] = []
        try:
            out = self._transform(transformer, X, y)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            out = X
        elapsed = (time.perf_counter() - t0) * 1000

        cols_after = set(out.columns)
        added = sorted(cols_after - cols_before)
        removed = sorted(cols_before - cols_after)

        # Update lineage for new columns
        for col in added:
            self._lineage[col] = LineageNode(
                name=col,
                source_columns=sorted(cols_before),
                transform_step=name,
                depth=depth,
            )

        stats: dict[str, dict[str, Any]] = {}
        if self.capture_stats and not errors:
            stats = self._compute_stats(out, added)

        return StepResult(
            step_name=name,
            n_features_in=len(cols_before),
            n_features_out=len(cols_after),
            features_added=added,
            features_removed=removed,
            elapsed_ms=elapsed,
            stats=stats,
            errors=errors,
        )

    def _transform(
        self,
        transformer: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series | None,
    ) -> pd.DataFrame:
        if hasattr(transformer, "transform"):
            if not self._is_fitted(transformer):
                transformer.fit(X, y)  # type: ignore[arg-type]
            result = transformer.transform(X)  # type: ignore[arg-type]
        else:
            result = X

        if isinstance(result, np.ndarray):
            result = pd.DataFrame(result)
        return result

    @staticmethod
    def _is_fitted(transformer: BaseEstimator) -> bool:
        fitted_attrs = [
            a for a in dir(transformer)
            if a.endswith("_") and not a.startswith("__")
        ]
        return len(fitted_attrs) > 0

    @staticmethod
    def _compute_stats(
        df: pd.DataFrame, columns: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        cols = columns or list(df.columns)
        stats: dict[str, dict[str, Any]] = {}
        for col in cols:
            if col not in df.columns:
                continue
            series = df[col]
            col_stats: dict[str, Any] = {
                "dtype": str(series.dtype),
                "null_count": int(series.isna().sum()),
            }
            if pd.api.types.is_numeric_dtype(series):
                col_stats["mean"] = float(series.mean())
                col_stats["std"] = float(series.std())
                col_stats["min"] = float(series.min())
                col_stats["max"] = float(series.max())
            else:
                col_stats["n_unique"] = int(series.nunique())
            stats[col] = col_stats
        return stats
