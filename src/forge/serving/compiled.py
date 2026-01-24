"""Compiled feature pipeline for low-latency serving.

Pre-compiles a fitted sklearn pipeline into an optimized execution
graph with lookup tables for categorical operations and pre-computed
constants, achieving sub-millisecond single-row transforms.

Example:
    >>> from forge.serving.compiled import CompiledPipeline
    >>> compiled = CompiledPipeline.from_transformer(fitted_pipeline)
    >>> result = compiled.transform_single({"age": 25, "income": 50000})
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TransformStep:
    """A single pre-compiled transformation step.

    Attributes:
        name: Step identifier.
        operation: Operation type (e.g., 'scale', 'onehot_lookup', 'interact').
        input_columns: Input column names.
        output_columns: Output column names.
        params: Pre-computed parameters for this step.
    """

    name: str
    operation: str
    input_columns: list[str]
    output_columns: list[str]
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompiledGraph:
    """Pre-compiled execution graph for transforms.

    Attributes:
        steps: Ordered list of transform steps.
        input_columns: Required input columns.
        output_columns: All output columns.
        metadata: Graph metadata (compilation time, step count, etc.).
    """

    steps: list[TransformStep] = field(default_factory=list)
    input_columns: list[str] = field(default_factory=list)
    output_columns: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ServingMetrics:
    """Metrics collected during serving.

    Attributes:
        total_requests: Total requests served.
        total_errors: Total failed requests.
        total_latency_ms: Cumulative latency in milliseconds.
        p50_latency_ms: 50th percentile latency.
        p99_latency_ms: 99th percentile latency.
    """

    total_requests: int = 0
    total_errors: int = 0
    total_latency_ms: float = 0.0
    _latencies: list[float] = field(default_factory=list)

    def record(self, latency_ms: float, error: bool = False) -> None:
        """Record a request metric."""
        self.total_requests += 1
        self.total_latency_ms += latency_ms
        self._latencies.append(latency_ms)
        if error:
            self.total_errors += 1
        # Keep only last 1000 latencies for percentile calculations
        if len(self._latencies) > 1000:
            self._latencies = self._latencies[-1000:]

    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    @property
    def p50_latency_ms(self) -> float:
        """50th percentile latency."""
        if not self._latencies:
            return 0.0
        return float(np.percentile(self._latencies, 50))

    @property
    def p99_latency_ms(self) -> float:
        """99th percentile latency."""
        if not self._latencies:
            return 0.0
        return float(np.percentile(self._latencies, 99))

    @property
    def error_rate(self) -> float:
        """Error rate as a fraction."""
        if self.total_requests == 0:
            return 0.0
        return self.total_errors / self.total_requests

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics to dictionary."""
        return {
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "avg_latency_ms": round(self.avg_latency_ms, 3),
            "p50_latency_ms": round(self.p50_latency_ms, 3),
            "p99_latency_ms": round(self.p99_latency_ms, 3),
            "error_rate": round(self.error_rate, 5),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.total_requests = 0
        self.total_errors = 0
        self.total_latency_ms = 0.0
        self._latencies.clear()


class CompiledPipeline:
    """Pre-compiled feature transformation pipeline for low-latency serving.

    Extracts parameters from a fitted sklearn transformer and compiles
    them into a fast execution graph. Avoids DataFrame overhead for
    single-row transforms.

    Parameters:
        transformer: A fitted sklearn-compatible transformer.

    Example:
        >>> compiled = CompiledPipeline.from_transformer(fitted_pipeline)
        >>> features = compiled.transform_single({"age": 25, "city": "NYC"})
        >>> batch = compiled.transform_batch(df)
    """

    def __init__(self, transformer: Any = None) -> None:
        self._transformer = transformer
        self._graph: CompiledGraph | None = None
        self._metrics = ServingMetrics()
        self._scale_params: dict[str, tuple[float, float]] = {}
        self._onehot_maps: dict[str, dict[str, int]] = {}
        self._feature_names: list[str] = []
        self._is_compiled = False

    @classmethod
    def from_transformer(cls, transformer: Any) -> CompiledPipeline:
        """Create a compiled pipeline from a fitted transformer.

        Args:
            transformer: A fitted sklearn-compatible transformer.

        Returns:
            Compiled pipeline ready for low-latency serving.
        """
        instance = cls(transformer)
        instance.compile()
        return instance

    def compile(self) -> CompiledGraph:
        """Compile the transformer into an optimized execution graph.

        Extracts parameters from the fitted transformer and pre-computes
        lookup tables and constants.

        Returns:
            The compiled execution graph.
        """
        start_time = time.time()
        graph = CompiledGraph()
        steps: list[TransformStep] = []

        if self._transformer is not None:
            self._extract_transform_params(self._transformer, steps)

            # Extract feature names
            if hasattr(self._transformer, "get_feature_names_out"):
                try:
                    self._feature_names = list(self._transformer.get_feature_names_out())
                except Exception:
                    self._feature_names = []

        graph.steps = steps
        graph.output_columns = self._feature_names
        graph.metadata = {
            "compilation_time_ms": (time.time() - start_time) * 1000,
            "step_count": len(steps),
            "transformer_type": type(self._transformer).__name__ if self._transformer else "none",
        }

        self._graph = graph
        self._is_compiled = True
        return graph

    def transform_single(self, data: dict[str, Any]) -> dict[str, Any]:
        """Transform a single row of data with minimal overhead.

        Uses pre-compiled parameters to avoid DataFrame creation
        when possible.

        Args:
            data: Dictionary of column name -> value.

        Returns:
            Dictionary of feature name -> value.
        """
        start_time = time.time()
        try:
            result: dict[str, Any] = {}

            # Apply scale params if available
            for col, (mean, std) in self._scale_params.items():
                if col in data:
                    val = data[col]
                    if std > 0:
                        result[col] = (val - mean) / std
                    else:
                        result[col] = 0.0

            # Apply onehot lookups if available
            for col, mapping in self._onehot_maps.items():
                if col in data:
                    val = str(data[col])
                    for cat, idx in mapping.items():
                        result[f"{col}_{cat}"] = 1.0 if val == cat else 0.0

            # Fallback to full transform if no compiled params
            if not result and self._transformer is not None:
                df = pd.DataFrame([data])
                transformed = self._transformer.transform(df)
                if isinstance(transformed, pd.DataFrame):
                    result = transformed.iloc[0].to_dict()
                else:
                    arr = np.asarray(transformed)
                    if len(self._feature_names) == arr.shape[1]:
                        result = dict(zip(self._feature_names, arr[0]))
                    else:
                        result = {f"f{i}": float(v) for i, v in enumerate(arr[0])}

            latency_ms = (time.time() - start_time) * 1000
            self._metrics.record(latency_ms)
            return result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._metrics.record(latency_ms, error=True)
            logger.warning("transform_single failed: %s", e)
            return {}

    def transform_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform a batch of data using the compiled pipeline.

        For batch operations, delegates to the original transformer
        which is optimized for vectorized operations.

        Args:
            X: Input DataFrame.

        Returns:
            Transformed DataFrame.
        """
        start_time = time.time()
        try:
            if self._transformer is not None:
                result = self._transformer.transform(X)
                if isinstance(result, pd.DataFrame):
                    output = result
                else:
                    output = pd.DataFrame(
                        result,
                        columns=self._feature_names or None,
                        index=X.index,
                    )
            else:
                output = X.copy()

            latency_ms = (time.time() - start_time) * 1000
            self._metrics.record(latency_ms)
            return output

        except Exception:
            latency_ms = (time.time() - start_time) * 1000
            self._metrics.record(latency_ms, error=True)
            raise

    @property
    def graph(self) -> CompiledGraph | None:
        """The compiled execution graph."""
        return self._graph

    @property
    def metrics(self) -> ServingMetrics:
        """Serving metrics."""
        return self._metrics

    @property
    def is_compiled(self) -> bool:
        """Whether the pipeline has been compiled."""
        return self._is_compiled

    @property
    def feature_names(self) -> list[str]:
        """Output feature names."""
        return self._feature_names

    def health_check(self) -> dict[str, Any]:
        """Return health status of the compiled pipeline.

        Returns:
            Dictionary with health information.
        """
        return {
            "status": "healthy" if self._is_compiled else "not_compiled",
            "is_compiled": self._is_compiled,
            "step_count": len(self._graph.steps) if self._graph else 0,
            "feature_count": len(self._feature_names),
            "metrics": self._metrics.to_dict(),
        }

    def _extract_transform_params(
        self, transformer: Any, steps: list[TransformStep]
    ) -> None:
        """Extract pre-computable parameters from a transformer."""
        # Extract StandardScaler parameters
        if hasattr(transformer, "mean_") and hasattr(transformer, "scale_"):
            mean = transformer.mean_
            scale = transformer.scale_
            if hasattr(transformer, "feature_names_in_"):
                names = list(transformer.feature_names_in_)
            else:
                names = [f"f{i}" for i in range(len(mean))]

            for i, name in enumerate(names):
                self._scale_params[name] = (float(mean[i]), float(scale[i]))

            steps.append(TransformStep(
                name="standard_scaler",
                operation="scale",
                input_columns=names,
                output_columns=names,
                params={"mean": list(mean), "scale": list(scale)},
            ))

        # Extract OneHotEncoder categories
        if hasattr(transformer, "categories_"):
            categories = transformer.categories_
            if hasattr(transformer, "feature_names_in_"):
                input_names = list(transformer.feature_names_in_)
            else:
                input_names = [f"f{i}" for i in range(len(categories))]

            for i, cats in enumerate(categories):
                col_name = input_names[i] if i < len(input_names) else f"f{i}"
                mapping = {str(cat): idx for idx, cat in enumerate(cats)}
                self._onehot_maps[col_name] = mapping

                out_names = [f"{col_name}_{cat}" for cat in cats]
                steps.append(TransformStep(
                    name=f"onehot_{col_name}",
                    operation="onehot_lookup",
                    input_columns=[col_name],
                    output_columns=out_names,
                    params={"categories": [str(c) for c in cats]},
                ))

        # Extract Pipeline steps
        if hasattr(transformer, "steps"):
            for name, step in transformer.steps:
                if step is not None and step != "passthrough":
                    self._extract_transform_params(step, steps)

        # Extract ColumnTransformer transformers
        if hasattr(transformer, "transformers_"):
            for name, step, columns in transformer.transformers_:
                if step != "remainder" and step is not None and step != "drop":
                    self._extract_transform_params(step, steps)
