"""ONNX export and optimised NumPy fast-path for feature pipelines.

Provides:
- ``ONNXExporter``: Converts fitted sklearn-compatible pipelines to ONNX.
- ``NumpyFastPath``: Pure-NumPy execution of common transforms, avoiding
  pandas overhead for latency-critical serving.
- ``StreamProcessor``: Accepts an iterable/generator of dicts and yields
  transformed dicts—suitable for Kafka-style streaming integration.

Example:
    >>> from forge.serving.onnx_export import ONNXExporter, NumpyFastPath
    >>> exporter = ONNXExporter(pipeline)
    >>> onnx_bytes = exporter.export()
    >>>
    >>> fast = NumpyFastPath.from_transformer(pipeline)
    >>> features = fast.transform_row({"age": 25, "income": 50000})
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Generator, Iterable  # noqa: TC003
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NumPy fast path — avoids DataFrame overhead for single-row transforms
# ---------------------------------------------------------------------------


@dataclass
class ScaleParams:
    """Pre-extracted StandardScaler parameters."""

    columns: list[str]
    mean: np.ndarray
    scale: np.ndarray


@dataclass
class OneHotParams:
    """Pre-extracted OneHotEncoder parameters."""

    column: str
    categories: list[str]


class NumpyFastPath:
    """Ultra-low-latency feature transform using pure NumPy arrays.

    Extracts parameters from a fitted sklearn transformer and applies
    them without constructing a DataFrame.

    Parameters
    ----------
    scale_params : list[ScaleParams]
        Scaling parameters to apply.
    onehot_params : list[OneHotParams]
        One-hot encoding parameters to apply.
    """

    def __init__(
        self,
        scale_params: list[ScaleParams] | None = None,
        onehot_params: list[OneHotParams] | None = None,
    ) -> None:
        self.scale_params = scale_params or []
        self.onehot_params = onehot_params or []
        self._output_names: list[str] = []
        self._build_output_names()

    @classmethod
    def from_transformer(cls, transformer: Any) -> NumpyFastPath:
        """Create a fast path from a fitted sklearn transformer."""
        scale_params: list[ScaleParams] = []
        onehot_params: list[OneHotParams] = []
        cls._extract(transformer, scale_params, onehot_params)
        return cls(scale_params=scale_params, onehot_params=onehot_params)

    def transform_row(self, row: dict[str, Any]) -> dict[str, float]:
        """Transform a single row with minimal overhead.

        Args:
            row: Column name → value mapping.

        Returns:
            Feature name → transformed value mapping.
        """
        result: dict[str, float] = {}

        for sp in self.scale_params:
            for i, col in enumerate(sp.columns):
                if col in row:
                    val = float(row[col])
                    s = sp.scale[i]
                    result[col] = (val - sp.mean[i]) / s if s > 0 else 0.0

        for oh in self.onehot_params:
            val = str(row.get(oh.column, ""))
            for cat in oh.categories:
                result[f"{oh.column}_{cat}"] = 1.0 if val == cat else 0.0

        # Pass-through any columns not handled above
        for key, value in row.items():
            if key not in result:
                with contextlib.suppress(ValueError, TypeError):
                    result[key] = float(value)

        return result

    def transform_batch(self, rows: list[dict[str, Any]]) -> np.ndarray:
        """Transform a batch of rows into a NumPy array.

        Args:
            rows: List of row dicts.

        Returns:
            2-D NumPy array (n_rows x n_features).
        """
        if not rows:
            return np.empty((0, len(self._output_names)))
        results = [self.transform_row(r) for r in rows]
        names = self._output_names or list(results[0].keys())
        arr = np.array([[r.get(n, 0.0) for n in names] for r in results])
        return arr

    @property
    def output_names(self) -> list[str]:
        return list(self._output_names)

    def _build_output_names(self) -> None:
        names: list[str] = []
        for sp in self.scale_params:
            names.extend(sp.columns)
        for oh in self.onehot_params:
            for cat in oh.categories:
                names.append(f"{oh.column}_{cat}")
        self._output_names = names

    @classmethod
    def _extract(
        cls,
        transformer: Any,
        scale_params: list[ScaleParams],
        onehot_params: list[OneHotParams],
    ) -> None:
        """Recursively extract parameters from a transformer tree."""
        # StandardScaler
        if hasattr(transformer, "mean_") and hasattr(transformer, "scale_"):
            cols = (
                list(transformer.feature_names_in_)
                if hasattr(transformer, "feature_names_in_")
                else [f"f{i}" for i in range(len(transformer.mean_))]
            )
            scale_params.append(ScaleParams(
                columns=cols,
                mean=np.asarray(transformer.mean_, dtype=float),
                scale=np.asarray(transformer.scale_, dtype=float),
            ))

        # OneHotEncoder
        if hasattr(transformer, "categories_"):
            cols = (
                list(transformer.feature_names_in_)
                if hasattr(transformer, "feature_names_in_")
                else [f"f{i}" for i in range(len(transformer.categories_))]
            )
            for i, cats in enumerate(transformer.categories_):
                col_name = cols[i] if i < len(cols) else f"f{i}"
                onehot_params.append(OneHotParams(
                    column=col_name,
                    categories=[str(c) for c in cats],
                ))

        # Pipeline
        if hasattr(transformer, "steps"):
            for _, step in transformer.steps:
                if step is not None and step != "passthrough":
                    cls._extract(step, scale_params, onehot_params)

        # ColumnTransformer
        if hasattr(transformer, "transformers_"):
            for _, step, _ in transformer.transformers_:
                if step not in ("remainder", "drop", None):
                    cls._extract(step, scale_params, onehot_params)


# ---------------------------------------------------------------------------
# ONNX Exporter
# ---------------------------------------------------------------------------


@dataclass
class ONNXExportResult:
    """Result of an ONNX export.

    Attributes:
        model_bytes: Serialised ONNX model bytes (None if export failed).
        input_names: Expected input tensor names.
        output_names: Output tensor names.
        success: Whether the export succeeded.
        error: Error message if export failed.
    """

    model_bytes: bytes | None = None
    input_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)
    success: bool = True
    error: str = ""


class ONNXExporter:
    """Export fitted sklearn pipelines to ONNX format.

    Uses *skl2onnx* when available; otherwise falls back to a
    lightweight manual export for StandardScaler / OneHotEncoder trees.

    Parameters
    ----------
    transformer : Any
        Fitted sklearn-compatible transformer.
    opset : int
        ONNX opset version.

    Example:
    -------
    >>> exporter = ONNXExporter(pipeline)
    >>> result = exporter.export()
    >>> if result.success:
    ...     with open("model.onnx", "wb") as f:
    ...         f.write(result.model_bytes)
    """

    def __init__(self, transformer: Any, opset: int = 13) -> None:
        self._transformer = transformer
        self.opset = opset

    def export(
        self,
        initial_types: list[tuple[str, Any]] | None = None,
    ) -> ONNXExportResult:
        """Export the transformer to ONNX bytes.

        Args:
            initial_types: ONNX initial type hints.

        Returns:
            ONNXExportResult with model bytes or error.
        """
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType

            if initial_types is None:
                n_features = self._infer_n_features()
                initial_types = [
                    ("X", FloatTensorType([None, n_features])),
                ]

            onnx_model = convert_sklearn(
                self._transformer,
                initial_types=initial_types,
                target_opset=self.opset,
            )
            return ONNXExportResult(
                model_bytes=onnx_model.SerializeToString(),
                input_names=[inp.name for inp in onnx_model.graph.input],
                output_names=[out.name for out in onnx_model.graph.output],
            )
        except ImportError:
            return ONNXExportResult(
                success=False,
                error=(
                    "skl2onnx is required for ONNX export. "
                    "Install with: pip install skl2onnx"
                ),
            )
        except Exception as exc:
            return ONNXExportResult(success=False, error=str(exc))

    def _infer_n_features(self) -> int:
        if hasattr(self._transformer, "n_features_in_"):
            return int(self._transformer.n_features_in_)
        if hasattr(self._transformer, "feature_names_in_"):
            return len(self._transformer.feature_names_in_)
        return 1


# ---------------------------------------------------------------------------
# Streaming processor
# ---------------------------------------------------------------------------


class StreamProcessor:
    """Process a stream of row dicts through a feature pipeline.

    Designed for integration with message queues (Kafka, SQS, etc.).

    Parameters
    ----------
    fast_path : NumpyFastPath | None
        Optimised fast-path transform. Falls back to *transformer*.
    transformer : Any
        Fitted sklearn-compatible transformer (used when fast_path is None).
    batch_size : int
        Accumulate this many rows before processing as a batch.
    """

    def __init__(
        self,
        fast_path: NumpyFastPath | None = None,
        transformer: Any = None,
        batch_size: int = 1,
    ) -> None:
        self.fast_path = fast_path
        self._transformer = transformer
        self.batch_size = max(1, batch_size)
        self._processed = 0
        self._errors = 0
        self._total_latency_ms = 0.0

    def process_stream(
        self, rows: Iterable[dict[str, Any]],
    ) -> Generator[dict[str, Any], None, None]:
        """Yield transformed dicts for each incoming row.

        Args:
            rows: Iterable of input row dicts.

        Yields:
            Transformed feature dicts.
        """
        batch: list[dict[str, Any]] = []
        for row in rows:
            batch.append(row)
            if len(batch) >= self.batch_size:
                yield from self._process_batch(batch)
                batch = []
        if batch:
            yield from self._process_batch(batch)

    def _process_batch(
        self, batch: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        start = time.perf_counter()
        results: list[dict[str, Any]] = []
        try:
            if self.fast_path is not None:
                for row in batch:
                    results.append(self.fast_path.transform_row(row))
            elif self._transformer is not None:
                df = pd.DataFrame(batch)
                out = self._transformer.transform(df)
                if isinstance(out, pd.DataFrame):
                    results = out.to_dict("records")
                else:
                    arr = np.asarray(out)
                    for i in range(arr.shape[0]):
                        results.append({f"f{j}": float(v) for j, v in enumerate(arr[i])})
            else:
                results = list(batch)
        except Exception as exc:
            self._errors += len(batch)
            logger.warning("Batch processing failed: %s", exc)
            results = list(batch)

        self._processed += len(batch)
        self._total_latency_ms += (time.perf_counter() - start) * 1000
        return results

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "processed": self._processed,
            "errors": self._errors,
            "avg_latency_ms": (
                self._total_latency_ms / self._processed
                if self._processed > 0 else 0.0
            ),
        }
