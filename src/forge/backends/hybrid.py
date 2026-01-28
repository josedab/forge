"""Hybrid execution dispatcher — routes operations to GPU or CPU based on data size.

Transparently chooses between GPU and CPU backends by estimating
whether GPU acceleration will be beneficial for the given data size
and operation type. Includes automatic fallback on GPU errors.

Example:
    >>> from forge.backends.hybrid import HybridDispatcher
    >>> dispatcher = HybridDispatcher(gpu_threshold_rows=10000)
    >>> result = dispatcher.transform(X, transformer)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from forge.backends.cpu import CPUBackend

logger = logging.getLogger(__name__)


@dataclass
class ExecutionStats:
    """Statistics from a hybrid execution.

    Attributes:
        device_used: Which device was used ("cpu" or "gpu").
        latency_ms: Execution time in milliseconds.
        rows_processed: Number of rows processed.
        fallback_occurred: Whether GPU failed and fell back to CPU.
        reason: Why this device was chosen.
    """

    device_used: str = "cpu"
    latency_ms: float = 0.0
    rows_processed: int = 0
    fallback_occurred: bool = False
    reason: str = ""


@dataclass
class HybridConfig:
    """Configuration for hybrid GPU/CPU dispatch.

    Attributes:
        gpu_threshold_rows: Minimum rows to use GPU (below → CPU).
        gpu_threshold_cols: Minimum columns to use GPU.
        gpu_threshold_elements: Minimum total elements for GPU.
        prefer_gpu: If True, always try GPU first above thresholds.
        max_gpu_memory_mb: Maximum GPU memory to use.
        fallback_to_cpu: If True, fall back to CPU on GPU errors.
    """

    gpu_threshold_rows: int = 10_000
    gpu_threshold_cols: int = 5
    gpu_threshold_elements: int = 50_000
    prefer_gpu: bool = True
    max_gpu_memory_mb: int = 2048
    fallback_to_cpu: bool = True


class HybridDispatcher:
    """Hybrid GPU/CPU execution dispatcher.

    Automatically routes operations to the most efficient backend
    based on data size, available hardware, and operation type.

    Args:
        config: Hybrid dispatch configuration.

    Example:
        >>> dispatcher = HybridDispatcher()
        >>> result, stats = dispatcher.execute(X, operation="transform", transformer=t)
        >>> print(f"Used {stats.device_used} in {stats.latency_ms:.1f}ms")
    """

    def __init__(self, config: HybridConfig | None = None) -> None:
        self.config = config or HybridConfig()
        self._cpu_backend = CPUBackend()
        self._gpu_available = self._check_gpu()
        self._execution_history: list[ExecutionStats] = []

    def _check_gpu(self) -> bool:
        """Check if GPU backend is available."""
        try:
            from forge.backends.gpu import GPUBackend
            return GPUBackend().is_available()
        except Exception:
            return False

    def should_use_gpu(self, X: pd.DataFrame) -> tuple[bool, str]:
        """Determine whether to use GPU for this data.

        Args:
            X: Input DataFrame.

        Returns:
            Tuple of (use_gpu, reason).
        """
        if not self._gpu_available:
            return False, "GPU not available"

        if not self.config.prefer_gpu:
            return False, "GPU not preferred"

        n_rows = len(X)
        n_cols = len(X.columns)
        n_elements = n_rows * n_cols

        if n_rows < self.config.gpu_threshold_rows:
            return False, f"Below row threshold ({n_rows} < {self.config.gpu_threshold_rows})"

        if n_elements < self.config.gpu_threshold_elements:
            return False, f"Below element threshold ({n_elements} < {self.config.gpu_threshold_elements})"

        # Estimate memory requirement
        memory_mb = (n_elements * 8) / (1024 * 1024)  # 8 bytes per float64
        if memory_mb > self.config.max_gpu_memory_mb:
            return False, f"Exceeds GPU memory limit ({memory_mb:.0f}MB > {self.config.max_gpu_memory_mb}MB)"

        return True, "Data size suitable for GPU acceleration"

    def transform(
        self, X: pd.DataFrame, transformer: Any, y: Any = None
    ) -> tuple[pd.DataFrame, ExecutionStats]:
        """Execute a transform operation with hybrid dispatch.

        Args:
            X: Input data.
            transformer: Fitted sklearn-compatible transformer.
            y: Optional target (for fit_transform).

        Returns:
            Tuple of (result DataFrame, execution statistics).
        """
        use_gpu, reason = self.should_use_gpu(X)
        stats = ExecutionStats(rows_processed=len(X), reason=reason)

        if use_gpu:
            result = self._try_gpu_transform(X, transformer, y, stats)
            if result is not None:
                self._execution_history.append(stats)
                return result, stats
            # GPU failed, fall back
            if not self.config.fallback_to_cpu:
                raise RuntimeError("GPU execution failed and fallback disabled")
            stats.fallback_occurred = True
            logger.warning("GPU failed, falling back to CPU")

        # CPU execution
        start = time.perf_counter()
        result = self._cpu_transform(X, transformer, y)
        stats.latency_ms = (time.perf_counter() - start) * 1000
        stats.device_used = "cpu"
        self._execution_history.append(stats)
        return result, stats

    def _try_gpu_transform(
        self, X: pd.DataFrame, transformer: Any, y: Any,
        stats: ExecutionStats,
    ) -> pd.DataFrame | None:
        """Attempt GPU transform, returning None on failure."""
        try:
            from forge.backends.gpu import GPUBackend
            gpu = GPUBackend()

            start = time.perf_counter()
            # Convert to GPU frame, transform, convert back
            X_gpu = gpu._to_cudf_frame(X)
            if hasattr(transformer, "transform"):
                result_gpu = transformer.transform(X_gpu)
            else:
                result_gpu = X_gpu

            if hasattr(result_gpu, "to_pandas"):
                result = result_gpu.to_pandas()
            elif isinstance(result_gpu, np.ndarray):
                result = pd.DataFrame(result_gpu, columns=X.columns, index=X.index)
            else:
                result = pd.DataFrame(result_gpu)

            stats.latency_ms = (time.perf_counter() - start) * 1000
            stats.device_used = "gpu"
            return result
        except Exception as exc:
            logger.debug("GPU transform failed: %s", exc)
            return None

    def _cpu_transform(
        self, X: pd.DataFrame, transformer: Any, y: Any = None
    ) -> pd.DataFrame:
        """Execute transform on CPU."""
        result = transformer.transform(X)
        if isinstance(result, pd.DataFrame):
            return result
        if isinstance(result, np.ndarray):
            cols = list(X.columns)
            if hasattr(transformer, "get_feature_names_out"):
                try:
                    cols = list(transformer.get_feature_names_out())
                except Exception:
                    pass
            return pd.DataFrame(result, columns=cols[:result.shape[1]], index=X.index)
        return pd.DataFrame(result)

    @property
    def gpu_available(self) -> bool:
        """Whether GPU backend is available."""
        return self._gpu_available

    def get_execution_summary(self) -> dict[str, Any]:
        """Get summary of all executions."""
        if not self._execution_history:
            return {"total_executions": 0}

        gpu_count = sum(1 for s in self._execution_history if s.device_used == "gpu")
        cpu_count = sum(1 for s in self._execution_history if s.device_used == "cpu")
        fallback_count = sum(1 for s in self._execution_history if s.fallback_occurred)
        avg_latency = np.mean([s.latency_ms for s in self._execution_history])

        return {
            "total_executions": len(self._execution_history),
            "gpu_executions": gpu_count,
            "cpu_executions": cpu_count,
            "fallback_count": fallback_count,
            "avg_latency_ms": round(float(avg_latency), 3),
        }
