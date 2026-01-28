"""Compute backend abstraction for CPU and GPU acceleration.

This module provides a transparent dispatch layer that routes
computation to CPU (NumPy/Pandas) or GPU (cuDF/cuPy) backends
depending on hardware availability.
"""

from __future__ import annotations

from forge.backends.base import ComputeBackend, ComputeDevice
from forge.backends.cpu import CPUBackend
from forge.backends.dispatch import get_backend, set_backend
from forge.backends.duckdb_backend import DuckDBBackend
from forge.backends.gpu import GPUBackend
from forge.backends.hybrid import ExecutionStats, HybridConfig, HybridDispatcher
from forge.backends.polars_backend import PolarsBackend

__all__ = [
    "CPUBackend",
    "ComputeBackend",
    "ComputeDevice",
    "DuckDBBackend",
    "ExecutionStats",
    "GPUBackend",
    "HybridConfig",
    "HybridDispatcher",
    "PolarsBackend",
    "get_backend",
    "set_backend",
]
