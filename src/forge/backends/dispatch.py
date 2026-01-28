"""Backend dispatch for transparent CPU/GPU selection."""

from __future__ import annotations

import logging

from forge.backends.base import ComputeBackend, ComputeDevice
from forge.backends.cpu import CPUBackend
from forge.backends.duckdb_backend import DuckDBBackend
from forge.backends.gpu import GPUBackend
from forge.backends.polars_backend import PolarsBackend
from forge.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

_CURRENT_BACKEND: ComputeBackend | None = None


def get_backend(device: str | ComputeDevice = ComputeDevice.AUTO) -> ComputeBackend:
    """Get the appropriate compute backend.

    Args:
        device: Preferred device. "auto" detects best available.
            Options: "cpu", "gpu", "polars", "duckdb", "auto".

    Returns:
        A ComputeBackend instance.

    Example:
        >>> backend = get_backend("polars")
        >>> result = backend.multiply(series_a, series_b)
    """
    global _CURRENT_BACKEND

    if isinstance(device, str):
        device = ComputeDevice(device.lower())

    if device == ComputeDevice.POLARS:
        polars_be = PolarsBackend()
        if polars_be.is_available():
            _CURRENT_BACKEND = polars_be
            logger.info("Using Polars backend")
            return polars_be
        raise ConfigurationError(
            "Polars backend requested but polars is not installed. "
            "Install with: pip install polars"
        )

    if device == ComputeDevice.DUCKDB:
        duckdb_be = DuckDBBackend()
        if duckdb_be.is_available():
            _CURRENT_BACKEND = duckdb_be
            logger.info("Using DuckDB backend")
            return duckdb_be
        raise ConfigurationError(
            "DuckDB backend requested but duckdb is not installed. "
            "Install with: pip install duckdb"
        )

    if device == ComputeDevice.GPU:
        gpu = GPUBackend()
        if gpu.is_available():
            _CURRENT_BACKEND = gpu
            logger.info("Using GPU backend (RAPIDS cuDF)")
            return gpu
        raise ConfigurationError(
            "GPU backend requested but RAPIDS (cudf) is not installed. "
            "Install with: pip install cudf-cu12"
        )

    if device == ComputeDevice.AUTO:
        gpu = GPUBackend()
        if gpu.is_available():
            _CURRENT_BACKEND = gpu
            logger.info("Auto-detected GPU backend (RAPIDS cuDF)")
            return gpu
        _CURRENT_BACKEND = CPUBackend()
        logger.debug("Using CPU backend (NumPy/Pandas)")
        return _CURRENT_BACKEND

    # CPU
    _CURRENT_BACKEND = CPUBackend()
    return _CURRENT_BACKEND


def set_backend(backend: ComputeBackend) -> None:
    """Set a custom compute backend globally.

    Args:
        backend: ComputeBackend instance to use.
    """
    global _CURRENT_BACKEND
    _CURRENT_BACKEND = backend


def current_backend() -> ComputeBackend:
    """Return the currently active backend, initializing if needed."""
    global _CURRENT_BACKEND
    if _CURRENT_BACKEND is None:
        _CURRENT_BACKEND = get_backend(ComputeDevice.AUTO)
    return _CURRENT_BACKEND
