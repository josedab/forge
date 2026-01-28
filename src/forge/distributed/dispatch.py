"""Dispatch for selecting distributed backends."""

from __future__ import annotations

from typing import Any

from forge.distributed.base import DistributedBackend
from forge.distributed.dask_backend import DaskBackend
from forge.distributed.ray_backend import RayBackend
from forge.distributed.spark_backend import SparkBackend
from forge.exceptions import ConfigurationError

_BACKENDS: dict[str, type[DistributedBackend]] = {
    "dask": DaskBackend,
    "ray": RayBackend,
    "spark": SparkBackend,
}


def get_distributed_backend(
    name: str = "auto",
    **kwargs: Any,
) -> DistributedBackend:
    """Get a distributed backend by name.

    Args:
        name: Backend name ("dask", "ray", "spark", or "auto").
            "auto" tries Dask, then Ray, then Spark.
        **kwargs: Backend-specific configuration.

    Returns:
        An unconnected DistributedBackend instance.

    Example:
        >>> backend = get_distributed_backend("dask", n_workers=8)
        >>> backend.connect()
    """
    if name == "auto":
        for backend_name in ["dask", "ray", "spark"]:
            backend_class = _BACKENDS[backend_name]
            backend = backend_class(**kwargs)
            if backend.is_available():
                return backend
        raise ConfigurationError(
            "No distributed backend available. Install one of: "
            "dask[distributed], ray, pyspark"
        )

    name_lower = name.lower()
    if name_lower not in _BACKENDS:
        raise ConfigurationError(
            f"Unknown distributed backend: {name!r}. "
            f"Available: {list(_BACKENDS.keys())}"
        )

    backend_class = _BACKENDS[name_lower]
    return backend_class(**kwargs)
