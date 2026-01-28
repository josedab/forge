"""Distributed computation backends for large-scale feature engineering.

Provides transparent scaling from single-machine to distributed
execution using Dask, Ray, or Spark with an identical API.
"""

from __future__ import annotations

from forge.distributed.base import DistributedBackend
from forge.distributed.dask_backend import DaskBackend
from forge.distributed.dispatch import get_distributed_backend
from forge.distributed.ray_backend import RayBackend
from forge.distributed.spark_backend import SparkBackend

__all__ = [
    "DaskBackend",
    "DistributedBackend",
    "RayBackend",
    "SparkBackend",
    "get_distributed_backend",
]
