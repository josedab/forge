"""Ray distributed backend."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from forge.distributed.base import DistributedBackend
from forge.exceptions import MissingDependencyError

logger = logging.getLogger(__name__)


class RayBackend(DistributedBackend):
    """Distributed backend using Ray.

    Uses Ray tasks for parallel execution of feature transformations.

    Args:
        num_cpus: Number of CPUs for local Ray cluster.
        address: Address of existing Ray cluster.

    Example:
        >>> backend = RayBackend(num_cpus=4)
        >>> backend.connect()
        >>> result = backend.map_partitions(df, my_transform)
        >>> backend.disconnect()
    """

    def __init__(
        self,
        num_cpus: int = 4,
        address: str | None = None,
    ) -> None:
        self.num_cpus = num_cpus
        self.address = address
        self._is_connected = False

    @property
    def name(self) -> str:
        return "ray"

    def is_available(self) -> bool:
        try:
            import ray  # noqa: F401
            return True
        except ImportError:
            return False

    def connect(self, **kwargs: Any) -> None:
        """Initialize Ray."""
        try:
            import ray
        except ImportError:
            raise MissingDependencyError("ray", "Ray distributed backend")

        if not ray.is_initialized():
            if self.address:
                ray.init(address=self.address, **kwargs)
            else:
                ray.init(num_cpus=self.num_cpus, **kwargs)
        self._is_connected = True
        logger.info("Connected to Ray cluster")

    def disconnect(self) -> None:
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
        except ImportError:
            pass
        self._is_connected = False

    def map_partitions(
        self,
        df: pd.DataFrame,
        func: Any,
        n_partitions: int | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        import ray

        npart = n_partitions or max(1, self.num_cpus)
        chunks = np.array_split(df, npart)

        @ray.remote  # type: ignore[misc]
        def _apply(chunk: pd.DataFrame) -> pd.DataFrame:
            return func(chunk, **kwargs)  # type: ignore[no-any-return]

        futures = [_apply.remote(chunk) for chunk in chunks if not chunk.empty]  # type: ignore[union-attr]
        results = ray.get(futures)
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()  # type: ignore[no-any-return]

    def groupby_apply(
        self,
        df: pd.DataFrame,
        group_col: str,
        func: Any,
        **kwargs: Any,
    ) -> pd.DataFrame:
        import ray

        groups = [group for _, group in df.groupby(group_col)]

        @ray.remote  # type: ignore[misc]
        def _apply_group(group: pd.DataFrame) -> pd.DataFrame:
            return func(group, **kwargs)  # type: ignore[no-any-return]

        futures = [_apply_group.remote(g) for g in groups]
        results = ray.get(futures)
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()  # type: ignore[no-any-return]

    def parallel_transform(
        self,
        df: pd.DataFrame,
        transformers: list[Any],
        **kwargs: Any,
    ) -> pd.DataFrame:
        import ray

        df_ref = ray.put(df)

        @ray.remote  # type: ignore[misc]
        def _transform(t: Any, data_ref: Any) -> pd.DataFrame:
            data = data_ref
            return t.transform(data)  # type: ignore[no-any-return]

        futures = [_transform.remote(t, df_ref) for t in transformers]
        results = ray.get(futures)
        return pd.concat(results, axis=1)  # type: ignore[no-any-return]
