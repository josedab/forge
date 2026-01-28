"""Dask distributed backend."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from forge.distributed.base import DistributedBackend
from forge.exceptions import MissingDependencyError

logger = logging.getLogger(__name__)


class DaskBackend(DistributedBackend):
    """Distributed backend using Dask.

    Converts pandas DataFrames to Dask DataFrames for parallel
    execution, then collects results back to pandas.

    Args:
        n_workers: Number of workers for local cluster.
        scheduler_address: Address of existing Dask scheduler.

    Example:
        >>> backend = DaskBackend(n_workers=4)
        >>> backend.connect()
        >>> result = backend.map_partitions(df, my_transform)
        >>> backend.disconnect()
    """

    def __init__(
        self,
        n_workers: int = 4,
        scheduler_address: str | None = None,
    ) -> None:
        self.n_workers = n_workers
        self.scheduler_address = scheduler_address
        self._client: Any = None

    @property
    def name(self) -> str:
        return "dask"

    def is_available(self) -> bool:
        try:
            import dask  # type: ignore[import-untyped]
            import dask.dataframe  # type: ignore[import-untyped] # noqa: F401
            return True
        except ImportError:
            return False

    def connect(self, **kwargs: Any) -> None:
        """Start or connect to a Dask cluster."""
        try:
            from dask.distributed import Client
        except ImportError:
            raise MissingDependencyError("dask[distributed]", "Dask distributed backend")

        if self.scheduler_address:
            self._client = Client(self.scheduler_address)
        else:
            self._client = Client(
                n_workers=self.n_workers,
                threads_per_worker=1,
                **kwargs,
            )
        logger.info("Connected to Dask cluster: %s", self._client.dashboard_link)

    def disconnect(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def map_partitions(
        self,
        df: pd.DataFrame,
        func: Any,
        n_partitions: int | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        import dask.dataframe as dd
        npart = n_partitions or max(1, self.n_workers)
        ddf = dd.from_pandas(df, npartitions=npart)
        result = ddf.map_partitions(func, **kwargs)
        return result.compute()  # type: ignore[no-any-return]

    def groupby_apply(
        self,
        df: pd.DataFrame,
        group_col: str,
        func: Any,
        **kwargs: Any,
    ) -> pd.DataFrame:
        import dask.dataframe as dd
        npart = max(1, self.n_workers)
        ddf = dd.from_pandas(df, npartitions=npart)
        result = ddf.groupby(group_col).apply(func, **kwargs)
        return result.compute().reset_index(drop=True)  # type: ignore[no-any-return]

    def parallel_transform(
        self,
        df: pd.DataFrame,
        transformers: list[Any],
        **kwargs: Any,
    ) -> pd.DataFrame:
        if self._client is None:
            # Fallback to sequential
            frames = [t.transform(df) for t in transformers]
            return pd.concat(frames, axis=1)  # type: ignore[no-any-return]

        from dask import delayed
        futures = [delayed(t.transform)(df) for t in transformers]
        results = self._client.compute(futures)
        frames = self._client.gather(results)
        return pd.concat(frames, axis=1)  # type: ignore[no-any-return]
