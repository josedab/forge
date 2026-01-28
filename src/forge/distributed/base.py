"""Abstract interface for distributed computation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class DistributedBackend(ABC):
    """Abstract interface for distributed feature engineering backends.

    Defines operations that can be executed on distributed
    DataFrames (Dask, Ray, or Spark). All methods accept and
    return pandas DataFrames, handling the distributed conversion
    internally.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend's dependencies are installed."""

    @abstractmethod
    def connect(self, **kwargs: Any) -> None:
        """Initialize the distributed runtime."""

    @abstractmethod
    def disconnect(self) -> None:
        """Shut down the distributed runtime."""

    @abstractmethod
    def map_partitions(
        self,
        df: pd.DataFrame,
        func: Any,
        n_partitions: int | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Apply a function to each partition of a DataFrame.

        The data is partitioned, the function is applied in parallel,
        and results are collected back to a single pandas DataFrame.

        Args:
            df: Input pandas DataFrame.
            func: Callable that takes a DataFrame partition and returns a DataFrame.
            n_partitions: Number of partitions. None for auto.
            **kwargs: Additional arguments passed to func.

        Returns:
            Combined result DataFrame.
        """

    @abstractmethod
    def groupby_apply(
        self,
        df: pd.DataFrame,
        group_col: str,
        func: Any,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Distributed group-by apply.

        Args:
            df: Input DataFrame.
            group_col: Column to group by.
            func: Function applied to each group.
            **kwargs: Additional arguments.

        Returns:
            Combined result DataFrame.
        """

    @abstractmethod
    def parallel_transform(
        self,
        df: pd.DataFrame,
        transformers: list[Any],
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Run multiple transformers in parallel and concatenate results.

        Args:
            df: Input DataFrame.
            transformers: sklearn-compatible transformers.
            **kwargs: Additional arguments.

        Returns:
            DataFrame with all transformer outputs concatenated.
        """
