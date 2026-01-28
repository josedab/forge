"""PySpark distributed backend."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from forge.distributed.base import DistributedBackend
from forge.exceptions import MissingDependencyError

logger = logging.getLogger(__name__)


class SparkBackend(DistributedBackend):
    """Distributed backend using PySpark.

    Converts pandas DataFrames to Spark DataFrames for
    distributed processing, then collects results back to pandas.

    Args:
        app_name: Spark application name.
        master: Spark master URL. "local[*]" for local mode.
        config: Additional Spark configuration.

    Example:
        >>> backend = SparkBackend(master="local[4]")
        >>> backend.connect()
        >>> result = backend.map_partitions(df, my_func)
        >>> backend.disconnect()
    """

    def __init__(
        self,
        app_name: str = "forge",
        master: str = "local[*]",
        config: dict[str, str] | None = None,
    ) -> None:
        self.app_name = app_name
        self.master = master
        self.config = config or {}
        self._spark: Any = None

    @property
    def name(self) -> str:
        return "spark"

    def is_available(self) -> bool:
        try:
            import pyspark  # noqa: F401
            return True
        except ImportError:
            return False

    def connect(self, **kwargs: Any) -> None:
        """Create or get a SparkSession."""
        try:
            from pyspark.sql import SparkSession
        except ImportError:
            raise MissingDependencyError("pyspark", "Spark distributed backend")

        builder = SparkSession.builder.appName(self.app_name).master(self.master)
        for k, v in self.config.items():
            builder = builder.config(k, v)
        self._spark = builder.getOrCreate()
        logger.info("Connected to Spark: %s", self._spark.sparkContext.uiWebUrl)

    def disconnect(self) -> None:
        if self._spark is not None:
            self._spark.stop()
            self._spark = None

    def map_partitions(
        self,
        df: pd.DataFrame,
        func: Any,
        n_partitions: int | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if self._spark is None:
            raise RuntimeError("SparkBackend not connected. Call connect() first.")

        sdf = self._spark.createDataFrame(df)
        if n_partitions:
            sdf = sdf.repartition(n_partitions)

        schema = sdf.schema

        def apply_func(iterator: Any) -> Any:
            for pdf in iterator:
                yield func(pdf, **kwargs)

        result = sdf.mapInPandas(apply_func, schema=schema)
        return result.toPandas()  # type: ignore[no-any-return]

    def groupby_apply(
        self,
        df: pd.DataFrame,
        group_col: str,
        func: Any,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if self._spark is None:
            raise RuntimeError("SparkBackend not connected.")

        sdf = self._spark.createDataFrame(df)
        schema = sdf.schema

        def apply_func(pdf: pd.DataFrame) -> pd.DataFrame:
            return func(pdf, **kwargs)  # type: ignore[no-any-return]

        result = sdf.groupby(group_col).applyInPandas(apply_func, schema=schema)
        return result.toPandas()  # type: ignore[no-any-return]

    def parallel_transform(
        self,
        df: pd.DataFrame,
        transformers: list[Any],
        **kwargs: Any,
    ) -> pd.DataFrame:
        # Spark doesn't easily parallelize arbitrary Python transformers,
        # so we fall back to sequential execution with Spark broadcast
        frames = [t.transform(df) for t in transformers]
        return pd.concat(frames, axis=1)  # type: ignore[no-any-return]
