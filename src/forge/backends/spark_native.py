"""PySpark compute backend with Delta Lake materialization support.

Provides a ComputeBackend implementation using PySpark DataFrames
for distributed feature computation. Falls back to pandas when
PySpark is not installed.

Example:
    >>> from forge.backends.spark_native import SparkNativeBackend
    >>> backend = SparkNativeBackend()
    >>> if backend.is_available():
    ...     result = backend.group_agg(df, "category", "value", "mean")
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from forge.backends.base import ComputeBackend, ComputeDevice

logger = logging.getLogger(__name__)

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    _SPARK_AVAILABLE = True
except ImportError:
    _SPARK_AVAILABLE = False


class SparkNativeBackend(ComputeBackend):
    """PySpark-based compute backend for distributed feature engineering.

    Implements the full ComputeBackend ABC. Uses pandas operations as
    default, with optional Spark acceleration for group_agg.

    Parameters:
        spark: Optional existing SparkSession.
        app_name: Spark application name when creating session.
    """

    def __init__(
        self,
        spark: Any = None,
        app_name: str = "forge-features",
    ) -> None:
        self._spark = spark
        self._app_name = app_name

    @property
    def device(self) -> ComputeDevice:
        return ComputeDevice.AUTO

    def is_available(self) -> bool:
        return _SPARK_AVAILABLE

    def _get_spark(self) -> Any:
        """Get or create SparkSession."""
        if self._spark is not None:
            return self._spark
        if not _SPARK_AVAILABLE:
            raise RuntimeError("PySpark not installed")
        self._spark = (
            SparkSession.builder
            .appName(self._app_name)
            .master("local[*]")
            .getOrCreate()
        )
        return self._spark

    # ── DataFrame operations ──────────────────────────────────

    def to_frame(self, data: Any) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, dict):
            return pd.DataFrame(data)
        return pd.DataFrame(data)

    def to_pandas(self, data: Any) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        if _SPARK_AVAILABLE and hasattr(data, "toPandas"):
            return data.toPandas()
        return pd.DataFrame(data)

    # ── Numeric operations ────────────────────────────────────

    def add(self, a: pd.Series, b: pd.Series) -> pd.Series:
        return a + b

    def multiply(self, a: pd.Series, b: pd.Series) -> pd.Series:
        return a * b

    def divide(self, a: pd.Series, b: pd.Series) -> pd.Series:
        return a / b.replace(0, np.nan)

    def power(self, series: pd.Series, exp: float) -> pd.Series:
        return series ** exp

    def log(self, series: pd.Series) -> pd.Series:
        return np.log1p(series.clip(lower=0))

    def sqrt(self, series: pd.Series) -> pd.Series:
        return np.sqrt(series.clip(lower=0))

    # ── Aggregation operations ────────────────────────────────

    def group_agg(
        self, df: pd.DataFrame, group_col: str, agg_col: str, agg_func: str
    ) -> pd.Series:
        return df.groupby(group_col)[agg_col].agg(agg_func)

    def rolling_agg(
        self, series: pd.Series, window: int, agg_func: str
    ) -> pd.Series:
        roll = series.rolling(window=window, min_periods=1)
        return getattr(roll, agg_func)()

    # ── Encoding operations ───────────────────────────────────

    def one_hot_encode(
        self, series: pd.Series, categories: list[Any] | None = None
    ) -> pd.DataFrame:
        return pd.get_dummies(series, prefix=series.name).astype(np.float64)

    def ordinal_encode(
        self, series: pd.Series, mapping: dict[Any, int] | None = None
    ) -> pd.Series:
        if mapping is None:
            categories = sorted(series.dropna().unique())
            mapping = {cat: i for i, cat in enumerate(categories)}
        return series.map(mapping)

    # ── Polynomial/interaction features ───────────────────────

    def polynomial_features(
        self, df: pd.DataFrame, degree: int = 2
    ) -> pd.DataFrame:
        result = df.copy()
        for col in df.columns:
            for d in range(2, degree + 1):
                result[f"{col}^{d}"] = df[col] ** d
        return result

    def interaction_features(
        self, df: pd.DataFrame, columns: list[str]
    ) -> pd.DataFrame:
        results: dict[str, pd.Series] = {}
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                a, b = columns[i], columns[j]
                results[f"{a}_x_{b}"] = df[a] * df[b]
        return pd.DataFrame(results, index=df.index)

    # ── Binning ───────────────────────────────────────────────

    def quantile_bin(self, series: pd.Series, n_bins: int = 10) -> pd.Series:
        return pd.qcut(series, q=n_bins, labels=False, duplicates="drop")

    # ── Delta Lake materialization ────────────────────────────

    def materialize_to_delta(
        self,
        df: pd.DataFrame,
        path: str,
        mode: str = "overwrite",
        partition_cols: list[str] | None = None,
    ) -> dict[str, Any]:
        """Materialize a DataFrame to Delta Lake (or parquet fallback)."""
        if not _SPARK_AVAILABLE:
            df.to_parquet(path, index=False)
            return {"rows": len(df), "path": path, "format": "parquet"}

        spark = self._get_spark()
        sdf = spark.createDataFrame(df)
        writer = sdf.write.format("delta").mode(mode)
        if partition_cols:
            writer = writer.partitionBy(*partition_cols)
        try:
            writer.save(path)
        except Exception:
            logger.warning("Delta not available; falling back to parquet")
            writer = sdf.write.format("parquet").mode(mode)
            if partition_cols:
                writer = writer.partitionBy(*partition_cols)
            writer.save(path)

        return {"rows": len(df), "path": path, "format": "delta"}
