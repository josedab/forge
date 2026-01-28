"""DuckDB compute backend for SQL-optimized feature engineering.

Uses DuckDB's in-process SQL engine for efficient aggregations,
joins, and window functions. Falls back gracefully when DuckDB
is not installed.

Example:
    >>> from forge.backends.duckdb_backend import DuckDBBackend
    >>> backend = DuckDBBackend()
    >>> if backend.is_available():
    ...     result = backend.group_agg(df, "category", "value", "mean")
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from forge.backends.base import ComputeBackend, ComputeDevice

logger = logging.getLogger(__name__)

_DUCKDB_AVAILABLE = False
try:
    import duckdb
    _DUCKDB_AVAILABLE = True
except ImportError:
    duckdb = None  # type: ignore[assignment]


class DuckDBBackend(ComputeBackend):
    """DuckDB-based compute backend for SQL-optimized operations.

    Leverages DuckDB's vectorized SQL execution engine for efficient
    aggregations, window functions, and data transformations.
    Particularly effective for group-by and analytical queries.
    """

    def __init__(self) -> None:
        self._conn: Any = None

    @property
    def device(self) -> ComputeDevice:
        return ComputeDevice.CPU

    def is_available(self) -> bool:
        return _DUCKDB_AVAILABLE

    def _get_conn(self) -> Any:
        """Get or create a DuckDB connection."""
        if not _DUCKDB_AVAILABLE:
            raise RuntimeError("DuckDB is not installed")
        if self._conn is None:
            self._conn = duckdb.connect(":memory:")
        return self._conn

    def _sql(self, query: str, **local_vars: Any) -> pd.DataFrame:
        """Execute a SQL query with local variable access and return a pandas DataFrame."""
        conn = self._get_conn()
        # Register local variables so DuckDB can reference them by name
        for name, val in local_vars.items():
            conn.register(name, val)
        try:
            return conn.execute(query).fetchdf()
        finally:
            for name in local_vars:
                try:
                    conn.unregister(name)
                except Exception:
                    pass

    def to_frame(self, data: Any) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        return pd.DataFrame(data)

    def to_pandas(self, data: Any) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        return pd.DataFrame(data)

    def add(self, a: pd.Series, b: pd.Series) -> pd.Series:
        if not _DUCKDB_AVAILABLE:
            return a + b  # type: ignore[return-value]
        df = pd.DataFrame({"a": a.values, "b": b.values})
        result = self._sql("SELECT a + b AS result FROM df", df=df)
        out = result["result"]
        out.index = a.index
        return out

    def multiply(self, a: pd.Series, b: pd.Series) -> pd.Series:
        if not _DUCKDB_AVAILABLE:
            return a * b  # type: ignore[return-value]
        df = pd.DataFrame({"a": a.values, "b": b.values})
        result = self._sql("SELECT a * b AS result FROM df", df=df)
        out = result["result"]
        out.index = a.index
        return out

    def divide(self, a: pd.Series, b: pd.Series) -> pd.Series:
        if not _DUCKDB_AVAILABLE:
            return a / b.replace(0, np.nan)  # type: ignore[return-value]
        df = pd.DataFrame({"a": a.values, "b": b.values})
        result = self._sql(
            "SELECT CASE WHEN b = 0 THEN NULL ELSE a / b END AS result FROM df",
            df=df,
        )
        out = result["result"]
        out.index = a.index
        return out

    def power(self, series: pd.Series, exp: float) -> pd.Series:
        if not _DUCKDB_AVAILABLE:
            return series ** exp  # type: ignore[return-value]
        df = pd.DataFrame({"val": series.values})
        result = self._sql(f"SELECT POWER(val, {exp}) AS result FROM df", df=df)
        out = result["result"]
        out.index = series.index
        return out

    def log(self, series: pd.Series) -> pd.Series:
        if not _DUCKDB_AVAILABLE:
            return np.log(series.clip(lower=1e-10))  # type: ignore[return-value]
        df = pd.DataFrame({"val": series.values})
        result = self._sql(
            "SELECT LN(GREATEST(val, 1e-10)) AS result FROM df", df=df,
        )
        out = result["result"]
        out.index = series.index
        return out

    def sqrt(self, series: pd.Series) -> pd.Series:
        if not _DUCKDB_AVAILABLE:
            return np.sqrt(series.clip(lower=0))  # type: ignore[return-value]
        df = pd.DataFrame({"val": series.values})
        result = self._sql(
            "SELECT SQRT(GREATEST(val, 0)) AS result FROM df", df=df,
        )
        out = result["result"]
        out.index = series.index
        return out

    def group_agg(
        self, df: pd.DataFrame, group_col: str, agg_col: str, agg_func: str
    ) -> pd.Series:
        if not _DUCKDB_AVAILABLE:
            return df.groupby(group_col)[agg_col].transform(agg_func)  # type: ignore[return-value]

        func_map = {
            "mean": "AVG",
            "sum": "SUM",
            "std": "STDDEV_SAMP",
            "min": "MIN",
            "max": "MAX",
            "count": "COUNT",
        }
        sql_func = func_map.get(agg_func, agg_func.upper())

        query = f"""
            SELECT {sql_func}("{agg_col}") OVER (PARTITION BY "{group_col}") AS result
            FROM input_df
        """
        result = self._sql(query, input_df=df)
        out = result["result"]
        out.index = df.index
        return out

    def rolling_agg(
        self, series: pd.Series, window: int, agg_func: str
    ) -> pd.Series:
        if not _DUCKDB_AVAILABLE:
            roller = series.rolling(window, min_periods=1)
            return getattr(roller, agg_func)()  # type: ignore[return-value]

        func_map = {
            "mean": "AVG",
            "sum": "SUM",
            "min": "MIN",
            "max": "MAX",
        }
        sql_func = func_map.get(agg_func)
        if sql_func is None:
            roller = series.rolling(window, min_periods=1)
            return getattr(roller, agg_func)()  # type: ignore[return-value]

        df = pd.DataFrame({"val": series.values, "idx": range(len(series))})
        query = f"""
            SELECT {sql_func}(val) OVER (
                ORDER BY idx
                ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW
            ) AS result
            FROM input_df
        """
        result = self._sql(query, input_df=df)
        out = result["result"]
        out.index = series.index
        return out

    def one_hot_encode(
        self, series: pd.Series, categories: list[Any] | None = None
    ) -> pd.DataFrame:
        # SQL-based one-hot is complex; delegate to pandas for this
        if categories is not None:
            series = pd.Categorical(series, categories=categories)  # type: ignore[assignment]
        return pd.get_dummies(series, prefix=series.name)  # type: ignore[arg-type]

    def ordinal_encode(
        self, series: pd.Series, mapping: dict[Any, int] | None = None
    ) -> pd.Series:
        if mapping is not None:
            return series.map(mapping)
        cats = sorted(series.dropna().unique())
        auto_map = {v: i for i, v in enumerate(cats)}
        return series.map(auto_map)

    def polynomial_features(
        self, df: pd.DataFrame, degree: int = 2
    ) -> pd.DataFrame:
        if not _DUCKDB_AVAILABLE:
            from forge.backends.cpu import CPUBackend
            return CPUBackend().polynomial_features(df, degree)

        numeric = df.select_dtypes(include=[np.number])
        cols = list(numeric.columns)
        selects = [f'"{c}"' for c in cols]

        for d in range(2, degree + 1):
            for col in cols:
                selects.append(f'POWER("{col}", {d}) AS "{col}^{d}"')

        if degree >= 2:
            for c1, c2 in combinations(cols, 2):
                selects.append(f'"{c1}" * "{c2}" AS "{c1}*{c2}"')

        query = f"SELECT {', '.join(selects)} FROM input_df"
        result = self._sql(query, input_df=numeric)
        result.index = df.index
        return result

    def interaction_features(
        self, df: pd.DataFrame, columns: list[str]
    ) -> pd.DataFrame:
        if not _DUCKDB_AVAILABLE:
            from forge.backends.cpu import CPUBackend
            return CPUBackend().interaction_features(df, columns)

        valid_cols = [c for c in columns if c in df.columns]
        if len(valid_cols) < 2:
            return pd.DataFrame(index=df.index)

        selects = []
        for c1, c2 in combinations(valid_cols, 2):
            selects.append(f'"{c1}" * "{c2}" AS "{c1}*{c2}"')

        query = f"SELECT {', '.join(selects)} FROM input_df"
        result = self._sql(query, input_df=df)
        result.index = df.index
        return result

    def quantile_bin(
        self, series: pd.Series, n_bins: int = 10
    ) -> pd.Series:
        return pd.qcut(series, q=n_bins, labels=False, duplicates="drop")  # type: ignore[return-value]

    def close(self) -> None:
        """Close the DuckDB connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
