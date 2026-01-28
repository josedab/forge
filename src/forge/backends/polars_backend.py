"""Polars compute backend for high-performance feature engineering.

Uses Polars DataFrames for significantly faster operations than Pandas
on medium-to-large datasets. Falls back gracefully when Polars is
not installed.

Example:
    >>> from forge.backends.polars_backend import PolarsBackend
    >>> backend = PolarsBackend()
    >>> if backend.is_available():
    ...     result = backend.multiply(series_a, series_b)
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from forge.backends.base import ComputeBackend, ComputeDevice

logger = logging.getLogger(__name__)

_POLARS_AVAILABLE = False
try:
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:
    pl = None  # type: ignore[assignment]


class PolarsBackend(ComputeBackend):
    """Polars-based compute backend for fast columnar operations.

    Leverages Polars' lazy evaluation and columnar processing for
    5-50x speedups over Pandas on datasets > 100K rows.
    Falls back to Pandas if Polars is not installed.
    """

    @property
    def device(self) -> ComputeDevice:
        return ComputeDevice.CPU

    def is_available(self) -> bool:
        return _POLARS_AVAILABLE

    def _to_polars(self, data: Any) -> Any:
        """Convert pandas data to Polars."""
        if not _POLARS_AVAILABLE:
            return data
        if isinstance(data, pd.DataFrame):
            return pl.from_pandas(data)
        if isinstance(data, pd.Series):
            return pl.from_pandas(data.to_frame()).to_series(0)
        return data

    def to_frame(self, data: Any) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        if _POLARS_AVAILABLE and isinstance(data, pl.DataFrame):
            return data.to_pandas()
        return pd.DataFrame(data)

    def to_pandas(self, data: Any) -> pd.DataFrame:
        if _POLARS_AVAILABLE and isinstance(data, pl.DataFrame):
            return data.to_pandas()
        if isinstance(data, pd.DataFrame):
            return data
        return pd.DataFrame(data)

    def add(self, a: pd.Series, b: pd.Series) -> pd.Series:
        if not _POLARS_AVAILABLE:
            return a + b  # type: ignore[return-value]
        pa = pl.from_pandas(a.to_frame()).to_series(0)
        pb = pl.from_pandas(b.to_frame()).to_series(0)
        result = (pa + pb).to_pandas()
        result.index = a.index
        return result

    def multiply(self, a: pd.Series, b: pd.Series) -> pd.Series:
        if not _POLARS_AVAILABLE:
            return a * b  # type: ignore[return-value]
        pa = pl.from_pandas(a.to_frame()).to_series(0)
        pb = pl.from_pandas(b.to_frame()).to_series(0)
        result = (pa * pb).to_pandas()
        result.index = a.index
        return result

    def divide(self, a: pd.Series, b: pd.Series) -> pd.Series:
        if not _POLARS_AVAILABLE:
            return a / b.replace(0, np.nan)  # type: ignore[return-value]
        pa = pl.from_pandas(a.to_frame()).to_series(0)
        pb = pl.from_pandas(b.to_frame()).to_series(0)
        # Replace zeros with null for safe division
        pb_safe = pb.map_elements(lambda x: None if x == 0 else x, return_dtype=pl.Float64)
        result = (pa / pb_safe).to_pandas()
        result.index = a.index
        return result

    def power(self, series: pd.Series, exp: float) -> pd.Series:
        if not _POLARS_AVAILABLE:
            return series ** exp  # type: ignore[return-value]
        ps = pl.from_pandas(series.to_frame()).to_series(0)
        result = ps.pow(exp).to_pandas()
        result.index = series.index
        return result

    def log(self, series: pd.Series) -> pd.Series:
        if not _POLARS_AVAILABLE:
            return np.log(series.clip(lower=1e-10))  # type: ignore[return-value]
        ps = pl.from_pandas(series.to_frame()).to_series(0)
        result = ps.clip(1e-10).log().to_pandas()
        result.index = series.index
        return result

    def sqrt(self, series: pd.Series) -> pd.Series:
        if not _POLARS_AVAILABLE:
            return np.sqrt(series.clip(lower=0))  # type: ignore[return-value]
        ps = pl.from_pandas(series.to_frame()).to_series(0)
        result = ps.clip(0).sqrt().to_pandas()
        result.index = series.index
        return result

    def group_agg(
        self, df: pd.DataFrame, group_col: str, agg_col: str, agg_func: str
    ) -> pd.Series:
        if not _POLARS_AVAILABLE:
            return df.groupby(group_col)[agg_col].transform(agg_func)  # type: ignore[return-value]
        pldf = pl.from_pandas(df)
        agg_map = {
            "mean": pl.col(agg_col).mean(),
            "sum": pl.col(agg_col).sum(),
            "std": pl.col(agg_col).std(),
            "min": pl.col(agg_col).min(),
            "max": pl.col(agg_col).max(),
            "count": pl.col(agg_col).count(),
        }
        agg_expr = agg_map.get(agg_func)
        if agg_expr is None:
            return df.groupby(group_col)[agg_col].transform(agg_func)  # type: ignore[return-value]

        result = pldf.select(
            pl.col(agg_col).over(group_col).alias(f"{agg_col}_{agg_func}")
        ).to_pandas().iloc[:, 0]
        result.index = df.index
        return result  # type: ignore[return-value]

    def rolling_agg(
        self, series: pd.Series, window: int, agg_func: str
    ) -> pd.Series:
        if not _POLARS_AVAILABLE:
            roller = series.rolling(window, min_periods=1)
            return getattr(roller, agg_func)()  # type: ignore[return-value]
        ps = pl.from_pandas(series.to_frame()).to_series(0)
        agg_map = {
            "mean": ps.rolling_mean(window, min_periods=1),
            "sum": ps.rolling_sum(window, min_periods=1),
            "std": ps.rolling_std(window, min_periods=1),
            "min": ps.rolling_min(window, min_periods=1),
            "max": ps.rolling_max(window, min_periods=1),
        }
        result_pl = agg_map.get(agg_func)
        if result_pl is None:
            roller = series.rolling(window, min_periods=1)
            return getattr(roller, agg_func)()  # type: ignore[return-value]
        result = result_pl.to_pandas()
        result.index = series.index
        return result

    def one_hot_encode(
        self, series: pd.Series, categories: list[Any] | None = None
    ) -> pd.DataFrame:
        if not _POLARS_AVAILABLE:
            if categories is not None:
                series = pd.Categorical(series, categories=categories)  # type: ignore[assignment]
            return pd.get_dummies(series, prefix=series.name)  # type: ignore[arg-type]
        # Polars one-hot via to_dummies
        name = series.name or "col"
        pldf = pl.DataFrame({name: series.values})
        result = pldf.to_dummies(columns=[name]).to_pandas()
        result.index = series.index
        return result

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
        if not _POLARS_AVAILABLE:
            from forge.backends.cpu import CPUBackend
            return CPUBackend().polynomial_features(df, degree)

        numeric = df.select_dtypes(include=[np.number])
        pldf = pl.from_pandas(numeric)
        cols = list(numeric.columns)
        exprs = [pl.col(c) for c in cols]

        for d in range(2, degree + 1):
            for col in cols:
                exprs.append(pl.col(col).pow(d).alias(f"{col}^{d}"))

        if degree >= 2:
            for c1, c2 in combinations(cols, 2):
                exprs.append((pl.col(c1) * pl.col(c2)).alias(f"{c1}*{c2}"))

        result = pldf.select(exprs).to_pandas()
        result.index = df.index
        return result

    def interaction_features(
        self, df: pd.DataFrame, columns: list[str]
    ) -> pd.DataFrame:
        if not _POLARS_AVAILABLE:
            from forge.backends.cpu import CPUBackend
            return CPUBackend().interaction_features(df, columns)

        pldf = pl.from_pandas(df)
        exprs = []
        for c1, c2 in combinations(columns, 2):
            if c1 in df.columns and c2 in df.columns:
                exprs.append((pl.col(c1) * pl.col(c2)).alias(f"{c1}*{c2}"))

        if not exprs:
            return pd.DataFrame(index=df.index)

        result = pldf.select(exprs).to_pandas()
        result.index = df.index
        return result

    def quantile_bin(
        self, series: pd.Series, n_bins: int = 10
    ) -> pd.Series:
        return pd.qcut(series, q=n_bins, labels=False, duplicates="drop")  # type: ignore[return-value]
