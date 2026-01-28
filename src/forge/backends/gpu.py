"""GPU compute backend using cuDF and cuPy (RAPIDS).

Falls back to CPU backend when RAPIDS is not available.
All operations accept pandas inputs and transparently
convert to/from cuDF as needed.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from forge.backends.base import ComputeBackend, ComputeDevice

logger = logging.getLogger(__name__)

_HAS_CUDF = False
try:
    import cudf
    import cupy as cp
    _HAS_CUDF = True
except ImportError:
    cudf = None  # type: ignore[assignment]
    cp = None  # type: ignore[assignment]


class GPUBackend(ComputeBackend):
    """GPU-accelerated backend using RAPIDS cuDF/cuPy.

    When RAPIDS is not installed, ``is_available()`` returns False.
    Use the dispatch module to automatically fall back to CPU.

    Example:
        >>> from forge.backends import get_backend
        >>> backend = get_backend("auto")  # GPU if available, else CPU
        >>> result = backend.multiply(series_a, series_b)
    """

    @property
    def device(self) -> ComputeDevice:
        return ComputeDevice.GPU

    def is_available(self) -> bool:
        return _HAS_CUDF

    def _to_cudf_series(self, s: pd.Series) -> Any:
        if _HAS_CUDF:
            return cudf.Series(s)
        return s

    def _to_cudf_frame(self, df: pd.DataFrame) -> Any:
        if _HAS_CUDF:
            return cudf.DataFrame(df)
        return df

    def _to_pandas_series(self, s: Any) -> pd.Series:
        if _HAS_CUDF and isinstance(s, cudf.Series):
            return s.to_pandas()  # type: ignore[no-any-return]
        return pd.Series(s)

    def _to_pandas_frame(self, df: Any) -> pd.DataFrame:
        if _HAS_CUDF and isinstance(df, cudf.DataFrame):
            return df.to_pandas()  # type: ignore[no-any-return]
        return pd.DataFrame(df)

    def to_frame(self, data: Any) -> pd.DataFrame:
        if _HAS_CUDF:
            if isinstance(data, cudf.DataFrame):
                return data.to_pandas()  # type: ignore[no-any-return]
        if isinstance(data, pd.DataFrame):
            return data
        return pd.DataFrame(data)

    def to_pandas(self, data: Any) -> pd.DataFrame:
        return self._to_pandas_frame(data)

    def add(self, a: pd.Series, b: pd.Series) -> pd.Series:
        if not _HAS_CUDF:
            return a + b  # type: ignore[no-any-return]
        ga = self._to_cudf_series(a)
        gb = self._to_cudf_series(b)
        return self._to_pandas_series(ga + gb)

    def multiply(self, a: pd.Series, b: pd.Series) -> pd.Series:
        if not _HAS_CUDF:
            return a * b  # type: ignore[no-any-return]
        ga = self._to_cudf_series(a)
        gb = self._to_cudf_series(b)
        return self._to_pandas_series(ga * gb)

    def divide(self, a: pd.Series, b: pd.Series) -> pd.Series:
        if not _HAS_CUDF:
            return a / b.replace(0, np.nan)  # type: ignore[no-any-return]
        ga = self._to_cudf_series(a)
        gb = self._to_cudf_series(b)
        gb = gb.replace(0, np.nan)
        return self._to_pandas_series(ga / gb)

    def power(self, series: pd.Series, exp: float) -> pd.Series:
        if not _HAS_CUDF:
            return series ** exp  # type: ignore[no-any-return]
        gs = self._to_cudf_series(series)
        return self._to_pandas_series(gs ** exp)

    def log(self, series: pd.Series) -> pd.Series:
        if not _HAS_CUDF:
            return np.log(series.clip(lower=1e-10))  # type: ignore[no-any-return]
        gs = self._to_cudf_series(series.clip(lower=1e-10))
        return self._to_pandas_series(cp.log(gs.values))

    def sqrt(self, series: pd.Series) -> pd.Series:
        if not _HAS_CUDF:
            return np.sqrt(series.clip(lower=0))  # type: ignore[no-any-return]
        gs = self._to_cudf_series(series.clip(lower=0))
        return self._to_pandas_series(cp.sqrt(gs.values))

    def group_agg(
        self, df: pd.DataFrame, group_col: str, agg_col: str, agg_func: str
    ) -> pd.Series:
        if not _HAS_CUDF:
            return df.groupby(group_col)[agg_col].transform(agg_func)  # type: ignore[no-any-return]
        gdf = self._to_cudf_frame(df)
        result = gdf.groupby(group_col)[agg_col].transform(agg_func)
        return self._to_pandas_series(result)

    def rolling_agg(
        self, series: pd.Series, window: int, agg_func: str
    ) -> pd.Series:
        if not _HAS_CUDF:
            roller = series.rolling(window, min_periods=1)
            return getattr(roller, agg_func)()  # type: ignore[no-any-return]
        gs = self._to_cudf_series(series)
        roller = gs.rolling(window, min_periods=1)
        result = getattr(roller, agg_func)()
        return self._to_pandas_series(result)

    def one_hot_encode(
        self, series: pd.Series, categories: list[Any] | None = None
    ) -> pd.DataFrame:
        if not _HAS_CUDF:
            if categories is not None:
                series = pd.Categorical(series, categories=categories)  # type: ignore[assignment]
            return pd.get_dummies(series, prefix=series.name)  # type: ignore[arg-type]
        gs = self._to_cudf_series(series)
        result = cudf.get_dummies(gs, prefix=series.name)
        return self._to_pandas_frame(result)

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
        numeric: pd.DataFrame = df.select_dtypes(include=[np.number])
        if not _HAS_CUDF:
            result = numeric.copy()
            cols = list(numeric.columns)
            for d in range(2, degree + 1):
                for col in cols:
                    result[f"{col}^{d}"] = numeric[col] ** d
            if degree >= 2:
                for c1, c2 in combinations(cols, 2):
                    result[f"{c1}*{c2}"] = numeric[c1] * numeric[c2]
            return result

        gdf = self._to_cudf_frame(numeric)
        result_dict: dict[str, Any] = {c: gdf[c] for c in gdf.columns}
        cols = list(gdf.columns)
        for d in range(2, degree + 1):
            for col in cols:
                result_dict[f"{col}^{d}"] = gdf[col] ** d
        if degree >= 2:
            for c1, c2 in combinations(cols, 2):
                result_dict[f"{c1}*{c2}"] = gdf[c1] * gdf[c2]
        return self._to_pandas_frame(cudf.DataFrame(result_dict))

    def interaction_features(
        self, df: pd.DataFrame, columns: list[str]
    ) -> pd.DataFrame:
        result: dict[str, pd.Series] = {}
        for c1, c2 in combinations(columns, 2):
            if c1 in df.columns and c2 in df.columns:
                result[f"{c1}*{c2}"] = self.multiply(df[c1], df[c2])
        return pd.DataFrame(result, index=df.index)

    def quantile_bin(
        self, series: pd.Series, n_bins: int = 10
    ) -> pd.Series:
        # cuDF cut/qcut support is limited; use pandas
        return pd.qcut(series, q=n_bins, labels=False, duplicates="drop")  # type: ignore[no-any-return]
