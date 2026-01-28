"""CPU compute backend using NumPy and Pandas."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from forge.backends.base import ComputeBackend, ComputeDevice


class CPUBackend(ComputeBackend):
    """Standard CPU backend using NumPy/Pandas.

    This is the default backend and is always available.
    """

    @property
    def device(self) -> ComputeDevice:
        return ComputeDevice.CPU

    def is_available(self) -> bool:
        return True

    def to_frame(self, data: Any) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        return pd.DataFrame(data)

    def to_pandas(self, data: Any) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        return pd.DataFrame(data)

    def add(self, a: pd.Series, b: pd.Series) -> pd.Series:
        return a + b  # type: ignore[no-any-return]

    def multiply(self, a: pd.Series, b: pd.Series) -> pd.Series:
        return a * b  # type: ignore[no-any-return]

    def divide(self, a: pd.Series, b: pd.Series) -> pd.Series:
        return a / b.replace(0, np.nan)  # type: ignore[no-any-return]

    def power(self, series: pd.Series, exp: float) -> pd.Series:
        return series ** exp  # type: ignore[no-any-return]

    def log(self, series: pd.Series) -> pd.Series:
        return np.log(series.clip(lower=1e-10))  # type: ignore[no-any-return]

    def sqrt(self, series: pd.Series) -> pd.Series:
        return np.sqrt(series.clip(lower=0))  # type: ignore[no-any-return]

    def group_agg(
        self, df: pd.DataFrame, group_col: str, agg_col: str, agg_func: str
    ) -> pd.Series:
        return df.groupby(group_col)[agg_col].transform(agg_func)  # type: ignore[no-any-return]

    def rolling_agg(
        self, series: pd.Series, window: int, agg_func: str
    ) -> pd.Series:
        roller = series.rolling(window, min_periods=1)
        return getattr(roller, agg_func)()  # type: ignore[no-any-return]

    def one_hot_encode(
        self, series: pd.Series, categories: list[Any] | None = None
    ) -> pd.DataFrame:
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
        numeric: pd.DataFrame = df.select_dtypes(include=[np.number])
        result = numeric.copy()
        cols = list(numeric.columns)
        for d in range(2, degree + 1):
            for col in cols:
                result[f"{col}^{d}"] = numeric[col] ** d
        if degree >= 2:
            for c1, c2 in combinations(cols, 2):
                result[f"{c1}*{c2}"] = numeric[c1] * numeric[c2]
        return result

    def interaction_features(
        self, df: pd.DataFrame, columns: list[str]
    ) -> pd.DataFrame:
        result: dict[str, pd.Series] = {}
        for c1, c2 in combinations(columns, 2):
            if c1 in df.columns and c2 in df.columns:
                result[f"{c1}*{c2}"] = df[c1] * df[c2]
        return pd.DataFrame(result, index=df.index)

    def quantile_bin(
        self, series: pd.Series, n_bins: int = 10
    ) -> pd.Series:
        return pd.qcut(series, q=n_bins, labels=False, duplicates="drop")  # type: ignore[no-any-return]
