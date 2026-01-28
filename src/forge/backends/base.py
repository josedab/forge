"""Abstract compute backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import pandas as pd


class ComputeDevice(str, Enum):
    """Available compute devices."""

    CPU = "cpu"
    GPU = "gpu"
    POLARS = "polars"
    DUCKDB = "duckdb"
    AUTO = "auto"


class ComputeBackend(ABC):
    """Abstract interface for compute backends.

    All feature computation operations are defined here and
    implemented by CPU and GPU backends. This enables transparent
    dispatch based on hardware availability.
    """

    @property
    @abstractmethod
    def device(self) -> ComputeDevice:
        """Return which device this backend uses."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on the current system."""

    # ── DataFrame operations ───────────────────────────────

    @abstractmethod
    def to_frame(self, data: Any) -> pd.DataFrame:
        """Convert data to a DataFrame (or GPU equivalent)."""

    @abstractmethod
    def to_pandas(self, data: Any) -> pd.DataFrame:
        """Convert backend-specific frame to pandas DataFrame."""

    # ── Numeric operations ─────────────────────────────────

    @abstractmethod
    def add(self, a: pd.Series, b: pd.Series) -> pd.Series:
        """Element-wise addition."""

    @abstractmethod
    def multiply(self, a: pd.Series, b: pd.Series) -> pd.Series:
        """Element-wise multiplication."""

    @abstractmethod
    def divide(self, a: pd.Series, b: pd.Series) -> pd.Series:
        """Element-wise division (safe, NaN for div-by-zero)."""

    @abstractmethod
    def power(self, series: pd.Series, exp: float) -> pd.Series:
        """Raise series to a power."""

    @abstractmethod
    def log(self, series: pd.Series) -> pd.Series:
        """Natural logarithm (clipped to avoid -inf)."""

    @abstractmethod
    def sqrt(self, series: pd.Series) -> pd.Series:
        """Square root (clipped to avoid NaN on negatives)."""

    # ── Aggregation operations ─────────────────────────────

    @abstractmethod
    def group_agg(
        self, df: pd.DataFrame, group_col: str, agg_col: str, agg_func: str
    ) -> pd.Series:
        """Grouped aggregation."""

    @abstractmethod
    def rolling_agg(
        self, series: pd.Series, window: int, agg_func: str
    ) -> pd.Series:
        """Rolling window aggregation."""

    # ── Encoding operations ────────────────────────────────

    @abstractmethod
    def one_hot_encode(
        self, series: pd.Series, categories: list[Any] | None = None
    ) -> pd.DataFrame:
        """One-hot encode a categorical series."""

    @abstractmethod
    def ordinal_encode(
        self, series: pd.Series, mapping: dict[Any, int] | None = None
    ) -> pd.Series:
        """Ordinal encode a categorical series."""

    # ── Polynomial/interaction features ────────────────────

    @abstractmethod
    def polynomial_features(
        self, df: pd.DataFrame, degree: int = 2
    ) -> pd.DataFrame:
        """Generate polynomial features."""

    @abstractmethod
    def interaction_features(
        self, df: pd.DataFrame, columns: list[str]
    ) -> pd.DataFrame:
        """Generate pairwise interaction features."""

    # ── Binning ────────────────────────────────────────────

    @abstractmethod
    def quantile_bin(
        self, series: pd.Series, n_bins: int = 10
    ) -> pd.Series:
        """Quantile-based binning."""
