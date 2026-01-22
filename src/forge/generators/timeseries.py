"""Time-series feature generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np
import pandas as pd

from forge.exceptions import NotFittedError, ValidationError
from forge.generators.base import BaseFeatureGenerator

if TYPE_CHECKING:
    from typing_extensions import Self


@dataclass
class LagConfig:
    """Configuration for lag features."""

    lags: list[int] = field(default_factory=lambda: [1, 2, 3, 7, 14, 30])
    columns: list[str] | None = None


@dataclass
class RollingConfig:
    """Configuration for rolling window features."""

    windows: list[int] = field(default_factory=lambda: [3, 7, 14, 30])
    functions: list[str] = field(default_factory=lambda: ["mean", "std", "min", "max"])
    columns: list[str] | None = None
    min_periods: int = 1


@dataclass
class DatetimeConfig:
    """Configuration for datetime features."""

    components: list[str] = field(default_factory=lambda: [
        "year", "month", "day", "dayofweek", "hour", "quarter", "dayofyear"
    ])
    cyclical: bool = True
    include_is_flags: bool = True


class TimeSeriesFeatureGenerator(BaseFeatureGenerator):
    """Generates time-series features from temporal data.

    Creates lag features, rolling window statistics, datetime components,
    and other time-series specific transformations.

    Example:
        >>> generator = TimeSeriesFeatureGenerator(
        ...     datetime_col="timestamp",
        ...     target_cols=["value"],
        ...     lags=[1, 7, 30],
        ...     rolling_windows=[7, 30],
        ... )
        >>> X_features = generator.fit_transform(X)
    """

    def __init__(
        self,
        datetime_col: str | None = None,
        target_cols: list[str] | None = None,
        entity_col: str | None = None,
        lags: list[int] | LagConfig | None = None,
        rolling_windows: list[int] | RollingConfig | None = None,
        expanding: bool = False,
        datetime_features: bool = True,
        datetime_config: DatetimeConfig | None = None,
        diff_orders: list[int] | None = None,
        pct_change: bool = True,
        ewm_spans: list[int] | None = None,
        include_original: bool = True,
        drop_na: bool = False,
    ) -> None:
        """Initialize time-series feature generator.

        Args:
            datetime_col: Column containing datetime. Auto-detected if None.
            target_cols: Columns for lag/rolling features. Numeric cols if None.
            entity_col: Column for grouping (for panel data).
            lags: Lag periods to create features for.
            rolling_windows: Window sizes for rolling statistics.
            expanding: Whether to include expanding window features.
            datetime_features: Whether to extract datetime components.
            datetime_config: Configuration for datetime features.
            diff_orders: Differencing orders (e.g., [1, 2] for first and second diff).
            pct_change: Whether to include percentage change features.
            ewm_spans: Spans for exponential weighted moving statistics.
            include_original: Whether to include original columns.
            drop_na: Whether to drop rows with NaN from lag/rolling.
        """
        super().__init__()
        self.datetime_col = datetime_col
        self.target_cols = target_cols
        self.entity_col = entity_col
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.expanding = expanding
        self.datetime_features = datetime_features
        self.datetime_config = datetime_config or DatetimeConfig()
        self.diff_orders = diff_orders
        self.pct_change = pct_change
        self.ewm_spans = ewm_spans
        self.include_original = include_original
        self.drop_na = drop_na

        self._datetime_col: str | None = None
        self._target_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the generator.

        Args:
            X: Input DataFrame with time-series data.
            y: Ignored.

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)
        self._input_columns = list(X.columns)

        # Find datetime column
        if self.datetime_col is not None:
            if self.datetime_col not in X.columns:
                raise ValidationError(f"Datetime column '{self.datetime_col}' not found")
            self._datetime_col = self.datetime_col
        else:
            # Auto-detect datetime column
            for col in X.columns:
                if pd.api.types.is_datetime64_any_dtype(X[col]):
                    self._datetime_col = col
                    break

        # Determine target columns for lag/rolling features
        if self.target_cols is not None:
            missing = [c for c in self.target_cols if c not in X.columns]
            if missing:
                raise ValidationError(f"Target columns not found: {missing}")
            self._target_cols = self.target_cols
        else:
            # Use numeric columns (excluding entity and datetime)
            exclude = set()
            if self._datetime_col:
                exclude.add(self._datetime_col)
            if self.entity_col:
                exclude.add(self.entity_col)
            self._target_cols = [
                c for c in X.select_dtypes(include=[np.number]).columns
                if c not in exclude
            ]

        # Build feature names
        self._feature_names_out = self._build_feature_names(X)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate time-series features.

        Args:
            X: Input DataFrame.

        Returns:
            DataFrame with generated features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result = X.copy() if self.include_original else pd.DataFrame(index=X.index)

        # Sort by datetime if available
        if self._datetime_col and self._datetime_col in X.columns:
            X = X.sort_values(self._datetime_col).copy()
            result = result.loc[X.index]

        # Datetime features
        if self.datetime_features and self._datetime_col:
            dt_features = self._generate_datetime_features(X)
            result = pd.concat([result, dt_features], axis=1)

        # Generate features (with entity grouping if specified)
        if self.entity_col and self.entity_col in X.columns:
            groups = X.groupby(self.entity_col)
            ts_features = []
            for _, group in groups:
                group_features = self._generate_ts_features(group)
                ts_features.append(group_features)
            ts_df = pd.concat(ts_features).loc[X.index]
        else:
            ts_df = self._generate_ts_features(X)

        result = pd.concat([result, ts_df], axis=1)

        # Remove duplicates
        result = result.loc[:, ~result.columns.duplicated()]

        if self.drop_na:
            result = result.dropna()

        return result

    def _build_feature_names(self, X: pd.DataFrame) -> list[str]:
        """Build list of output feature names."""
        names = []

        if self.include_original:
            names.extend(X.columns.tolist())

        # Datetime features
        if self.datetime_features and self._datetime_col:
            prefix = self._datetime_col
            config = self.datetime_config
            for comp in config.components:
                names.append(f"{prefix}_{comp}")
                if config.cyclical and comp in ["month", "dayofweek", "hour", "dayofyear"]:
                    names.append(f"{prefix}_{comp}_sin")
                    names.append(f"{prefix}_{comp}_cos")
            if config.include_is_flags:
                names.extend([
                    f"{prefix}_is_weekend",
                    f"{prefix}_is_month_start",
                    f"{prefix}_is_month_end",
                    f"{prefix}_is_quarter_start",
                    f"{prefix}_is_quarter_end",
                    f"{prefix}_is_year_start",
                    f"{prefix}_is_year_end",
                ])

        # Lag features
        if self.lags:
            lag_config = self._parse_lag_config()
            cols = lag_config.columns or self._target_cols
            for col in cols:
                for lag in lag_config.lags:
                    names.append(f"{col}_lag_{lag}")

        # Rolling features
        if self.rolling_windows:
            roll_config = self._parse_rolling_config()
            cols = roll_config.columns or self._target_cols
            for col in cols:
                for window in roll_config.windows:
                    for func in roll_config.functions:
                        names.append(f"{col}_rolling_{window}_{func}")

        # Expanding features
        if self.expanding:
            for col in self._target_cols:
                for func in ["mean", "std", "min", "max"]:
                    names.append(f"{col}_expanding_{func}")

        # Diff features
        if self.diff_orders:
            for col in self._target_cols:
                for order in self.diff_orders:
                    names.append(f"{col}_diff_{order}")

        # Pct change
        if self.pct_change:
            for col in self._target_cols:
                names.append(f"{col}_pct_change")

        # EWM features
        if self.ewm_spans:
            for col in self._target_cols:
                for span in self.ewm_spans:
                    names.append(f"{col}_ewm_{span}_mean")
                    names.append(f"{col}_ewm_{span}_std")

        return names

    def _parse_lag_config(self) -> LagConfig:
        """Parse lag configuration."""
        if isinstance(self.lags, LagConfig):
            return self.lags
        elif isinstance(self.lags, list):
            return LagConfig(lags=self.lags)
        return LagConfig()

    def _parse_rolling_config(self) -> RollingConfig:
        """Parse rolling configuration."""
        if isinstance(self.rolling_windows, RollingConfig):
            return self.rolling_windows
        elif isinstance(self.rolling_windows, list):
            return RollingConfig(windows=self.rolling_windows)
        return RollingConfig()

    def _generate_datetime_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate datetime component features."""
        features = {}
        dt_col = X[self._datetime_col]

        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(dt_col):
            dt_col = pd.to_datetime(dt_col)

        prefix = self._datetime_col
        config = self.datetime_config

        for comp in config.components:
            if comp == "year":
                features[f"{prefix}_year"] = dt_col.dt.year
            elif comp == "month":
                features[f"{prefix}_month"] = dt_col.dt.month
            elif comp == "day":
                features[f"{prefix}_day"] = dt_col.dt.day
            elif comp == "dayofweek":
                features[f"{prefix}_dayofweek"] = dt_col.dt.dayofweek
            elif comp == "hour":
                features[f"{prefix}_hour"] = dt_col.dt.hour
            elif comp == "minute":
                features[f"{prefix}_minute"] = dt_col.dt.minute
            elif comp == "quarter":
                features[f"{prefix}_quarter"] = dt_col.dt.quarter
            elif comp == "dayofyear":
                features[f"{prefix}_dayofyear"] = dt_col.dt.dayofyear
            elif comp == "weekofyear":
                features[f"{prefix}_weekofyear"] = dt_col.dt.isocalendar().week

        # Cyclical encoding
        if config.cyclical:
            if "month" in config.components:
                month = features.get(f"{prefix}_month", dt_col.dt.month)
                features[f"{prefix}_month_sin"] = np.sin(2 * np.pi * month / 12)
                features[f"{prefix}_month_cos"] = np.cos(2 * np.pi * month / 12)

            if "dayofweek" in config.components:
                dow = features.get(f"{prefix}_dayofweek", dt_col.dt.dayofweek)
                features[f"{prefix}_dayofweek_sin"] = np.sin(2 * np.pi * dow / 7)
                features[f"{prefix}_dayofweek_cos"] = np.cos(2 * np.pi * dow / 7)

            if "hour" in config.components:
                hour = features.get(f"{prefix}_hour", dt_col.dt.hour)
                features[f"{prefix}_hour_sin"] = np.sin(2 * np.pi * hour / 24)
                features[f"{prefix}_hour_cos"] = np.cos(2 * np.pi * hour / 24)

            if "dayofyear" in config.components:
                doy = features.get(f"{prefix}_dayofyear", dt_col.dt.dayofyear)
                features[f"{prefix}_dayofyear_sin"] = np.sin(2 * np.pi * doy / 365)
                features[f"{prefix}_dayofyear_cos"] = np.cos(2 * np.pi * doy / 365)

        # Is-flags
        if config.include_is_flags:
            features[f"{prefix}_is_weekend"] = dt_col.dt.dayofweek.isin([5, 6]).astype(int)
            features[f"{prefix}_is_month_start"] = dt_col.dt.is_month_start.astype(int)
            features[f"{prefix}_is_month_end"] = dt_col.dt.is_month_end.astype(int)
            features[f"{prefix}_is_quarter_start"] = dt_col.dt.is_quarter_start.astype(int)
            features[f"{prefix}_is_quarter_end"] = dt_col.dt.is_quarter_end.astype(int)
            features[f"{prefix}_is_year_start"] = dt_col.dt.is_year_start.astype(int)
            features[f"{prefix}_is_year_end"] = dt_col.dt.is_year_end.astype(int)

        return pd.DataFrame(features, index=X.index)

    def _generate_ts_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate time-series features (lag, rolling, etc.)."""
        features = {}

        # Lag features
        if self.lags:
            lag_config = self._parse_lag_config()
            cols = lag_config.columns or self._target_cols
            for col in cols:
                if col in X.columns:
                    for lag in lag_config.lags:
                        features[f"{col}_lag_{lag}"] = X[col].shift(lag)

        # Rolling features
        if self.rolling_windows:
            roll_config = self._parse_rolling_config()
            cols = roll_config.columns or self._target_cols
            for col in cols:
                if col in X.columns:
                    for window in roll_config.windows:
                        rolling = X[col].rolling(window, min_periods=roll_config.min_periods)
                        for func in roll_config.functions:
                            if hasattr(rolling, func):
                                features[f"{col}_rolling_{window}_{func}"] = getattr(rolling, func)()

        # Expanding features
        if self.expanding:
            for col in self._target_cols:
                if col in X.columns:
                    expanding = X[col].expanding(min_periods=1)
                    features[f"{col}_expanding_mean"] = expanding.mean()
                    features[f"{col}_expanding_std"] = expanding.std()
                    features[f"{col}_expanding_min"] = expanding.min()
                    features[f"{col}_expanding_max"] = expanding.max()

        # Differencing
        if self.diff_orders:
            for col in self._target_cols:
                if col in X.columns:
                    for order in self.diff_orders:
                        features[f"{col}_diff_{order}"] = X[col].diff(order)

        # Percentage change
        if self.pct_change:
            for col in self._target_cols:
                if col in X.columns:
                    features[f"{col}_pct_change"] = X[col].pct_change()

        # Exponential weighted moving features
        if self.ewm_spans:
            for col in self._target_cols:
                if col in X.columns:
                    for span in self.ewm_spans:
                        ewm = X[col].ewm(span=span, min_periods=1)
                        features[f"{col}_ewm_{span}_mean"] = ewm.mean()
                        features[f"{col}_ewm_{span}_std"] = ewm.std()

        return pd.DataFrame(features, index=X.index)


class SeasonalDecomposer(BaseFeatureGenerator):
    """Extracts seasonal decomposition features.

    Decomposes time series into trend, seasonal, and residual
    components using STL or classical decomposition.

    Example:
        >>> decomposer = SeasonalDecomposer(
        ...     datetime_col="date",
        ...     value_col="sales",
        ...     period=12,
        ... )
        >>> X_decomposed = decomposer.fit_transform(X)
    """

    def __init__(
        self,
        datetime_col: str,
        value_cols: list[str] | str,
        period: int | None = None,
        model: Literal["additive", "multiplicative"] = "additive",
        method: Literal["stl", "classical"] = "stl",
        robust: bool = True,
        include_original: bool = True,
    ) -> None:
        """Initialize seasonal decomposer.

        Args:
            datetime_col: Datetime column name.
            value_cols: Columns to decompose.
            period: Seasonal period (auto-detected if None).
            model: Decomposition model type.
            method: Decomposition method.
            robust: Use robust estimation (STL only).
            include_original: Include original columns.
        """
        super().__init__()
        self.datetime_col = datetime_col
        self.value_cols = [value_cols] if isinstance(value_cols, str) else value_cols
        self.period = period
        self.model = model
        self.method = method
        self.robust = robust
        self.include_original = include_original

        self._periods: dict[str, int] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the decomposer.

        Args:
            X: Input DataFrame.
            y: Ignored.

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)

        if self.datetime_col not in X.columns:
            raise ValidationError(f"Datetime column '{self.datetime_col}' not found")

        for col in self.value_cols:
            if col not in X.columns:
                raise ValidationError(f"Value column '{col}' not found")

        self._input_columns = list(X.columns)

        # Detect period if not specified
        if self.period is not None:
            for col in self.value_cols:
                self._periods[col] = self.period
        else:
            # Auto-detect based on datetime frequency
            dt = pd.to_datetime(X[self.datetime_col])
            freq = pd.infer_freq(dt.sort_values())
            if freq:
                if freq.startswith(("D", "B")):
                    period = 7  # Weekly
                elif freq.startswith("W"):
                    period = 52  # Yearly
                elif freq.startswith("M"):
                    period = 12  # Monthly
                elif freq.startswith("H"):
                    period = 24  # Daily
                else:
                    period = 7
            else:
                period = 7
            for col in self.value_cols:
                self._periods[col] = period

        # Build feature names
        names = []
        if self.include_original:
            names.extend(X.columns.tolist())
        for col in self.value_cols:
            names.extend([
                f"{col}_trend",
                f"{col}_seasonal",
                f"{col}_residual",
                f"{col}_seasonal_strength",
            ])
        self._feature_names_out = names
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply seasonal decomposition.

        Args:
            X: Input DataFrame.

        Returns:
            DataFrame with decomposition features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result = X.copy() if self.include_original else pd.DataFrame(index=X.index)

        # Sort by datetime
        X_sorted = X.sort_values(self.datetime_col)
        sort_idx = X_sorted.index

        for col in self.value_cols:
            series = X_sorted[col].values
            period = self._periods[col]

            if len(series) < 2 * period:
                # Not enough data for decomposition
                result[f"{col}_trend"] = np.nan
                result[f"{col}_seasonal"] = np.nan
                result[f"{col}_residual"] = np.nan
                result[f"{col}_seasonal_strength"] = np.nan
                continue

            try:
                if self.method == "stl":
                    from statsmodels.tsa.seasonal import STL
                    stl = STL(series, period=period, robust=self.robust)
                    decomp = stl.fit()
                    trend = decomp.trend
                    seasonal = decomp.seasonal
                    residual = decomp.resid
                else:
                    from statsmodels.tsa.seasonal import seasonal_decompose
                    decomp = seasonal_decompose(
                        series, model=self.model, period=period
                    )
                    trend = decomp.trend
                    seasonal = decomp.seasonal
                    residual = decomp.resid

                # Calculate seasonal strength
                var_resid = np.nanvar(residual)
                var_resid_seasonal = np.nanvar(residual + seasonal)
                strength = max(0, 1 - var_resid / var_resid_seasonal) if var_resid_seasonal > 0 else 0

                # Create features aligned with original index
                result.loc[sort_idx, f"{col}_trend"] = trend
                result.loc[sort_idx, f"{col}_seasonal"] = seasonal
                result.loc[sort_idx, f"{col}_residual"] = residual
                result.loc[sort_idx, f"{col}_seasonal_strength"] = strength

            except Exception:
                # Fallback to NaN if decomposition fails
                result[f"{col}_trend"] = np.nan
                result[f"{col}_seasonal"] = np.nan
                result[f"{col}_residual"] = np.nan
                result[f"{col}_seasonal_strength"] = np.nan

        return result


class FourierFeatureGenerator(BaseFeatureGenerator):
    """Generates Fourier features for capturing seasonality.

    Creates sine and cosine features at various frequencies
    to capture periodic patterns in time series.

    Example:
        >>> fourier = FourierFeatureGenerator(
        ...     datetime_col="date",
        ...     periods={"yearly": 365.25, "weekly": 7},
        ...     n_terms=5,
        ... )
        >>> X_fourier = fourier.fit_transform(X)
    """

    def __init__(
        self,
        datetime_col: str,
        periods: dict[str, float] | None = None,
        n_terms: int = 3,
        include_original: bool = True,
    ) -> None:
        """Initialize Fourier feature generator.

        Args:
            datetime_col: Datetime column name.
            periods: Dict of {name: period_in_days}.
            n_terms: Number of Fourier terms per period.
            include_original: Include original columns.
        """
        super().__init__()
        self.datetime_col = datetime_col
        self.periods = periods or {"yearly": 365.25, "weekly": 7, "monthly": 30.44}
        self.n_terms = n_terms
        self.include_original = include_original

        self._base_date: pd.Timestamp | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the generator.

        Args:
            X: Input DataFrame.
            y: Ignored.

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)

        if self.datetime_col not in X.columns:
            raise ValidationError(f"Datetime column '{self.datetime_col}' not found")

        dt = pd.to_datetime(X[self.datetime_col])
        self._base_date = dt.min()
        self._input_columns = list(X.columns)

        # Build feature names
        names = []
        if self.include_original:
            names.extend(X.columns.tolist())
        for period_name in self.periods:
            for k in range(1, self.n_terms + 1):
                names.append(f"fourier_{period_name}_sin_{k}")
                names.append(f"fourier_{period_name}_cos_{k}")
        self._feature_names_out = names
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate Fourier features.

        Args:
            X: Input DataFrame.

        Returns:
            DataFrame with Fourier features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result = X.copy() if self.include_original else pd.DataFrame(index=X.index)

        dt = pd.to_datetime(X[self.datetime_col])
        days_since_start = (dt - self._base_date).dt.total_seconds() / (24 * 3600)

        for period_name, period_days in self.periods.items():
            t = days_since_start / period_days
            for k in range(1, self.n_terms + 1):
                result[f"fourier_{period_name}_sin_{k}"] = np.sin(2 * np.pi * k * t)
                result[f"fourier_{period_name}_cos_{k}"] = np.cos(2 * np.pi * k * t)

        return result


def generate_timeseries_features(
    X: pd.DataFrame,
    datetime_col: str | None = None,
    target_cols: list[str] | None = None,
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, TimeSeriesFeatureGenerator]:
    """Convenience function for time-series feature generation.

    Args:
        X: Input DataFrame.
        datetime_col: Datetime column.
        target_cols: Columns for lag/rolling features.
        lags: Lag periods.
        rolling_windows: Rolling window sizes.
        **kwargs: Additional arguments.

    Returns:
        Tuple of (features DataFrame, fitted generator).
    """
    generator = TimeSeriesFeatureGenerator(
        datetime_col=datetime_col,
        target_cols=target_cols,
        lags=lags,
        rolling_windows=rolling_windows,
        **kwargs,
    )
    X_features = generator.fit_transform(X)
    return X_features, generator
