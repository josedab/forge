"""Date/time component extraction generator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from forge.generators.base import BaseFeatureGenerator

if TYPE_CHECKING:
    from typing_extensions import Self


class DateTimeComponents(BaseFeatureGenerator):
    """Extracts components from datetime columns.

    Creates features like year, month, day, hour, minute, day of week
    and cyclical encodings for temporal patterns.

    Example:
        >>> gen = DateTimeComponents(
        ...     columns=["timestamp"]
        ...     components=["year", "month", "dayofweek", "hour"]
        ... )
        >>> X_time = gen.fit_transform(X)
    """

    SUPPORTED_COMPONENTS = [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "dayofweek",
        "dayofyear",
        "weekofyear",
        "quarter",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "is_quarter_start",
        "is_quarter_end",
        "is_year_start",
        "is_year_end",
    ]

    def __init__(
        self,
        columns: list[str] | None = None,
        components: list[str] | None = None,
        cyclical_encode: bool = True,
        drop_original: bool = True
    ) -> None:
        """Initialize the datetime components generator.

        Args:
            columns: DateTime columns to process. If None, auto-detect.,
            components: Components to extract. Default: common components.,
            cyclical_encode: Whether to add cyclical (sin/cos) encodings.,
            drop_original: Whether to exclude original datetime columns.
        """
        super().__init__()
        self.columns = columns
        self.components = components or [
            "year",
            "month",
            "day",
            "dayofweek",
            "hour",
            "is_weekend",
        ]
        self.cyclical_encode = cyclical_encode
        self.drop_original = drop_original

        # Validate components,
        invalid = set(self.components) - set(self.SUPPORTED_COMPONENTS)
        if invalid:
            raise ValueError(f"Unsupported components: {invalid}")

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the generator.

        Args:
            X: Input DataFrame.,
            y: Ignored.,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)

        if self.columns is None:
            cols = []
            for col in X.columns:
                if pd.api.types.is_datetime64_any_dtype(X[col]):
                    cols.append(col)
                elif X[col].dtype == object:
                    # Try to parse as datetime
                    try:
                        pd.to_datetime(X[col].head(100))
                        cols.append(col)
                    except (ValueError, TypeError):
                        pass
        else:
            cols = self._validate_columns(X, self.columns)

        self._columns_fitted = cols
        self._input_columns = list(X.columns)
        self._feature_names_out = []

        # Build feature names
        for col in cols:
            for comp in self.components:
                self._feature_names_out.append(f"{col}_{comp}")

            if self.cyclical_encode:
                # Add cyclical features for periodic components,
                cyclical_comps = {
                    "month": 12,
                    "day": 31,
                    "hour": 24,
                    "minute": 60,
                    "dayofweek": 7,
                    "dayofyear": 365
                }
                for comp in self.components:
                    if comp in cyclical_comps:
                        self._feature_names_out.append(f"{col}_{comp}_sin")
                        self._feature_names_out.append(f"{col}_{comp}_cos")

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract datetime components.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with extracted components.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result_data: dict[str, np.ndarray] = {}

        for col in self._columns_fitted:
            # Convert to datetime if needed
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                dt = X[col]
            else:
                dt = pd.to_datetime(X[col], errors="coerce")

            dt_accessor = dt.dt

            for comp in self.components:
                feature_name = f"{col}_{comp}"

                if comp == "year":
                    values = dt_accessor.year
                elif comp == "month":
                    values = dt_accessor.month
                elif comp == "day":
                    values = dt_accessor.day
                elif comp == "hour":
                    values = dt_accessor.hour
                elif comp == "minute":
                    values = dt_accessor.minute
                elif comp == "second":
                    values = dt_accessor.second
                elif comp == "dayofweek":
                    values = dt_accessor.dayofweek
                elif comp == "dayofyear":
                    values = dt_accessor.dayofyear
                elif comp == "weekofyear":
                    values = dt_accessor.isocalendar().week
                elif comp == "quarter":
                    values = dt_accessor.quarter
                elif comp == "is_weekend":
                    values = (dt_accessor.dayofweek >= 5).astype(int)
                elif comp == "is_month_start":
                    values = dt_accessor.is_month_start.astype(int)
                elif comp == "is_month_end":
                    values = dt_accessor.is_month_end.astype(int)
                elif comp == "is_quarter_start":
                    values = dt_accessor.is_quarter_start.astype(int)
                elif comp == "is_quarter_end":
                    values = dt_accessor.is_quarter_end.astype(int)
                elif comp == "is_year_start":
                    values = dt_accessor.is_year_start.astype(int)
                elif comp == "is_year_end":
                    values = dt_accessor.is_year_end.astype(int)
                else:
                    continue

                result_data[feature_name] = values.values

            # Add cyclical encodings
            if self.cyclical_encode:
                cyclical_comps = {
                    "month": (dt_accessor.month, 12),
                    "day": (dt_accessor.day, 31),
                    "hour": (dt_accessor.hour, 24),
                    "minute": (dt_accessor.minute, 60),
                    "dayofweek": (dt_accessor.dayofweek, 7),
                    "dayofyear": (dt_accessor.dayofyear, 365),
                }

                for comp in self.components:
                    if comp in cyclical_comps:
                        values, period = cyclical_comps[comp]
                        values_arr = values.values.astype(float)

                        sin_vals = np.sin(2 * np.pi * values_arr / period)
                        cos_vals = np.cos(2 * np.pi * values_arr / period)

                        result_data[f"{col}_{comp}_sin"] = sin_vals
                        result_data[f"{col}_{comp}_cos"] = cos_vals

        return pd.DataFrame(result_data, index=X.index)


class HolidayFeatures(BaseFeatureGenerator):
    """Generates holiday-related features.

    Creates binary indicators for holidays and days around holidays.
    Supports US holidays by default.

    Example:
        >>> gen = HolidayFeatures(columns=["date"], country="US")
        >>> X_holidays = gen.fit_transform(X)
    """

    # US Federal holidays (simplified)
    US_HOLIDAYS = {
        (1, 1): "new_years",
        (7, 4): "independence_day",
        (12, 25): "christmas",
        (12, 31): "new_years_eve"
    }

    def __init__(
        self,
        columns: list[str] | None = None,
        country: str = "US",
        days_before: int = 2,
        days_after: int = 2
    ) -> None:
        """Initialize holiday feature generator.

        Args:
            columns: DateTime columns. If None, auto-detect.,
            country: Country for holidays (currently only "US").,
            days_before: Days before holiday to flag.,
            days_after: Days after holiday to flag.
        """
        super().__init__()
        self.columns = columns
        self.country = country
        self.days_before = days_before
        self.days_after = days_after

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the generator.

        Args:
            X: Input DataFrame.,
            y: Ignored.,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)

        if self.columns is None:
            cols = X.select_dtypes(include=["datetime64"]).columns.tolist()
        else:
            cols = self._validate_columns(X, self.columns)

        self._columns_fitted = cols
        self._input_columns = list(X.columns)

        # Build feature names,
        self._feature_names_out = []
        for col in cols:
            self._feature_names_out.extend([
                f"{col}_is_holiday"
                f"{col}_days_to_holiday"
                f"{col}_days_from_holiday"
                f"{col}_near_holiday"
            ])

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate holiday features.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with holiday features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result_data: dict[str, np.ndarray] = {}

        for col in self._columns_fitted:
            dt = pd.to_datetime(X[col], errors="coerce")

            # Check if date is a holiday,
            month_day = list(zip(dt.dt.month, dt.dt.day))
            is_holiday = np.array([
                1 if (m, d) in self.US_HOLIDAYS else 0
                for m, d in month_day
            ])

            result_data[f"{col}_is_holiday"] = is_holiday
            result_data[f"{col}_days_to_holiday"] = np.zeros(len(X))
            result_data[f"{col}_days_from_holiday"] = np.zeros(len(X))
            result_data[f"{col}_near_holiday"] = (
                is_holiday |
                (np.roll(is_holiday, self.days_before) == 1) |
                (np.roll(is_holiday, -self.days_after) == 1)
            ).astype(int)

        return pd.DataFrame(result_data, index=X.index)
