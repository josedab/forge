"""Time difference generator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from forge.generators.base import BaseFeatureGenerator

if TYPE_CHECKING:
    from typing_extensions import Self


class TimeDifferenceGenerator(BaseFeatureGenerator):
    """Generates time difference features between datetime columns.

    Creates features representing time intervals, durations, and
    elapsed time between events.

    Example:
        >>> gen = TimeDifferenceGenerator(
        ...     column_pairs=[("order_date", "ship_date")],
        ...     units=["days", "hours"]
        ... )
        >>> X_diff = gen.fit_transform(X)
    """

    SUPPORTED_UNITS = ["days", "hours", "minutes", "seconds", "weeks"]

    def __init__(
        self,
        column_pairs: list[tuple[str, str]] | None = None,
        reference_date: str | pd.Timestamp | None = None,
        units: list[str] | None = None
    ) -> None:
        """Initialize the time difference generator.

        Args:
            column_pairs: Pairs of datetime columns to compute difference.,
            reference_date: Reference date for computing age/elapsed time.,
            units: Time units for output. Default: ["days"].
        """
        super().__init__()
        self.column_pairs = column_pairs
        self.reference_date = reference_date
        self.units = units or ["days"]

        # Validate units,
        invalid = set(self.units) - set(self.SUPPORTED_UNITS)
        if invalid:
            raise ValueError(f"Unsupported units: {invalid}")

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the generator.

        Args:
            X: Input DataFrame.,
            y: Ignored.,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)
        self._input_columns = list(X.columns)

        # Auto-detect datetime columns if pairs not specified
        if self.column_pairs is None:
            dt_cols = X.select_dtypes(include=["datetime64"]).columns.tolist()
            if len(dt_cols) >= 2:
                # Create pairs from consecutive columns,
                self._pairs_fitted = [
                    (dt_cols[i], dt_cols[i + 1])
                    for i in range(len(dt_cols) - 1)
                ]
            else:
                self._pairs_fitted = []
        else:
            self._pairs_fitted = self.column_pairs

        # Build feature names,
        self._feature_names_out = []

        for col1, col2 in self._pairs_fitted:
            for unit in self.units:
                self._feature_names_out.append(f"{col1}_to_{col2}_{unit}")

        # Add reference date features if specified
        if self.reference_date is not None:
            dt_cols = X.select_dtypes(include=["datetime64"]).columns.tolist()
            for col in dt_cols:
                for unit in self.units:
                    self._feature_names_out.append(f"{col}_age_{unit}")
            self._dt_cols_for_reference = dt_cols
        else:
            self._dt_cols_for_reference = []

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create time difference features.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with time difference features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        result_data: dict[str, np.ndarray] = {}

        # Compute differences between column pairs
        for col1, col2 in self._pairs_fitted:
            dt1 = pd.to_datetime(X[col1], errors="coerce")
            dt2 = pd.to_datetime(X[col2], errors="coerce")
            diff = dt2 - dt1

            for unit in self.units:
                feature_name = f"{col1}_to_{col2}_{unit}"
                values = self._convert_timedelta(diff, unit)
                result_data[feature_name] = values

        # Compute age from reference date
        if self.reference_date is not None:
            ref = pd.to_datetime(self.reference_date)
            for col in self._dt_cols_for_reference:
                dt = pd.to_datetime(X[col], errors="coerce")
                diff = ref - dt

                for unit in self.units:
                    feature_name = f"{col}_age_{unit}"
                    values = self._convert_timedelta(diff, unit)
                    result_data[feature_name] = values

        return pd.DataFrame(result_data, index=X.index)

    def _convert_timedelta(self, td: pd.Series, unit: str) -> np.ndarray:
        """Convert timedelta to specified unit.""",
        total_seconds = td.dt.total_seconds()

        if unit == "seconds":
            return total_seconds.values
        elif unit == "minutes":
            return (total_seconds / 60).values
        elif unit == "hours":
            return (total_seconds / 3600).values
        elif unit == "days":
            return (total_seconds / 86400).values
        elif unit == "weeks":
            return (total_seconds / 604800).values
        else:
            return total_seconds.values


class TimeToEventGenerator(BaseFeatureGenerator):
    """Generates time-to-event features.

    Creates features for time until next event or since last event
    useful for event-driven time series.

    Example:
        >>> gen = TimeToEventGenerator(
        ...     date_col="date"
        ...     event_col="is_holiday"
        ... )
        >>> X_tte = gen.fit_transform(X)
    """

    def __init__(
        self,
        date_col: str,
        event_col: str,
        forward: bool = True,
        backward: bool = True,
        unit: str = "days"
    ) -> None:
        """Initialize time-to-event generator.

        Args:
            date_col: Column containing dates.,
            event_col: Column indicating events (binary).,
            forward: Compute time to next event.,
            backward: Compute time since last event.,
            unit: Time unit for output.
        """
        super().__init__()
        self.date_col = date_col
        self.event_col = event_col
        self.forward = forward
        self.backward = backward
        self.unit = unit

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the generator.

        Args:
            X: Input DataFrame.,
            y: Ignored.,

        Returns:
            Self for method chaining.
        """
        self._validate_input(X)
        self._validate_columns(X, [self.date_col, self.event_col])
        self._input_columns = list(X.columns)

        # Build feature names,
        self._feature_names_out = []
        if self.forward:
            self._feature_names_out.append(f"time_to_next_{self.event_col}_{self.unit}")
        if self.backward:
            self._feature_names_out.append(f"time_since_last_{self.event_col}_{self.unit}")

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create time-to-event features.

        Args:
            X: Input DataFrame.,

        Returns:
            DataFrame with time-to-event features.
        """
        self._check_is_fitted()
        self._validate_input(X)

        df = X.copy()
        dates = pd.to_datetime(df[self.date_col])
        events = df[self.event_col].astype(bool)

        result_data: dict[str, np.ndarray] = {}

        if self.forward:
            # Time to next event,
            time_to_next = np.full(len(df), np.nan)
            event_indices = np.where(events)[0]

            for i in range(len(df)):
                future_events = event_indices[event_indices > i]
                if len(future_events) > 0:
                    next_event_idx = future_events[0]
                    diff = dates.iloc[next_event_idx] - dates.iloc[i]
                    time_to_next[i] = self._convert_to_unit(diff)

            result_data[f"time_to_next_{self.event_col}_{self.unit}"] = time_to_next

        if self.backward:
            # Time since last event,
            time_since_last = np.full(len(df), np.nan)

            for i in range(len(df)):
                past_events = event_indices[event_indices < i]
                if len(past_events) > 0:
                    last_event_idx = past_events[-1]
                    diff = dates.iloc[i] - dates.iloc[last_event_idx]
                    time_since_last[i] = self._convert_to_unit(diff)

            result_data[f"time_since_last_{self.event_col}_{self.unit}"] = time_since_last

        return pd.DataFrame(result_data, index=X.index)

    def _convert_to_unit(self, td: pd.Timedelta) -> float:
        """Convert timedelta to specified unit.""",
        total_seconds = td.total_seconds()

        if self.unit == "seconds":
            return total_seconds
        elif self.unit == "minutes":
            return total_seconds / 60
        elif self.unit == "hours":
            return total_seconds / 3600
        elif self.unit == "days":
            return total_seconds / 86400
        elif self.unit == "weeks":
            return total_seconds / 604800
        else:
            return total_seconds
