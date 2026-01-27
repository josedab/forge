"""IoT domain feature pack — sensor and device data features.

Generates features commonly used in IoT, industrial monitoring,
predictive maintenance, and sensor data analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forge.packs.base import FeaturePack


class IoTFeaturePack(FeaturePack):
    """Feature pack for IoT and sensor data.

    Generates features from time-series sensor readings including
    signal statistics, change detection, operating range indicators,
    and degradation patterns.

    Args:
        features: Specific features to generate. None for all.
        prefix: Prefix for generated feature names.
        sensor_columns: Columns containing sensor readings.
        timestamp_column: Column containing timestamps (optional).
        window_size: Rolling window size for temporal features.

    Example:
        >>> pack = IoTFeaturePack(sensor_columns=["temp", "pressure", "vibration"])
        >>> X_features = pack.fit_transform(sensor_data)
    """

    domain = "iot"

    def __init__(
        self,
        features: list[str] | None = None,
        prefix: str | None = None,
        sensor_columns: list[str] | None = None,
        timestamp_column: str | None = None,
        window_size: int = 10,
    ) -> None:
        super().__init__(features=features, prefix=prefix)
        self.sensor_columns = sensor_columns
        self.timestamp_column = timestamp_column
        self.window_size = window_size
        self._sensor_stats: dict[str, dict[str, float]] = {}

    def available_features(self) -> list[str]:
        """Return available IoT feature names."""
        return [
            "rolling_mean", "rolling_std", "rolling_min", "rolling_max",
            "rate_of_change", "abs_deviation", "signal_energy",
            "zero_crossing_rate", "peak_count", "operating_range",
            "signal_to_noise", "cumulative_sum",
        ]

    def _generate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate IoT features from sensor data."""
        sensor_cols = self.sensor_columns or list(
            X.select_dtypes(include=[np.number]).columns
        )

        selected = self.features or self.available_features()
        result = pd.DataFrame(index=X.index)

        for col in sensor_cols:
            if col not in X.columns:
                continue

            series = X[col].astype(float)

            if "rolling_mean" in selected:
                result[f"{self.prefix}{col}_rolling_mean"] = (
                    series.rolling(self.window_size, min_periods=1).mean()
                )

            if "rolling_std" in selected:
                result[f"{self.prefix}{col}_rolling_std"] = (
                    series.rolling(self.window_size, min_periods=1).std().fillna(0)
                )

            if "rolling_min" in selected:
                result[f"{self.prefix}{col}_rolling_min"] = (
                    series.rolling(self.window_size, min_periods=1).min()
                )

            if "rolling_max" in selected:
                result[f"{self.prefix}{col}_rolling_max"] = (
                    series.rolling(self.window_size, min_periods=1).max()
                )

            if "rate_of_change" in selected:
                result[f"{self.prefix}{col}_rate_of_change"] = series.diff().fillna(0)

            if "abs_deviation" in selected:
                mean = series.mean()
                result[f"{self.prefix}{col}_abs_deviation"] = (series - mean).abs()

            if "signal_energy" in selected:
                result[f"{self.prefix}{col}_signal_energy"] = (
                    (series ** 2).rolling(self.window_size, min_periods=1).mean()
                )

            if "zero_crossing_rate" in selected:
                signs = np.sign(series - series.mean())
                crossings = (signs.diff().abs() > 0).astype(float)
                result[f"{self.prefix}{col}_zero_crossing_rate"] = (
                    crossings.rolling(self.window_size, min_periods=1).mean()
                )

            if "operating_range" in selected:
                result[f"{self.prefix}{col}_operating_range"] = (
                    series.rolling(self.window_size, min_periods=1).max()
                    - series.rolling(self.window_size, min_periods=1).min()
                )

            if "signal_to_noise" in selected:
                rm = series.rolling(self.window_size, min_periods=1).mean()
                rs = series.rolling(self.window_size, min_periods=1).std().fillna(1)
                result[f"{self.prefix}{col}_snr"] = rm / rs.replace(0, 1)

            if "cumulative_sum" in selected:
                result[f"{self.prefix}{col}_cumsum"] = series.cumsum()

            # Store sensor statistics for reference
            self._sensor_stats[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()) if series.std() == series.std() else 0.0,
                "min": float(series.min()),
                "max": float(series.max()),
            }

        return result

    def get_sensor_stats(self) -> dict[str, dict[str, float]]:
        """Get computed sensor statistics."""
        return dict(self._sensor_stats)


class MarketingFeaturePack(FeaturePack):
    """Feature pack for marketing and customer analytics.

    Generates features for customer behavior analysis, campaign
    effectiveness, and engagement metrics.

    Args:
        features: Specific features to generate. None for all.
        prefix: Prefix for generated feature names.
        revenue_column: Column containing revenue/purchase amounts.
        frequency_column: Column containing purchase/visit frequency.
        recency_column: Column containing days since last interaction.
        segment_column: Column containing customer segments.

    Example:
        >>> pack = MarketingFeaturePack(
        ...     revenue_column="total_spend",
        ...     frequency_column="n_purchases",
        ...     recency_column="days_since_last",
        ... )
        >>> X_features = pack.fit_transform(customer_data)
    """

    domain = "marketing"

    def __init__(
        self,
        features: list[str] | None = None,
        prefix: str | None = None,
        revenue_column: str | None = None,
        frequency_column: str | None = None,
        recency_column: str | None = None,
        segment_column: str | None = None,
    ) -> None:
        super().__init__(features=features, prefix=prefix)
        self.revenue_column = revenue_column
        self.frequency_column = frequency_column
        self.recency_column = recency_column
        self.segment_column = segment_column
        self._rfm_quantiles: dict[str, list[float]] = {}

    def available_features(self) -> list[str]:
        """Return available marketing feature names."""
        return [
            "rfm_score", "monetary_rank", "frequency_rank",
            "recency_rank", "revenue_per_visit", "engagement_score",
            "log_revenue", "is_high_value", "revenue_zscore",
        ]

    def _generate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate marketing features."""
        selected = self.features or self.available_features()
        result = pd.DataFrame(index=X.index)

        revenue = self._get_column(X, self.revenue_column, "revenue")
        frequency = self._get_column(X, self.frequency_column, "frequency")
        recency = self._get_column(X, self.recency_column, "recency")

        if revenue is not None:
            if "monetary_rank" in selected:
                result[f"{self.prefix}monetary_rank"] = revenue.rank(pct=True)

            if "log_revenue" in selected:
                result[f"{self.prefix}log_revenue"] = np.log1p(revenue.clip(lower=0))

            if "is_high_value" in selected:
                threshold = revenue.quantile(0.75)
                result[f"{self.prefix}is_high_value"] = (revenue >= threshold).astype(int)

            if "revenue_zscore" in selected:
                std = revenue.std()
                if std > 0:
                    result[f"{self.prefix}revenue_zscore"] = (
                        (revenue - revenue.mean()) / std
                    )
                else:
                    result[f"{self.prefix}revenue_zscore"] = 0.0

        if frequency is not None:
            if "frequency_rank" in selected:
                result[f"{self.prefix}frequency_rank"] = frequency.rank(pct=True)

        if recency is not None:
            if "recency_rank" in selected:
                result[f"{self.prefix}recency_rank"] = (
                    1 - recency.rank(pct=True)  # Inverse: lower recency = higher rank
                )

        if revenue is not None and frequency is not None:
            if "revenue_per_visit" in selected:
                safe_freq = frequency.replace(0, 1)
                result[f"{self.prefix}revenue_per_visit"] = revenue / safe_freq

        # RFM score
        if "rfm_score" in selected:
            rfm_parts: list[pd.Series] = []
            if revenue is not None:
                rfm_parts.append(revenue.rank(pct=True))
            if frequency is not None:
                rfm_parts.append(frequency.rank(pct=True))
            if recency is not None:
                rfm_parts.append(1 - recency.rank(pct=True))
            if rfm_parts:
                rfm = sum(rfm_parts) / len(rfm_parts)  # type: ignore[arg-type]
                result[f"{self.prefix}rfm_score"] = rfm

        # Engagement score (composite)
        if "engagement_score" in selected and frequency is not None:
            eng = frequency.rank(pct=True)
            if recency is not None:
                eng = eng * (1 - recency.rank(pct=True))
            result[f"{self.prefix}engagement_score"] = eng

        return result

    def _get_column(
        self, X: pd.DataFrame, explicit: str | None, keyword: str
    ) -> pd.Series | None:
        """Get a column by explicit name or keyword search."""
        if explicit and explicit in X.columns:
            return X[explicit].astype(float)

        # Try to find by keyword
        for col in X.columns:
            if keyword.lower() in col.lower():
                return X[col].astype(float)
        return None
