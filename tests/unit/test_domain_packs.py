"""Tests for IoT and Marketing feature packs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.packs.iot import IoTFeaturePack, MarketingFeaturePack


@pytest.fixture
def sensor_data() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "temperature": rng.randn(100) * 10 + 50,
        "pressure": rng.randn(100) * 5 + 100,
        "vibration": rng.randn(100) * 2,
    })


@pytest.fixture
def customer_data() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "total_revenue": rng.exponential(100, 200),
        "purchase_frequency": rng.poisson(5, 200).astype(float),
        "days_since_last_recency": rng.exponential(30, 200),
    })


class TestIoTFeaturePack:
    def test_fit_transform(self, sensor_data: pd.DataFrame) -> None:
        pack = IoTFeaturePack(sensor_columns=["temperature", "pressure"])
        result = pack.fit_transform(sensor_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sensor_data)
        assert result.shape[1] > 0

    def test_generates_rolling_features(self, sensor_data: pd.DataFrame) -> None:
        pack = IoTFeaturePack(
            sensor_columns=["temperature"],
            features=["rolling_mean", "rolling_std"],
        )
        result = pack.fit_transform(sensor_data)
        assert "iot_temperature_rolling_mean" in result.columns
        assert "iot_temperature_rolling_std" in result.columns

    def test_rate_of_change(self, sensor_data: pd.DataFrame) -> None:
        pack = IoTFeaturePack(
            sensor_columns=["temperature"], features=["rate_of_change"]
        )
        result = pack.fit_transform(sensor_data)
        assert "iot_temperature_rate_of_change" in result.columns

    def test_signal_energy(self, sensor_data: pd.DataFrame) -> None:
        pack = IoTFeaturePack(
            sensor_columns=["vibration"], features=["signal_energy"]
        )
        result = pack.fit_transform(sensor_data)
        col = "iot_vibration_signal_energy"
        assert col in result.columns
        assert (result[col] >= 0).all()

    def test_auto_detect_sensor_columns(self, sensor_data: pd.DataFrame) -> None:
        pack = IoTFeaturePack()
        result = pack.fit_transform(sensor_data)
        assert result.shape[1] > 0

    def test_custom_window_size(self, sensor_data: pd.DataFrame) -> None:
        pack = IoTFeaturePack(
            sensor_columns=["temperature"],
            features=["rolling_mean"],
            window_size=5,
        )
        result = pack.fit_transform(sensor_data)
        assert result.shape[1] == 1

    def test_get_sensor_stats(self, sensor_data: pd.DataFrame) -> None:
        pack = IoTFeaturePack(sensor_columns=["temperature"])
        pack.fit_transform(sensor_data)
        stats = pack.get_sensor_stats()
        assert "temperature" in stats
        assert "mean" in stats["temperature"]

    def test_available_features(self) -> None:
        pack = IoTFeaturePack()
        features = pack.available_features()
        assert len(features) >= 10
        assert "rolling_mean" in features

    def test_get_feature_names_out(self, sensor_data: pd.DataFrame) -> None:
        pack = IoTFeaturePack(sensor_columns=["temperature"])
        pack.fit(sensor_data)
        names = pack.get_feature_names_out()
        assert len(names) > 0
        assert all(n.startswith("iot_") for n in names)


class TestMarketingFeaturePack:
    def test_fit_transform(self, customer_data: pd.DataFrame) -> None:
        pack = MarketingFeaturePack(
            revenue_column="total_revenue",
            frequency_column="purchase_frequency",
            recency_column="days_since_last_recency",
        )
        result = pack.fit_transform(customer_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(customer_data)

    def test_rfm_score(self, customer_data: pd.DataFrame) -> None:
        pack = MarketingFeaturePack(
            revenue_column="total_revenue",
            frequency_column="purchase_frequency",
            recency_column="days_since_last_recency",
            features=["rfm_score"],
        )
        result = pack.fit_transform(customer_data)
        assert "marketing_rfm_score" in result.columns
        assert result["marketing_rfm_score"].between(0, 1).all()

    def test_monetary_rank(self, customer_data: pd.DataFrame) -> None:
        pack = MarketingFeaturePack(
            revenue_column="total_revenue",
            features=["monetary_rank"],
        )
        result = pack.fit_transform(customer_data)
        assert "marketing_monetary_rank" in result.columns

    def test_high_value_flag(self, customer_data: pd.DataFrame) -> None:
        pack = MarketingFeaturePack(
            revenue_column="total_revenue",
            features=["is_high_value"],
        )
        result = pack.fit_transform(customer_data)
        assert set(result["marketing_is_high_value"].unique()).issubset({0, 1})

    def test_revenue_per_visit(self, customer_data: pd.DataFrame) -> None:
        pack = MarketingFeaturePack(
            revenue_column="total_revenue",
            frequency_column="purchase_frequency",
            features=["revenue_per_visit"],
        )
        result = pack.fit_transform(customer_data)
        assert "marketing_revenue_per_visit" in result.columns

    def test_auto_detect_columns(self, customer_data: pd.DataFrame) -> None:
        pack = MarketingFeaturePack()
        result = pack.fit_transform(customer_data)
        assert result.shape[1] > 0

    def test_available_features(self) -> None:
        pack = MarketingFeaturePack()
        features = pack.available_features()
        assert "rfm_score" in features
        assert "engagement_score" in features
