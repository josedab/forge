"""Tests for domain-specific feature packs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.packs import (
    EcommerceFeaturePack,
    FeaturePackRegistry,
    FinanceFeaturePack,
    GeospatialFeaturePack,
    HealthcareFeaturePack,
)


class TestFinanceFeaturePack:
    @pytest.fixture
    def stock_data(self):
        rng = np.random.RandomState(42)
        n = 100
        return pd.DataFrame({
            "close": 100 + rng.randn(n).cumsum(),
            "volume": rng.randint(1000, 10000, n).astype(float),
        })

    def test_basic_generation(self, stock_data):
        pack = FinanceFeaturePack(windows=[5, 10])
        result = pack.fit_transform(stock_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(stock_data)
        assert any("sma" in c for c in result.columns)
        assert any("rsi" in c for c in result.columns)

    def test_selected_features(self, stock_data):
        pack = FinanceFeaturePack(features=["returns", "sma"], windows=[5])
        result = pack.fit_transform(stock_data)
        assert any("returns" in c for c in result.columns)
        assert any("sma" in c for c in result.columns)
        assert not any("rsi" in c for c in result.columns)

    def test_volume_features(self, stock_data):
        pack = FinanceFeaturePack(features=["vwap", "obv"])
        result = pack.fit_transform(stock_data)
        assert any("vwap" in c for c in result.columns)
        assert any("obv" in c for c in result.columns)

    def test_custom_columns(self, stock_data):
        data = stock_data.rename(columns={"close": "px", "volume": "vol"})
        pack = FinanceFeaturePack(
            price_col="px", volume_col="vol", features=["returns"]
        )
        result = pack.fit_transform(data)
        assert len(result.columns) >= 1

    def test_available_features(self):
        pack = FinanceFeaturePack()
        available = pack.available_features()
        assert "rsi" in available
        assert "macd" in available
        assert len(available) >= 15

    def test_get_feature_names_out(self, stock_data):
        pack = FinanceFeaturePack(features=["returns"], windows=[5])
        pack.fit(stock_data)
        names = pack.get_feature_names_out()
        assert len(names) >= 1


class TestHealthcareFeaturePack:
    @pytest.fixture
    def patient_data(self):
        rng = np.random.RandomState(42)
        n = 50
        return pd.DataFrame({
            "age": rng.randint(20, 90, n),
            "weight": rng.uniform(50, 120, n),
            "height": rng.uniform(150, 200, n),  # cm
            "systolic": rng.randint(90, 200, n),
            "diastolic": rng.randint(60, 120, n),
            "lab_glucose": rng.uniform(70, 300, n),
            "lab_cholesterol": rng.uniform(100, 350, n),
        })

    def test_bmi_calculation(self, patient_data):
        pack = HealthcareFeaturePack(features=["bmi"])
        result = pack.fit_transform(patient_data)
        assert any("bmi" in c for c in result.columns)
        # BMI should be in reasonable range
        bmi_col = [c for c in result.columns if "bmi" in c][0]
        assert result[bmi_col].median() > 15
        assert result[bmi_col].median() < 50

    def test_age_features(self, patient_data):
        pack = HealthcareFeaturePack(features=["age_group", "is_elderly", "age_decades"])
        result = pack.fit_transform(patient_data)
        assert any("age_group" in c for c in result.columns)
        assert any("is_elderly" in c for c in result.columns)

    def test_bp_features(self, patient_data):
        pack = HealthcareFeaturePack(features=["pulse_pressure", "map_pressure"])
        result = pack.fit_transform(patient_data)
        assert any("pulse_pressure" in c for c in result.columns)
        assert any("map_pressure" in c for c in result.columns)

    def test_lab_features(self, patient_data):
        pack = HealthcareFeaturePack(features=["lab_zscore", "lab_critical_flag"])
        result = pack.fit_transform(patient_data)
        assert any("zscore" in c for c in result.columns)
        assert any("critical" in c for c in result.columns)

    def test_risk_score(self, patient_data):
        pack = HealthcareFeaturePack(features=["risk_score_simple", "bmi"])
        result = pack.fit_transform(patient_data)
        assert any("risk_score" in c for c in result.columns)


class TestEcommerceFeaturePack:
    @pytest.fixture
    def transaction_data(self):
        rng = np.random.RandomState(42)
        n = 100
        return pd.DataFrame({
            "customer_id": rng.choice(["C1", "C2", "C3", "C4", "C5"], n),
            "amount": rng.uniform(10, 500, n),
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "product_id": rng.choice(["P1", "P2", "P3", "P4"], n),
        })

    def test_basic_generation(self, transaction_data):
        pack = EcommerceFeaturePack()
        result = pack.fit_transform(transaction_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(transaction_data)

    def test_rfm_features(self, transaction_data):
        pack = EcommerceFeaturePack(features=["total_spend", "order_count", "days_since_last"])
        result = pack.fit_transform(transaction_data)
        assert any("total_spend" in c for c in result.columns)
        assert any("order_count" in c for c in result.columns)

    def test_product_features(self, transaction_data):
        pack = EcommerceFeaturePack(features=["unique_products", "basket_size"])
        result = pack.fit_transform(transaction_data)
        assert any("unique_products" in c for c in result.columns)

    def test_repeat_customer(self, transaction_data):
        pack = EcommerceFeaturePack(features=["is_repeat_customer"])
        result = pack.fit_transform(transaction_data)
        assert any("repeat" in c for c in result.columns)


class TestGeospatialFeaturePack:
    @pytest.fixture
    def location_data(self):
        rng = np.random.RandomState(42)
        n = 50
        return pd.DataFrame({
            "latitude": rng.uniform(40.5, 41.0, n),
            "longitude": rng.uniform(-74.2, -73.7, n),
        })

    def test_basic_generation(self, location_data):
        pack = GeospatialFeaturePack()
        result = pack.fit_transform(location_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(location_data)

    def test_cyclical_encoding(self, location_data):
        pack = GeospatialFeaturePack(features=["lat_sin", "lat_cos", "lon_sin", "lon_cos"])
        result = pack.fit_transform(location_data)
        assert any("lat_sin" in c for c in result.columns)
        # Sin values should be between -1 and 1
        sin_col = [c for c in result.columns if "lat_sin" in c][0]
        assert result[sin_col].max() <= 1.0
        assert result[sin_col].min() >= -1.0

    def test_distance_to_reference(self, location_data):
        pack = GeospatialFeaturePack(
            features=["haversine_distance"],
            reference_points={"nyc": (40.7128, -74.0060)},
        )
        result = pack.fit_transform(location_data)
        dist_col = [c for c in result.columns if "dist" in c][0]
        assert result[dist_col].min() >= 0

    def test_spatial_clustering(self, location_data):
        pack = GeospatialFeaturePack(features=["spatial_cluster"], n_clusters=3)
        result = pack.fit_transform(location_data)
        assert any("cluster" in c for c in result.columns)
        cluster_col = [c for c in result.columns if "cluster" in c][0]
        assert result[cluster_col].nunique() == 3

    def test_cartesian_coords(self, location_data):
        pack = GeospatialFeaturePack(features=["cartesian_x", "cartesian_y", "cartesian_z"])
        result = pack.fit_transform(location_data)
        assert any("cartesian_x" in c for c in result.columns)


class TestFeaturePackRegistry:
    def test_register_and_create(self):
        registry = FeaturePackRegistry()
        registry.register("finance", FinanceFeaturePack)
        pack = registry.create("finance")
        assert isinstance(pack, FinanceFeaturePack)

    def test_list_packs(self):
        registry = FeaturePackRegistry()
        registry.register("finance", FinanceFeaturePack)
        registry.register("healthcare", HealthcareFeaturePack)
        assert set(registry.list_packs()) == {"finance", "healthcare"}

    def test_unknown_pack(self):
        registry = FeaturePackRegistry()
        with pytest.raises(Exception, match="Unknown pack"):
            registry.create("nonexistent")
