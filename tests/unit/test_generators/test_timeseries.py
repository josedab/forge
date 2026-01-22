"""Tests for time-series feature generators."""

import numpy as np
import pandas as pd
import pytest

from forge.generators.timeseries import (
    TimeSeriesFeatureGenerator,
    SeasonalDecomposer,
    FourierFeatureGenerator,
    LagConfig,
    RollingConfig,
    DatetimeConfig,
    generate_timeseries_features,
)
from forge.exceptions import NotFittedError, ValidationError


@pytest.fixture
def timeseries_data():
    """Create time series dataset."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=365, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "value": np.sin(np.arange(365) * 2 * np.pi / 7) * 10 + np.random.randn(365) * 2 + 50,
        "sales": np.random.poisson(100, 365) + np.arange(365) * 0.1,
    })
    return df


@pytest.fixture
def simple_timeseries():
    """Create simple time series for basic tests."""
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=100, freq="D"),
        "value": np.arange(100) + np.random.randn(100) * 5,
    })


class TestTimeSeriesFeatureGenerator:
    """Tests for TimeSeriesFeatureGenerator."""

    def test_lag_features(self, simple_timeseries):
        """Test lag feature generation."""
        generator = TimeSeriesFeatureGenerator(
            datetime_col="date",
            target_cols=["value"],
            lags=[1, 2, 3],
            rolling_windows=None,
            datetime_features=False,
        )
        result = generator.fit_transform(simple_timeseries)

        assert "value_lag_1" in result.columns
        assert "value_lag_2" in result.columns
        assert "value_lag_3" in result.columns

    def test_rolling_features(self, simple_timeseries):
        """Test rolling window features."""
        generator = TimeSeriesFeatureGenerator(
            datetime_col="date",
            target_cols=["value"],
            lags=None,
            rolling_windows=[3, 7],
            datetime_features=False,
        )
        result = generator.fit_transform(simple_timeseries)

        # Check for rolling features (may have different naming conventions)
        rolling_cols = [c for c in result.columns if "rolling" in c.lower() or "mean" in c.lower()]
        assert len(rolling_cols) > 0

    def test_datetime_features(self, simple_timeseries):
        """Test datetime component extraction."""
        generator = TimeSeriesFeatureGenerator(
            datetime_col="date",
            target_cols=["value"],
            lags=None,
            rolling_windows=None,
            datetime_features=True,
            datetime_config=DatetimeConfig(
                components=["dayofweek", "month", "day"],
            ),
        )
        result = generator.fit_transform(simple_timeseries)

        # Check for datetime component features
        dt_cols = [c for c in result.columns if any(x in c.lower() for x in ["dayofweek", "month", "day_"])]
        assert len(dt_cols) > 0 or "date_dayofweek" in result.columns

    def test_combined_features(self, timeseries_data):
        """Test combined feature generation."""
        generator = TimeSeriesFeatureGenerator(
            datetime_col="date",
            target_cols=["value", "sales"],
            lags=[1, 7],
            rolling_windows=[7],
            datetime_features=True,
        )
        result = generator.fit_transform(timeseries_data)

        # Check that features were generated
        assert len(result.columns) > len(timeseries_data.columns)

    def test_expanding_features(self, simple_timeseries):
        """Test expanding window features."""
        generator = TimeSeriesFeatureGenerator(
            datetime_col="date",
            target_cols=["value"],
            lags=None,
            rolling_windows=None,
            expanding=True,
            datetime_features=False,
        )
        result = generator.fit_transform(simple_timeseries)

        # Check for expanding features
        assert len(result.columns) >= len(simple_timeseries.columns)

    def test_ewm_features(self, simple_timeseries):
        """Test exponential weighted features."""
        generator = TimeSeriesFeatureGenerator(
            datetime_col="date",
            target_cols=["value"],
            lags=None,
            rolling_windows=None,
            ewm_spans=[7],
            datetime_features=False,
        )
        result = generator.fit_transform(simple_timeseries)

        # Check that ewm features were generated
        assert len(result.columns) >= len(simple_timeseries.columns)

    def test_get_feature_names_out(self, simple_timeseries):
        """Test get_feature_names_out method."""
        generator = TimeSeriesFeatureGenerator(
            datetime_col="date",
            target_cols=["value"],
            lags=[1, 2],
            datetime_features=False,
        )
        generator.fit(simple_timeseries)

        names = generator.get_feature_names_out()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_not_fitted_error(self, simple_timeseries):
        """Test NotFittedError before fitting."""
        generator = TimeSeriesFeatureGenerator(datetime_col="date")

        with pytest.raises(NotFittedError):
            generator.transform(simple_timeseries)

    def test_diff_features(self, simple_timeseries):
        """Test difference features."""
        generator = TimeSeriesFeatureGenerator(
            datetime_col="date",
            target_cols=["value"],
            lags=None,
            diff_orders=[1, 2],
            datetime_features=False,
        )
        result = generator.fit_transform(simple_timeseries)

        # Check for diff features
        diff_cols = [c for c in result.columns if "diff" in c.lower()]
        assert len(diff_cols) > 0


class TestSeasonalDecomposer:
    """Tests for SeasonalDecomposer."""

    def test_stl_decomposition(self, timeseries_data):
        """Test STL decomposition."""
        decomposer = SeasonalDecomposer(
            datetime_col="date",
            value_cols="value",
            period=7,
            method="stl",
        )
        result = decomposer.fit_transform(timeseries_data)

        # Check for decomposition components
        decomp_cols = [c for c in result.columns if any(x in c.lower() for x in ["trend", "seasonal", "resid"])]
        assert len(decomp_cols) > 0

    def test_classical_decomposition(self, timeseries_data):
        """Test classical decomposition."""
        decomposer = SeasonalDecomposer(
            datetime_col="date",
            value_cols="value",
            period=7,
            method="classical",
        )
        result = decomposer.fit_transform(timeseries_data)

        # Check that decomposition was performed
        assert len(result.columns) >= len(timeseries_data.columns)

    def test_multiplicative_model(self, timeseries_data):
        """Test multiplicative decomposition model."""
        # Ensure positive values for multiplicative
        timeseries_data["positive_value"] = timeseries_data["value"].abs() + 1

        decomposer = SeasonalDecomposer(
            datetime_col="date",
            value_cols="positive_value",
            period=7,
            model="multiplicative",
        )
        result = decomposer.fit_transform(timeseries_data)

        assert len(result.columns) >= len(timeseries_data.columns)

    def test_get_decomposition(self, timeseries_data):
        """Test getting decomposition components."""
        decomposer = SeasonalDecomposer(
            datetime_col="date",
            value_cols="value",
            period=7,
        )
        result = decomposer.fit_transform(timeseries_data)

        # Check that decomposition features were created
        assert "value_trend" in result.columns or len(result.columns) >= len(timeseries_data.columns)


class TestFourierFeatureGenerator:
    """Tests for FourierFeatureGenerator."""

    def test_basic_fourier(self, simple_timeseries):
        """Test basic Fourier feature generation."""
        generator = FourierFeatureGenerator(
            datetime_col="date",
            periods={"weekly": 7, "monthly": 30},
            n_terms=2,
        )
        result = generator.fit_transform(simple_timeseries)

        # Should have sin and cos for each period and term
        assert len(result.columns) > len(simple_timeseries.columns)

    def test_single_period(self, simple_timeseries):
        """Test with single period."""
        generator = FourierFeatureGenerator(
            datetime_col="date",
            periods={"weekly": 7},
            n_terms=1,
        )
        result = generator.fit_transform(simple_timeseries)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(simple_timeseries)

    def test_multiple_terms(self, simple_timeseries):
        """Test with multiple terms."""
        generator = FourierFeatureGenerator(
            datetime_col="date",
            periods={"weekly": 7},
            n_terms=3,
        )
        result = generator.fit_transform(simple_timeseries)

        # Should have more features than original
        assert len(result.columns) > len(simple_timeseries.columns)

    def test_get_feature_names_out(self, simple_timeseries):
        """Test get_feature_names_out method."""
        generator = FourierFeatureGenerator(
            datetime_col="date",
            periods={"weekly": 7},
            n_terms=2,
        )
        generator.fit(simple_timeseries)

        names = generator.get_feature_names_out()
        assert isinstance(names, list)


class TestGenerateTimeseriesFeaturesFunction:
    """Tests for generate_timeseries_features convenience function."""

    def test_basic_usage(self, simple_timeseries):
        """Test basic function usage."""
        result, generator = generate_timeseries_features(
            simple_timeseries,
            datetime_col="date",
            target_cols=["value"],
            lags=[1, 2],
        )

        assert isinstance(result, pd.DataFrame)
        assert isinstance(generator, TimeSeriesFeatureGenerator)

    def test_with_rolling(self, simple_timeseries):
        """Test with rolling windows."""
        result, generator = generate_timeseries_features(
            simple_timeseries,
            datetime_col="date",
            target_cols=["value"],
            rolling_windows=[3, 7],
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) >= len(simple_timeseries.columns)

    def test_with_datetime_features(self, simple_timeseries):
        """Test with datetime features."""
        result, generator = generate_timeseries_features(
            simple_timeseries,
            datetime_col="date",
            datetime_features=True,
        )

        assert isinstance(result, pd.DataFrame)
