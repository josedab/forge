"""Tests for monitoring and drift detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.monitoring import (
    DriftDetector,
    DriftReport,
    PSICalculator,
    calculate_psi,
)


class TestCalculatePSI:
    """Tests for calculate_psi function."""

    def test_identical_distributions(self):
        """Test PSI is ~0 for identical distributions."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 10000)

        psi = calculate_psi(data, data)
        assert psi < 0.01  # Should be very close to 0

    def test_similar_distributions(self):
        """Test PSI is low for similar distributions."""
        np.random.seed(42)
        expected = np.random.normal(0, 1, 10000)
        actual = np.random.normal(0.05, 1, 10000)  # Slight shift

        psi = calculate_psi(expected, actual)
        assert psi < 0.1  # Should indicate no significant change

    def test_shifted_distribution(self):
        """Test PSI detects shifted distribution."""
        np.random.seed(42)
        expected = np.random.normal(0, 1, 10000)
        actual = np.random.normal(1, 1, 10000)  # Significant shift

        psi = calculate_psi(expected, actual)
        assert psi > 0.1  # Should indicate change

    def test_different_variance(self):
        """Test PSI detects different variance."""
        np.random.seed(42)
        expected = np.random.normal(0, 1, 10000)
        actual = np.random.normal(0, 2, 10000)  # Different variance

        psi = calculate_psi(expected, actual)
        assert psi > 0.05  # Should detect the change

    def test_handles_nan(self):
        """Test that NaN values are handled."""
        np.random.seed(42)
        expected = np.array([1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10] * 100)
        actual = np.array([1, 2, 3, 4, 5, np.nan, 7, 8, 9, 10] * 100)

        psi = calculate_psi(expected, actual)
        assert not np.isnan(psi)

    def test_uniform_strategy(self):
        """Test uniform binning strategy."""
        np.random.seed(42)
        expected = np.random.uniform(0, 10, 10000)
        actual = np.random.uniform(0, 10, 10000)

        psi = calculate_psi(expected, actual, strategy="uniform")
        assert psi < 0.1

    def test_custom_bins(self):
        """Test with custom number of bins."""
        np.random.seed(42)
        expected = np.random.normal(0, 1, 10000)
        actual = np.random.normal(0.5, 1, 10000)

        psi_5 = calculate_psi(expected, actual, bins=5)
        psi_20 = calculate_psi(expected, actual, bins=20)

        # Both should detect drift, but values may differ
        assert psi_5 > 0
        assert psi_20 > 0


class TestDriftReport:
    """Tests for DriftReport dataclass."""

    def test_str_representation(self):
        """Test string representation."""
        report = DriftReport(
            feature_name="test_col",
            psi_value=0.15,
            drift_level="minor",
            bins=10,
            expected_percentages=np.array([0.1] * 10),
            actual_percentages=np.array([0.1] * 10),
            recommendation="Monitor closely.",
        )

        string = str(report)
        assert "test_col" in string
        assert "0.15" in string
        assert "minor" in string

    def test_drift_levels(self):
        """Test all drift levels have icons."""
        for level in ["none", "minor", "moderate", "significant"]:
            report = DriftReport(
                feature_name="test",
                psi_value=0.1,
                drift_level=level,
                bins=10,
                expected_percentages=np.array([]),
                actual_percentages=np.array([]),
                recommendation="Test",
            )
            string = str(report)
            assert len(string) > 0


class TestPSICalculator:
    """Tests for PSICalculator class."""

    def test_fit_transform(self, sample_numeric_df: pd.DataFrame):
        """Test basic fit and transform."""
        calculator = PSICalculator()
        calculator.fit(sample_numeric_df)
        result = calculator.transform(sample_numeric_df)

        # Should return input unchanged
        assert len(result) == len(sample_numeric_df)

        # Should have PSI values
        assert hasattr(calculator, "psi_values_")
        assert len(calculator.psi_values_) > 0

    def test_detects_drift(self):
        """Test that drift is detected when distributions change."""
        np.random.seed(42)
        X_train = pd.DataFrame({
            "stable": np.random.normal(0, 1, 1000),
            "drifting": np.random.normal(0, 1, 1000),
        })

        X_prod = pd.DataFrame({
            "stable": np.random.normal(0, 1, 1000),
            "drifting": np.random.normal(2, 1, 1000),  # Shifted
        })

        calculator = PSICalculator(threshold=0.2)
        calculator.fit(X_train)
        calculator.transform(X_prod)

        # Drifting column should have higher PSI
        assert calculator.psi_values_["drifting"] > calculator.psi_values_["stable"]

    def test_get_drift_summary(self, sample_numeric_df: pd.DataFrame):
        """Test drift summary."""
        calculator = PSICalculator()
        calculator.fit(sample_numeric_df)
        calculator.transform(sample_numeric_df)

        summary = calculator.get_drift_summary()

        assert "total_features" in summary
        assert "none" in summary
        assert "flagged_features" in summary
        assert "average_psi" in summary

    def test_specific_columns(self, sample_numeric_df: pd.DataFrame):
        """Test monitoring specific columns only."""
        calculator = PSICalculator(columns=["age", "income"])
        calculator.fit(sample_numeric_df)
        calculator.transform(sample_numeric_df)

        assert len(calculator.feature_names_in_) == 2
        assert "age" in calculator.feature_names_in_
        assert "income" in calculator.feature_names_in_

    def test_threshold_parameter(self):
        """Test threshold parameter."""
        np.random.seed(42)
        X_train = pd.DataFrame({"col": np.random.normal(0, 1, 1000)})
        X_prod = pd.DataFrame({"col": np.random.normal(0.5, 1, 1000)})

        calculator_strict = PSICalculator(threshold=0.05)
        calculator_loose = PSICalculator(threshold=0.5)

        calculator_strict.fit(X_train)
        calculator_loose.fit(X_train)

        calculator_strict.transform(X_prod)
        calculator_loose.transform(X_prod)

        summary_strict = calculator_strict.get_drift_summary()
        summary_loose = calculator_loose.get_drift_summary()

        # Strict threshold should flag more
        assert len(summary_strict["flagged_features"]) >= len(summary_loose["flagged_features"])

    def test_get_feature_names_out(self, sample_numeric_df: pd.DataFrame):
        """Test feature names output."""
        calculator = PSICalculator()
        calculator.fit(sample_numeric_df)

        names = calculator.get_feature_names_out()
        assert len(names) > 0

    def test_error_before_fit(self, sample_numeric_df: pd.DataFrame):
        """Test error when transform called before fit."""
        calculator = PSICalculator()

        with pytest.raises(RuntimeError):
            calculator.transform(sample_numeric_df)


class TestDriftDetector:
    """Tests for DriftDetector class."""

    def test_basic_drift_detection(self, sample_numeric_df: pd.DataFrame):
        """Test basic drift detection."""
        detector = DriftDetector()
        detector.fit(sample_numeric_df)

        # Same data should not have drift
        has_drift = detector.check_drift(sample_numeric_df)
        assert has_drift is False

    def test_detects_drift_with_shift(self):
        """Test drift detection with shifted data."""
        np.random.seed(42)
        X_train = pd.DataFrame({"col": np.random.normal(0, 1, 1000)})
        X_prod = pd.DataFrame({"col": np.random.normal(2, 1, 1000)})  # Shifted

        detector = DriftDetector(threshold=0.2)
        detector.fit(X_train)

        has_drift = detector.check_drift(X_prod)
        assert has_drift is True

    def test_multiple_methods(self):
        """Test with multiple detection methods."""
        np.random.seed(42)
        X_train = pd.DataFrame({"col": np.random.normal(0, 1, 1000)})
        X_prod = pd.DataFrame({"col": np.random.normal(0.5, 1, 1000)})

        detector = DriftDetector(methods=["psi", "ks", "js"])
        detector.fit(X_train)
        detector.check_drift(X_prod)

        report = detector.get_report()
        feature_results = report["feature_results"]["col"]

        assert "psi" in feature_results
        assert "ks_statistic" in feature_results
        assert "js_divergence" in feature_results

    def test_get_report(self, sample_numeric_df: pd.DataFrame):
        """Test detailed report."""
        detector = DriftDetector()
        detector.fit(sample_numeric_df)
        detector.check_drift(sample_numeric_df)

        report = detector.get_report()

        assert "features_analyzed" in report
        assert "drift_detected" in report
        assert "threshold" in report
        assert "feature_results" in report
        assert "flagged_features" in report

    def test_specific_columns(self, sample_numeric_df: pd.DataFrame):
        """Test monitoring specific columns."""
        detector = DriftDetector(columns=["age"])
        detector.fit(sample_numeric_df)
        detector.check_drift(sample_numeric_df)

        report = detector.get_report()
        assert report["features_analyzed"] == 1

    def test_error_before_fit(self, sample_numeric_df: pd.DataFrame):
        """Test error when check_drift called before fit."""
        detector = DriftDetector()

        with pytest.raises(RuntimeError):
            detector.check_drift(sample_numeric_df)


class TestSklearnCompatibility:
    """Test sklearn compatibility for monitoring classes."""

    def test_psi_calculator_clone(self):
        """Test that PSICalculator can be cloned."""
        from sklearn.base import clone

        calculator = PSICalculator(bins=20, threshold=0.15)
        cloned = clone(calculator)

        assert cloned.bins == 20
        assert cloned.threshold == 0.15

    def test_drift_detector_clone(self):
        """Test that DriftDetector can be cloned."""
        from sklearn.base import clone

        detector = DriftDetector(threshold=0.25, methods=["psi", "ks"])
        cloned = clone(detector)

        assert cloned.threshold == 0.25
        assert cloned.methods == ["psi", "ks"]
