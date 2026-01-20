"""Population Stability Index (PSI) and drift detection.

This module provides tools for detecting feature drift between
training and production data using the Population Stability Index (PSI)
and other statistical measures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

if TYPE_CHECKING:
    import pandas as pd
    from typing_extensions import Self


@dataclass
class DriftReport:
    """Report of drift detection results.

    Attributes
    ----------
    feature_name : str
        Name of the feature analyzed.
    psi_value : float
        Population Stability Index value.
    drift_level : str
        Drift severity: 'none', 'minor', 'moderate', or 'significant'.
    bins : int
        Number of bins used in calculation.
    expected_percentages : np.ndarray
        Expected (baseline) distribution percentages.
    actual_percentages : np.ndarray
        Actual (current) distribution percentages.
    recommendation : str
        Suggested action based on drift level.
    """

    feature_name: str
    psi_value: float
    drift_level: str
    bins: int
    expected_percentages: np.ndarray
    actual_percentages: np.ndarray
    recommendation: str

    def __str__(self) -> str:
        """Return human-readable summary."""
        status_icons = {
            "none": "✓",
            "minor": "⚠",
            "moderate": "⚠⚠",
            "significant": "🚨",
        }
        icon = status_icons.get(self.drift_level, "?")
        return (
            f"{icon} {self.feature_name}: PSI = {self.psi_value:.4f} "
            f"({self.drift_level} drift)\n"
            f"   {self.recommendation}"
        )


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
    strategy: Literal["quantile", "uniform"] = "quantile",
    epsilon: float = 1e-6
) -> float:
    """Calculate Population Stability Index (PSI) between two distributions.

    PSI measures how much a distribution has shifted from a baseline.
    It's commonly used in credit scoring and ML monitoring.

    Interpretation:
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.2: Minor change, monitor closely
    - 0.2 <= PSI < 0.25: Moderate change, investigate
    - PSI >= 0.25: Significant change, action required

    Parameters
    ----------
    expected : np.ndarray
        Baseline (training) distribution values.
    actual : np.ndarray
        Current (production) distribution values.
    bins : int
        Number of bins for discretization.
    strategy : {'quantile', 'uniform'}
        Binning strategy. 'quantile' creates equal-frequency bins,
        'uniform' creates equal-width bins.
    epsilon : float
        Small value to avoid division by zero and log(0).

    Returns
    -------
    float
        PSI value.

    Examples
    --------
    >>> import numpy as np
    >>> expected = np.random.normal(0, 1, 10000)
    >>> actual = np.random.normal(0.5, 1, 10000)  # Shifted distribution
    >>> psi = calculate_psi(expected, actual)
    >>> print(f"PSI: {psi:.4f}")
    PSI: 0.1234

    Notes
    -----
    PSI formula: sum((actual_% - expected_%) * ln(actual_% / expected_%))
    """
    expected = np.asarray(expected).flatten()
    actual = np.asarray(actual).flatten()

    # Remove NaN values
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    # Determine bin edges based on expected distribution
    if strategy == "quantile":
        percentiles = np.linspace(0, 100, bins + 1)
        bin_edges = np.percentile(expected, percentiles)
        # Ensure unique bin edges
        bin_edges = np.unique(bin_edges)
    else:  # uniform,
        bin_edges = np.linspace(expected.min(), expected.max(), bins + 1)

    # Calculate bin counts
    expected_counts = np.histogram(expected, bins=bin_edges)[0]
    actual_counts = np.histogram(actual, bins=bin_edges)[0]

    # Convert to percentages
    expected_pct = expected_counts / len(expected) + epsilon
    actual_pct = actual_counts / len(actual) + epsilon

    # Calculate PSI
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

    return float(psi)


def interpret_psi(psi_value: float) -> tuple[str, str]:
    """Interpret PSI value and provide recommendation.

    Parameters
    ----------
    psi_value : float
        PSI value to interpret.

    Returns
    -------
    tuple[str, str]
        Drift level and recommendation.
    """
    if np.isnan(psi_value):
        return "unknown", "Unable to calculate PSI (missing data)."

    if psi_value < 0.1:
        return "none", "No action required. Distribution is stable."
    elif psi_value < 0.2:
        return "minor", "Monitor closely. Consider investigating if trend continues."
    elif psi_value < 0.25:
        return "moderate", "Investigate the cause. Model retraining may be needed."
    else:
        return "significant", "Action required! Significant drift detected. Retrain model."


class PSICalculator(BaseEstimator, TransformerMixin):
    """Calculate PSI for features in a DataFrame.

    This transformer fits on baseline data and computes PSI when
    transforming new data, enabling drift monitoring in pipelines.

    Parameters
    ----------
    bins : int
        Number of bins for discretization.
    strategy : {'quantile', 'uniform'}
        Binning strategy.
    columns : list[str] | None
        Columns to monitor. If None, monitors all numeric columns.
    threshold : float
        PSI threshold for flagging drift (default 0.2).

    Attributes
    ----------
    baseline_distributions_ : dict
        Stored baseline distributions for each column.
    feature_names_in_ : list[str]
        Names of features seen during fit.
    psi_values_ : dict
        Most recent PSI values (populated after transform).
    drift_reports_ : dict
        Detailed drift reports (populated after transform).

    Examples
    --------
    >>> from forge.monitoring import PSICalculator
    >>> import pandas as pd
    >>>
    >>> # Fit on training data
    >>> calculator = PSICalculator(threshold=0.2)
    >>> calculator.fit(X_train)
    >>>
    >>> # Check drift on new data
    >>> X_transformed = calculator.transform(X_prod)
    >>>
    >>> # Get drift reports
    >>> for col, report in calculator.drift_reports_.items():
    ...     if report.drift_level != 'none':
    ...         print(report)
    """

    def __init__(
        self,
        bins: int = 10,
        strategy: Literal["quantile", "uniform"] = "quantile",
        columns: list[str] | None = None,
        threshold: float = 0.2
    ) -> None:
        self.bins = bins
        self.strategy = strategy
        self.columns = columns
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        """Fit the calculator on baseline data.

        Parameters
        ----------
        X : pd.DataFrame
            Baseline (training) data.
        y : Any
            Ignored. Present for sklearn compatibility.

        Returns
        -------
        Self
            Fitted calculator.
        """
        import pandas as pd

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Determine columns to monitor
        if self.columns is not None:
            self.feature_names_in_ = list(self.columns)
        else:
            self.feature_names_in_ = list(X.select_dtypes(include=[np.number]).columns)

        # Store baseline distributions
        self.baseline_distributions_: dict[str, np.ndarray] = {}
        for col in self.feature_names_in_:
            if col in X.columns:
                self.baseline_distributions_[col] = X[col].dropna().values

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data and calculate PSI.

        Parameters
        ----------
        X : pd.DataFrame
            New data to check for drift.

        Returns
        -------
        pd.DataFrame
            Input data unchanged (PSI values stored in attributes).
        """
        import pandas as pd

        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise RuntimeError("PSICalculator must be fitted before transform.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.psi_values_: dict[str, float] = {}
        self.drift_reports_: dict[str, DriftReport] = {}

        for col in self.feature_names_in_:
            if col not in X.columns or col not in self.baseline_distributions_:
                continue

            expected = self.baseline_distributions_[col]
            actual = X[col].dropna().values

            psi = calculate_psi(
                expected,
                actual,
                bins = self.bins,
                strategy=self.strategy
            )

            self.psi_values_[col] = psi
            drift_level, recommendation = interpret_psi(psi)

            # Calculate percentages for report
            if self.strategy == "quantile":
                percentiles = np.linspace(0, 100, self.bins + 1)
                bin_edges = np.percentile(expected, percentiles)
                bin_edges = np.unique(bin_edges)
            else:
                bin_edges = np.linspace(expected.min(), expected.max(), self.bins + 1)

            expected_pct = np.histogram(expected, bins=bin_edges)[0] / len(expected)
            actual_pct = np.histogram(actual, bins=bin_edges)[0] / max(len(actual), 1)

            self.drift_reports_[col] = DriftReport(
                feature_name = col,
                psi_value = psi,
                drift_level = drift_level,
                bins = len(bin_edges) - 1,
                expected_percentages = expected_pct,
                actual_percentages = actual_pct,
                recommendation=recommendation
            )

        return X

    def get_drift_summary(self) -> dict[str, Any]:
        """Get summary of drift detection results.

        Returns
        -------
        dict
            Summary with counts by drift level and flagged features.
        """
        if not hasattr(self, "drift_reports_"):
            return {"error": "No drift analysis performed yet. Call transform() first."}

        summary = {
            "total_features": len(self.drift_reports_),
            "none": 0,
            "minor": 0,
            "moderate": 0,
            "significant": 0,
            "flagged_features": [],
            "average_psi": 0.0,
        }

        psi_values = []
        for col, report in self.drift_reports_.items():
            summary[report.drift_level] += 1
            psi_values.append(report.psi_value)
            if report.psi_value >= self.threshold:
                summary["flagged_features"].append(col)

        summary["average_psi"] = np.mean(psi_values) if psi_values else 0.0
        return summary

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Get output feature names.

        Parameters
        ----------
        input_features : list[str] | None
            Input feature names (ignored, uses fitted names).

        Returns
        -------
        list[str]
            Feature names being monitored.
        """
        return self.feature_names_in_


class DriftDetector(BaseEstimator):
    """Comprehensive drift detector using multiple methods.

    This class provides a unified interface for detecting various types
    of data drift including distribution shift, concept drift, and
    covariate shift.

    Parameters
    ----------
    methods : list[str]
        Detection methods to use. Options: ['psi', 'ks', 'chi2', 'js'].
        Default is ['psi'].
    threshold : float
        Threshold for flagging drift.
    columns : list[str] | None
        Columns to monitor. If None, monitors all numeric columns.

    Attributes
    ----------
    baseline_data_ : pd.DataFrame
        Stored baseline data.
    drift_results_ : dict
        Results from the most recent drift check.

    Examples
    --------
    >>> from forge.monitoring import DriftDetector
    >>>
    >>> detector = DriftDetector(methods=['psi', 'ks'])
    >>> detector.fit(X_train)
    >>>
    >>> # Check drift
    >>> drift_detected = detector.check_drift(X_new)
    >>> if drift_detected:
    ...     print("Drift detected!")
    ...     print(detector.get_report())
    """

    def __init__(
        self,
        methods: list[str] | None = None,
        threshold: float = 0.2,
        columns: list[str] | None = None
    ) -> None:
        self.methods = methods or ["psi"]
        self.threshold = threshold
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        """Fit the detector on baseline data.

        Parameters
        ----------
        X : pd.DataFrame
            Baseline data.
        y : Any
            Ignored.

        Returns
        -------
        Self
            Fitted detector.
        """
        import pandas as pd

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.baseline_data_ = X.copy()

        if self.columns is not None:
            self.feature_names_in_ = list(self.columns)
        else:
            self.feature_names_in_ = list(X.select_dtypes(include=[np.number]).columns)

        self._is_fitted = True
        return self

    def check_drift(self, X: pd.DataFrame) -> bool:
        """Check for drift in new data.

        Parameters
        ----------
        X : pd.DataFrame
            New data to check.

        Returns
        -------
        bool
            True if significant drift is detected.
        """
        import pandas as pd

        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise RuntimeError("DriftDetector must be fitted before check_drift.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.drift_results_ = {}

        for col in self.feature_names_in_:
            if col not in X.columns or col not in self.baseline_data_.columns:
                continue

            col_results: dict[str, float] = {}
            baseline = self.baseline_data_[col].dropna().values
            current = X[col].dropna().values

            if "psi" in self.methods:
                col_results["psi"] = calculate_psi(baseline, current)

            if "ks" in self.methods:
                from scipy.stats import ks_2samp

                statistic, p_value = ks_2samp(baseline, current)
                col_results["ks_statistic"] = statistic
                col_results["ks_pvalue"] = p_value

            if "js" in self.methods:
                col_results["js_divergence"] = self._calculate_js(baseline, current)

            self.drift_results_[col] = col_results

        # Check if any feature exceeds threshold
        return self._has_significant_drift()

    def _calculate_js(
        self, baseline: np.ndarray, current: np.ndarray, bins: int = 10
    ) -> float:
        """Calculate Jensen-Shannon divergence."""
        # Create histograms
        min_val = min(baseline.min(), current.min())
        max_val = max(baseline.max(), current.max())
        bin_edges = np.linspace(min_val, max_val, bins + 1)

        p = np.histogram(baseline, bins=bin_edges, density=True)[0] + 1e-10
        q = np.histogram(current, bins=bin_edges, density=True)[0] + 1e-10

        # Normalize
        p = p / p.sum()
        q = q / q.sum()

        # JS divergence = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = (P+Q)/2
        m = 0.5 * (p + q)
        js = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))

        return float(js)

    def _has_significant_drift(self) -> bool:
        """Check if any feature has significant drift."""
        for col, results in self.drift_results_.items():
            if "psi" in results and results["psi"] >= self.threshold:
                return True
            if "ks_pvalue" in results and results["ks_pvalue"] < 0.05:
                return True
            if "js_divergence" in results and results["js_divergence"] >= 0.1:
                return True
        return False

    def get_report(self) -> dict[str, Any]:
        """Get detailed drift report.

        Returns
        -------
        dict
            Detailed drift analysis results.
        """
        if not hasattr(self, "drift_results_"):
            return {"error": "No drift check performed yet."}

        report = {
            "features_analyzed": len(self.drift_results_),
            "drift_detected": self._has_significant_drift(),
            "threshold": self.threshold,
            "feature_results": self.drift_results_,
            "flagged_features": [],
        }

        for col, results in self.drift_results_.items():
            if "psi" in results and results["psi"] >= self.threshold:
                report["flagged_features"].append(
                    {"feature": col, "method": "psi", "value": results["psi"]}
                )
            if "ks_pvalue" in results and results["ks_pvalue"] < 0.05:
                report["flagged_features"].append(
                    {"feature": col, "method": "ks", "p_value": results["ks_pvalue"]}
                )

        return report
